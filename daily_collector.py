# -*- coding: utf-8 -*-
"""
daily_collector.py — сбор статей за сутки (или за неделю, см. --week ниже)
из дампов GDELT GKG, отсев мусора (короткие/нерелевантные/не-новостные
тексты) и запись в БД с embedding-вектором и региональной меткой.

Запускается по cron каждые 6 часов в режиме "day" (см. crontab).
Режим "week"/"backfill" (--week или --backfill в argv) используется
только для ручного бэкафилла на более широком окне GKG-дампов.

Пайплайн одного query_index (см. config.QUERIES_GKG):
    0. seen_store          — очередь pending разбирается ДО новых дампов, а
                             URL с окончательным вердиктом (accepted/rejected/
                             short/non_event) повторно не скачивается. Раньше
                             проверка шла по таблице articles, где лежат только
                             ПРИНЯТЫЕ статьи, и 61.7% работы прогона уходило
                             на повторную обработку уже отклонённых URL.
    1. fetch_gdelt()      — сканирует 15-минутные дампы GKG (EN + переводной
                             поток); gkg_filter отбирает строки по СВЯЗИ темы
                             и страны через символьные офсеты полей V2.
    2. fetch_and_extract() — скачивает страницу, извлекает текст/заголовок
                             (trafilatura) и дату публикации (htmldate).
    3. is_non_event_article() — быстрый regex-отсев не-новостных материалов
                             (интервью/колонки/мнения на нескольких языках).
    4. embed_texts()       — embedding считается ОДИН раз на кандидата и
                             переиспользуется на стадиях 5, 6 и 8.
    5. cluster_duplicates() — синдикация одного сюжета схлопывается в кластер,
                             в LLM уходит один представитель; в БД пишутся все.
    6. prefilter           — локальный классификатор-дистиллят отбрасывает
                             заведомый мусор без запроса к LLM (если обучен).
    7. judge_parallel()    — ОДИН LLM-вызов отдаёт релевантность, регион
                             (TR/CA/SC/MIX) и масштаб события (1..3) одной
                             строкой. Нет ответа -> pending, не отказ.
    8. flush_batch()       — пишет принятые статьи с готовым embedding в БД.
"""
import os
os.environ["ATEN_CPU_CAPABILITY"] = "default"
os.environ["OMP_NUM_THREADS"] = "1"

import io
import json
import logging
import re
import sqlite3
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
from htmldate import find_date
from langdetect import DetectorFactory, LangDetectException, detect
from trafilatura import bare_extraction

import archive_fetch
import config
import gkg_filter
import prefilter
import seen_store
from model_rotation import _call_openrouter_raw

os.environ.setdefault("HF_HOME", "/opt/digest/.cache/huggingface")
# Модель лежит в кэше целиком, ходить за ней в сеть незачем. Без этого каждый
# прогон делает ~20 HEAD-запросов к huggingface.co: лишняя задержка и жёсткая
DetectorFactory.seed = 0  # детерминированный langdetect

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
_model = None                # embedding-модель, лениво грузится в get_model()
BATCH_SIZE = 16               # размер батча для embedding и вставки в БД
LLM_FILTER_BATCH = 10         # статей в одном LLM-запросе на релевантность (было 20 —
                               # уменьшено, чтобы модели было проще держать в фокусе
                               # каждую статью и не путать индексы в батче)
GKG_FETCH_DELAY = 0.5         # пауза между запросами GKG-дампов, сек

GKG_URL_EN = "http://data.gdeltproject.org/gdeltv2/{ts}.gkg.csv.zip"
GKG_URL_TL = "http://data.gdeltproject.org/gdeltv2/{ts}.translation.gkg.csv.zip"
# Колонки 8 и 10 (V2EnhancedThemes / V2EnhancedLocations) несут символьные
# офсеты каждого упоминания — по ним gkg_filter восстанавливает связь темы и
# страны, которой в V1-полях (7 и 9) физически нет.
# Колонки 11 и 13 (V1Persons / V1Organizations) — NER самого GDELT, наш
# «словарь политических субъектов» против локального шума (см. entities.py).
# Порядок обязан быть возрастающим: pandas раздаёт names по позиции выбранных
# колонок в файле.
GKG_USECOLS = [1, 3, 4, 7, 8, 9, 10, 11, 13]
GKG_COLNAMES = ["date", "source", "url", "v1t", "v2t", "v1l", "v2l", "v1p", "v1o"]

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept-Language": "en,ru,tr,az,kk,uz,ky,tg,tk,hy,ka,fa,ar;q=0.9,*;q=0.5",
}


def setup_logging():
    os.makedirs(config.LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(config.LOG_DIR, "collector.log"), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("collector")


log = setup_logging()


def _is_week_mode() -> bool:
    return "--week" in sys.argv or "--backfill" in sys.argv


def _run_mode_label() -> str:
    return "week" if _is_week_mode() else "day"


# ---------------------------------------------------------------------------
# LLM: один вызов = релевантность + региональная метка.
#
# Раньше это были две стадии по двум пулам моделей: булев фильтр, затем
# классификация TR/CA/SC/MIX. Вторая тратила ещё ~26 запросов на прогон, хотя
# судит тот же текст по тем же критериям. Теперь один вызов отдаёт либо "NO",
# либо код региона.
#
# Список отказов намеренно перечисляет конкретные жанры, а не просит «оценить
# значимость». Прежняя формулировка «purely local/municipal/ceremonial» не
# ловила самый частый класс шума: рутину ОДНОГО заведения, у которого актор
# при этом национальный (министерство, госкомиссия). Проходной балл вуза,
# график приёма, проводы школьной команды на олимпиаду — по формальным
# признакам это не «местная новость», и судья говорил да. В тонких корзинах
# CA/SC такие заметки потом занимали половину квоты: ранжирование их не
# отсеивает — все четыре фактора важности меряют тематичность и охват, а не
# масштаб события, и фактор topic даже поднимает их наверх, потому что текст
# приёмной кампании — самый «канонично образовательный» в корзине.
#
# Формат ответа — ОБЪЕКТ {"1":"CA","2":"NO"}, а не позиционный массив. Массив
# ломался незаметно: модель теряла или добавляла элемент, индексы съезжали, и
# вердикт доставался чужой статье. С номерами-ключами съехать нельзя, а
# недостающий номер виден явно и уводит статью в pending, а не в отказ.
# ---------------------------------------------------------------------------
_JUDGE_SYSTEM = (
    "You are a strict relevance filter and region classifier for a research "
    "digest published by the Institute of Oriental Studies (Russian Academy of "
    "Sciences). The digest covers ONLY two kinds of news:\n"
    "\n"
    "T1 — Science, education, or youth policy as the MAIN SUBJECT of the "
    "article, happening IN Turkey, Central Asia (Kazakhstan, Uzbekistan, "
    "Turkmenistan, Kyrgyzstan, Tajikistan), or the South Caucasus (Georgia, "
    "Armenia, Azerbaijan).\n"
    "\n"
    "T2 — A concrete science/education/youth action, program, agreement, or "
    "institution — funded, launched, or run by the EU, USA, Iran, India, China, "
    "Japan, South Korea, or Turkey — that is PHYSICALLY TAKING PLACE IN Central "
    "Asia or the South Caucasus (e.g. a foreign-funded university campus opening "
    "in Uzbekistan, a Chinese scholarship program for Kazakh students, a Turkish "
    "school built in Kyrgyzstan). A diplomatic visit, trade deal, or general "
    "foreign-policy statement with no science/education/youth substance does "
    "NOT count, even if it involves these same countries.\n"
    "\n"
    "A relevant article has a national actor or a national effect: a law or a "
    "reform, a state programme or its funding, a new university, campus, "
    "laboratory or research centre, published research results, an "
    "intergovernmental agreement, national statistics, or a ranking or "
    "assessment covering a country's universities as a whole. Such an article "
    "stays IN even if it also mentions exams, enrolment or a ceremony.\n"
    "\n"
    "Answer NO for an article if ANY of these apply:\n"
    "  - the topic is only mentioned in passing — not what the article is "
    "actually about;\n"
    "  - it reports the ROUTINE OPERATION of a single school, university, "
    "faculty or agency instead of a change of national scale: admission score "
    "thresholds, enrolment and application schedules, exam-result "
    "announcements, tuition or benefit terms, one institution's own curricula "
    "and course offerings, staff appointments, anniversaries, graduations, "
    "open days, seminars, trainings, courses, bootcamps and summer camps, or a "
    "memorandum with nothing concrete behind it;\n"
    "  - it is about pupils or students as individuals — a team leaving for or "
    "preparing for an olympiad or a contest, personal prizes, medals and "
    "congratulations, a scholarship awarded to named people;\n"
    "  - the news is local/municipal/ceremonial, a minor award, a corporate "
    "sponsorship or a press release with no policy or research substance;\n"
    "  - it's an opinion piece, interview, or analysis rather than a report of "
    "an actual event.\n"
    "\n"
    "For every article that is NOT rejected, assign exactly ONE region:\n"
    "  TR  - primarily about Turkey, including domestic Turkish "
    "science/education/STEM/youth developments;\n"
    "  CA  - primarily about Central Asia (KZ/UZ/TM/KG/TJ);\n"
    "  SC  - primarily about South Caucasus (GE/AM/AZ);\n"
    "  MIX - relevant, but the region is unclear.\n"
    "If Turkey is the main actor or location, choose TR.\n"
    "\n"
    "Right after the region, append ONE digit — how much this event CHANGES the "
    "country's science, education or youth system:\n"
    "  3 - the country as a whole is different afterwards: a law or reform "
    "adopted, a national programme launched or funded, an intergovernmental "
    "agreement signed, a new university, national laboratory or research centre "
    "opened, national statistics or a country-wide ranking published, a research "
    "result of national or international significance;\n"
    "  2 - a real change inside ONE institution or ONE sector, or an official "
    "announcement that such a change is being prepared: a large grant, a new "
    "faculty or department, a partnership with concrete money or obligations "
    "behind it, an announced revision of an exam or a curriculum;\n"
    "  1 - nothing in the system changes: a meeting, a visit, a forum, a "
    "competition or its results, a corporate or charitable initiative, and any "
    "seasonal or annual routine — enrolment, school uniforms, timetables, "
    "scholarship lists, grant winners of a regular contest.\n"
    "Be strict. In a batch of ten articles usually no more than two deserve 3, "
    "and 1 is the most common answer. A ministry, a minister or a national "
    "company as the ACTOR does not by itself make the event national, and "
    "neither does the fact that it takes place across the whole country: ask "
    "what would be DIFFERENT in the country if this news had not happened. "
    "\"NO\" carries no digit.\n"
    "\n"
    "The article may be in any language — judge by content, not language."
)

_JUDGE_USER_TMPL = (
    "Below are {n} articles, each preceded by its number in square brackets.\n"
    "Reply ONLY with a JSON object mapping every article number to its verdict, "
    'no extra text — e.g. {{"1": "CA2", "2": "NO", "3": "TR3"}}.\n'
    'Allowed values: "NO", or one of "TR", "CA", "SC", "MIX" immediately '
    "followed by the scale digit 1, 2 or 3. "
    "Every number from 1 to {n} must appear exactly once.\n\n"
    "{articles}"
)

JUDGE_TEXT_CHARS = 2000   # сколько символов статьи уходит в промпт
JUDGE_BATCH = LLM_FILTER_BATCH

# Маркер «решил предфильтр, а не LLM». Отдельный объект, а не строка "NO",
# чтобы вердикт нельзя было спутать с ответом судьи ни здесь, ни в журнале.
_PREFILTERED = object()
_VERDICTS = ("NO", "TR", "CA", "SC", "MIX")
# Регион и цифра масштаба приезжают одним значением ("TR3"): второй вызов LLM
# ради масштаба стоил бы ещё один суточный лимит, а судья статью уже прочитал.
# Цифра необязательна — старый ответ без неё разбирается как раньше.
#
# Рубрика масштаба калибрована замером 08.08.2026, а не написана на глаз.
# Первая редакция («насколько событие национально важно» + перечень признаков)
# дала 45 троек из 57 оценённых: судья читал «министерство» и «по всей стране»
# как масштаб сам по себе, и в тройки попадали продажа школьной формы и
# корпоративная программа Turkcell, а «меняются задания национального экзамена
# YKS» получало двойку. Помогли три добавки: вопрос «что было бы в стране
# ИНАЧЕ, если бы этой новости не было», явный список рутины в пункте 1 и
# указание доли (не больше двух троек на десяток). На той же выборке стало
# 8/11/12 при 1/2/3 — и переехало именно то, что должно было.
_VERDICT_RE = re.compile(r"(NO|TR|CA|SC|MIX)\s*([1-3])?$")


def _parse_judge(content, n):
    """-> список пар (регион, масштаб) длины n; None на позиции = решения нет.

    Масштаб — 1..3 или None, если модель цифру не поставила (масштаб тогда
    считается средним, см. sunday_processor_mmr._story_scale)."""
    m = re.search(r"\{.*\}", content or "", re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    out = []
    for i in range(1, n + 1):
        v = data.get(str(i), data.get(i))
        m = _VERDICT_RE.match(str(v).strip().upper()) if v is not None else None
        out.append((m.group(1), int(m.group(2)) if m.group(2) else None)
                   if m else None)
    return out


def _judge_call(texts):
    block = "".join(
        f"[{i}] {t.replace(chr(10), ' ')[:JUDGE_TEXT_CHARS]}\n\n"
        for i, t in enumerate(texts, 1))
    raw = _call_openrouter_raw(
        _JUDGE_SYSTEM,
        _JUDGE_USER_TMPL.format(n=len(texts), articles=block),
        ref_url="judge")
    if raw is None:
        return None
    return _parse_judge(raw, len(texts))


def judge_parallel(texts):
    """-> список вердиктов той же длины и порядка.

    None на позиции означает "решения нет" и ведёт в pending, а НЕ в отказ.
    Прежний код в этом месте возвращал [False]*len(chunk) и не логировал
    ничего: прогон week_swap 21.07 отклонил 6006 статей из 6006 на
    исчерпанных лимитах, и в логе не осталось ни следа."""
    chunks = [texts[i:i + JUDGE_BATCH] for i in range(0, len(texts), JUDGE_BATCH)]
    if not chunks:
        return []
    results = [None] * len(chunks)
    workers = min(len(chunks), config.LLM_PARALLEL_BATCHES)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_judge_call, c): i for i, c in enumerate(chunks)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:
                log.warning("Батч %d упал: %s", i, exc)

    flat, undecided = [], 0
    for i, chunk in enumerate(chunks):
        r = results[i]
        if r is None:
            log.warning("Батч %d: ответа нет ни от одного провайдера — "
                        "%d статей уходят в pending", i, len(chunk))
            flat.extend([None] * len(chunk))
            undecided += len(chunk)
        else:
            flat.extend(r)
            undecided += sum(1 for v in r if v is None)
    if undecided:
        log.warning("Без решения LLM: %d из %d — остаются в очереди pending",
                    undecided, len(texts))
    return flat


# ---------------------------------------------------------------------------
# Embedding-модель (sentence-transformers), грузится один раз лениво.
# ---------------------------------------------------------------------------
def get_model():
    """Движок эмбеддингов. С 2026-07 это embedder.py: onnxruntime напрямую,
    без sentence-transformers и без torch.

    Связка sentence-transformers 5.6 / optimum 2.1 при local_files_only=True
    резолвила onnx/model.onnx в None и уходила конвертировать модель сама —
    через torch, который на этом CPU (нет AVX) убивается ядром по SIGILL.
    Прогон вис навсегда, держа ~2 ГБ. В /opt/gdelt_rss это уже случилось.

    Вектора идентичны прежним (косинус 1.000000 на выборке из БД), так что
    переиндексация не нужна. Старый путь сохранён как _get_model_st().
    """
    import embedder
    return embedder.get_engine()


def _get_model_st():
    """НЕ ИСПОЛЬЗУЕТСЯ. Прежняя реализация, оставлена для отката.
    Требует рабочего torch, которого на этом CPU нет."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading embedding model %s ...", config.EMBEDDING_MODEL)
        _model = SentenceTransformer(config.EMBEDDING_MODEL, backend="onnx",
                                     model_kwargs={"file_name": "onnx/model.onnx"},
                                     local_files_only=True)
        log.info("Model loaded.")
    return _model


# ---------------------------------------------------------------------------
# GDELT GKG: список 15-минутных дампов и их загрузка/фильтрация
# ---------------------------------------------------------------------------
def gkg_timestamps(hours=24):
    now = datetime.now(timezone.utc)
    aligned = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
    return [(aligned - timedelta(minutes=15 * i)).strftime("%Y%m%d%H%M%S")
            for i in range(hours * 4)]


def gkg_timestamps_week():
    """168 часов = 7 дней, каждые 15 минут = 672 дампа."""
    now = datetime.now(timezone.utc)
    aligned = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
    return [(aligned - timedelta(minutes=15 * i)).strftime("%Y%m%d%H%M%S")
            for i in range(168 * 4)]


def fetch_gkg_file(ts, translation=False):
    url = (GKG_URL_TL if translation else GKG_URL_EN).format(ts=ts)
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            with z.open(z.namelist()[0]) as f:
                return pd.read_csv(
                    f,
                    sep="\t",
                    header=None,
                    usecols=GKG_USECOLS,
                    names=GKG_COLNAMES,
                    on_bad_lines="skip",
                    low_memory=False,
                    dtype=str,
                    encoding_errors="replace",
                )
    except Exception as exc:
        log.warning("GKG fetch failed %s (translation=%s): %s", ts, translation, exc)
        return None


def fetch_gdelt(query_index, week_mode=False):
    cfg = config.QUERIES_GKG[query_index]
    timestamps = gkg_timestamps_week() if week_mode else gkg_timestamps(hours=24)
    mode_label = "WEEK" if week_mode else "DAY"
    log.info(
        "GKG query #%d [%s]: %d тиков x 2 потока | темы: %s | страны: %s | "
        "макс. разрыв тема-локация: %s симв. | мин. доля страны: %s",
        query_index, mode_label, len(timestamps),
        ",".join(cfg.get("themes", [])[:3]) + "...",
        ",".join(cfg.get("locations", [])),
        cfg.get("max_theme_loc_gap"), cfg.get("min_country_share"),
    )
    seen = {}
    stats = {}

    for i, ts in enumerate(timestamps, start=1):
        for translation in (False, True):
            df = fetch_gkg_file(ts, translation=translation)
            for url, gkg_date, ents in gkg_filter.select(df, cfg, stats):
                seen.setdefault(url, (gkg_date, ents))

        if i % 20 == 0 or i == len(timestamps):
            log.info("  [%s] %d/%d тиков, URL найдено: %d",
                     mode_label, i, len(timestamps), len(seen))
        time.sleep(GKG_FETCH_DELAY)

    log.info(
        "GKG query #%d [%s] готово: строк %d, прошло %d, отсеяно %d, "
        "откат на V1 %d, уникальных URL %d",
        query_index, mode_label, stats.get("rows", 0), stats.get("passed", 0),
        stats.get("dropped", 0), stats.get("v1_fallback", 0), len(seen),
    )
    # Распределения печатаются, чтобы max_theme_loc_gap и min_country_share
    # подбирались по факту, а не оставались догадкой.
    gaps = stats.get("gaps")
    if gaps:
        p = np.percentile(gaps, [50, 75, 90, 95])
        log.info("  разрыв тема-локация у прошедших: p50=%.0f p75=%.0f p90=%.0f p95=%.0f", *p)
    shares = stats.get("shares")
    if shares:
        p = np.percentile(shares, [10, 25, 50])
        log.info("  доля целевой страны у прошедших: p10=%.2f p25=%.2f p50=%.2f", *p)
    return [{"url": u, "gkg_date": d, "entities": e} for u, (d, e) in seen.items()]


# ---------------------------------------------------------------------------
# Скачивание страницы + извлечение текста/заголовка/даты публикации
# ---------------------------------------------------------------------------
def fetch_and_extract(url):
    """Единый HTTP-запрос: возвращает (text, title, publish_date)."""
    try:
        resp = requests.get(url, proxies=config.PROXIES, headers=HTTP_HEADERS, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        log.warning("Failed to download %s: %s", url, exc)
        return None, None, None
    # requests при отсутствии charset в заголовке Content-Type берёт
    # ISO-8859-1 (так предписывает RFC 2616), хотя страница объявляет UTF-8
    # в мета-теге. Турецкие сайты заголовок часто не ставят, и текст приезжал
    # искажённым: «BAÅVURU» вместо «BAŞVURU», «GEÃMÄ°Å» вместо «GEÇMİŞ».
    # Замер 14.08.2026: 34 статьи из 1496, из них 30 турецких, три домена —
    # bursahakimiyet.com.tr, hurhaber.com, turktime.com. Модель читала этот
    # текст и по нему писала пересказ.
    if not resp.encoding or resp.encoding.lower() in ("iso-8859-1", "latin-1", "latin1"):
        resp.encoding = resp.apparent_encoding or "utf-8"
    return extract_page(resp.text, url)


# Страница-заглушка антибота: щит отдаёт двухсотку с человеческим текстом, и
# для trafilatura это обычная статья на 300–500 символов — то есть ровно выше
# MIN_TEXT_LENGTH. Особенно бьёт по спасательному проходу: браузер щит
# проходит не всегда, и половина «спасённого» им оказывалась вот этим.
ANTIBOT_PATTERNS = re.compile(
    r"security service to protect|verifies you are not a bot|performing security "
    r"verification|triggered the security solution|you have been blocked|"
    r"just a moment|attention required|checking your browser|enable javascript "
    r"and cookies|cloudflare ray id|are you a robot|проверка безопасности", re.I)


def _pull(html, url, recall):
    try:
        kw = {"favor_recall": True} if recall else {"favor_precision": True}
        data = bare_extraction(html, url=url, with_metadata=True,
                               include_comments=False, **kw)
    except Exception as exc:
        log.warning("trafilatura failed %s: %s", url, exc)
        return None
    if not data:
        return None
    if hasattr(data, "text"):
        return (data.text or "").strip(), (data.title or "").strip()
    return (data.get("text", "") or "").strip(), (data.get("title", "") or "").strip()


def extract_page(html, url):
    """HTML -> (text, title, publish_date). Отдельно от загрузки: тем же
    разбором живут спасательные проходы (архив, браузер), и правила извлечения
    у всех обязаны совпадать.

    Разбор идёт в два захода. Первый — на точность: он отрезает всё, в чём не
    уверен, и на статье с непривычной вёрсткой запросто оставляет пустоту.
    Второй, на полноту, включается только когда от первого не осталось текста
    выше порога — терять на этом нечего, страница и так была бы отброшена.
    Замер на ста «пустых» адресах: 26 против 36 (см. archive_fetch).
    """
    got = _pull(html, url, recall=False)
    if not got or len(got[0]) < config.MIN_TEXT_LENGTH:
        got = _pull(html, url, recall=True) or got
    text, title = got if got else (None, None)

    if ANTIBOT_PATTERNS.search((title or "") + " " + (text or "")[:1500]):
        log.info("Антибот вместо статьи: %s", url)
        return None, None, None

    publish_date = None
    try:
        publish_date = find_date(
            html, url=url, extensive_search=True, outputformat="%Y-%m-%d"
        )
    except Exception as exc:
        log.debug("htmldate failed %s: %s", url, exc)

    return text, title, publish_date


def take_for_browser(shorts, budget):
    """Отрезает от бюджета прогона столько адресов, сколько ещё дозволено.

    Бюджет общий на прогон, а не на запрос GKG, и платится за открытые
    страницы, а не за спасённые: считать по спасённым значило бы «щит не
    пустил — открываем ещё одну», и одна трудная страна съедала бы браузер
    целиком. budget — список из одного числа, чтобы убывать между запросами.
    """
    take = shorts[:max(budget[0], 0)]
    budget[0] -= len(take)
    return take


def rescue_from_archive(urls):
    """Спасение снимком из Web Archive — первая ступень, до браузера.

    Берёт вдвое больше браузера и стоит секунды вместо пятнадцати, поэтому
    бюджета ему не нужно: замер и вся лестница способов — в archive_fetch.
    Идёт подряд, без пула: архив чужой и некоммерческий, пачка потоков
    заканчивается рванными соединениями. -> {url: (text, title, publish_date)}
    """
    out = {}
    for url in urls:
        raw = archive_fetch.snapshot(url)
        if not raw:
            continue
        # Снимок сырой (id_): баннера архива с его собственной датой в
        # разметке нет, дату публикации читаем из меток исходной страницы.
        text, title, pdate = extract_page(raw, url)
        if text and len(text) >= config.MIN_TEXT_LENGTH:
            out[url] = (text, title, pdate)
    if urls:
        log.info("Архивом спасено %d из %d", len(out), len(urls))
    return out


def rescue_with_browser(urls):
    """Последняя попытка: открыть страницу настоящим браузером.

    Вердикт "short" здесь окончательный с первого раза (seen_store.FINAL) —
    второй попытки у статьи не будет никогда, поэтому спасать надо сразу или
    не спасать вовсе. А спасать есть что: "текста нет" наполовину означает не
    отсутствие статьи, а отказ чужого сервера — 403 антибот-щиту, пустую
    двухсотку там, где текст рисует скрипт.

    Замер на сотне адресов, отброшенных как "short" (08.08.2026), считая
    спасением только настоящую статью (см. ANTIBOT_PATTERNS): обычный
    requests, повторённый днями позже, вытащил 23, curl_cffi с подделкой
    TLS-отпечатка Chrome — тоже 23 (то есть подделка отпечатка этим сайтам не
    указ), camoufox — 32. Без сторожа заглушек те же замеры давали 27, 25 и
    45: щит отвечает браузеру охотнее, чем скрипту, и его страница длиннее
    порога. Платит за это прогон пятнадцатью секундами на страницу, поэтому
    браузер идёт отдельным проходом в конце загрузки, одним экземпляром, и не
    больше config.CAMOUFOX_MAX страниц за прогон.

    -> {url: (text, title, publish_date)}
    """
    out = {}
    if not urls:
        return out
    try:
        from camoufox.sync_api import Camoufox
    except ImportError:
        log.warning("camoufox не установлен — спасать нечем")
        return out
    try:
        with Camoufox(headless=True, humanize=False) as br:
            for url in urls:
                try:
                    page = br.new_page()
                    page.goto(url, timeout=30_000, wait_until="domcontentloaded")
                    page.wait_for_timeout(1500)   # дорисовка после DOMContentLoaded
                    html = page.content()
                    page.close()
                except Exception as exc:
                    log.debug("camoufox %s: %s", url, exc)
                    continue
                text, title, pdate = extract_page(html, url)
                if text and len(text) >= config.MIN_TEXT_LENGTH:
                    out[url] = (text, title, pdate)
    except Exception as exc:            # браузер может и не подняться
        log.warning("camoufox не запустился: %s", exc)
    log.info("Браузером спасено %d из %d", len(out), len(urls))
    return out


# ---------------------------------------------------------------------------
# Быстрый (без LLM) отсев не-новостных материалов: интервью, колонки мнений,
# аналитика и т.п. — по ключевым словам на языках региона.
# ---------------------------------------------------------------------------
NON_EVENT_PATTERNS = [
    # Английский
    r"\binterview\b", r"\bin conversation\b", r"\bq&a\b", r"\banalysis\b",
    r"\bopinion\b", r"\bcommentary\b", r"\beditorial\b", r"\bguest essay\b",
    r"\bcolumn\b", r"\bexplainer\b", r"\bpodcast\b", r"\bnewsletter\b",

    # Русский
    r"\bинтервью\b", r"\bмнение\b", r"\bколонка\b", r"\bредакционн(?:ая|ое|ый)\b",
    r"\bанализ\b", r"\bкомментарий\b", r"\bразбор\b", r"\bэссе\b",

    # Турецкий (tr)
    r"\bröportaj\b", r"\bgörüş\b", r"\byorum\b", r"\banaliz\b",
    r"\bköşe yazısı\b", r"\beditöryal\b",

    # Азербайджанский (az)
    r"\bmüsahib[əe]\b", r"\brəy\b", r"\bşərh\b", r"\banaliz\b",

    # Узбекский (uz)
    r"\bsuhbat\b", r"\bfikr\b", r"\bsharh\b", r"\btahlil\b",

    # Казахский (kk)
    r"\bпікір\b", r"\bсұхбат\b", r"\bталдау\b",

    # Армянский (hy)
    r"\bհարցազրույց\b", r"\bկարծիք\b", r"\bմեկնաբանություն\b", r"\bվերլուծություն\b",

    # Грузинский (ka)
    r"\bინტერვიუ\b", r"\bმოსაზრება\b", r"\bკომენტარი\b", r"\bანალიზი\b",

    # Персидский (fa)
    r"\bمصاحبه\b", r"\bتحلیل\b", r"\bیادداشت\b", r"\bسرمقاله\b",

    # Арабский (ar)
    r"\bمقابلة\b", r"\bتحليل\b", r"\bرأي\b", r"\bافتتاحية\b",

    # Таджикский (tg, кириллица)
    r"\bмусоҳиба\b", r"\bтаҳлил\b", r"\bшарҳ\b", r"\bандеша\b",

    # Туркменский (tk, латиница)
    r"\bsöhbetdeşlik\b", r"\bpikir\b", r"\bteswir\b", r"\bseljerme\b",
]


def detect_text_language(text: str) -> str | None:
    """langdetect по первым ~2000 символам; None, если текст слишком короткий
    или язык не определился.

    Метка НИКУДА не идёт дальше колонки articles.language и одной строки в
    логе. Ни выбор глоссария, ни промпты, ни перевод, ни предфильтр, ни
    важность её не читают: load_week_articles её даже не выбирает.

    Проверено 16.08.2026 после аудита. Повод был: у 46 армянских (1in.am,
    panorama.am) и грузинских (sputnik-georgia.com) страниц стоит
    language='et'. Это не ошибка кода — у langdetect ровно 55 профилей, и ни
    hy, ни ka среди них нет, так что армянский и грузинский он обязан на
    что-то свалить. Чинить нечем и незачем: пока метка никем не читается,
    неверная метка ничего не портит. Если она когда-нибудь понадобится —
    сначала завести детектор, знающий эти письменности (у них свои алфавиты,
    хватит проверки диапазона Unicode), и только потом на неё опираться.
    """
    if not text:
        return None
    sample = re.sub(r"\s+", " ", text.strip())[:2000]
    if len(sample) < 80:
        return None
    try:
        return detect(sample)
    except LangDetectException:
        return None
    except Exception:
        return None


def looks_like_non_news(url: str, title: str, text: str) -> bool:
    haystack = " ".join([
        (url or "").lower(),
        (title or "").lower(),
        (text or "")[:2000].lower(),
    ])
    return any(re.search(pat, haystack, re.IGNORECASE) for pat in NON_EVENT_PATTERNS)


def is_non_event_article(url, title, text):
    """True — статью нужно отсеять до LLM-фильтра (слишком короткая или
    похожа на интервью/колонку/аналитику по NON_EVENT_PATTERNS)."""
    if not text or len(text) < config.MIN_TEXT_LENGTH:
        return True
    return looks_like_non_news(url, title, text)


# ---------------------------------------------------------------------------
# Запись батча в БД
# ---------------------------------------------------------------------------
def embed_texts(texts):
    """Эмбеддинги считаются ОДИН раз на кандидата и переиспользуются: для
    дедупликации синдикации, для локального предфильтра и для записи в БД.
    e5 всё равно обрезает вход по 512 токенов, поэтому отдельного укороченного
    прохода ради дедупа не нужно."""
    if not texts:
        return np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32)
    return get_model().encode(
        ["passage: " + t for t in texts], batch_size=BATCH_SIZE,
        convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


def cluster_duplicates(embs, threshold):
    """Жадная кластеризация по косинусу. GDELT индексирует один сюжет через
    десятки изданий, и LLM отвечала "да" про один и тот же текст по тридцать
    раз. В LLM теперь идёт один представитель, но в БД пишутся ВСЕ члены:
    LexRank выражает важность через плотность графа схожестей, и удаление
    дублей обрушило бы этот сигнал.
    Сходства считаются ПОСТРОЧНО, а не матрицей n*n: в недельном режиме
    кандидатов ~8.5 тыс., и полная матрица float32 заняла бы 293 МБ одним
    куском — на 3.9 ГБ с активным свопом это и убивало прогон. Построчно
    пик держится на O(n) (десятки килобайт), а число умножений то же.
    # ponytail: жадная склейка по первому непомеченному — порядок кандидатов
    # влияет на выбор центра кластера, но при пороге 0.95 состав от этого
    # практически не зависит."""
    n = len(embs)
    if n == 0:
        return np.zeros(0, dtype=int)
    E = np.asarray(embs, dtype=np.float32)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-10)
    labels = np.full(n, -1, dtype=int)
    nxt = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        sims = E @ E[i]  # один вектор длины n вместо матрицы n*n
        labels[(sims >= threshold) & (labels < 0)] = nxt
        nxt += 1
    return labels


def flush_batch(conn, batch):
    """batch items: (qi, url, fetch_date, publish_date, title, text, region, lang, entities, emb, scale)"""
    if not batch:
        return 0
    inserted = 0
    for (qi, url, fd, pdate, title, text, bucket, lang, ents, emb, scale) in batch:
        try:
            conn.execute(
                "INSERT OR IGNORE INTO articles "
                "(query_index,url,fetch_date,publish_date,title,text,embedding,region_bucket,language,entities,scale) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (qi, url, fd, pdate, title, text, emb.tobytes(), bucket, lang, ents, scale))
            inserted += 1
        except Exception as exc:
            log.warning("INSERT error %s: %s", url, exc)
    conn.commit()
    log.info("Flushed %d/%d rows to DB.", inserted, len(batch))
    return inserted


# ---------------------------------------------------------------------------
# Основной сценарий
# ---------------------------------------------------------------------------
def main():
    log.info("=== Start daily_collector | mode=%s ===", _run_mode_label())
    if not os.path.exists(config.DB_PATH):
        log.error("DB not found: %s. Run init_db.py first.", config.DB_PATH)
        sys.exit(1)

    conn = sqlite3.connect(config.DB_PATH)
    seen_store.ensure(conn)
    fetch_date = time.strftime("%Y-%m-%d")
    total_new = 0
    week_mode = _is_week_mode()
    budget = [config.CAMOUFOX_MAX]   # страниц браузеру на весь прогон, на все запросы

    try:
        for qi in range(len(config.QUERIES_GKG)):
            log.info("--- Query #%d | mode=%s ---", qi, "week" if week_mode else "day")

            # 1) Очередь нерешённых разбирается ДО новых дампов: это статьи,
            #    по которым в прошлый раз молчали все провайдеры.
            queue = [(u, None, "") for u in seen_store.pending_urls(conn, qi)]
            if queue:
                log.info("Query #%d: %d URL из очереди pending", qi, len(queue))

            # 2) Новые кандидаты минус те, по которым решение уже окончательно.
            #    Раньше проверка шла по таблице articles, где лежат только
            #    ПРИНЯТЫЕ статьи, поэтому отклонённые возвращались каждые 6
            #    часов — 61.7% работы прогона уходило на повтор.
            fresh = [(a["url"], a["gkg_date"], a["entities"])
                     for a in fetch_gdelt(qi, week_mode=week_mode)]
            decided = seen_store.final_urls(conn, [u for u, _, _ in fresh])
            queue += [(u, d, e) for u, d, e in fresh if u not in decided]
            log.info("Query #%d: найдено %d, из них решены ранее %d, к обработке %d",
                     qi, len(fresh), len(decided), len(queue))

            # 3) Загрузка страниц и дешёвые правила.
            cands, marks, shorts = [], [], []

            def sift(item, got):
                """Правила отбора одной статьи -> вердикт. Принятая попутно
                оседает в cands. Вызывается дважды — из общей загрузки и ещё
                раз для того, что вытащил браузер, — поэтому и вынесена."""
                url, gkg_date, ents = item
                text, title, html_date = got
                if not text or len(text) < config.MIN_TEXT_LENGTH:
                    return "short"
                lang = detect_text_language(text) or "unk"
                if is_non_event_article(url, title, text):
                    return "non_event"
                # GDELT для этих запросов сущностей почти не отдаёт: колонка
                # entities в БД пустая на всех 3428 статьях. Достаём их сами
                # по общему газеттиру (glossary: топонимы Азии и Африки,
                # институты, персоналии) — тем же, что питает перевод.
                if not (ents or "").strip():
                    try:
                        import glossary
                        found = glossary.extract_entities(
                            (title or "") + " " + (text or "")[:3000])
                        ents = ";".join(found)
                    except Exception:
                        log.exception("извлечение сущностей не удалось")
                cands.append((url, text, html_date or gkg_date, title, lang, ents))
                return "accepted"

            for item in queue:
                v = sift(item, fetch_and_extract(item[0]))
                if v == "short":
                    shorts.append(item)             # вердикт после спасения
                elif v != "accepted":
                    marks.append((item[0], qi, v, None))

            # Кому обычный запрос текста не отдал — того спасаем: другой
            # попытки не будет, "short" окончателен с первого раза. Сперва
            # архив — он дешевле и берёт больше; браузер тратит свой скудный
            # бюджет только на то, чего в архиве не нашлось.
            hard = [u for u, _, _ in shorts]
            rescued = rescue_from_archive(hard)
            rescued.update(rescue_with_browser(take_for_browser(
                [u for u in hard if u not in rescued], budget)))
            for item in shorts:
                got = rescued.get(item[0])
                if not (got and sift(item, got) == "accepted"):
                    marks.append((item[0], qi, "short", None))
            seen_store.mark(conn, marks, fetch_date)
            if not cands:
                log.info("Query #%d: кандидатов после правил не осталось", qi)
                continue
            log.info("Query #%d: %d кандидатов после правил", qi, len(cands))

            # 4) Эмбеддинги — один раз, дальше переиспользуются везде.
            embs = embed_texts([c[1] for c in cands])

            # 5) Дедуп синдикации: судится один представитель кластера
            #    (самый длинный текст), вердикт затем идёт всем членам.
            labels = cluster_duplicates(embs, config.DEDUP_COSINE)
            reps = {}
            for i, lab in enumerate(labels):
                j = reps.get(lab)
                if j is None or len(cands[i][1]) > len(cands[j][1]):
                    reps[lab] = i
            rep_idx = sorted(reps.values())
            log.info("Query #%d: %d кандидатов -> %d кластеров (порог %.2f), "
                     "сэкономлено %d судейств",
                     qi, len(cands), len(rep_idx), config.DEDUP_COSINE,
                     len(cands) - len(rep_idx))

            # 6) Локальный предфильтр — только если артефакт обучен.
            to_llm, auto_drop = rep_idx, []
            if prefilter.is_ready(config.PREFILTER_PATH):
                mask = prefilter.drop_mask(config.PREFILTER_PATH, embs[rep_idx])
                auto_drop = [rep_idx[k] for k in np.where(mask)[0]]
                to_llm = [rep_idx[k] for k in np.where(~mask)[0]]
                log.info("Предфильтр: отброшено без LLM %d, уходит в LLM %d",
                         len(auto_drop), len(to_llm))

            # 7) LLM: релевантность и регион одним вызовом.
            #    У предфильтрованных вердикт СВОЙ, не "NO": их решение приняла
            #    модель, которая учится на этой же разметке (см. seen_store).
            verdicts = {i: _PREFILTERED for i in auto_drop}
            if to_llm:
                log.info("Query #%d: %d судейств -> ~%d запросов (батч %d)",
                         qi, len(to_llm), -(-len(to_llm) // JUDGE_BATCH), JUDGE_BATCH)
                verdicts.update(zip(to_llm, judge_parallel([cands[i][1] for i in to_llm])))

            # 8) Вердикт кластера распространяется на всех его членов.
            marks, batch = [], []
            for i, (url, text, pdate, title, lang, ents) in enumerate(cands):
                v = verdicts.get(reps[labels[i]])
                if v is None:
                    marks.append((url, qi, "pending", None))
                elif v is _PREFILTERED:
                    marks.append((url, qi, "prefiltered", None))
                elif v[0] == "NO":
                    marks.append((url, qi, "rejected", embs[i]))
                else:
                    marks.append((url, qi, "accepted", embs[i]))
                    batch.append((qi, url, fetch_date, pdate, title, text, v[0],
                                  lang, ents, embs[i], v[1]))
            seen_store.mark(conn, marks, fetch_date)
            for k in range(0, len(batch), BATCH_SIZE):
                total_new += flush_batch(conn, batch[k:k + BATCH_SIZE])

            log.info("Query #%d: принято %d, отклонено %d, предфильтр %d, в pending %d",
                     qi,
                     sum(1 for m in marks if m[2] == "accepted"),
                     sum(1 for m in marks if m[2] == "rejected"),
                     sum(1 for m in marks if m[2] == "prefiltered"),
                     sum(1 for m in marks if m[2] == "pending"))

        log.info("=== Done. Total new articles: %d ===", total_new)
        log.info("Журнал seen_urls: %s", seen_store.stats(conn))
        pos, neg = seen_store.label_counts(conn)
        log.info("Разметка для предфильтра: принятых %d, отклонённых %d "
                 "(для обучения нужно %d суммарно)",
                 pos, neg, config.PREFILTER_MIN_LABELS)
        removed = seen_store.prune(conn, config.SEEN_KEEP_DAYS)
        if removed:
            log.info("Журнал seen_urls: удалено %d старых служебных строк", removed)
        for row in conn.execute(
            "SELECT language, COUNT(*) FROM articles WHERE fetch_date = ? GROUP BY language",
            (fetch_date,)
        ):
            log.info("  lang=%s: %d articles", row[0], row[1])
    finally:
        conn.close()


if __name__ == "__main__":
    main()
