# -*- coding: utf-8 -*-
"""
daily_collector.py — сбор статей за сутки (или за неделю, см. --week ниже)
из дампов GDELT GKG, отсев мусора (короткие/нерелевантные/не-новостные
тексты) и запись в БД с embedding-вектором и региональной меткой.

Запускается по cron каждые 6 часов в режиме "day" (см. crontab).
Режим "week"/"backfill" (--week или --backfill в argv) используется
только для ручного бэкафилла на более широком окне GKG-дампов.

Пайплайн одного query_index (см. config.QUERIES_GKG):
    1. fetch_gdelt()      — сканирует 15-минутные дампы GKG (EN + переводной
                             поток), фильтрует по themes/locations, отдаёт
                             набор URL-кандидатов с датой из GKG.
    2. fetch_and_extract() — скачивает страницу, извлекает текст/заголовок
                             (trafilatura) и дату публикации (htmldate).
    3. is_non_event_article() — быстрый regex-отсев не-новостных материалов
                             (интервью/колонки/мнения на нескольких языках).
    4. llm_filter_parallel()  — строгая тематическая релевантность через LLM:
                             тема должна быть ГЛАВНЫМ предметом статьи (не
                             случайным упоминанием) и новость — национального
                             или крупного масштаба, не местячковой (батчами,
                             параллельно по нескольким моделям).
    5. classify_region_parallel() — региональная метка (TR/CA/SC/MIX) через LLM.
    6. flush_batch()       — считает embedding (sentence-transformers) и
                             пишет батч в БД.
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

import config

os.environ.setdefault("HF_HOME", "/opt/digest/.cache/huggingface")
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
GKG_USECOLS = [1, 3, 4, 7, 9]
GKG_COLNAMES = ["date", "source", "url", "themes", "locations"]

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
# OpenRouter: общий пул ":free"-моделей для LLM-фильтров ниже
# (тематическая релевантность + региональная классификация).
# ---------------------------------------------------------------------------
# Переопределяется через .env — см. комментарий в model_rotation.py.
_OR_API = os.environ.get(
    "OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions"
)
_OR_MODELS = "https://openrouter.ai/api/v1/models"
_FB_MODELS = ["google/gemma-3-27b-it:free", "meta-llama/llama-3.1-8b-instruct:free"]
_POOL_SIZE = 5


def _build_pool(existing, force=False):
    """
    Возвращает список ":free"-моделей OpenRouter (топ-_POOL_SIZE по context_length).
    Если existing уже заполнен и force=False — возвращает его как есть (не дёргает API).
    При ошибке запроса — фолбэк на _FB_MODELS (или на existing, если он был).
    """
    if existing and not force:
        return existing
    try:
        r = requests.get(
            _OR_MODELS,
            headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"},
            proxies=config.PROXIES,
            timeout=30,
        )
        r.raise_for_status()
        free = [(m["id"], m.get("context_length") or 0)
                for m in r.json().get("data", []) if m.get("id", "").endswith(":free")]
        free.sort(key=lambda x: x[1], reverse=True)
        pool = [mid for mid, _ in free[:_POOL_SIZE]]
        if not pool:
            raise ValueError("empty")
        log.info("Pool updated: %s", pool)
        return pool
    except Exception as exc:
        log.warning("Pool refresh failed: %s", exc)
        return existing or list(_FB_MODELS)


def _fmodel_call(mid, prompt, system_prompt):
    """Один запрос к конкретной модели. Один повтор при 429/5xx, иначе 'switch'."""
    for attempt in range(2):
        try:
            r = requests.post(
                _OR_API,
                json={"model": mid, "temperature": 0.0,
                      "messages": [{"role": "system", "content": system_prompt},
                                   {"role": "user", "content": prompt}]},
                headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                         "Content-Type": "application/json"},
                proxies=config.PROXIES,
                timeout=60,
            )
        except Exception as exc:
            log.warning("LLM call net error (%s): %s", mid, exc)
            return "switch", None
        if r.status_code == 200:
            try:
                return "ok", r.json()["choices"][0]["message"]["content"]
            except Exception:
                return "switch", None
        if r.status_code == 429 or 500 <= r.status_code < 600:
            if attempt == 0:
                time.sleep(5)
                continue
            return "switch", None
        return "switch", None
    return "switch", None


def _run_batches_parallel(chunks, pool, call_fn, empty_result_fn):
    """
    chunks       – список списков текстов (каждый до 20 шт.)
    pool         – список моделей (>=1)
    call_fn(model, chunk) -> raw_response_or_None
    empty_result_fn(chunk) -> список дефолтных значений при полном отказе
    Возвращает список результатов той же длины/порядка, что chunks.
    """
    if not chunks:
        return []
    results = [None] * len(chunks)
    if not pool:
        pool = list(_FB_MODELS)

    with ThreadPoolExecutor(max_workers=min(len(chunks), len(pool))) as ex:
        futures = {}
        for i, chunk in enumerate(chunks):
            model = pool[i % len(pool)]          # round-robin по разным моделям
            futures[ex.submit(call_fn, model, chunk)] = i
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:
                log.warning("Parallel batch %d failed: %s", i, exc)
                results[i] = None

    for i, chunk in enumerate(chunks):
        if results[i] is None:
            results[i] = empty_result_fn(chunk)
    return results


# ---------------------------------------------------------------------------
# LLM-фильтр тематической релевантности — главный сдерживающий барьер против
# мусора: GKG theme/location пред-фильтр в fetch_gdelt() ловит любое СОВПАДЕНИЕ
# ключевых слов и локации в статье, а не то, что статья ДЕЙСТВИТЕЛЬНО про это.
# Поэтому здесь явно требуем, чтобы тема была ГЛАВНЫМ предметом статьи, а не
# случайным упоминанием, и чтобы новость была нацио­нального/крупного масштаба,
# а не местячковой.
# ---------------------------------------------------------------------------
_FILTER_SYSTEM = (
    "You are a strict relevance filter for a research digest published by the "
    "Institute of Oriental Studies (Russian Academy of Sciences). The digest "
    "covers ONLY two kinds of news:\n"
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
    "For BOTH categories, reject the article (answer false) if ANY of these apply:\n"
    "  - the topic is only mentioned in passing — not what the article is "
    "actually about;\n"
    "  - the news is purely local/municipal/ceremonial (a single school event, "
    "a minor local award, a routine press release) rather than of national or "
    "large-scale significance;\n"
    "  - it's an opinion piece, interview, or analysis rather than a report of "
    "an actual event.\n"
    "\n"
    "For each article, decide: does it match T1 or T2? The article may be in "
    "any language — judge by content, not language."
)
_FILTER_USER_TMPL = (
    "Below are {n} articles. Reply ONLY with a JSON array of {n} booleans, "
    "same order as the articles, no extra text — e.g. [true, false, true].\n\n"
    "{articles}"
)
FILTER_TEXT_CHARS = 2000  # сколько символов статьи уходит в промпт: обычно
                           # достаточно, чтобы понять главный предмет статьи,
                           # и это же естественно наказывает случаи, когда
                           # нужная тема всплывает лишь мельком в середине текста

_fpool = []  # пул моделей для _llm_filter_call, см. _build_pool()


def _coerce_bool(v) -> bool:
    """Терпимо к лёгким отклонениям модели от чистого true/false в массиве
    (например, объект {"relevant": true} вместо true)."""
    if isinstance(v, bool):
        return v
    if isinstance(v, dict):
        return any(bool(x) for x in v.values())
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1")
    return bool(v)


def _llm_filter_call(model, texts):
    status, content = _fmodel_call(
        model,
        _FILTER_USER_TMPL.format(
            n=len(texts),
            articles="".join(
                f"[{i}] {t.replace(chr(10), ' ')[:FILTER_TEXT_CHARS]}\n\n"
                for i, t in enumerate(texts, 1)
            ),
        ),
        system_prompt=_FILTER_SYSTEM,
    )
    if status != "ok":
        return None
    try:
        m = re.search(r"\[.*\]", content, re.DOTALL)
        data = json.loads(m.group(0) if m else content)
        return [_coerce_bool(v) for v in data[:len(texts)]]
    except Exception:
        return None


def llm_filter_parallel(pending_texts):
    """pending_texts: список текстов всех кандидатов запроса.
    При полном отказе всех моделей на батч — статьи батча ОТСЕИВАЮТСЯ
    (fail-closed: лучше недобрать, чем протащить непроверенный мусор в БД).
    Это не риск полноты: тот же URL снова попадёт в окно выборки следующего
    прогона коллектора (cron каждые 6 часов, окно GKG — последние 24 часа),
    так что статья получит повторный шанс пройти фильтр."""
    global _fpool
    _fpool = _build_pool(_fpool)
    chunks = [pending_texts[i:i + LLM_FILTER_BATCH]
              for i in range(0, len(pending_texts), LLM_FILTER_BATCH)]
    results = _run_batches_parallel(
        chunks, _fpool,
        call_fn=_llm_filter_call,
        empty_result_fn=lambda c: [False] * len(c),
    )
    flat = []
    for r in results:
        flat.extend(r)
    return flat


# ---------------------------------------------------------------------------
# LLM: региональная классификация (TR/CA/SC/MIX)
# ---------------------------------------------------------------------------
_REGION_SYSTEM_BATCH = (
    "Classify EACH article into exactly ONE region bucket:\n"
    "  TR  - primarily about Turkey, including domestic Turkish science/education/STEM/youth developments;\n"
    "  CA  - primarily about Central Asia (KZ/UZ/TM/KG/TJ);\n"
    "  SC  - primarily about South Caucasus (GE/AM/AZ);\n"
    "  MIX - unclear.\n"
    "If Turkey is the main actor or location, choose TR.\n"
    "Reply ONLY with a JSON array of strings (TR/CA/SC/MIX), same order, same length as input."
)
REGION_BATCH_SIZE = 20

_rpool = []  # пул моделей для _classify_region_batch_single_model, см. _build_pool()


def _classify_region_batch_single_model(model, texts):
    n = len(texts)
    block = "".join(f"[{i}] {t.replace(chr(10), ' ')[:3000]}\n\n" for i, t in enumerate(texts, 1))
    prompt = f"Below are {n} articles. Reply with a JSON array of {n} region codes.\n\n{block}"
    status, content = _fmodel_call(model, prompt, system_prompt=_REGION_SYSTEM_BATCH)
    if status != "ok":
        return None
    try:
        m = re.search(r"\[.*\]", content, re.DOTALL)
        data = json.loads(m.group(0) if m else content)
        codes = [str(c).strip().upper()[:3] for c in data[:n]]
        return [c if c in ("TR", "CA", "SC", "MIX") else "MIX" for c in codes]
    except Exception:
        return None


def classify_region_parallel(texts):
    global _rpool
    _rpool = _build_pool(_rpool)
    chunks = [texts[i:i + REGION_BATCH_SIZE] for i in range(0, len(texts), REGION_BATCH_SIZE)]
    results = _run_batches_parallel(
        chunks, _rpool,
        call_fn=_classify_region_batch_single_model,
        empty_result_fn=lambda c: ["MIX"] * len(c),
    )
    flat = []
    for r in results:
        flat.extend(r)
    return flat


# ---------------------------------------------------------------------------
# Embedding-модель (sentence-transformers), грузится один раз лениво.
# ---------------------------------------------------------------------------
def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading embedding model %s ...", config.EMBEDDING_MODEL)
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
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


def filter_gkg(df, theme_filters, loc_filters, extra_terms=None):
    if df is None or df.empty:
        return []
    themes = df["themes"].fillna("")
    locs = df["locations"].fillna("")
    mask = (
        themes.str.contains("|".join(theme_filters), case=False, na=False)
        & locs.str.contains("|".join(f"#{c}#" for c in loc_filters), na=False)
    )
    if extra_terms:
        src = df["source"].fillna("") + " " + df["url"].fillna("")
        mask = mask & src.str.contains("|".join(extra_terms), case=False, na=False)
    sub = df.loc[mask, ["url", "date"]].dropna(subset=["url"])
    seen = {}
    for _, row in sub.iterrows():
        url = row["url"]
        if not isinstance(url, str) or not url.startswith("http"):
            continue
        if url not in seen:
            raw_ts = str(row.get("date", "") or "")
            try:
                gkg_date = datetime.strptime(raw_ts[:8], "%Y%m%d").strftime("%Y-%m-%d")
            except ValueError:
                gkg_date = None
            seen[url] = gkg_date
    return list(seen.items())


def fetch_gdelt(query_index, week_mode=False):
    cfg = config.QUERIES_GKG[query_index]
    timestamps = gkg_timestamps_week() if week_mode else gkg_timestamps(hours=24)
    mode_label = "WEEK" if week_mode else "DAY"
    log.info(
        "GKG query #%d [%s]: %d ticks × 2 streams | themes: %s | locs: %s",
        query_index, mode_label, len(timestamps),
        ",".join(cfg.get("themes", [])[:3]) + "...",
        ",".join(cfg.get("locations", [])),
    )
    seen: dict[str, str | None] = {}

    for i, ts in enumerate(timestamps, start=1):
        df_en = fetch_gkg_file(ts, translation=False)
        for url, gkg_date in filter_gkg(df_en, cfg["themes"], cfg["locations"], cfg.get("extra_terms")):
            seen.setdefault(url, gkg_date)

        df_tl = fetch_gkg_file(ts, translation=True)
        for url, gkg_date in filter_gkg(df_tl, cfg["themes"], cfg["locations"], cfg.get("extra_terms")):
            seen.setdefault(url, gkg_date)

        if i % 20 == 0 or i == len(timestamps):
            log.info(
                "  [%s] %d/%d ticks processed, URLs found: %d",
                mode_label, i, len(timestamps), len(seen),
            )
        time.sleep(GKG_FETCH_DELAY)

    log.info("GKG query #%d [%s] done: %d unique URLs.", query_index, mode_label, len(seen))
    return [{"url": u, "gkg_date": d} for u, d in seen.items()]


def url_exists(conn, url):
    return conn.execute("SELECT 1 FROM articles WHERE url = ? LIMIT 1", (url,)).fetchone() is not None


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

    html = resp.text
    text, title = None, None
    try:
        data = bare_extraction(
            html, url=url, with_metadata=True,
            include_comments=False, favor_precision=True,
        )
        if data:
            if hasattr(data, "text"):
                text  = (data.text  or "").strip()
                title = (data.title or "").strip()
            else:
                text  = (data.get("text",  "") or "").strip()
                title = (data.get("title", "") or "").strip()
    except Exception as exc:
        log.warning("trafilatura failed %s: %s", url, exc)

    publish_date = None
    try:
        publish_date = find_date(
            html, url=url, extensive_search=True, outputformat="%Y-%m-%d"
        )
    except Exception as exc:
        log.debug("htmldate failed %s: %s", url, exc)

    return text, title, publish_date


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
    или язык не определился. Используется только для метки language в БД."""
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
def flush_batch(conn, batch):
    """batch items: (qi, url, fetch_date, publish_date, title, text, region_bucket, language)"""
    if not batch:
        return 0
    model = get_model()
    embeddings = model.encode(
        ["passage: " + r[5] for r in batch],
        batch_size=BATCH_SIZE,
        convert_to_numpy=True, show_progress_bar=False,
    ).astype(np.float32)
    inserted = 0
    for (qi, url, fd, pdate, title, text, bucket, lang), emb in zip(batch, embeddings):
        try:
            conn.execute(
                "INSERT OR IGNORE INTO articles "
                "(query_index,url,fetch_date,publish_date,title,text,embedding,region_bucket,language) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (qi, url, fd, pdate, title, text, emb.tobytes(), bucket, lang))
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
    log.info("=== Start daily_collector (GKG + LLM filter) | mode=%s ===", _run_mode_label())
    if not os.path.exists(config.DB_PATH):
        log.error("DB not found: %s. Run init_db.py first.", config.DB_PATH)
        sys.exit(1)

    conn = sqlite3.connect(config.DB_PATH)
    fetch_date = time.strftime("%Y-%m-%d")
    total_new = 0
    week_mode = _is_week_mode()

    try:
        for qi in range(len(config.QUERIES_GKG)):
            log.info("--- Query #%d | mode=%s ---", qi, "week" if week_mode else "day")
            pending = []
            for art in fetch_gdelt(qi, week_mode=week_mode):
                url = art.get("url")
                gkg_date = art.get("gkg_date")
                if not url or url_exists(conn, url):
                    continue
                text, title, html_date = fetch_and_extract(url)
                if not text or len(text) < config.MIN_TEXT_LENGTH:
                    log.info("Skip (short/empty): %s", url)
                    continue

                lang = detect_text_language(text) or "unk"
                if is_non_event_article(url, title, text):
                    log.info("Skip (non-event/rule) [lang=%s]: %s", lang, url)
                    continue

                log.info("Accept (rule-pass) [lang=%s]: %s | title=%.80s", lang, url, title or "")
                publish_date = html_date or gkg_date
                pending.append((url, text, publish_date, title, lang))

            passed = []
            if pending:
                lang_stats = (
                    pd.Series([lang for _, _, _, _, lang in pending])
                    .value_counts().to_dict()
                )
                log.info(
                    "Query #%d: %d candidates before LLM filter (langs: %s)",
                    qi, len(pending), json.dumps(lang_stats, ensure_ascii=False)
                )
                texts_only = [t for _, t, _, _, _ in pending]
                decisions = llm_filter_parallel(texts_only)
                passed = [item for item, ok in zip(pending, decisions) if ok]

            log.info("Query #%d: %d articles passed LLM filter", qi, len(passed))

            tagged = []
            if passed:
                region_texts = [t for _, t, _, _, _ in passed]
                regions = classify_region_parallel(region_texts)
                tagged = [(url, text, pd_, title, region, lang)
                          for (url, text, pd_, title, lang), region in zip(passed, regions)]

            batch = []
            for url, text, publish_date, title, bucket, lang in tagged:
                batch.append((qi, url, fetch_date, publish_date, title, text, bucket, lang))
                if len(batch) >= BATCH_SIZE:
                    total_new += flush_batch(conn, batch)
                    batch = []
            total_new += flush_batch(conn, batch)

        log.info("=== Done. Total new articles: %d ===", total_new)
        for row in conn.execute(
            "SELECT language, COUNT(*) FROM articles WHERE fetch_date = ? GROUP BY language",
            (fetch_date,)
        ):
            log.info("  lang=%s: %d articles", row[0], row[1])
    finally:
        conn.close()


if __name__ == "__main__":
    main()
