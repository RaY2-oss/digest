# -*- coding: utf-8 -*-
"""apply_patches.py — правки существующих файлов /opt/digest.

Каждая замена проверяется на единственность вхождения. При любом промахе
ничего не пишется на диск и печатается, что не совпало.
"""
import io
import os
import sys

BASE = "/opt/digest"
SEP = "# ---------------------------------------------------------------------------\n"
errors = []


def rd(name):
    with io.open(os.path.join(BASE, name), encoding="utf-8") as fh:
        return fh.read()


def wr(name, text):
    with io.open(os.path.join(BASE, name), "w", encoding="utf-8") as fh:
        fh.write(text)


def sub(src, old, new, tag):
    n = src.count(old)
    if n != 1:
        errors.append(f"{tag}: вхождений {n}, ожидалось 1")
        return src
    return src.replace(old, new)


def splice(src, start_marker, end_marker, new, tag, frame=True):
    """Вырезает от рамки комментария перед start_marker до рамки перед
    end_marker и подставляет new."""
    try:
        i = src.index(start_marker)
        j = src.index(end_marker)
    except ValueError:
        errors.append(f"{tag}: маркер не найден")
        return src
    if frame:
        i = src.rindex(SEP, 0, i)
        j = src.rindex(SEP, 0, j)
    if not i < j:
        errors.append(f"{tag}: маркеры в неверном порядке")
        return src
    return src[:i] + new + src[j:]


# ===========================================================================
# model_rotation.py
# ===========================================================================
mr = rd("model_rotation.py")

mr = sub(mr, """import logging
import os
import time

import requests""", """import logging
import os
import threading
import time

import requests""", "MR/import")

mr = sub(mr, """# --- Глобальное состояние пула (переживает несколько вызовов подряд) ---
_free_models_pool = []
_batches_processed = 0
""", '''# --- Глобальное состояние пула (переживает несколько вызовов подряд) ---
_free_models_pool = []
_batches_processed = 0

# daily_collector зовёт _call_openrouter_raw из нескольких потоков сразу,
# поэтому мутации пула идут под замком: без него два потока могли пройти
# проверку `if x in pool` одновременно, и второй remove() падал с ValueError.
# Сам HTTP-запрос остаётся ВНЕ замка, иначе параллельность батчей потерялась бы.
_POOL_LOCK = threading.RLock()


def _demote(model_id):
    """Неудачная модель уходит в конец пула."""
    with _POOL_LOCK:
        if model_id in _free_models_pool:
            _free_models_pool.remove(model_id)
            _free_models_pool.append(model_id)


def _promote(model_id):
    """Успешная модель встаёт в начало пула."""
    with _POOL_LOCK:
        if model_id in _free_models_pool:
            _free_models_pool.remove(model_id)
        _free_models_pool.insert(0, model_id)
''', "MR/lock")

mr = sub(mr, "BETWEEN_MODEL_PAUSE = 10      # пауза перед сменой модели при 429/5xx, сек",
         """# BETWEEN_MODEL_PAUSE убран намеренно: 429 у ":free"-моделей почти всегда
# означает исчерпанный СУТОЧНЫЙ лимит, и пауза в секундах его не лечит — она
# лишь удлиняет прогон и тратит время впустую. Следующая модель пробуется
# сразу, а исчерпанный провайдер целиком уходит в кулдаун (PROVIDER_COOLDOWN).""",
         "MR/const")

mr = sub(mr, """            # Сетевая ошибка — сразу следующая модель, без паузы.
            if model_id in _free_models_pool:
                _free_models_pool.remove(model_id)
                _free_models_pool.append(model_id)
            continue""", """            # Сетевая ошибка — сразу следующая модель, без паузы.
            _demote(model_id)
            continue""", "MR/demote-net")

mr = sub(mr, """                if model_id in _free_models_pool:
                    _free_models_pool.remove(model_id)
                _free_models_pool.insert(0, model_id)
                _batches_processed += 1""", """                _promote(model_id)
                _batches_processed += 1""", "MR/promote")

mr = sub(mr, """        # Пессимизация: неудачная модель уходит в конец пула.
        if model_id in _free_models_pool:
            _free_models_pool.remove(model_id)
            _free_models_pool.append(model_id)

        if need_pause:
            log.info(
                "Пауза %ds перед переходом к следующей модели (после %s)",
                BETWEEN_MODEL_PAUSE, model_id,
            )
            time.sleep(BETWEEN_MODEL_PAUSE)""", """        _demote(model_id)  # пессимизация: неудачная модель уходит в конец пула
        if need_pause:
            log.debug("Смена модели после %s без паузы", model_id)""", "MR/sleep")


# ===========================================================================
# init_db.py
# ===========================================================================
db = rd("init_db.py")
db = sub(db, """import os
import sqlite3

import config""", """import os
import sqlite3

import config
import seen_store""", "DB/import")
db = sub(db, """        conn.executescript(SCHEMA_SQL)
        migrate(conn)""", """        conn.executescript(SCHEMA_SQL)
        migrate(conn)
        seen_store.ensure(conn)""", "DB/ensure")


# ===========================================================================
# config.py
# ===========================================================================
cf = rd("config.py")

GAP_NOTE = """        # Тема и страна должны быть названы в пределах одного абзаца — это и
        # есть та связь, которой в GKG нет напрямую (см. gkg_filter.py).
        # Оба порога логируются каждым прогоном в виде перцентилей, поэтому
        # подбираются по наблюдаемым распределениям, а не на глаз.
        "max_theme_loc_gap": 400,
        "min_country_share": 0.30,
"""

cf = sub(cf, """        "locations": ["TU", "KZ", "UZ", "TX", "KG", "TI", "GG", "AM", "AJ"],
    },""", """        "locations": ["TU", "KZ", "UZ", "TX", "KG", "TI", "GG", "AM", "AJ"],
""" + GAP_NOTE + "    },", "CF/q0")

cf = sub(cf, """        "locations": ["KZ", "UZ", "TX", "KG", "TI", "GG", "AM", "AJ"],
    },""", """        "locations": ["KZ", "UZ", "TX", "KG", "TI", "GG", "AM", "AJ"],
""" + GAP_NOTE + "    },", "CF/q1")

cf += '''

# ---------------------------------------------------------------------------
# Отсев кандидатов ДО обращения к LLM
# ---------------------------------------------------------------------------
# Порог косинуса, выше которого две статьи считаются одним сюжетом.
# Важно: все члены кластера всё равно пишутся в БД. LexRank в
# sunday_processor_mmr выражает важность через плотность графа схожестей —
# тридцать изданий, перепечатавших один сюжет, образуют плотную клику, и
# именно она даёт высокий скор. Удаление дублей обнулило бы этот сигнал.
# Экономятся только LLM-запросы: судится один представитель кластера.
DEDUP_COSINE = 0.95

# Сколько батчей уходит в LLM параллельно.
LLM_PARALLEL_BATCHES = 5

# Журнал seen_urls: служебные строки без эмбеддинга (short/non_event/pending)
# старше стольких дней удаляются. Строки с эмбеддингом — обучающая выборка
# для предфильтра, они не трогаются.
SEEN_KEEP_DAYS = 30

# ---------------------------------------------------------------------------
# Локальный предфильтр (train_prefilter.py -> prefilter.py)
# ---------------------------------------------------------------------------
# Обучается на вердиктах самой LLM из seen_urls. Пороги допуска намеренно
# консервативные: чистая разметка начала копиться только с внедрением
# seen_urls, а прежние логи брать нельзя — отказ провайдеров молча помечал
# батч отклонённым (week_swap 21.07: 6006 кандидатов, 0 принятых).
PREFILTER_PATH = os.path.join(BASE_DIR, "data", "prefilter.joblib")
PREFILTER_MIN_LABELS = 3000      # меньше — тренер отказывается учиться
PREFILTER_MIN_MINORITY = 500     # минимальный размер класса меньшинства
PREFILTER_TARGET_RECALL = 0.97   # какую долю нужных статей обязан сохранить
PREFILTER_MIN_GAIN = 0.30        # режет меньше мусора — стадия не окупается
'''


# ===========================================================================
# daily_collector.py
# ===========================================================================
dc = rd("daily_collector.py")

dc = sub(dc, """from trafilatura import bare_extraction

import config""", """from trafilatura import bare_extraction

import config
import gkg_filter
import prefilter
import seen_store
from model_rotation import _call_openrouter_raw""", "DC/import")

dc = sub(dc, """GKG_USECOLS = [1, 3, 4, 7, 9]
GKG_COLNAMES = ["date", "source", "url", "themes", "locations"]""",
         """# Колонки 8 и 10 (V2EnhancedThemes / V2EnhancedLocations) несут символьные
# офсеты каждого упоминания — по ним gkg_filter восстанавливает связь темы и
# страны, которой в V1-полях (7 и 9) физически нет.
GKG_USECOLS = [1, 3, 4, 7, 8, 9, 10]
GKG_COLNAMES = ["date", "source", "url", "v1t", "v2t", "v1l", "v2l"]""",
         "DC/usecols")

JUDGE = SEP + '''# LLM: один вызов = релевантность + региональная метка.
#
# Раньше это были две стадии по двум пулам моделей: булев фильтр, затем
# классификация TR/CA/SC/MIX. Вторая тратила ещё ~26 запросов на прогон, хотя
# судит тот же текст по тем же критериям. Теперь один вызов отдаёт либо "NO",
# либо код региона.
#
# Формат ответа — ОБЪЕКТ {"1":"CA","2":"NO"}, а не позиционный массив. Массив
# ломался незаметно: модель теряла или добавляла элемент, индексы съезжали, и
# вердикт доставался чужой статье. С номерами-ключами съехать нельзя, а
# недостающий номер виден явно и уводит статью в pending, а не в отказ.
''' + SEP + '''_JUDGE_SYSTEM = (
    "You are a strict relevance filter and region classifier for a research "
    "digest published by the Institute of Oriental Studies (Russian Academy of "
    "Sciences). The digest covers ONLY two kinds of news:\\n"
    "\\n"
    "T1 — Science, education, or youth policy as the MAIN SUBJECT of the "
    "article, happening IN Turkey, Central Asia (Kazakhstan, Uzbekistan, "
    "Turkmenistan, Kyrgyzstan, Tajikistan), or the South Caucasus (Georgia, "
    "Armenia, Azerbaijan).\\n"
    "\\n"
    "T2 — A concrete science/education/youth action, program, agreement, or "
    "institution — funded, launched, or run by the EU, USA, Iran, India, China, "
    "Japan, South Korea, or Turkey — that is PHYSICALLY TAKING PLACE IN Central "
    "Asia or the South Caucasus (e.g. a foreign-funded university campus opening "
    "in Uzbekistan, a Chinese scholarship program for Kazakh students, a Turkish "
    "school built in Kyrgyzstan). A diplomatic visit, trade deal, or general "
    "foreign-policy statement with no science/education/youth substance does "
    "NOT count, even if it involves these same countries.\\n"
    "\\n"
    "Answer NO for an article if ANY of these apply:\\n"
    "  - the topic is only mentioned in passing — not what the article is "
    "actually about;\\n"
    "  - the news is purely local/municipal/ceremonial (a single school event, "
    "a minor local award, a routine press release) rather than of national or "
    "large-scale significance;\\n"
    "  - it's an opinion piece, interview, or analysis rather than a report of "
    "an actual event.\\n"
    "\\n"
    "For every article that is NOT rejected, assign exactly ONE region:\\n"
    "  TR  - primarily about Turkey, including domestic Turkish "
    "science/education/STEM/youth developments;\\n"
    "  CA  - primarily about Central Asia (KZ/UZ/TM/KG/TJ);\\n"
    "  SC  - primarily about South Caucasus (GE/AM/AZ);\\n"
    "  MIX - relevant, but the region is unclear.\\n"
    "If Turkey is the main actor or location, choose TR.\\n"
    "\\n"
    "The article may be in any language — judge by content, not language."
)

_JUDGE_USER_TMPL = (
    "Below are {n} articles, each preceded by its number in square brackets.\\n"
    "Reply ONLY with a JSON object mapping every article number to its verdict, "
    'no extra text — e.g. {{"1": "CA", "2": "NO", "3": "TR"}}.\\n'
    'Allowed values: "NO", "TR", "CA", "SC", "MIX". '
    "Every number from 1 to {n} must appear exactly once.\\n\\n"
    "{articles}"
)

JUDGE_TEXT_CHARS = 2000   # сколько символов статьи уходит в промпт
JUDGE_BATCH = LLM_FILTER_BATCH
_VERDICTS = ("NO", "TR", "CA", "SC", "MIX")


def _parse_judge(content, n):
    """-> список длины n; None на позиции = решения по этой статье нет."""
    m = re.search(r"\\{.*\\}", content or "", re.DOTALL)
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
        v = str(v).strip().upper() if v is not None else ""
        out.append(v if v in _VERDICTS else None)
    return out


def _judge_call(texts):
    block = "".join(
        f"[{i}] {t.replace(chr(10), ' ')[:JUDGE_TEXT_CHARS]}\\n\\n"
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


'''

dc = splice(dc, '# OpenRouter: общий пул ":free"-моделей для LLM-фильтров ниже',
            "# Embedding-модель (sentence-transformers)", JUDGE, "DC/judge")

FETCH = '''def fetch_gdelt(query_index, week_mode=False):
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
            for url, gkg_date in gkg_filter.select(df, cfg, stats):
                seen.setdefault(url, gkg_date)

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
    return [{"url": u, "gkg_date": d} for u, d in seen.items()]


'''

i0 = dc.find("def filter_gkg(")
i1 = dc.find("def url_exists(")
if i0 < 0 or i1 < 0 or i0 >= i1:
    errors.append("DC/fetch: границы filter_gkg..url_exists не найдены")
else:
    dc = dc[:i0] + FETCH + dc[i1:]

dc = sub(dc, '''def flush_batch(conn, batch):
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
    for (qi, url, fd, pdate, title, text, bucket, lang), emb in zip(batch, embeddings):''',
         '''def embed_texts(texts):
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
    # ponytail: O(n^2) по памяти; на 400-600 кандидатах это мегабайты и доли
    # секунды, иерархическая кластеризация тут не окупается."""
    n = len(embs)
    if n == 0:
        return np.zeros(0, dtype=int)
    E = np.asarray(embs, dtype=np.float32)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-10)
    sim = E @ E.T
    labels = np.full(n, -1, dtype=int)
    nxt = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        labels[(sim[i] >= threshold) & (labels < 0)] = nxt
        nxt += 1
    return labels


def flush_batch(conn, batch):
    """batch items: (qi, url, fetch_date, publish_date, title, text, region, lang, emb)"""
    if not batch:
        return 0
    inserted = 0
    for (qi, url, fd, pdate, title, text, bucket, lang, emb) in batch:''',
         "DC/flush")

MAIN = '''def main():
    log.info("=== Start daily_collector | mode=%s ===", _run_mode_label())
    if not os.path.exists(config.DB_PATH):
        log.error("DB not found: %s. Run init_db.py first.", config.DB_PATH)
        sys.exit(1)

    conn = sqlite3.connect(config.DB_PATH)
    seen_store.ensure(conn)
    fetch_date = time.strftime("%Y-%m-%d")
    total_new = 0
    week_mode = _is_week_mode()

    try:
        for qi in range(len(config.QUERIES_GKG)):
            log.info("--- Query #%d | mode=%s ---", qi, "week" if week_mode else "day")

            # 1) Очередь нерешённых разбирается ДО новых дампов: это статьи,
            #    по которым в прошлый раз молчали все провайдеры.
            queue = [(u, None) for u in seen_store.pending_urls(conn, qi)]
            if queue:
                log.info("Query #%d: %d URL из очереди pending", qi, len(queue))

            # 2) Новые кандидаты минус те, по которым решение уже окончательно.
            #    Раньше проверка шла по таблице articles, где лежат только
            #    ПРИНЯТЫЕ статьи, поэтому отклонённые возвращались каждые 6
            #    часов — 61.7% работы прогона уходило на повтор.
            fresh = [(a["url"], a["gkg_date"])
                     for a in fetch_gdelt(qi, week_mode=week_mode)]
            decided = seen_store.final_urls(conn, [u for u, _ in fresh])
            queue += [(u, d) for u, d in fresh if u not in decided]
            log.info("Query #%d: найдено %d, из них решены ранее %d, к обработке %d",
                     qi, len(fresh), len(decided), len(queue))

            # 3) Загрузка страниц и дешёвые правила.
            cands, marks = [], []
            for url, gkg_date in queue:
                text, title, html_date = fetch_and_extract(url)
                if not text or len(text) < config.MIN_TEXT_LENGTH:
                    marks.append((url, qi, "short", None))
                    continue
                lang = detect_text_language(text) or "unk"
                if is_non_event_article(url, title, text):
                    marks.append((url, qi, "non_event", None))
                    continue
                cands.append((url, text, html_date or gkg_date, title, lang))
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
            verdicts = {i: "NO" for i in auto_drop}
            if to_llm:
                log.info("Query #%d: %d судейств -> ~%d запросов (батч %d)",
                         qi, len(to_llm), -(-len(to_llm) // JUDGE_BATCH), JUDGE_BATCH)
                verdicts.update(zip(to_llm, judge_parallel([cands[i][1] for i in to_llm])))

            # 8) Вердикт кластера распространяется на всех его членов.
            marks, batch = [], []
            for i, (url, text, pdate, title, lang) in enumerate(cands):
                v = verdicts.get(reps[labels[i]])
                if v is None:
                    marks.append((url, qi, "pending", None))
                elif v == "NO":
                    marks.append((url, qi, "rejected", embs[i]))
                else:
                    marks.append((url, qi, "accepted", embs[i]))
                    batch.append((qi, url, fetch_date, pdate, title, text, v, lang, embs[i]))
            seen_store.mark(conn, marks, fetch_date)
            for k in range(0, len(batch), BATCH_SIZE):
                total_new += flush_batch(conn, batch[k:k + BATCH_SIZE])

            log.info("Query #%d: принято %d, отклонено %d, в pending %d",
                     qi,
                     sum(1 for m in marks if m[2] == "accepted"),
                     sum(1 for m in marks if m[2] == "rejected"),
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


'''

i0 = dc.find("def main():")
i1 = dc.find('if __name__ == "__main__":')
if i0 < 0 or i1 < 0 or i0 >= i1:
    errors.append("DC/main: границы main..__main__ не найдены")
else:
    dc = dc[:i0] + MAIN + dc[i1:]


# ===========================================================================
if errors:
    print("НЕ ПРИМЕНЕНО, ошибки:")
    for e in errors:
        print("  -", e)
    sys.exit(1)

wr("model_rotation.py", mr)
wr("init_db.py", db)
wr("config.py", cf)
wr("daily_collector.py", dc)
print("Патчи применены: model_rotation.py, init_db.py, config.py, daily_collector.py")
