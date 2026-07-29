# /opt/digest/sunday_processor_mmr.py
# -*- coding: utf-8 -*-
"""
sunday_processor_mmr.py — воскресная обработка: отбор 20 репрезентативных статей
методом MMR (Maximal Marginal Relevance), пересказ через OpenRouter,
сборка .docx и отправка в Telegram.

Запускается по cron в воскресенье 08:00 UTC.

Пайплайн:
1. SELECT статей за последние 7 дней (publish_date или fetch_date).
2. Десериализация BLOB -> numpy float32 (dim = EMBEDDING_DIM).
3. Квотированный MMR по корзинам TR=7, CA=7, SC=6.
   Дефицит квот покрывается из leftover + MIX.
4. Для каждого слота из пула — пересказ через _call_openrouter_raw
   (system-промпт _SYSTEM_PROMPT_RETELL передаётся напрямую, без глобального
   состояния — раньше это делалось через несработавшую подмену config.SYSTEM_PROMPT).
   Пересказ формируется на АНГЛИЙСКОМ — независимо от языка источника.
4a. _validate_summary_fast: regex + langdetect проверяют, что вывод
    действительно английский, и отсутствие смешения скриптов.
4b. _validate_topic_llm: LLM проверяет тематическую релевантность и чистоту
    темы ГОТОВОГО пересказа (а не только исходной статьи, см. _validate_topic_pre_llm).
    При провале 4b — берём следующую статью из того же регионального пула,
    пока не найдём подходящую.
5. Локальный перевод английского пересказа на русский (translate_ru.translate_doc,
   тот же движок opus-mt-tc-big-en-zle, что и в /opt/gdelt_rss) — после
   _postprocess_digest, чтобы дедуп дубликатов события считался по стабильному
   английскому тексту, а не по возможно разошедшимся переводам.
6. build_digest() -> .docx.
7. send_document() -> Telegram.
8. Очистка старых строк: DELETE WHERE fetch_date < now-8.

MMR-формула (lambda=0.5):
score(d) = lambda * importance(d) - (1-lambda) * max_{s in S} sim(d, s)
где importance(d) — см. _importance(): LexRank-скор статьи в графе косинусных
схожестей корзины (степенная итерация, аналог PageRank), нормализованный
в [0, 1], смешанный с политическим весом статьи,
S = уже выбранные статьи (нарастающий список),
sim = косинусное сходство.

Почему не центроид: похожесть на центроид корзины отражает "типичность"
статьи для средней темы недели, а не её значимость — заметный выброс
(нетипичная, но важная новость) получал бы заниженный score. LexRank
вместо этого учитывает всю структуру графа схожести: статья считается
важной, если на неё "похожи" другие статьи, которые сами хорошо связаны
с остальными (т.е. её сюжет широко перекликается с другими материалами
недели), без жёсткого порога отсечения, как при кластеризации.

LexRank считается ПО КОРЗИНЕ (региону) — см. select_representatives: граф
строится внутри TR/CA/SC отдельно, поэтому крупный регион не топит малый.
Рёбра внутри одного домена обнулены (см. _lexrank_scores): пять версий
текста на одном сайте — один голос, а не клика из пяти.

Почему одного LexRank мало: он меряет ТОЛЬКО плотность связей, поэтому пачка
однотипных локальных заметок (десяток текстов «школьник сдал экзамен на
максимум баллов») образует такую же плотную клику, как настоящий
общенациональный сюжет. Второй фактор смотрит на субъектов статьи —
персоны/организации, которых GDELT уже извлёк за нас, — и поднимает те, где
фигурируют субъекты, о которых пишет и вся остальная лента (см. entities.py,
config.ENTITY_*).
"""

import json
import logging
import os
import random
import re
import sqlite3
import sys

import numpy as np
from langdetect import DetectorFactory, LangDetectException, detect

import config
import entities
import importance
import translate_ru
from model_rotation import _call_openrouter_raw
from word_generator import build_digest
from telegram_sender import send_document

DetectorFactory.seed = 0  # детерминированный langdetect, как в daily_collector.py

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------------
def setup_logging():
    os.makedirs(config.LOG_DIR, exist_ok=True)
    log_path = os.path.join(config.LOG_DIR, "processor.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("processor")

log = setup_logging()

# ---------------------------------------------------------------------------
# Квоты
# ---------------------------------------------------------------------------
QUOTA = {"TR": 7, "CA": 7, "SC": 6}
MMR_LAMBDA = 0.5
DEFICIT_POOL_CAP = 500  # см. select_representatives(): защита от OOM на LexRank
LEXRANK_DAMPING = 0.85
LEXRANK_MAX_ITER = 100
LEXRANK_TOL = 1e-6

# ---------------------------------------------------------------------------
# Выборка недельного корпуса
# ---------------------------------------------------------------------------
def load_week_articles(conn):
    rows = conn.execute(
        "SELECT url, text, embedding, publish_date, region_bucket, entities FROM articles "
        "WHERE ("
        " (publish_date IS NOT NULL AND publish_date >= date('now', '-7 days'))"
        " OR"
        " (publish_date IS NULL AND fetch_date >= date('now', '-7 days'))"
        ") "
        "AND embedding IS NOT NULL"
    ).fetchall()

    # Док-частота субъектов по всему недельному корпусу — самонастраивающийся
    # «словарь» заметных персон/организаций (см. entities.py). Пока в нём нет
    # ни одного субъекта выше порога (старая база без колонки entities), фактор
    # молча выключается: политический вес у всех статей = 0.
    ent_df = entities.document_freq(r[5] for r in rows)
    known = sum(1 for v in ent_df.values() if v >= config.ENTITY_MIN_DF)
    if not known:
        ent_df = None

    articles = []
    for url, text, blob, pub_date, bucket, ents in rows:
        if not blob:
            continue
        emb = np.frombuffer(blob, dtype=np.float32).copy()
        if emb.shape[0] != config.EMBEDDING_DIM:
            log.warning(
                "Пропуск %s: размерность %d != %d",
                url, emb.shape[0], config.EMBEDDING_DIM,
            )
            continue
        prom = 0.0 if ent_df is None else entities.weight(
            entities.prominent(ents, ent_df, config.ENTITY_MIN_DF),
            config.ENTITY_FULL_AT)
        articles.append((url, text, emb, pub_date or "", bucket or "MIX", prom))

    log.info("Загружено статей за неделю: %d | заметных субъектов в словаре: %d",
             len(articles), known)
    return articles

# ---------------------------------------------------------------------------
# MMR-ядро
# ---------------------------------------------------------------------------
def _cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return A_norm @ B_norm.T

def _lexrank_scores(
    embeddings: np.ndarray,
    damping: float = LEXRANK_DAMPING,
    max_iter: int = LEXRANK_MAX_ITER,
    tol: float = LEXRANK_TOL,
    domains: list | None = None,
) -> np.ndarray:
    """
    LexRank-важность статей корзины: степенная итерация (аналог PageRank)
    по графу косинусных схожестей между всеми статьями корзины.

    Статья получает высокий скор, если на неё похожи другие статьи,
    которые сами хорошо связаны с остальным графом — т.е. её сюжет
    широко перекликается с другими материалами недели, а не просто
    близок к "средней теме" (как при similarity к центроиду).

    Возвращает вектор скоров (сумма = 1), без нормализации в [0,1] —
    нормализация делается на стороне вызывающего кода.
    """
    n = embeddings.shape[0]
    if n <= 1:
        return np.ones(n, dtype=np.float64)

    sim = _cosine_matrix(embeddings, embeddings).astype(np.float64)
    np.fill_diagonal(sim, 0.0)
    sim = np.clip(sim, 0.0, None)  # отрицательную "схожесть" не переносим как вес ребра

    # Рёбра между статьями ОДНОГО издания обнуляются. LexRank меряет плотность
    # связей, и издание, выкатившее пять версий одного текста, образует плотную
    # клику, которая накачивает сама себя, — ровно тот локальный шум, против
    # которого фактор и вводился. Пять версий на одном сайте должны весить как
    # один голос, а не как пять.
    if domains is not None:
        d = np.asarray(domains, dtype=object)
        sim[(d[:, None] == d[None, :]) & (d[:, None] != "")] = 0.0

    row_sums = sim.sum(axis=1, keepdims=True)
    dangling = (row_sums.flatten() == 0.0)
    row_sums_safe = np.where(row_sums == 0.0, 1.0, row_sums)
    trans = sim / row_sums_safe
    if dangling.any():
        # узел без связей (ни на кого не похож) — равномерно раздаём вес,
        # иначе он "проглатывает" скор и никуда его не возвращает
        trans[dangling, :] = 1.0 / n

    scores = np.full(n, 1.0 / n, dtype=np.float64)
    teleport = (1.0 - damping) / n
    for _ in range(max_iter):
        new_scores = teleport + damping * (trans.T @ scores)
        if np.abs(new_scores - scores).sum() < tol:
            scores = new_scores
            break
        scores = new_scores

    return scores


def _importance(subset: list, embeddings: np.ndarray) -> np.ndarray:
    """Важность статей корзины в [0, 1]: LexRank + политический вес.

    LexRank нормализуется в [0,1] и смешивается с уже посчитанным при загрузке
    политическим весом статьи (subset[i][5], см. load_week_articles):

        importance = (1 - W) * lexrank + W * политический_вес,  W = ENTITY_WEIGHT

    Если фактор выключен (W=0) или по корзине нет ни одного заметного субъекта,
    возвращается чистый LexRank — прежнее поведение.
    """
    rel = _lexrank_scores(embeddings,
                          domains=[importance.domain(a[0]) for a in subset])
    r_min, r_max = float(rel.min()), float(rel.max())
    if r_max - r_min > 1e-12:
        rel = (rel - r_min) / (r_max - r_min)
    else:
        rel = np.full_like(rel, 0.5)

    w = config.ENTITY_WEIGHT
    if w <= 0:
        return rel
    prom = np.array([a[5] for a in subset], dtype=np.float64)
    if not prom.any():
        return rel
    return (1.0 - w) * rel + w * prom


def _max_cosine_to_selected(emb: np.ndarray, selected_embs: list[np.ndarray]) -> float:
    """
    Возвращает максимальное косинусное сходство кандидата
    с уже принятыми в дайджест статьями.
    """
    if not selected_embs:
        return -1.0

    A = emb.reshape(1, -1).astype(np.float32)
    B = np.vstack(selected_embs).astype(np.float32)
    sims = _cosine_matrix(A, B).flatten()
    return float(sims.max())


def _is_semantic_duplicate(
    emb: np.ndarray,
    selected_embs: list[np.ndarray],
    threshold: float = 0.93,
) -> tuple[bool, float]:
    """
    threshold:
    - 0.90..0.93  -> мягкий фильтр
    - 0.93..0.96  -> нормальный фильтр
    - 0.96+       -> только почти идентичные материалы
    """
    max_sim = _max_cosine_to_selected(emb, selected_embs)
    return max_sim >= threshold, max_sim

def _mmr_pick(subset: list, n_pick: int, lam: float = MMR_LAMBDA) -> list:
    """
    MMR-отбор n_pick статей из subset.
    Возвращает list of (url, text, emb, pub_date, bucket).
    """
    if not subset:
        return []
    n_pick = min(n_pick, len(subset))
    if n_pick <= 0:
        return []
    if n_pick == len(subset):
        return list(subset)

    embeddings = np.vstack([a[2] for a in subset]).astype(np.float32)
    rel_scores = _importance(subset, embeddings)

    selected_idx = []
    candidate_mask = np.ones(len(subset), dtype=bool)

    for step in range(n_pick):
        if step == 0:
            mmr_scores = rel_scores.copy()
        else:
            sel_emb = embeddings[selected_idx]
            sim_to_sel = _cosine_matrix(embeddings, sel_emb)
            max_sim = sim_to_sel.max(axis=1)
            mmr_scores = lam * rel_scores - (1.0 - lam) * max_sim

        mmr_scores_masked = np.where(candidate_mask, mmr_scores, -np.inf)
        best = int(np.argmax(mmr_scores_masked))
        selected_idx.append(best)
        candidate_mask[best] = False

        log.info(
            " MMR step %d/%d: idx=%d mmr=%.4f imp=%.4f url=%s",
            step + 1, n_pick, best,
            float(mmr_scores[best]), float(rel_scores[best]),
            subset[best][0],
        )

    return [subset[i] for i in selected_idx]


def _rank_by_importance(subset: list) -> list:
    """
    Сортирует статьи по важности (LexRank + политический вес, см. _importance),
    БЕЗ диверсити-члена MMR.

    Используется для регионального резерва (замена нерелевантных статей):
    диверсити тут не нужен, т.к. семантические дубликаты уже отсекаются
    отдельно в _process_slot() через _is_semantic_duplicate() на этапе
    перебора резерва — значит, важно просто отдавать кандидатов в порядке
    "следующий самый значимый", а не в порядке из SQL-запроса.
    """
    if len(subset) <= 1:
        return list(subset)

    embeddings = np.vstack([a[2] for a in subset]).astype(np.float32)
    scores = _importance(subset, embeddings)
    order = np.argsort(-scores)
    return [subset[i] for i in order]

# ---------------------------------------------------------------------------
# Квотированный отбор — возвращает основной список + региональные резервы
# ---------------------------------------------------------------------------
def select_representatives(articles: list) -> tuple[list, dict]:
    """
    Возвращает (reps, regional_reserves).

    reps — список (url, text, emb, pub_date, bucket) длиной <= config.N_CLUSTERS.
    regional_reserves — dict {bucket: [(url, text, emb, pub_date, bucket), ...]}
                        остатки по каждому региону, отсортированные по
                        LexRank-важности (убывание, см. _rank_by_importance).
                        Используются для замены нерелевантных статей —
                        замена берётся из той же корзины, что и основная статья,
                        в порядке "следующая самая значимая".
    """
    if not articles:
        return [], {}

    buckets = {"TR": [], "CA": [], "SC": [], "MIX": []}
    for art in articles:
        b = art[4] if art[4] in buckets else "MIX"
        buckets[b].append(art)

    log.info(
        "Корзины: TR=%d CA=%d SC=%d MIX=%d",
        len(buckets["TR"]), len(buckets["CA"]),
        len(buckets["SC"]), len(buckets["MIX"]),
    )

    result = []
    regional_reserves: dict[str, list] = {k: [] for k in buckets}
    selected_urls: set[str] = set()

    for key, quota in QUOTA.items():
        picked = _mmr_pick(buckets[key], quota)
        log.info(" [%s] выбрано %d / квота %d", key, len(picked), quota)
        result.extend(picked)
        picked_urls = {p[0] for p in picked}
        selected_urls.update(picked_urls)
        # Остаток корзины → региональный резерв для замен, по важности
        leftover_bucket = [a for a in buckets[key] if a[0] not in picked_urls]
        regional_reserves[key] = _rank_by_importance(leftover_bucket)

    leftover_mix = [a for a in buckets["MIX"] if a[0] not in selected_urls]
    # MIX может быть большим (десятки тысяч статей) — LexRank требует плотную
    # матрицу схожести N×N, что на слабом VPS уводит процесс в OOM (см. инцидент
    # 2026-07-20). Резерв MIX используется только как fallback при дефиците,
    # поэтому ранжирование по важности здесь не считаем — берём как есть.
    regional_reserves["MIX"] = leftover_mix

    deficit = config.N_CLUSTERS - len(result)
    if deficit > 0:
        all_leftover = [a for a in articles if a[0] not in selected_urls]
        if all_leftover:
            # Дефицит — редкий случай, точность MMR тут не критична, а
            # прогонять LexRank/MMR по всей неделе (десятки тысяч статей)
            # уже роняло процесс OOM'ом (см. инцидент 2026-07-20). Поэтому
            # сперва режем пул случайной выборкой до разумного размера.
            if len(all_leftover) > DEFICIT_POOL_CAP:
                all_leftover = random.sample(all_leftover, DEFICIT_POOL_CAP)
            log.info("Дефицит %d статей, добираем из leftover+MIX (%d)", deficit, len(all_leftover))
            extra = _mmr_pick(all_leftover, deficit)
            result.extend(extra)
            selected_urls.update(p[0] for p in extra)

    log.info("Выбрано представителей: %d", len(result))
    return result, regional_reserves

# ---------------------------------------------------------------------------
# Системные промпты
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT_RETELL = """\
Retell the source article in English for an academic digest, in about 3–4 sentences.
The source article may be in Turkish, Russian, English, or another language — \
regardless of the source language, your ENTIRE output (both title and summary) \
must be written in English. A separate local translation step converts this \
English text to Russian afterwards, so keep all proper names in their normal \
English/original spelling — do not transliterate anything.

Rules:
- Length: the summary must be about 3–4 sentences — concise, only the essential facts.
- Do not leave ANY word, phrase, or clause untranslated in Turkish, Russian, or any \
other language — translate everything into English.
- English grammar must be correct.
- Do not invent facts.
- Keep ONLY content related to science, education, youth policy in Turkey, Central Asia, South Caucasus, or external engagement involving Central Asia and South Caucasus. \
If the source article also covers other, unrelated topics or events, SILENTLY DROP \
them — do not summarize, mention, or even briefly reference the unrelated parts. \
The final summary must read as if the unrelated content never existed in the source.
- On first mention, expand abbreviations: full form first, then abbreviation in parentheses.
- ALWAYS make the geographic or institutional context explicit.
- The title MUST explicitly indicate the place or institution the news is about, \
for example: "In Turkey ...", "In Uzbekistan ...", "In Baku ...", \
"At Tashkent State University ...", "Kazakhstan's Ministry of Education ...".
- The summary MUST explicitly mention the place, country, city, institution, or \
organisation the article is about in the first 1–2 sentences.

Return ONLY valid JSON, no markdown fences:
{"title": "<title>", "summary": "<3–4 sentence summary>"}
"""

_SYSTEM_PROMPT_VALIDATE_TOPIC = """\
You are a relevance-check assistant for an academic digest.

The digest covers ONLY the following topics:
- Science, education, and youth policy IN Turkey, Central Asia \
  (Kazakhstan, Uzbekistan, Kyrgyzstan, Tajikistan, Turkmenistan), \
  or the South Caucasus (Georgia, Armenia, Azerbaijan).
- Actions, policies, or initiatives OF the EU, USA, Iran, India, China, Japan, \
  or South Korea directed AT Central Asia or the South Caucasus.

Given a JSON with "title" and "summary" fields, decide whether the content \
matches at least one of these topics, AND stays focused ONLY on such topics \
throughout. If the item is mostly on-topic but also mentions, describes, or \
drifts into an unrelated event or subject anywhere in the title or summary, \
that counts as a FAIL — the digest must never mix in unrelated topics, even briefly.

Respond ONLY with valid JSON, no markdown fences:
{"ok": true}
or
{"ok": false, "reason": "<short explanation in English>"}
"""

_USER_PROMPT_RETELL_TEMPLATE = "Article URL: {url}\nArticle text:\n{text}"

# ---------------------------------------------------------------------------
# Вспомогательный парсер JSON из ответа LLM
# ---------------------------------------------------------------------------
def _parse_llm_json(raw: str | None) -> dict | None:
    """
    Принимает сырой текстовый ответ _call_openrouter_raw().
    Убирает markdown-обёртки ```json ... ``` и парсит JSON.
    Возвращает dict или None при ошибке.
    """
    if raw is None:
        return None
    cleaned = re.sub(r"^```[a-z]*\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        log.warning("JSON parse error: %s | raw: %r", exc, raw[:200])
        return None

# ---------------------------------------------------------------------------
# Валидация темы — LLM (один вызов, только когда формат уже прошёл)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Промпт для ПРЕДВАРИТЕЛЬНОЙ проверки темы — до пересказа, на сырой статье
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Промпты предварительной проверки темы (ДО пересказа)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TOPIC_PRE_TR = """\
You are a relevance-check assistant for an academic digest about Turkey.

The digest accepts articles that cover ANY of the following topics \
related to Turkey (Türkiye):
- Science, research, universities, or academic institutions in Turkey.
- Education policy, school reforms, curriculum, or student programs in Turkey.
- Youth policy, youth organizations, or youth-related government programs in Turkey.
- Turkish academic or educational cooperation with any foreign country.

IMPORTANT: The article must be of NATIONAL significance — i.e. it concerns \
state-level policy, major institutions, large-scale programs, or events with \
nationwide impact. Reject local, municipal, or purely ceremonial news \
(e.g. a single school event, a minor local award ceremony).

The article may be in any language. Judge by content, not language.

Respond ONLY with valid JSON, no markdown fences:
{"ok": true}
or
{"ok": false, "reason": "<short explanation in English>"}
"""

_SYSTEM_PROMPT_TOPIC_PRE_INTL = """\
You are a relevance-check assistant for an academic digest.

The digest accepts articles that cover ANY of the following topics:
- Science, education, or youth policy IN Central Asia \
(Kazakhstan, Uzbekistan, Kyrgyzstan, Tajikistan, Turkmenistan).
- Science, education, or youth policy IN the South Caucasus \
(Georgia, Armenia, Azerbaijan).
- Actions, policies, or initiatives OF the EU, USA, Iran, India, China, Japan, \
or South Korea directed AT Central Asia or the South Caucasus.

IMPORTANT: The article must be of NATIONAL significance — i.e. it concerns \
state-level policy, major institutions, large-scale programs, or events with \
nationwide impact. Reject local, municipal, or purely ceremonial news \
(e.g. a single school event, a minor local award ceremony).

The article may be in any language. Judge by content, not language.

Respond ONLY with valid JSON, no markdown fences:
{"ok": true}
or
{"ok": false, "reason": "<short explanation in English>"}
"""

def _validate_topic_pre_llm(text: str, url: str, bucket: str) -> tuple[bool, str]:
    """
    Предварительная тематическая проверка на сыром тексте статьи.
    Задаём вопрос явно в user-сообщении, чтобы модель не уходила в пересказ.
    """
    system = _SYSTEM_PROMPT_TOPIC_PRE_TR if bucket == "TR" else _SYSTEM_PROMPT_TOPIC_PRE_INTL
    snippet = text.replace("\n", " ").strip()

    # Формируем user-сообщение как явный вопрос, а не просто текст статьи
    user_msg = (
        "Does the following article match the digest topics? "
        "Reply ONLY with JSON {\"ok\": true} or {\"ok\": false, \"reason\": \"...\"}.\n\n"
        f"Article text:\n{snippet}"
    )

    raw    = _call_openrouter_raw(system, user_msg, ref_url=url)
    parsed = _parse_llm_json(raw)
    if parsed is None:
        # Парс-фейл (LLM недоступна / вернула не-JSON) — это поломка валидатора, а
        # не сигнал "не по теме". Пропускаем статью дальше (fail-open): её тему ещё
        # раз проверит _validate_topic_llm на готовом пересказе.
        log.warning("_validate_topic_pre_llm: ответ не распарсился для %s | raw=%r — пропускаем (fail-open)", url, str(raw)[:200])
        return True, "pre-check недоступен — fail-open"
    if parsed.get("ok") is True:
        return True, "ok"
    return False, parsed.get("reason", "LLM: тема не соответствует дайджесту")


# ---------------------------------------------------------------------------
# Валидация формата и языка пересказа — БЫСТРАЯ (regex, без LLM)
# Ограничение по длине УБРАНО.
# ---------------------------------------------------------------------------
_MIXED_SCRIPT_RE = re.compile(
    r"[а-яёА-ЯЁ][a-zA-Z][а-яёА-ЯЁ]"
    r"|[a-zA-Z][а-яёА-ЯЁ][a-zA-Z]",
    flags=re.UNICODE,
)

# ponytail: цена ложных срабатываний langdetect на коротком тексте (3-4
# предложения) выше цены пропуска редкого нецелевого языка — fail-open.
def _validate_summary_fast(result: dict) -> tuple[bool, str]:
    title   = result.get("title", "")
    summary = result.get("summary", "")
    full    = f"{title} {summary}"

    if not title or not summary:
        return False, "пустой title или summary"

    # Заголовок-обрубок: слабая модель иногда выдаёт только префикс-локацию
    # ("In Turkey", "Uzbekistan:") без сути новости. Требуем минимум 3 значимых
    # слова — иначе перевыбор модели/статьи.
    title_words = [w for w in re.findall(r"\w+", title, re.UNICODE) if len(w) > 1]
    if len(title_words) < 3:
        return False, f"заголовок-обрубок: «{title}»"

    m = _MIXED_SCRIPT_RE.search(full)
    if m:
        return False, f"смешение скриптов внутри слова: «{m.group()}»"

    try:
        lang = detect(full)
    except LangDetectException:
        lang = None  # слишком коротко/неоднозначно — fail-open, не браковать
    if lang is not None and lang != "en":
        return False, f"пересказ не на английском (langdetect: {lang})"

    return True, "ok"


# ---------------------------------------------------------------------------
# Валидация ГОТОВОГО пересказа на чистоту темы (LLM) — уже после пересказа,
# т.к. модель-пересказчик иногда всё равно подмешивает лишнее из источника,
# даже если сама статья была одобрена на этапе _validate_topic_pre_llm.
# ---------------------------------------------------------------------------
def _validate_topic_llm(result: dict, url: str) -> tuple[bool, str]:
    payload = {"title": result.get("title", ""), "summary": result.get("summary", "")}
    user_msg = (
        "Does the following digest item match the digest topics, and does it stay "
        "focused ONLY on those topics with no unrelated topics mixed in? "
        "Reply ONLY with JSON {\"ok\": true} or {\"ok\": false, \"reason\": \"...\"}.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    raw    = _call_openrouter_raw(_SYSTEM_PROMPT_VALIDATE_TOPIC, user_msg, ref_url=url)
    parsed = _parse_llm_json(raw)
    if parsed is None:
        # Парс-фейл валидатора не должен ронять уже сгенерированный пересказ —
        # это поломка проверки, а не доказательство "не по теме". Принимаем (fail-open).
        log.warning("_validate_topic_llm: ответ не распарсился для %s | raw=%r — принимаем (fail-open)", url, str(raw)[:200])
        return True, "topic-check недоступен — fail-open"
    if parsed.get("ok") is True:
        return True, "ok"
    return False, parsed.get("reason", "LLM: тема пересказа не соответствует дайджесту")


# ---------------------------------------------------------------------------
# Пересказ одной статьи: LLM-пересказ + быстрая regex-валидация формата +
# LLM-валидация чистоты темы готового пересказа.
# Тематическая релевантность ИСХОДНОЙ статьи проверяется заранее, в _process_slot.
# ---------------------------------------------------------------------------
def _retell_article(url: str, text: str, pub_date: str, attempt_label: str) -> dict | None:
    user_msg = _USER_PROMPT_RETELL_TEMPLATE.format(url=url, text=text[:3000])

    for attempt in range(1, MAX_RETRIES + 1):
        log.info(" %s попытка %d/%d: %s", attempt_label, attempt, MAX_RETRIES, url)
        raw    = _call_openrouter_raw(_SYSTEM_PROMPT_RETELL, user_msg, ref_url=url)
        result = _parse_llm_json(raw)

        if result is None:
            log.warning(" %s попытка %d: не удалось распарсить JSON", attempt_label, attempt)
            continue

        ok, reason = _validate_summary_fast(result)
        if not ok:
            log.warning(" %s попытка %d: %s", attempt_label, attempt, reason)
            continue

        ok_topic, reason_topic = _validate_topic_llm(result, url)
        if not ok_topic:
            log.warning(" %s попытка %d: тема пересказа не прошла проверку (%s)", attempt_label, attempt, reason_topic)
            continue

        result["url"]          = url
        result["publish_date"] = pub_date
        return result

    log.error(" %s: все %d попытки исчерпаны", attempt_label, MAX_RETRIES)
    return None


# ---------------------------------------------------------------------------
# Обработка одного слота: сначала тема (pre-check), потом пересказ
# При провале темы или формата — следующая статья из регионального MMR-резерва
# ---------------------------------------------------------------------------
MAX_RESERVE_SCAN = 50

def _process_slot(
    primary_url: str,
    primary_text: str,
    primary_emb: np.ndarray,
    primary_pub_date: str,
    primary_bucket: str,
    regional_reserves: dict,
    processed_urls: set,
    accepted_embeddings: list[np.ndarray],
    slot_label: str,
) -> dict | None:
    scanned = 0

    candidates = [(primary_url, primary_text, primary_emb, primary_pub_date)]
    for item in regional_reserves.get(primary_bucket, []):
        r_url, r_text, r_emb, r_pub_date = item[:4]
        if r_url not in processed_urls:
            candidates.append((r_url, r_text, r_emb, r_pub_date))

    for url, text, emb, pub_date in candidates:
        if scanned >= MAX_RESERVE_SCAN:
            log.warning(
                " %s: достигнут лимит сканирования %d — слот остаётся пустым.",
                slot_label, MAX_RESERVE_SCAN,
            )
            break

        if url in processed_urls:
            scanned += 1
            continue

        processed_urls.add(url)
        scanned += 1

        is_dup, max_sim = _is_semantic_duplicate(
            emb=emb,
            selected_embs=accepted_embeddings,
            threshold=0.90,
        )

        log.info(
            " %s: duplicate-check url=%s max_sim=%.4f threshold=%.2f",
            slot_label, url, max_sim, 0.90,
        )

        if is_dup:
            log.warning(
                " %s: пропуск %s как семантического дубликата (max_sim=%.4f) — берём следующую из корзины %s",
                slot_label, url, max_sim, primary_bucket,
            )
            continue

        ok_topic, reason_topic = _validate_topic_pre_llm(text, url, primary_bucket)
        if not ok_topic:
            log.warning(
                " %s: тема не подходит для %s (%s) — берём следующую из корзины %s",
                slot_label, url, reason_topic, primary_bucket,
            )
            continue

        result = _retell_article(url, text, pub_date, slot_label)
        if result is None:
            log.warning(
                " %s: пересказ/формат не прошёл для %s — берём следующую из корзины %s",
                slot_label, url, primary_bucket,
            )
            continue

        result["embedding"] = emb
        log.info(" %s: принята статья %s", slot_label, url)
        return result

    log.error(" %s: резерв корзины %s исчерпан — слот остаётся пустым.", slot_label, primary_bucket)
    return None

# ---------------------------------------------------------------------------
# Промпт для пост-обработки дайджеста
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT_POSTPROCESS = """You are a quality-control editor for an English-language academic digest.

You will receive a JSON array of digest items. Each item has:
  "idx"     — original index (integer, preserve it),
  "title"   — headline in English,
  "summary" — body text in English.

Apply ONE check and return a corrected JSON array:

1. DUPLICATE EVENTS
   If two or more items describe the same real-world event (same fact, same
   figures, same actors — even if worded differently), keep only the item
   with the higher idx value and remove the others.
   Do NOT remove items just because they share a topic — only remove true
   duplicates of the same specific event.

Return ONLY a valid JSON array (no markdown fences, no extra keys):
[
  {"idx": <int>, "title": "<corrected title>", "summary": "<corrected summary>"},
  ...
]
"""


def _postprocess_digest(summaries: list[dict]) -> list[dict]:
    """
    Пост-обработка сформированного дайджеста LLM-редактором:
    - устраняет смешение скриптов внутри слов;
    - удаляет дубликаты одного события.
    При ошибке LLM — возвращает исходный список без изменений.
    """
    items_for_llm = [
        {"idx": i, "title": s["title"], "summary": s["summary"]}
        for i, s in enumerate(summaries)
    ]
    user_msg = (
        "Apply both checks (mixed-script, duplicates) "
        "to the following digest items and return the corrected JSON array.\n\n"
        + json.dumps(items_for_llm, ensure_ascii=False, indent=2)
    )

    raw = _call_openrouter_raw(_SYSTEM_PROMPT_POSTPROCESS, user_msg, ref_url="postprocess")
    if raw is None:
        log.warning("Пост-обработка: LLM вернул None — оставляем дайджест без изменений")
        return summaries

    parsed = _parse_llm_json(raw)
    if not isinstance(parsed, list):
        log.warning("Пост-обработка: неверный формат ответа (type=%s) — оставляем без изменений | raw=%r",
                    type(parsed).__name__, str(raw)[:300])
        return summaries

    idx_to_corrected: dict[int, dict] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        idx = item.get("idx")
        if not isinstance(idx, int) or idx < 0 or idx >= len(summaries):
            continue
        idx_to_corrected[idx] = item

    result = []
    removed = []
    for i, original in enumerate(summaries):
        if i not in idx_to_corrected:
            removed.append(i)
            log.info("Пост-обработка: статья idx=%d ('%s') удалена как дубликат",
                     i, original["title"][:60])
            continue
        corrected = idx_to_corrected[i]
        merged = {**original}
        merged["title"]   = corrected.get("title",   original["title"])
        merged["summary"] = corrected.get("summary", original["summary"])
        result.append(merged)

    if removed:
        log.info("Пост-обработка: удалено %d дубликат(ов), осталось %d статей", len(removed), len(result))
    else:
        log.info("Пост-обработка: дубликатов не найдено, тексты исправлены")
    return result


# ---------------------------------------------------------------------------
# Локальный перевод английского пересказа на русский (translate_ru, тот же
# движок opus-mt-tc-big-en-zle, что и /opt/gdelt_rss — см. translate_worker.py).
# ---------------------------------------------------------------------------
def _translate_to_russian(summaries: list[dict]) -> list[dict]:
    result = []
    for item in summaries:
        try:
            title_ru, summary_ru, _route = translate_ru.translate_doc(
                item["title"], item["summary"])
        except Exception:
            log.exception("Перевод на русский упал для %s", item.get("url"))
            continue
        if title_ru is None or summary_ru is None:
            log.warning("Перевод на русский не удался для %s — статья пропущена",
                        item.get("url"))
            continue
        item["title"]   = title_ru
        item["summary"] = summary_ru
        result.append(item)
    translate_ru.unload_all()
    return result


# ---------------------------------------------------------------------------
# Очистка старых данных
# ---------------------------------------------------------------------------
def cleanup_old(conn):
    cur = conn.execute("DELETE FROM articles WHERE fetch_date < date('now', '-8 days')")
    conn.commit()
    log.info("Удалено устаревших строк: %d", cur.rowcount)

# ---------------------------------------------------------------------------
# Основной сценарий
# ---------------------------------------------------------------------------
def main():
    log.info("=== Старт sunday_processor (MMR) ===")

    if not os.path.exists(config.DB_PATH):
        log.error("БД не найдена: %s", config.DB_PATH)
        sys.exit(1)

    conn = sqlite3.connect(config.DB_PATH)
    try:
        articles = load_week_articles(conn)
        if not articles:
            log.warning("Нет статей за неделю — дайджест не формируется.")
            return

        reps, regional_reserves = select_representatives(articles)
        if not reps:
            log.warning("Не удалось выбрать представителей.")
            return

        summaries: list[dict] = []
        processed_urls: set[str] = set()
        accepted_embeddings: list[np.ndarray] = []
        target = config.N_CLUSTERS

        for url, text, emb, pub_date, bucket, _prom in reps:
            if len(summaries) >= target:
                break
            if url in processed_urls:
                continue

            slot_label = f"[{len(summaries)+1}/{target}]"
            result = _process_slot(
                primary_url=url,
                primary_text=text,
                primary_emb=emb,
                primary_pub_date=pub_date,
                primary_bucket=bucket,
                regional_reserves=regional_reserves,
                processed_urls=processed_urls,
                accepted_embeddings=accepted_embeddings,
                slot_label=slot_label,
            )

            if result is not None:
                accepted_embeddings.append(result["embedding"])
                summaries.append(result)
            else:
                log.warning("Слот %d не заполнен.", len(summaries) + 1)

        log.info("Итого пересказано статей: %d", len(summaries))

        if not summaries:
            log.error("Нет пересказанных статей — дайджест не формируется.")
            return

        summaries = _postprocess_digest(summaries)
        log.info("После пост-обработки статей: %d", len(summaries))

        if not summaries:
            log.error("После пост-обработки дайджест пуст.")
            return

        for item in summaries:
            item.pop("embedding", None)

        summaries = _translate_to_russian(summaries)
        log.info("После перевода на русский: %d", len(summaries))

        if not summaries:
            log.error("После перевода дайджест пуст.")
            return

        docx_path = build_digest(summaries)
        log.info("Документ сохранён: %s", docx_path)

        target_chat_id = os.environ.get("TARGET_CHAT_ID")
        ok = send_document(
            docx_path,
            chat_id=int(target_chat_id) if target_chat_id else None,
        )
        if ok:
            log.info("Документ отправлен в Telegram.")
        else:
            log.error("Не удалось отправить документ в Telegram (см. errors.log).")

        cleanup_old(conn)
        log.info("=== Готово ===")
    finally:
        conn.close()

if __name__ == "__main__":
    main()