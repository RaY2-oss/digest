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
    действительно английский, и отсутствие смешения скриптов. При провале —
    берём следующую статью из того же регионального пула.
    LLM-проверок темы здесь БОЛЬШЕ НЕТ: релевантность и регион статья получает
    один раз, при сборе (daily_collector._JUDGE_SYSTEM), а два прежних
    перепроверочных вызова (на сырой статье и на готовом пересказе) судили то же
    самое теми же критериями — и оба были fail-open, т.е. при мёртвых провайдерах
    молча пропускали всё. Стоили они по вызову на кандидата и по вызову на
    попытку: 492 + ~500 обращений за прогон 2026-07-28 при квоте порядка тысячи.
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
import prefilter
import translate_ru
from model_rotation import _call_openrouter_raw, providers_exhausted
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


def _versions(art: tuple) -> list:
    """Все версии сюжета: [(url, text, ...), ...]. У несгруппированной
    статьи — она сама, поэтому вызывать можно на чём угодно."""
    return art[6] if len(art) > 6 else [art]


def _group_stories(subset: list) -> list:
    """Почти-дубликаты одной новости -> один сюжет.

    Раньше каждая перепечатка шла в отбор отдельным кандидатом. Это стоило
    дважды: MMR мог отдать половину квоты шести зеркалам одной ленты, а
    «сколько редакций вообще написали о событии» — сильнейший бесплатный
    признак масштаба — нигде не учитывалось, хотя кластеры считались и
    выбрасывались. Теперь сюжет идёт одной записью, а список его версий
    едет седьмым полем: он даёт охват для _importance и тексты для пересказа.

    Представителем берём САМЫЙ ДЛИННЫЙ текст: по нему и эмбеддинг честнее,
    и пересказ полнее, если остальные версии не влезут в промпт.
    """
    if len(subset) <= 1:
        return [tuple(a) + (list(subset),) for a in subset]

    # ponytail: жадная однопроходная кластеризация, O(n^2) по схожести.
    # Корзина — до ~2 тыс. статей, это секунды; если вырастет на порядок,
    # менять на LSH/faiss, а не подкручивать порог.
    E = np.vstack([a[2] for a in subset]).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-10
    label = np.full(len(subset), -1)
    n_lab = 0
    for i in range(len(subset)):
        if label[i] >= 0:
            continue
        label[(E @ E[i] >= config.STORY_COSINE) & (label < 0)] = n_lab
        n_lab += 1

    stories = []
    for k in range(n_lab):
        members = [subset[i] for i in np.flatnonzero(label == k)]
        members.sort(key=lambda a: len(a[1] or ""), reverse=True)
        stories.append(tuple(members[0]) + (members,))
    return stories


def _importance(subset: list, embeddings: np.ndarray) -> np.ndarray:
    """Важность сюжетов корзины в [0, 1] — четыре фактора, без LLM.

        lexrank  — тематическая центральность в недельном окне;
        coverage — сколько РАЗНЫХ изданий написали о сюжете;
        entity   — политический вес субъектов (см. entities.py);
        topic    — вероятность локального предфильтра «это наша тема».

    Веса — в config.IMPORTANCE_W_*; ноль выключает фактор. Фактор topic
    выключается сам, пока артефакт предфильтра не обучен.

    Почему одного LexRank было мало: он меряет плотность связей, а корзина
    на 40% состоит из однотипных заметок «школьник сдал экзамен» из разных
    городов — они образуют такую же плотную клику, как настоящий сюжет, и
    заметка про одну школу в Кадыкёе получала важность 0.966.
    """
    lex = importance.minmax(_lexrank_scores(
        embeddings, domains=[importance.domain(a[0]) for a in subset]))

    cov = np.array([importance.coverage_weight(
        importance.distinct_domains(v[0] for v in _versions(a)),
        config.COVERAGE_FULL_AT) for a in subset], dtype=np.float64)

    ent = np.array([max(v[5] for v in _versions(a)) for a in subset],
                   dtype=np.float64)
    w_ent = config.IMPORTANCE_W_ENTITY if ent.any() else 0.0

    factors = [(config.IMPORTANCE_W_LEXRANK, lex),
               (config.IMPORTANCE_W_COVERAGE, cov),
               (w_ent, ent)]

    p = prefilter.score(config.PREFILTER_PATH, embeddings)
    if p is not None:
        factors.append((config.IMPORTANCE_W_TOPIC, importance.pct_rank(p)))

    return importance.blend(factors)


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
        stories = _group_stories(buckets[key])
        log.info(" [%s] %d статей -> %d сюжетов (порог %.2f)",
                 key, len(buckets[key]), len(stories), config.STORY_COSINE)
        picked = _mmr_pick(stories, quota)
        log.info(" [%s] выбрано %d / квота %d", key, len(picked), quota)
        result.extend(picked)
        # Занятыми считаются ВСЕ версии выбранного сюжета, а не только
        # представитель: иначе его же перепечатка вернётся из резерва как
        # «другая статья».
        picked_urls = {v[0] for p in picked for v in _versions(p)}
        selected_urls.update(picked_urls)
        # Остаток корзины → региональный резерв для замен, по важности.
        # Резерв тоже из сюжетов: замена приходит со своими версиями для
        # пересказа, как и основной кандидат.
        leftover_bucket = [s for s in stories if s[0] not in picked_urls]
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
            extra = _mmr_pick(_group_stories(all_leftover), deficit)
            result.extend(extra)
            selected_urls.update(v[0] for p in extra for v in _versions(p))

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
- You may receive SEVERAL versions of the SAME story, filed by different outlets \
and separated by a line of dashes. They are one news event, not several. Merge \
them into ONE summary: use a fact if any version reports it, and where versions \
disagree prefer the one reported by more of them. Never present them as separate \
items and never invent a link between them that no version states.

Return ONLY valid JSON, no markdown fences:
{"title": "<title>", "summary": "<3–4 sentence summary>"}
"""

_USER_PROMPT_RETELL_TEMPLATE = "Article URL: {url}\nArticle text:\n{text}"


def _retell_sources(versions: list) -> str:
    """Тексты версий сюжета в один промпт, под общим потолком по символам.

    Разные редакции сохраняют разные детали (цифры, имена, цитаты), поэтому
    пересказ по нескольким версиям полнее. Потолки — из config: одна версия
    получает столько же символов, сколько получала раньше единственная
    статья, дальше бюджет делится, чтобы промпт не вылезал из контекста
    бесплатных моделей.
    """
    versions = versions[:config.RETELL_MAX_VERSIONS]
    per = min(config.RETELL_CHARS_PER_VERSION,
              config.RETELL_CHARS_TOTAL // len(versions))
    return "\n\n----------\n\n".join(
        _USER_PROMPT_RETELL_TEMPLATE.format(url=v[0], text=(v[1] or "")[:per])
        for v in versions)

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
# Пересказ одной статьи: LLM-пересказ + быстрая regex-валидация формата.
# Тематическая релевантность статьи решена ещё при сборе (daily_collector),
# здесь её не перепроверяем — см. п.4a в шапке модуля.
# ---------------------------------------------------------------------------
def _retell_article(url: str, versions: list, pub_date: str, attempt_label: str) -> dict | None:
    user_msg = _retell_sources(versions)
    urls = [v[0] for v in versions[:config.RETELL_MAX_VERSIONS]]
    if len(urls) > 1:
        log.info(" %s: пересказ по %d версиям сюжета", attempt_label, len(urls))

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

        result["url"]          = url
        result["urls"]         = urls      # все источники, попавшие в промпт
        result["publish_date"] = pub_date
        return result

    log.error(" %s: все %d попытки исчерпаны", attempt_label, MAX_RETRIES)
    return None


# ---------------------------------------------------------------------------
# Обработка одного слота: дедуп-проверка, затем пересказ.
# При провале формата — следующая статья из регионального MMR-резерва
# ---------------------------------------------------------------------------
MAX_RESERVE_SCAN = 50

def _process_slot(
    primary: tuple,
    regional_reserves: dict,
    processed_urls: set,
    accepted_embeddings: list[np.ndarray],
    slot_label: str,
) -> dict | None:
    scanned = 0
    primary_bucket = primary[4]

    candidates = [primary]
    candidates.extend(item for item in regional_reserves.get(primary_bucket, [])
                      if item[0] not in processed_urls)

    for cand in candidates:
        url, _text, emb, pub_date = cand[:4]
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

        result = _retell_article(url, _versions(cand), pub_date, slot_label)
        if result is None:
            # Провал пересказа при мёртвых провайдерах — это не "статья плохая",
            # а "LLM недоступна". Перебор резерва в таком состоянии просто
            # выжигает корзину за секунды и оставляет слот пустым; выходим сразу,
            # чтобы следующие слоты не потеряли ещё и свои кандидаты.
            if providers_exhausted():
                log.error(" %s: LLM недоступна — прекращаем перебор резерва %s.",
                          slot_label, primary_bucket)
                return None
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

        for rep in reps:
            if len(summaries) >= target:
                break
            if rep[0] in processed_urls:
                continue

            slot_label = f"[{len(summaries)+1}/{target}]"
            result = _process_slot(
                primary=rep,
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
                if providers_exhausted():
                    log.error("LLM недоступна — набор слотов прерван на %d из %d.",
                              len(summaries), target)
                    break

        log.info("Итого пересказано статей: %d из %d", len(summaries), target)

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
        # Машиночитаемый итог в stdout: telegram_bot_listener читает его, чтобы
        # не отвечать бодрым "✅ сформирован", когда слотов набралось 5 из 20.
        print(f"DIGEST_ARTICLES={len(summaries)}/{target}", flush=True)
        log.info("=== Готово ===")
    finally:
        conn.close()

if __name__ == "__main__":
    main()