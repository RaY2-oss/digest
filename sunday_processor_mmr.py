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
   Пересказ формируется СРАЗУ ПО-РУССКИ — независимо от языка источника.

   Раньше он писался по-английски, а на русский его переводила локальная
   opus-mt-tc-big-en-zle. Замер на шести живых статьях (07.08.2026) показал,
   что плечо перевода не столько переводит, сколько портит: «восемь школьных
   зданий будут усилены землетрясением» вместо «усилены НА СЛУЧАЙ
   землетрясения», адрес портала İŞKUR, развалившийся на «Искур. Гоув. tr»,
   и предложения по девяносто слов, потому что модель переводит фразу за
   фразой и не умеет их разбивать. У пересказа сразу по-русски этих поломок
   нет: тот же вызов LLM, но одним плечом вместо двух.

   Что при этом потеряно, честно: латинские имена больше не проходят через
   маркеры словаря нетронутыми, и модель иногда сама выбирает написание
   («şehit» как «шейх»). Лечится списком принятых написаний в промпте, см.
   _glossary_block.
4a. _validate_summary_fast: regex проверяет формат, долю кириллицы,
    отсутствие смешения скриптов и то, что слово источника не переписано
    кириллицей вместо перевода (translit_guard). При провале — берём следующую статью из
    того же регионального пула.
    LLM-проверок темы здесь БОЛЬШЕ НЕТ: релевантность и регион статья получает
    один раз, при сборе (daily_collector._JUDGE_SYSTEM), а два прежних
    перепроверочных вызова (на сырой статье и на готовом пересказе) судили то же
    самое теми же критериями — и оба были fail-open, т.е. при мёртвых провайдерах
    молча пропускали всё. Стоили они по вызову на кандидата и по вызову на
    попытку: 492 + ~500 обращений за прогон 2026-07-28 при квоте порядка тысячи.
4b. Дата пункта — дата СОБЫТИЯ, а не публикации: её называет та же LLM тем же
    вызовом (поле event_date), а печатается она, только если её подтвердил
    сторож event_date.choose — дата в недельном окне, не позже первой
    публикации и буквально стоящая в тексте, который модель читала. Не
    подтвердилась — остаётся дата первой перепечатки (_story_date). Старше
    недели в дайджест не попадает ничего: потолок стоит и в выборке (SQL), и
    на печати (event_date.clamp).
5. _postprocess_digest — LLM-редактор снимает дубликаты одного события.
   Отдельного шага перевода больше нет: п.4 уже отдаёт русский текст.
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
import unicodedata
from datetime import date

import numpy as np

import config
import entities
import event_date
import importance
import prefilter
import retell_select
import translit_guard
from model_rotation import _call_openrouter_raw, providers_exhausted
from word_generator import build_digest
from telegram_sender import send_document

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
        "SELECT url, text, embedding, publish_date, region_bucket, entities, scale "
        "FROM articles "
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
    graded = 0
    for url, text, blob, pub_date, bucket, ents, scale in rows:
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
        graded += scale in _SCALE_VALUE
        articles.append((url, text, emb, pub_date or "", bucket or "MIX", prom,
                         scale))

    log.info("Загружено статей за неделю: %d | заметных субъектов в словаре: %d "
             "| с оценкой масштаба: %d", len(articles), known, graded)
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


_ART_FIELDS = 7          # url, text, emb, pub_date, bucket, prom, scale
_SCALE_VALUE = {1: 0.0, 2: 0.5, 3: 1.0}
_SCALE_UNGRADED = 0.5    # старая строка без оценки не должна ни падать, ни расти


def _versions(art: tuple) -> list:
    """Все версии сюжета: [(url, text, ...), ...]. У несгруппированной
    статьи — она сама, поэтому вызывать можно на чём угодно."""
    return art[_ART_FIELDS] if len(art) > _ART_FIELDS else [art]


def _story_scale(art: tuple):
    """Национальный масштаб сюжета в [0,1] или None, если его никто не оценивал.

    По версиям берётся СРЕДНЕЕ, а не максимум. Максимум казался естественным
    («хоть одна редакция описала событие как национальное»), но замер 08.08.2026
    показал, в чём подвох: судья ставит тройку примерно каждой третьей принятой
    статье, поэтому у сюжета из тринадцати перепечаток тройка находится почти
    наверняка — и максимум превращает фактор в тот же охват, только шумный.
    Среднее пользуется перепечатками ровно наоборот: это несколько независимых
    оценок одного события, и разброс судьи они гасят.

    Независимых — поэтому голос считается ПО ИЗДАНИЮ, а не по версии: сайт,
    выложивший новость и её же обновление, судью не переспрашивает, он повторяет
    один и тот же текст, и судья отвечает на него одинаково. Так же считается
    охват (importance.publisher) и так же обнулены внутридоменные рёбра
    LexRank — масштаб был последним фактором, где одно издание голосовало
    столько раз, сколько опубликовало. За неделю 09.08.2026 две версии одного
    издания несут 75 сюжетов из 725, масштаб сдвигается у 26, в среднем на
    0.062 и до 0.225 — при весе 0.40 этого хватает на пару мест в отборе.
    """
    by_pub = {}
    for v in _versions(art):
        if v[6] in _SCALE_VALUE:
            by_pub.setdefault(importance.publisher(v[0]), []).append(
                _SCALE_VALUE[v[6]])
    if not by_pub:
        return None
    return sum(sum(g) / len(g) for g in by_pub.values()) / len(by_pub)


def _story_date(art: tuple, fallback: str) -> str:
    """Дата ПУБЛИКАЦИИ сюжета — самая ранняя из дат его версий.

    Ни GKG, ни разметка страницы даты события не знают: оба отвечают на вопрос
    «когда опубликовано». Но перепечатки одного сюжета растянуты на несколько
    дней, и первая из них — ближайшее к событию, что даёт корпус: первое
    издание пишет по факту, а не через неделю.

    Саму дату события называет модель при пересказе, и печатается она, только
    если её подтвердил сторож (event_date.choose); эта дата — запасной вариант
    и потолок: позже неё событие быть не может.

    Пустые даты в счёт не идут (у части статей publish_date не определился);
    если их нет ни у одной версии, остаётся дата представителя.
    """
    dates = [v[3] for v in _versions(art) if v[3]]
    return min(dates) if dates else fallback


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
    """Важность сюжетов корзины в [0, 1] — пять факторов, без отдельного LLM.

        scale    — национальный масштаб события глазами судьи (1..3);
        coverage — сколько РАЗНЫХ изданий написали о сюжете;
        topic    — вероятность локального предфильтра «это наша тема»;
        entity   — политический вес субъектов (см. entities.py);
        lexrank  — тематическая центральность в недельном окне.

    Веса — в config.IMPORTANCE_W_*; ноль выключает фактор. Факторы scale и
    topic выключаются сами, пока нет оценок и артефакта предфильтра.

    Почему одного LexRank было мало: он меряет плотность связей, а корзина
    на 40% состоит из однотипных заметок «школьник сдал экзамен» из разных
    городов — они образуют такую же плотную клику, как настоящий сюжет, и
    заметка про одну школу в Кадыкёе получала важность 0.966. Замер 08.08.2026
    добил вопрос: корреляция LexRank с числом написавших изданий −0.117, то
    есть он систематически поднимает то, о чём не написал больше никто, и его
    вес выставлен в ноль (тогда он и не считается).
    """
    w_lex = config.IMPORTANCE_W_LEXRANK
    lex = (importance.minmax(_lexrank_scores(
        embeddings, domains=[importance.domain(a[0]) for a in subset]))
        if w_lex else np.zeros(len(subset)))

    # Охват нормируется ПО КОРЗИНЕ, а не по абсолютной шкале. Абсолютный
    # потолок в 12 изданий насыщал турецкую корзину (сюжеты с 30, 18 и 13
    # изданиями были неразличимы — все 1.0) и не достигался в CA/SC, где
    # максимум за неделю 9 и 5: один и тот же вес значил в корзинах разное.
    n_dom = [importance.distinct_domains(v[0] for v in _versions(a))
             for a in subset]
    full_at = max(max(n_dom, default=0), config.COVERAGE_FULL_AT_MIN)
    cov = np.array([importance.coverage_weight(n, full_at) for n in n_dom],
                   dtype=np.float64)

    ent = np.array([max(v[5] for v in _versions(a)) for a in subset],
                   dtype=np.float64)
    w_ent = config.IMPORTANCE_W_ENTITY if ent.any() else 0.0

    raw_scale = [_story_scale(a) for a in subset]
    sc = np.array([_SCALE_UNGRADED if s is None else s for s in raw_scale],
                  dtype=np.float64)
    w_scale = (config.IMPORTANCE_W_SCALE
               if any(s is not None for s in raw_scale) else 0.0)

    factors = [(w_lex, lex),
               (config.IMPORTANCE_W_COVERAGE, cov),
               (w_ent, ent),
               (w_scale, sc)]

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
Retell the source article in RUSSIAN for an academic digest, in 3–4 sentences \
forming ONE paragraph.
The reader follows science, education and youth policy but knows NOTHING about \
this particular country's institutions, exams, agencies or political parties. \
Write so that such a reader understands the news without looking anything up.
The source article may be in Turkish, Russian, English, or another language — \
regardless of the source language, your ENTIRE output (both title and summary) \
must be written in natural, idiomatic Russian.
RUSSIAN means Russian specifically, NOT Ukrainian, Belarusian, Bulgarian or any \
other Cyrillic language. Never mirror the language of the source article.

Proper names:
- Organisations, universities, programmes, portals and laws keep their original \
Latin spelling (İŞKUR, TÜBİTAK, Erasmus+). Do not transliterate them.
- Personal names, cities and regions use the established Russian form and are \
inflected for case as Russian grammar requires.
- When the established Russian form is not obvious, keep the Latin original \
rather than inventing a transliteration.

Words that are NOT names:
- A COMMON noun stays a common noun and gets TRANSLATED. Never spell a foreign \
word out in Cyrillic letters: kontenjan -> «квота мест» (not «контэнджан»), \
puan -> «балл» (not «пуан»), görevli -> «сотрудник» (not «гёревли»). If you \
cannot translate it, keep the Latin original and explain it in the same \
sentence — Cyrillic letters do not make a foreign word Russian, they only hide \
that it was never translated.
- An abbreviation written in Latin capitals stays in Latin capitals: MEB, \
TENMAK, YÖK — never «Меб», never «Тенмак».
- Text that will physically READ in the source language — an inscription, a \
sign, a badge, a slogan — keeps its original spelling, with a Russian gloss \
right after it: надпись «Bahçe Görevlisi» («работник по уходу за территорией»). \
Putting a Russian phrase in the quotation marks instead invents a quote that \
nobody will see on the object.

Context for a newcomer:
- On first mention, expand EVERY abbreviation and add, in the same sentence, a \
short apposition saying what it is: "Совет по высшему образованию Турции (YÖK)", \
"LGS, экзамен при переходе в лицей". Same for institutions, exams, programmes, \
agencies, parties and rankings that only a specialist would recognise.
- NEVER invent an expansion or a description. Use the "Established renderings" \
block above and the source article, and nothing else. If neither says what an \
abbreviation stands for, leave it as it is — a wrong expansion is far worse than \
a missing one.
- Do NOT explain what everybody knows: ООН, ЕС, США, НАТО, ЮНЕСКО, ВОЗ, GPS, \
искусственный интеллект and the like go without a gloss.
- Turkish abbreviations of countries and unions are NOT names and must be \
rendered in Russian, not kept: AB -> ЕС, ABD -> США, BM -> ООН, TC -> Турция.
- A gloss is an apposition or a subordinate clause inside an existing sentence, \
never a separate explanatory sentence — the facts of the news must not lose room \
to them.

Rules:
- Length: the summary must be 3–4 sentences — concise, only the essential facts. \
It is ONE continuous paragraph: no line breaks, no bullet points, no numbered list.
- Do not leave ANY word, phrase, or clause untranslated in Turkish, English, or any \
other language — render everything in Russian.
- Russian grammar, cases, agreement and word order must be correct and natural. \
Do not calque English syntax: no chains of participles where a Russian writer \
would use a subordinate clause, no strings of nouns in the genitive.
- Do not invent facts.
- Keep ONLY content related to science, education, youth policy in Turkey, Central Asia, South Caucasus, or external engagement involving Central Asia and South Caucasus. \
If the source article also covers other, unrelated topics or events, SILENTLY DROP \
them — do not summarize, mention, or even briefly reference the unrelated parts. \
The final summary must read as if the unrelated content never existed in the source.
- ALWAYS make the geographic or institutional context explicit.
- The title MUST explicitly indicate the place or institution the news is about, \
for example: "В Турции ...", "В Узбекистане ...", "В Баку ...", \
"В Ташкентском государственном университете ...", \
"Министерство просвещения Казахстана ...".
- The summary MUST explicitly mention the place, country, city, institution, or \
organisation the article is about in the first 1–2 sentences.
- You may receive SEVERAL versions of the SAME story, filed by different outlets \
and separated by a line of dashes. They are one news event, not several. Merge \
them into ONE summary: use a fact if any version reports it, and where versions \
disagree prefer the one reported by more of them. Never present them as separate \
items and never invent a link between them that no version states.
- Each block is a SELECTION of sentences from its article, not a continuous \
excerpt: sentences that stood far apart in the original now stand side by side. \
Take each sentence as a separate fact. Do not read cause, sequence or contrast \
into two sentences merely because they are adjacent.
- Prefer facts that only ONE of the blocks reports — figures, dates, names, \
conditions, quotes. What every outlet repeats is the headline the reader can \
guess; what one outlet adds is why several of them were read at all.

Date of the event:
- "event_date" is the day the event REPORTED here actually happened, exactly as \
the article states it: "29 Temmuz'da düzenlenen zirve" -> "2026-07-29". \
The year is usually omitted in the text — take it from the article URL or the \
publication date.
- Give the date ONLY if the article names it. Never infer it from the \
publication date, never guess, never use a date the article mentions for \
something else (a deadline, an anniversary, a previous event, a schedule). \
When in doubt return null — null costs nothing, a wrong date is printed in the \
digest as a fact.
- A future date is not an event date: "applications close on 1 September" is a \
deadline, not the day something happened. Return null for those.
- For a multi-day event give its FIRST day.

Return ONLY valid JSON, no markdown fences:
{"title": "<заголовок по-русски>", "summary": "<пересказ по-русски, 3–4 предложения, один абзац>", "event_date": "YYYY-MM-DD или null"}
"""

_USER_PROMPT_RETELL_TEMPLATE = "Article URL: {url}\nArticle text:\n{text}"


def _retell_sources(versions: list) -> tuple[str, list]:
    """Тексты версий сюжета в один промпт → (промпт, попавшие в него адреса).

    Разные редакции сохраняют разные детали (цифры, имена, цитаты), но
    начинаются одинаково: перепечатка повторяет лид агентства слово в слово.
    Поэтому в промпт идёт не начало каждой статьи, а ОТБОР по всем версиям
    сразу (retell_select) — предложения, которых ещё не было, от как можно
    большего числа изданий. Бюджет тот же, фактов в нём на четверть больше.

    Издание, у которого не нашлось ничего своего, в промпт не попадает: пустой
    блок с адресом модели ничего не даёт, а место занимает. Адреса возвращаются
    отдельно, потому что под пересказом в документе печатаются источники, а
    источником называться может только тот, кого модель читала.
    """
    versions = versions[:config.RETELL_MAX_VERSIONS]
    picked = retell_select.pick([(v[0], v[1] or "") for v in versions],
                                config.RETELL_CHARS_TOTAL, config.RETELL_SCAN_CHARS)
    urls = [v[0] for v in versions if picked.get(v[0])]
    body = "\n\n----------\n\n".join(
        _USER_PROMPT_RETELL_TEMPLATE.format(url=u, text=" ".join(picked[u]))
        for u in urls)
    # Без текста словарь звать нельзя: на пустой строке glossary отдаёт первые
    # 60 терминов вообще всех, и в промпт уезжает список имён без статьи.
    return (_glossary_block(body) + body if body else ""), urls


def _glossary_block(text):
    """Принятые написания имён, встретившихся в этом тексте, — в промпт.

    Правило старое (glossary.py): к выводу LLM словарь НЕ применяется
    постобработкой, она ломает падежи. Пока пересказ писался по-английски,
    словарь был не нужен вовсе — имена шли латиницей, а русскую форму им потом
    подбирал локальный переводчик. Теперь русскую форму выбирает сама модель, и
    выбирает наугад: в замере она приняла турецкое şehit («павший», в названиях
    школ и больниц) за «шейха». Список принятых написаний это чинит там же, где
    возникает ошибка, и модель сама ставит нужный падеж.
    """
    try:
        import glossary
        block = glossary.as_prompt_block(text)
    except Exception:
        log.exception("Словарь имён не подгрузился — пересказ пойдёт без него")
        return ""
    return block + "\n\n" if block else ""

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

# Доля кириллицы среди букв. Спрашиваем письменность, а не язык: вопрос здесь
# ровно один — ответила модель по-русски или сорвалась в английский с турецким.
# langdetect на 3–4 предложениях с латинскими названиями организаций отвечает то
# «bg», то «mk», и его вердикт пришлось бы прощать так широко, что он перестал бы
# что-либо значить. Порог низкий: в пересказе про İŞKUR и TÜBİTAK латиницы
# законно много, а английский текст даёт ноль кириллицы, и промахнуться мимо
# половины невозможно.
_MIN_CYRILLIC = 0.5

# Кириллица бывает не только русская, и одной доли кириллицы мало: 07.08.2026
# в дайджест ушёл пункт целиком по-украински («У місті Тбілісі та Астані
# розгорнуто інтенсивну підготовку...»). Общих букв в тексте подавляющее
# большинство, доля кириллицы вышла 0.93 — верхний порог такое пропускает и
# будет пропускать всегда.
#
# Ловим по буквам, которых в русском алфавите нет: украинские і/ї/є/ґ,
# белорусская ў, сербские ђ/ћ/ј/љ/њ, казахские ә/ғ/қ/ң/ө/ұ/ү/һ.
#
# Порог не нулевой: имя в исходном написании посреди русского текста — ещё не
# смена языка, а перевыбор стоит слота. Замер по всем 20 сохранённым дайджестам
# (1717 абзацев) даёт два порядка зазора: у украинских абзацев 11% и 17%, у
# худшего русского — 0.12% («министр просвещения Кыргызстана Догдуркүл
# Кендирбаева», киргизская ү в одном имени). Порог стоит в этом промежутке.
_RU_ALPHABET = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
_MAX_FOREIGN_CYRILLIC = 0.02

# Третий алфавит. Модель иногда вставляет кусок исходного письма прямо в
# русское слово: «Обучающий этап نظمил Министерство энергетики» (замер
# 07.08.2026), а 25.07.2026 читателю ушёл пункт с «фонду
# «Գյումրիի տեղեկատվա...». Ни одно правило промпта такого не разрешает:
# имена идут либо принятой русской формой, либо латиницей.
#
# Допуска здесь нет, в отличие от кириллицы: одна арабская буква посреди слова
# — это уже сломанное слово, а не заимствованное написание. Греческий оставлен
# белым: β-распад и α-частицы в научном тексте законны. Замер по 20 дайджестам
# (1717 абзацев): ровно два попадания, оба — эта поломка.
_SCRIPTS_OK = ("LATIN", "CYRILLIC", "GREEK")


def _cyrillic_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum("а" <= c.lower() <= "я" or c.lower() == "ё" for c in letters) / len(letters)


def _third_script_letters(text: str) -> list:
    """Буквы не из латиницы, кириллицы и греческого — в порядке появления."""
    return [c for c in text
            if c.isalpha()
            and not unicodedata.name(c, "").startswith(_SCRIPTS_OK)]


def _foreign_cyrillic_share(text: str) -> float:
    """Доля нерусских кириллических букв среди кириллических."""
    cyr = [c for c in text.lower() if "Ѐ" <= c <= "ӿ"]
    if not cyr:
        return 0.0
    return sum(c not in _RU_ALPHABET for c in cyr) / len(cyr)


def _validate_summary_fast(result: dict, source: str = "") -> tuple[bool, str]:
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

    third = _third_script_letters(full)
    if third:
        return False, f"третий алфавит в тексте: «{''.join(third[:12])}»"

    # Обрыв на полуслове. Потолок токенов запросу не задаётся, его ставит
    # провайдер, и finish_reason до нас не доходит — _call_openrouter_raw
    # отдаёт один текст. Зато обрыв виден по концу: JSON при этом остаётся
    # валидным, поэтому парсер такое пропускает («...сохранение личных выплат
    # контрактных преподавателей» — замер 07.08.2026).
    if not re.search(r"[.!?…][»\"')\]]*$", summary.strip()):
        return False, f"пересказ оборван: «...{summary.strip()[-40:]}»"

    share = _cyrillic_share(full)
    if share < _MIN_CYRILLIC:
        return False, f"пересказ не на русском (кириллицы {share:.0%})"

    foreign = _foreign_cyrillic_share(full)
    if foreign > _MAX_FOREIGN_CYRILLIC:
        return False, (f"пересказ на другом кириллическом языке "
                       f"(нерусских букв {foreign:.0%})")

    # Слово источника, переписанное кириллицей вместо перевода: «увеличен
    # контэнджан стипендиальной программы» (09.08.2026), «пуаны» вместо баллов.
    # Промпт это запрещает, но запрет промпта модель соблюдает вероятностно —
    # см. translit_guard.
    copied = translit_guard.copied_words(full, source)
    if copied:
        w, src = copied[0]
        return False, (f"слово источника кириллицей вместо перевода: "
                       f"«{w}» ← {src}")

    return True, "ok"


# ---------------------------------------------------------------------------
# Пересказ одной статьи: LLM-пересказ + быстрая regex-валидация формата.
# Тематическая релевантность статьи решена ещё при сборе (daily_collector),
# здесь её не перепроверяем — см. п.4a в шапке модуля.
# ---------------------------------------------------------------------------
def _retell_article(url: str, versions: list, pub_date: str, attempt_label: str) -> dict | None:
    user_msg, urls = _retell_sources(versions)
    if not user_msg:
        # Ни в одной версии не нашлось предложения — просить пересказ нечего.
        log.warning(" %s: пустой промпт, слот пропущен: %s", attempt_label, url)
        return None
    if len(urls) > 1:
        log.info(" %s: пересказ по %d версиям сюжета", attempt_label, len(urls))

    hint = ""
    for attempt in range(1, MAX_RETRIES + 1):
        log.info(" %s попытка %d/%d: %s", attempt_label, attempt, MAX_RETRIES, url)
        raw    = _call_openrouter_raw(_SYSTEM_PROMPT_RETELL, user_msg + hint, ref_url=url)
        result = _parse_llm_json(raw)

        if result is None:
            log.warning(" %s попытка %d: не удалось распарсить JSON", attempt_label, attempt)
            continue

        ok, reason = _validate_summary_fast(result, user_msg)
        if not ok:
            log.warning(" %s попытка %d: %s", attempt_label, attempt, reason)
            # Повторять тот же промпт — значит надеяться, что модель передумает
            # сама. Причина отказа уже сформулирована одной строкой, и слот
            # стоит дороже лишней строки в запросе.
            hint = ("\n\nПредыдущий ответ забракован: %s. "
                    "Исправь именно это и перепиши ответ целиком." % reason)
            continue

        # Дата события: её называет модель, но печатается она только пройдя
        # сторожа (event_date) — в недельном окне, не позже публикации и
        # буквально стоящая в тексте, который модель читала.
        day, why = event_date.choose(result.get("event_date"), pub_date,
                                     user_msg, date.today())
        if why:
            log.info(" %s: дата события отклонена (%s), остаётся публикация %s",
                     attempt_label, why, pub_date)
        elif day and day != (pub_date or "")[:10]:
            log.info(" %s: дата события %s вместо публикации %s",
                     attempt_label, day, pub_date)

        result["url"]          = url
        result["urls"]         = urls      # все источники, попавшие в промпт
        result["publish_date"] = day
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

        # Дата сюжета — дата САМОЙ РАННЕЙ его версии, а не представителя (тот
        # выбран отбором MMR, а не по времени). Даты события в корпусе нет, но
        # «когда о сюжете написали впервые» ближе к ней всего: перепечатка на
        # третий день не должна датировать событие третьим днём.
        result = _retell_article(url, _versions(cand), _story_date(cand, pub_date),
                                 slot_label)
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
_SYSTEM_PROMPT_POSTPROCESS = """You are a quality-control editor for a Russian-language academic digest.

You will receive a JSON array of digest items. Each item has:
  "idx"     — original index (integer, preserve it),
  "title"   — headline in Russian,
  "summary" — body text in Russian.

Keep every item in Russian. Do not translate, rewrite or "improve" the wording.

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