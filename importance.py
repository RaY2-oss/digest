# -*- coding: utf-8 -*-
"""importance.py — структурная важность сюжета, считается БЕЗ обращения к LLM.

Зачем отдельный модуль. Оценка LLM (`articles.importance`) остаётся ГЕЙТОМ
релевантности: «по теме / не по теме» плюс грубая значимость. Ранжировать по
ней нельзя — она квантована: 96 % значений кратны пяти, и в окне одной страны
сотни статей делят одно и то же число (замер 27.07: у Китая 245 статей ровно
с 85). При TOP_N-отборе порядок внутри такой связки произволен.

Здесь важность считается по структуре корпуса и потому непрерывна — связок
не возникает вовсе. Три фактора:

    1. LexRank по графу косинусных схожестей (степенная итерация, аналог
       PageRank). Статья весома, если на неё похожи другие статьи, которые
       сами хорошо связаны с остальным окном. Считается по ЭМБЕДДИНГУ ТЕЛА
       статьи (`articles.embedding_body`), а не заголовка: по заголовкам
       LexRank мерил бы частоту перепечатки, а не тематическую центральность.

    2. Охват — сколько РАЗНЫХ доменов написали о сюжете, логарифмически.
       Линейная надбавка («+K за каждую статью») упирается в потолок: при
       базе 85 хватало четырёх перепечаток, дальше сюжет с 4 и сюжет с 40
       публикациями неразличимы. Логарифм насыщается, но не обнуляется и
       потолка не имеет: первая пара изданий даёт много, десятая почти
       ничего, но добавка продолжает расти.
       Домены дедуплицируются: одно издание, выкатившее пять версий текста,
       считается за один домен, а не за пять.

    3. Политический вес субъектов (entities.py) — как и раньше.

Веса факторов — в settings.IMPORTANCE_W_*; выставить любой в 0 значит
выключить фактор, не трогая код.
"""
import math
import re
from urllib.parse import urlsplit

import numpy as np

# www., m., amp. и подобные технические префиксы не делают издание другим.
_HOST_PREFIX = re.compile(r"^(?:www|m|amp|mobile|edition|en|ru)\.")


def domain(url: str) -> str:
    """URL -> нормализованный домен издания ('' если разобрать не удалось).

    Именно домен, а не URL, — единица охвата: пять версий одного текста на
    одном сайте не должны весить как пять изданий.
    """
    try:
        host = urlsplit(url or "").netloc.lower()
    except ValueError:
        return ""
    host = host.split("@")[-1].split(":")[0]
    return _HOST_PREFIX.sub("", host)


def distinct_domains(urls) -> int:
    return len({d for d in (domain(u) for u in urls) if d})


def coverage_weight(n_domains: int, full_at: int) -> float:
    """Насыщающийся охват в [0, 1].

    n_domains=1 -> 0.0 (одно издание — охвата нет)
    n_domains=full_at -> 1.0
    дальше растёт, но обрезается единицей: разница между 40 и 80 изданиями
    уже не должна перевешивать всё остальное.
    """
    if full_at <= 1 or n_domains <= 1:
        return 0.0
    return min(1.0, math.log1p(n_domains - 1) / math.log1p(full_at - 1))


def _normalize_rows(embs):
    return embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)


def lexrank(embeddings, damping, max_iter, tol, domains=None):
    """LexRank-скоры (сумма = 1) по графу косинусных схожестей.

    domains (опц.): если задан, рёбра МЕЖДУ СТАТЬЯМИ ОДНОГО ДОМЕНА обнуляются.
    Без этого издание, опубликовавшее пять версий одного текста, образует
    плотную клику и накачивает само себя — ровно тот локальный шум, ради
    которого фактор и вводился.
    """
    E = np.asarray(embeddings, dtype=np.float32)
    n = E.shape[0]
    if n <= 1:
        return np.ones(n, dtype=np.float64)

    E = _normalize_rows(E)
    sim = (E @ E.T).astype(np.float64)
    np.fill_diagonal(sim, 0.0)
    np.clip(sim, 0.0, None, out=sim)

    if domains is not None:
        d = np.asarray(domains, dtype=object)
        same = (d[:, None] == d[None, :]) & (d[:, None] != "")
        sim[same] = 0.0

    row_sums = sim.sum(axis=1, keepdims=True)
    dangling = (row_sums.ravel() == 0.0)
    trans = sim / np.where(row_sums == 0.0, 1.0, row_sums)
    if dangling.any():
        # Узел без связей ни на кого не похож — раздаём его вес равномерно,
        # иначе он проглатывает скор и не возвращает его в граф.
        trans[dangling, :] = 1.0 / n

    scores = np.full(n, 1.0 / n, dtype=np.float64)
    teleport = (1.0 - damping) / n
    for _ in range(max_iter):
        nxt = teleport + damping * (trans.T @ scores)
        if np.abs(nxt - scores).sum() < tol:
            return nxt
        scores = nxt
    return scores


def minmax(v):
    """Приводит вектор к [0,1]; вырожденный случай -> 0.5 у всех."""
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return v
    lo, hi = float(v.min()), float(v.max())
    if hi - lo <= 1e-12:
        return np.full_like(v, 0.5)
    return (v - lo) / (hi - lo)


def blend(lex, cov, ent, w_lex, w_cov, w_ent):
    """Свёртка трёх нормированных факторов в итоговую важность [0,1].

    Веса нормируются суммой, поэтому выключение фактора (вес 0) не требует
    пересчёта остальных вручную.
    """
    total = float(w_lex + w_cov + w_ent)
    if total <= 0:
        return np.asarray(lex, dtype=np.float64)
    return (w_lex * np.asarray(lex, dtype=np.float64)
            + w_cov * np.asarray(cov, dtype=np.float64)
            + w_ent * np.asarray(ent, dtype=np.float64)) / total


def _selfcheck():
    assert domain("https://www.example.com/a/b?x=1") == "example.com"
    assert domain("http://m.gazeta.ru/news") == "gazeta.ru"
    assert domain("https://sub.example.co.uk:8443/x") == "sub.example.co.uk"
    assert domain("не-url") == ""
    assert distinct_domains(["https://a.com/1", "https://www.a.com/2",
                             "https://b.com/1"]) == 2

    # Охват: насыщается, но не упирается в потолок раньше времени.
    assert coverage_weight(1, 12) == 0.0
    assert coverage_weight(12, 12) == 1.0
    assert coverage_weight(40, 12) == 1.0
    c2, c4, c8 = (coverage_weight(k, 12) for k in (2, 4, 8))
    assert c2 < c4 < c8
    assert (c4 - c2) > (c8 - c4)          # прирост убывает — это и есть насыщение

    # LexRank: центральная статья весомее периферийной.
    core = np.array([1.0, 0.0], np.float32)
    embs = np.array([core, core * 0.99 + [0, 0.01], core, [0.0, 1.0]], np.float32)
    s = lexrank(embs, 0.85, 100, 1e-6)
    assert s[3] < s[0], s
    assert abs(s.sum() - 1.0) < 1e-6

    # Подавление внутридоменных рёбер: три клона с одного домена не должны
    # накачивать друг друга сильнее, чем при честном подсчёте.
    doms = ["a.com", "a.com", "a.com", "b.com"]
    s_dom = lexrank(embs, 0.85, 100, 1e-6, domains=doms)
    assert s_dom[0] < s[0], (s_dom, s)

    assert np.allclose(minmax(np.array([5.0, 5.0])), 0.5)
    assert np.allclose(minmax(np.array([0.0, 1.0, 2.0])), [0.0, 0.5, 1.0])
    b = blend([1.0, 0.0], [0.0, 1.0], [0.0, 0.0], 1.0, 1.0, 0.0)
    assert np.allclose(b, [0.5, 0.5])
    b0 = blend([1.0, 0.0], [0.0, 1.0], [0.0, 0.0], 1.0, 0.0, 0.0)
    assert np.allclose(b0, [1.0, 0.0])
    print("importance: ok")


if __name__ == "__main__":
    _selfcheck()
