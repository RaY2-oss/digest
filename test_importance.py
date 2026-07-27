# -*- coding: utf-8 -*-
"""Проверка второго фактора важности — политического веса статьи.

LexRank меряет только плотность связей, поэтому пачка однотипных локальных
заметок даёт такую же клику, как настоящий сюжет. Здесь проверяется, что
_importance() поднимает статью с заметными субъектами и что при выключенном
факторе (или пустых entities) поведение остаётся прежним, чисто LexRank'овым.

Запуск: ./venv/bin/python test_importance.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import entities
import sunday_processor_mmr as spm

DIM = config.EMBEDDING_DIM


def _art(url, vec, prom):
    return (url, "text " + url, vec.astype(np.float32), "2026-07-27", "TR", prom)


def _orthogonal(n):
    """n взаимно ортогональных векторов — LexRank у всех одинаковый."""
    out = []
    for i in range(n):
        v = np.zeros(DIM, dtype=np.float32)
        v[i] = 1.0
        out.append(v)
    return out


def test_political_weight_breaks_the_tie():
    """При равном LexRank выигрывает статья с заметными субъектами."""
    vecs = _orthogonal(3)
    subset = [_art("local", vecs[0], 0.0),
              _art("mid", vecs[1], 0.5),
              _art("national", vecs[2], 1.0)]
    embs = np.vstack([a[2] for a in subset])

    scores = spm._importance(subset, embs)
    assert scores[2] > scores[1] > scores[0], scores

    ranked = [a[0] for a in spm._rank_by_importance(subset)]
    assert ranked == ["national", "mid", "local"], ranked


def test_factor_off_keeps_pure_lexrank():
    """W=0 и корпус без entities дают ровно прежний, чисто LexRank'овый скор."""
    vecs = _orthogonal(3)
    embs = np.vstack(vecs)
    weighted = [_art("a", vecs[0], 0.0), _art("b", vecs[1], 0.5),
                _art("c", vecs[2], 1.0)]
    flat = [_art("a", vecs[0], 0.0), _art("b", vecs[1], 0.0),
            _art("c", vecs[2], 0.0)]

    base = spm._importance(flat, embs)          # ни одного заметного субъекта
    old_weight = config.ENTITY_WEIGHT
    try:
        config.ENTITY_WEIGHT = 0.0
        assert np.allclose(spm._importance(weighted, embs), base)
    finally:
        config.ENTITY_WEIGHT = old_weight
    assert not np.allclose(spm._importance(weighted, embs), base), \
        "с ненулевым весом и заметными субъектами скор обязан отличаться"


def test_local_clique_does_not_outrank_national_story():
    """Пять перепечаток локальной заметки образуют плотную клику и забирают
    весь LexRank; одиночный общенациональный сюжет вытягивается субъектами."""
    clique_dir = np.zeros(DIM, dtype=np.float32); clique_dir[0] = 1.0
    subset = []
    for i in range(5):
        v = clique_dir.copy()
        v[i + 1] = 0.05           # почти дубли друг друга
        subset.append(_art(f"local{i}", v, 0.0))
    alone = np.zeros(DIM, dtype=np.float32); alone[100] = 1.0
    subset.append(_art("national", alone, 1.0))
    embs = np.vstack([a[2] for a in subset])

    old_weight = config.ENTITY_WEIGHT
    try:
        config.ENTITY_WEIGHT = 0.0
        pure = spm._importance(subset, embs)       # прежнее поведение
    finally:
        config.ENTITY_WEIGHT = old_weight
    blended = spm._importance(subset, embs)

    assert pure[-1] < pure[:-1].max(), "клика обязана выигрывать по чистому LexRank"
    assert (blended[:-1].max() - blended[-1]) < (pure[:-1].max() - pure[-1]), \
        "политический вес обязан сокращать отрыв клики"


def test_prominence_pipeline_from_raw_gkg_cells():
    """Сквозная связка entities.parse -> document_freq -> weight, как в
    load_week_articles."""
    corpus = [entities.parse("Ali Veli", "Ministry of Education"),
              entities.parse("", "Ministry of Education;TUBITAK;ASELSAN"),
              entities.parse("Recep Tayyip Erdogan",
                             "TUBITAK;ASELSAN;Ministry of Industry"),
              entities.parse("", "TUBITAK;ASELSAN;Ministry of Industry")]
    df = entities.document_freq(corpus)

    prom = [entities.weight(entities.prominent(c, df, config.ENTITY_MIN_DF),
                            config.ENTITY_FULL_AT) for c in corpus]
    assert prom[0] == 0.0, "локальная заметка: ни одного субъекта выше порога"
    assert prom[3] > prom[0]


def demo():
    test_political_weight_breaks_the_tie()
    test_factor_off_keeps_pure_lexrank()
    test_local_clique_does_not_outrank_national_story()
    test_prominence_pipeline_from_raw_gkg_cells()


if __name__ == "__main__":
    demo()
    print("ok")
