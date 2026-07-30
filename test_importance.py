# -*- coding: utf-8 -*-
"""Проверка второго фактора важности — политического веса статьи.

LexRank меряет только плотность связей, поэтому пачка однотипных локальных
заметок даёт такую же клику, как настоящий сюжет. Здесь проверяется, что
_importance() поднимает статью с заметными субъектами и что при выключенном
факторе (или пустых entities) поведение остаётся прежним, чисто LexRank'овым.

Запуск: ./venv/bin/python test_importance.py
"""
import contextlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import entities
import importance
import sunday_processor_mmr as spm

DIM = config.EMBEDDING_DIM


@contextlib.contextmanager
def weights(**kw):
    """Временно переопределяет config.IMPORTANCE_W_* — чтобы проверка смотрела
    на СВОЙ фактор, а не на сумму четырёх. Фактор topic глушится почти везде:
    он зависит от обученного артефакта, которого на чужой машине может не быть.
    """
    old = {k: getattr(config, k) for k in kw}
    for k, v in kw.items():
        setattr(config, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(config, k, v)


def _art(url, vec, prom):
    return (url, "text " + url, vec.astype(np.float32), "2026-07-27", "TR", prom)


def _story(arts):
    """Сюжет в том виде, в каком его отдаёт _group_stories: представитель
    плюс седьмым полем список версий."""
    return tuple(arts[0]) + (list(arts),)


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

    with weights(IMPORTANCE_W_TOPIC=0.0):
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

    with weights(IMPORTANCE_W_TOPIC=0.0):
        base = spm._importance(flat, embs)      # ни одного заметного субъекта
        with weights(IMPORTANCE_W_ENTITY=0.0):
            assert np.allclose(spm._importance(weighted, embs), base)
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

    with weights(IMPORTANCE_W_TOPIC=0.0):
        with weights(IMPORTANCE_W_ENTITY=0.0):
            pure = spm._importance(subset, embs)   # прежнее поведение
        blended = spm._importance(subset, embs)

    assert pure[-1] < pure[:-1].max(), "клика обязана выигрывать по чистому LexRank"
    assert (blended[:-1].max() - blended[-1]) < (pure[:-1].max() - pure[-1]), \
        "политический вес обязан сокращать отрыв клики"


def test_coverage_beats_the_local_clique():
    """Главная правка: пять городских заметок «школьник сдал экзамен» против
    одного сюжета, перепечатанного двенадцатью изданиями. По LexRank выигрывает
    клика — по охвату обязан выигрывать сюжет."""
    clique_dir = np.zeros(DIM, dtype=np.float32); clique_dir[0] = 1.0
    locals_ = []
    for i in range(5):
        v = clique_dir.copy()
        v[i + 1] = 0.05
        locals_.append(_story([_art(f"https://kadikoy{i}.com/a", v, 0.0)]))

    alone = np.zeros(DIM, dtype=np.float32); alone[100] = 1.0
    national = _story([_art(f"https://outlet{k}.com/a", alone, 0.0)
                       for k in range(config.COVERAGE_FULL_AT)])

    subset = locals_ + [national]
    embs = np.vstack([s[2] for s in subset])

    with weights(IMPORTANCE_W_TOPIC=0.0):
        with weights(IMPORTANCE_W_COVERAGE=0.0):
            pure = spm._importance(subset, embs)
        blended = spm._importance(subset, embs)

    assert pure[-1] < pure[:-1].max(), "без охвата клика обязана выигрывать"
    assert blended[-1] > blended[:-1].max(), \
        f"с охватом общенациональный сюжет обязан выигрывать: {blended}"


def test_wire_mirrors_do_not_inflate_coverage():
    """Шесть доменов одной сети-перепечатки — это одно издание, а не шесть:
    иначе контент-ферма обходит настоящий сюжет по охвату."""
    v = np.zeros(DIM, dtype=np.float32); v[0] = 1.0
    farm = _story([_art(f"https://{d}/news/279212771/x", v, 0.0) for d in
                   ("chinanationalnews.com", "russiaherald.com", "shanghaisun.com",
                    "hongkongherald.com", "shanghainews.net", "asiabulletin.com")])
    real = _story([_art(f"https://outlet{k}.com/a", v, 0.0) for k in range(3)])

    assert importance.distinct_domains(a[0] for a in farm[6]) == 1
    assert importance.distinct_domains(a[0] for a in real[6]) == 3


def test_group_stories_collapses_reprints():
    """_group_stories: перепечатки схлопываются, представителем становится
    самый длинный текст, версии едут дальше."""
    v = np.zeros(DIM, dtype=np.float32); v[0] = 1.0
    other = np.zeros(DIM, dtype=np.float32); other[7] = 1.0
    short = ("https://a.com/x", "коротко", v, "2026-07-27", "TR", 0.0)
    long_ = ("https://b.com/x", "текст подлиннее", v.copy(), "2026-07-27", "TR", 0.0)
    solo = ("https://c.com/y", "другое", other, "2026-07-27", "TR", 0.0)

    stories = spm._group_stories([short, long_, solo])
    assert len(stories) == 2, stories
    merged = next(s for s in stories if len(s[6]) == 2)
    assert merged[0] == "https://b.com/x", "представитель — самый длинный текст"
    assert {v[0] for v in merged[6]} == {"https://a.com/x", "https://b.com/x"}
    # Одиночная статья тоже приходит в формате сюжета — версия одна, она сама.
    assert spm._versions(next(s for s in stories if len(s[6]) == 1))[0][0] == solo[0]


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
    test_coverage_beats_the_local_clique()
    test_wire_mirrors_do_not_inflate_coverage()
    test_group_stories_collapses_reprints()
    test_prominence_pipeline_from_raw_gkg_cells()


if __name__ == "__main__":
    demo()
    print("ok")
