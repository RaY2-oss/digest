# -*- coding: utf-8 -*-
"""Проверка стадий отсева ДО обращения к LLM.

Запуск: ./venv/bin/python test_prefilter_stack.py
"""
import os
import sqlite3
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gkg_filter
import seen_store

# Реальная запись из дампа 20260722124500 (ирландская статья). Тема
# образования стоит на офсете 2081; ближайшее упоминание UK — 2235, то есть
# в 154 символах, примерно один абзац. Ирландия же сгруппирована в начале
# текста (61/261/503) и к теме образования отношения не имеет.
V2T = ("GOV_LOCALGOV,2081;WB_470_EDUCATION,2081;"
       "UNGP_FORESTS_RIVERS_OCEANS,1427;TAX_ETHNICITY_IRISH,598")
V2L = ("1#United Kingdom#UK#UK##54#-4#UK#105;"
       "1#United Kingdom#UK#UK##54#-4#UK#2235;"
       "1#Ireland#EI#EI##53#-8#EI#61;"
       "1#Ireland#EI#EI##53#-8#EI#261;"
       "1#Ireland#EI#EI##53#-8#EI#503")
EDU = {"WB_470_EDUCATION"}


def test_offsets_parse():
    th = gkg_filter.parse_themes(V2T)
    lo = gkg_filter.parse_locations(V2L)
    assert th["WB_470_EDUCATION"] == [2081], th
    assert lo["UK"] == [105, 2235], lo
    assert lo["EI"] == [61, 261, 503], lo
    assert gkg_filter.min_gap([2081], [105, 2235]) == 154
    print("gkg_filter: разбор офсетов и минимальный разрыв — OK")


def test_theme_must_be_near_country():
    """Главное свойство: страна должна обсуждаться ВМЕСТЕ с темой."""
    ok, gap, _ = gkg_filter.judge_v2(V2T, V2L, EDU, {"UK"}, 400, None)
    assert ok and gap == 154, (ok, gap)

    ok, gap, _ = gkg_filter.judge_v2(V2T, V2L, EDU, {"EI"}, 400, None)
    assert not ok and gap == 1578, (ok, gap)

    # Старое правило "есть тема И есть страна" пропустило бы и Ирландию —
    # ровно так в выборку попадали эстонские и корейские статьи.
    assert gkg_filter.judge_v1("WB_470_EDUCATION", V2L, EDU, {"EI"})
    print("gkg_filter: тема обязана стоять рядом со страной — OK")


def test_exact_theme_not_substring():
    """contains("EDUCATION") ловил WB_1497_EDUCATION_MANAGEMENT и соседей."""
    v2t = "WB_1497_EDUCATION_MANAGEMENT_AND_ADMINISTRATION,100"
    v2l = "1#Ireland#EI#EI##53#-8#EI#110"
    ok, _, _ = gkg_filter.judge_v2(v2t, v2l, EDU, {"EI"}, 400, None)
    assert not ok
    assert not gkg_filter.judge_v1(v2t, v2l, EDU, {"EI"})
    print("gkg_filter: точное совпадение темы вместо подстроки — OK")


def test_country_dominance():
    ok, _, share = gkg_filter.judge_v2(V2T, V2L, EDU, {"UK"}, 400, 0.30)
    assert ok and abs(share - 0.4) < 1e-9, share
    ok, _, _ = gkg_filter.judge_v2(V2T, V2L, EDU, {"UK"}, 400, 0.60)
    assert not ok
    print("gkg_filter: доля целевой страны среди упоминаний — OK")


def test_v2_absent_falls_back():
    assert gkg_filter.judge_v2("", "", EDU, {"EI"}, 400, None) is None
    print("gkg_filter: пустые поля V2 -> откат на V1 — OK")


def test_cluster_duplicates():
    import daily_collector as dc
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.02, 0.0], dtype=np.float32)   # перепечатка
    c = np.array([0.0, 1.0, 0.0], dtype=np.float32)    # другой сюжет
    labels = dc.cluster_duplicates(np.stack([a, b, c]), 0.95)
    assert labels[0] == labels[1] != labels[2], labels
    assert len(dc.cluster_duplicates(np.zeros((0, 3), dtype=np.float32), 0.95)) == 0
    print("cluster_duplicates: синдикация схлопывается, чужой сюжет — нет — OK")


def test_seen_store_verdicts():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    try:
        seen_store.ensure(conn)
        emb = np.ones(4, dtype=np.float32)
        seen_store.mark(conn, [
            ("http://a", 0, "accepted", emb),
            ("http://b", 0, "rejected", emb),
            ("http://c", 0, "pending", None),
            ("http://d", 0, "short", None),
        ], "2026-07-22")

        final = seen_store.final_urls(
            conn, ["http://a", "http://b", "http://c", "http://d"])
        assert final == {"http://a", "http://b", "http://d"}, final

        # Ключевое: нерешённое НЕ становится чёрным списком, а возвращается
        # в очередь — именно этого не хватало при исчерпанных лимитах.
        assert seen_store.pending_urls(conn, 0) == ["http://c"]
        assert seen_store.label_counts(conn) == (1, 1)

        seen_store.mark(conn, [("http://c", 0, "rejected", emb)], "2026-07-23")
        assert seen_store.pending_urls(conn, 0) == []
        assert seen_store.label_counts(conn) == (1, 2)

        # Чистка не трогает строки с эмбеддингом — это обучающая выборка.
        conn.execute("UPDATE seen_urls SET last_seen='2020-01-01'")
        conn.commit()
        seen_store.prune(conn, 30)
        assert seen_store.label_counts(conn) == (1, 2)
        assert conn.execute(
            "SELECT COUNT(*) FROM seen_urls WHERE embedding IS NULL"
        ).fetchone()[0] == 0
        print("seen_store: вердикты, очередь pending, сохранность разметки — OK")
    finally:
        conn.close()
        os.unlink(path)


def demo():
    test_offsets_parse()
    test_theme_must_be_near_country()
    test_exact_theme_not_substring()
    test_country_dominance()
    test_v2_absent_falls_back()
    test_cluster_duplicates()
    test_seen_store_verdicts()


if __name__ == "__main__":
    demo()
    print("ok")
