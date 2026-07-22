# -*- coding: utf-8 -*-
"""Калибровка zero-shot эмбеддингового предфильтра на уже принятых статьях.

Идея: e5 обучен на парах (query, passage). Строим набор прототипов-запросов
(положительных = наши темы, отрицательных = типовой шум) и считаем для каждой
исторически ПРИНЯТОЙ статьи score = max_cos(pos) - max_cos(neg).
Порог выбираем по нижним перцентилям — это прямая оценка потери полноты.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("HF_HOME", "/opt/digest/.cache/huggingface")

import sqlite3, sys
import numpy as np

sys.path.insert(0, "/opt/digest")
import config

POS = [
    "science, research, academic institutions in Turkey, Central Asia or the South Caucasus",
    "education policy, universities, schools, curriculum reform in Turkey, Kazakhstan, Uzbekistan, Kyrgyzstan, Tajikistan, Turkmenistan, Georgia, Armenia, Azerbaijan",
    "youth policy, students, scholarships, academic mobility and exchange programmes",
    "foreign-funded university campus, scientific cooperation agreement or scholarship programme in Central Asia or the South Caucasus",
    "наука, образование, университеты и молодёжная политика в Турции, Центральной Азии и на Южном Кавказе",
]
NEG = [
    "football match, sports results, championship, transfer of a player",
    "crime, murder, court trial, arrest, police investigation",
    "war, military strike, casualties, front line, armed conflict",
    "stock market, currency rate, oil price, corporate earnings",
    "celebrity, entertainment, television show, music release",
    "weather forecast, traffic accident, local municipal event",
    "elections, party politics, parliamentary session, diplomatic visit",
]


def main():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(config.EMBEDDING_MODEL)

    p = m.encode(["query: " + t for t in POS], convert_to_numpy=True).astype(np.float32)
    n = m.encode(["query: " + t for t in NEG], convert_to_numpy=True).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    n /= np.linalg.norm(n, axis=1, keepdims=True)

    conn = sqlite3.connect(config.DB_PATH)
    rows = conn.execute(
        "SELECT region_bucket, embedding FROM articles WHERE embedding IS NOT NULL"
    ).fetchall()
    E = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    buckets = np.array([r[0] or "?" for r in rows])

    pos_max = (E @ p.T).max(axis=1)
    neg_max = (E @ n.T).max(axis=1)
    margin = pos_max - neg_max

    print(f"Статей в БД (все прошли LLM-фильтр): {len(E)}")
    print("\n-- max cos к ПОЛОЖИТЕЛЬНЫМ прототипам --")
    for q in [1, 2, 5, 10, 25, 50, 75, 90]:
        print(f"  p{q:02d} = {np.percentile(pos_max, q):.4f}")
    print("\n-- margin = max_pos - max_neg --")
    for q in [1, 2, 5, 10, 25, 50, 75, 90]:
        print(f"  p{q:02d} = {np.percentile(margin, q):+.4f}")
    print(f"\n  доля с margin > 0 : {100*(margin > 0).mean():.1f}%")
    for thr in [0.00, 0.01, 0.02, 0.03, 0.05]:
        print(f"  доля с margin > {thr:.2f}: {100*(margin > thr).mean():.1f}%")
    print("\n-- по регионам (медиана margin) --")
    for b in sorted(set(buckets)):
        sel = buckets == b
        print(f"  {b:4s} n={sel.sum():4d}  med={np.median(margin[sel]):+.4f}")


if __name__ == "__main__":
    main()
