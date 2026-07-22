# -*- coding: utf-8 -*-
"""Оценка precision эмбеддингового предфильтра на РЕАЛЬНЫХ негативах.

Негативы = URL, помеченные в логе как "Accept (rule-pass)" (т.е. дошедшие до
LLM), но отсутствующие в БД -> LLM их отверг. Скачиваем выборку, считаем
margin = max_cos(pos_prototypes) - max_cos(neg_prototypes) и смотрим, какую
долю из них порог margin>0 отсёк бы ДО обращения к LLM.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("HF_HOME", "/opt/digest/.cache/huggingface")

import random, re, sqlite3, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np, requests
from trafilatura import bare_extraction

sys.path.insert(0, "/opt/digest")
import config
from embed_probe import POS, NEG

LOG = "/opt/digest/logs/cron_collector.log"
N_SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 150
HDR = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}


def get_text(url):
    try:
        r = requests.get(url, headers=HDR, timeout=10)
        r.raise_for_status()
        d = bare_extraction(r.text, url=url, with_metadata=True,
                            include_comments=False, favor_precision=True)
        if not d:
            return None
        t = (d.text if hasattr(d, "text") else d.get("text", "")) or ""
        return t.strip() if len(t.strip()) >= config.MIN_TEXT_LENGTH else None
    except Exception:
        return None


def main():
    conn = sqlite3.connect(config.DB_PATH)
    in_db = {r[0] for r in conn.execute("SELECT url FROM articles")}

    urls = []
    pat = re.compile(r"Accept \(rule-pass\).*?: (https?://\S+) \|")
    with open(LOG, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = pat.search(line)
            if m and m.group(1) not in in_db:
                urls.append(m.group(1))
    urls = list(dict.fromkeys(urls))
    print(f"Кандидатов-негативов в логе: {len(urls)}; берём выборку {N_SAMPLE}")
    random.seed(42)
    sample = random.sample(urls, min(N_SAMPLE, len(urls)))

    with ThreadPoolExecutor(max_workers=8) as ex:
        texts = [t for t in ex.map(get_text, sample) if t]
    print(f"Успешно скачано и извлечено: {len(texts)}")
    if not texts:
        return

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(config.EMBEDDING_MODEL)
    p = m.encode(["query: " + t for t in POS], convert_to_numpy=True).astype(np.float32)
    n = m.encode(["query: " + t for t in NEG], convert_to_numpy=True).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    n /= np.linalg.norm(n, axis=1, keepdims=True)

    E = m.encode(["passage: " + t for t in texts], batch_size=8,
                 convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    margin = (E @ p.T).max(axis=1) - (E @ n.T).max(axis=1)

    print("\n-- margin НЕГАТИВОВ (отвергнутых LLM) --")
    for q in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{q:02d} = {np.percentile(margin, q):+.4f}")
    print()
    for thr in [-0.01, 0.00, 0.005, 0.01, 0.02, 0.03]:
        keep = 100 * (margin > thr).mean()
        print(f"  порог {thr:+.3f}: прошло бы дальше {keep:5.1f}% негативов "
              f"(отсечено {100-keep:5.1f}%)")


if __name__ == "__main__":
    main()
