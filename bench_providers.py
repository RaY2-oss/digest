# -*- coding: utf-8 -*-
"""bench_providers.py — стоит ли добавлять бесплатного провайдера в кольцо.

Вопрос не «какая модель умнее», а «выдержит ли провайдер настоящую работу
судьи». Работа такая: батч из десяти статей, системный промпт на 4 КБ, ответ
строго объектом {"1": {"w","r","s"}, ...}. Провайдер, который не держит
формат, бесполезен при любом качестве: батч уходит в pending целиком.

Меряется поэтому в таком порядке:

  1. держит формат — доля разобранных батчей и решённых статей;
  2. согласие с боевым судьёй — регион и масштаб на корзине bench/basket.npz
     (там только ПРИНЯТЫЕ статьи, поэтому доля NO здесь — строгость, а не
     ошибка: судья, перечитывая свои же принятые статьи, отклоняет около
     половины, см. README);
  3. пропускная способность — медиана задержки и на каком запросе прилетел 429.

Квоту облачных провайдеров кольца стенд НЕ трогает: сравниваются только те,
у кого бесплатный доступ, плюс локальная модель как точка отсчёта.

    python bench_providers.py --n 60
    python bench_providers.py --n 60 --only llm7:gpt-oss:20b,local
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time

import numpy as np
import requests

from daily_collector import _JUDGE_SYSTEM, _JUDGE_USER_TMPL, _parse_judge

BASE = os.path.dirname(os.path.abspath(__file__))
BATCH = 10
TEXT_CHARS = 2000

# Локальный слот digest у gpu-broker. Общий 11434 отвечает 503.
LOCAL_URL = "http://10.13.13.3:11431/v1"

# (имя, база, модель, ключ). Ключ None = запрос уходит без Authorization.
PROVIDERS = [
    ("llm7:gpt-oss", "https://api.llm7.io/v1", "gpt-oss:20b", None),
    ("llm7:deepseek-v3", "https://api.llm7.io/v1", "deepseek-v3", None),
    ("llm7:gemini-lite", "https://api.llm7.io/v1", "gemini-3.1-flash-lite", None),
    ("llm7:minimax", "https://api.llm7.io/v1", "minimax-m2.7", None),
    ("llm7:mistral-nemo", "https://api.llm7.io/v1",
     "mistral-Nemo-Instruct-2407", None),
    ("ovh:gpt-oss-120b", "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
     "gpt-oss-120b", None),
    ("ovh:qwen3-32b", "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
     "Qwen3-32B", None),
    ("local:qwen3-1.7b", LOCAL_URL, "qwen3:1.7b", "x"),
]


def sample(n, seed=7):
    """n статей корзины, поровну из каждого разряда масштаба."""
    b = dict(np.load(os.path.join(BASE, "bench", "basket.npz"),
                     allow_pickle=True))
    rnd = random.Random(seed)
    picked = []
    for g in (1, 3, 5):
        pool = [i for i in range(len(b["scale"]))
                if int(b["scale"][i]) == g and len(str(b["text"][i])) > 400]
        picked += rnd.sample(pool, min(n // 3, len(pool)))
    rnd.shuffle(picked)
    texts = ["%s\n%s" % (b["title"][i], b["text"][i]) for i in picked]
    return (texts,
            [str(b["bucket"][i]) for i in picked],
            [int(b["scale"][i]) for i in picked])


def ask(base, model, key, system, user, timeout=180, retries=0, log=None):
    """-> (текст ответа | None, http-код, секунды).

    retries=0 меряет провайдера без подпорок. retries>0 отсиживает Retry-After,
    как это делает кольцо в model_rotation: без этого частотный лимит
    засчитывается в «не держит формат», а это разные беды.
    """
    t0 = time.time()
    for attempt in range(retries + 1):
        try:
            r = requests.post(
                base.rstrip("/") + "/chat/completions",
                headers={"Content-Type": "application/json",
                         **({"Authorization": "Bearer " + key} if key else {})},
                json={"model": model,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": user}],
                      "temperature": 0.0,
                      "max_tokens": 4000},
                timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            return None, str(exc)[:60], time.time() - t0
        if r.status_code in (429, 503) and attempt < retries:
            try:
                wait = float(r.headers.get("Retry-After", 10))
            except ValueError:
                wait = 10.0
            time.sleep(min(wait, 65))
            continue
        break
    dt = time.time() - t0
    if r.status_code != 200:
        return None, r.status_code, dt
    try:
        ch = r.json()["choices"][0]
        msg = ch["message"]
        # gpt-oss отдаёт цепочку рассуждения отдельным полем; content бывает
        # пустым, если весь лимит токенов ушёл в неё — это тоже провал формата,
        # но по другой причине, поэтому finish_reason попадает в журнал.
        content = (msg.get("content") or "").rsplit("</think>", 1)[-1].strip()
        if log is not None and not content:
            log.append({"finish": ch.get("finish_reason"),
                        "usage": r.json().get("usage"),
                        "reasoning_chars": len(msg.get("reasoning") or "")})
        return content, 200, dt
    except Exception:  # noqa: BLE001
        return None, "bad-json", dt


def run(name, base, model, key, texts, gold_region, gold_scale, pause,
        retries=0):
    chunks = [list(range(i, min(i + BATCH, len(texts))))
              for i in range(0, len(texts), BATCH)]
    got = [None] * len(texts)
    lat, codes, first_429, empty = [], [], None, []
    for bi, chunk in enumerate(chunks):
        block = "".join("[%d] %s\n\n"
                        % (j + 1, texts[k].replace("\n", " ")[:TEXT_CHARS])
                        for j, k in enumerate(chunk))
        raw, code, dt = ask(base, model, key, _JUDGE_SYSTEM,
                            _JUDGE_USER_TMPL.format(n=len(chunk),
                                                    articles=block),
                            retries=retries, log=empty)
        lat.append(dt)
        codes.append(code)
        if code == 429 and first_429 is None:
            first_429 = bi + 1
        out = _parse_judge(raw, len(chunk)) if raw else None
        if out:
            for k, v in zip(chunk, out):
                got[k] = v
        if pause:
            time.sleep(pause)

    ok_batches = sum(1 for c in chunks
                     if all(got[k] is not None for k in c))
    decided = [i for i, v in enumerate(got) if v is not None]
    accepted = [i for i in decided if got[i][0] != "NO"]
    region_hit = sum(1 for i in accepted if got[i][0] == gold_region[i])
    scaled = [i for i in accepted if got[i][1] is not None]
    mae = (statistics.fmean(abs(got[i][1] - gold_scale[i]) for i in scaled)
           if scaled else float("nan"))
    dist = {g: sum(1 for i in scaled if got[i][1] == g) for g in range(1, 6)}
    return {
        "provider": name, "model": model,
        "batches": len(chunks), "batches_ok": ok_batches,
        "decided": len(decided), "n": len(texts),
        "accepted": len(accepted),
        "region_agree": region_hit / len(accepted) if accepted else float("nan"),
        "scale_mae": mae, "scale_dist": dist,
        "lat_med": statistics.median(lat), "lat_max": max(lat),
        "first_429": first_429,
        "codes": sorted({str(c) for c in codes}),
        "empty_answers": empty,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--only", default="")
    ap.add_argument("--pause", type=float, default=0.0,
                    help="пауза между батчами, с (для лимита 2 req/min — 31)")
    ap.add_argument("--retries", type=int, default=0,
                    help="отсиживать Retry-After столько раз (как кольцо)")
    ap.add_argument("--out", default=os.path.join(BASE, "bench",
                                                  "providers.json"))
    a = ap.parse_args()

    texts, gold_region, gold_scale = sample(a.n)
    print("статей %d, батчей по %d: %d"
          % (len(texts), BATCH, (len(texts) + BATCH - 1) // BATCH))

    want = [s for s in a.only.split(",") if s]
    rows = []
    for name, base, model, key in PROVIDERS:
        if want and not any(w in name for w in want):
            continue
        pause = a.pause if name.startswith("ovh:") else 0.0
        t0 = time.time()
        row = run(name, base, model, key, texts, gold_region, gold_scale, pause,
                  retries=a.retries)
        row["wall_s"] = round(time.time() - t0)
        rows.append(row)
        print("%-18s батчей %d/%d, решено %3d/%d, принято %3d, регион %s, "
              "MAE %s, медиана %.1fс, коды %s"
              % (name, row["batches_ok"], row["batches"], row["decided"],
                 row["n"], row["accepted"],
                 ("%.2f" % row["region_agree"]) if row["accepted"] else "-",
                 ("%.2f" % row["scale_mae"]) if row["accepted"] else "-",
                 row["lat_med"], ",".join(row["codes"])))
        sys.stdout.flush()

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)
    print("-> %s" % a.out)


def _selfcheck():
    """Разбор и подсчёт метрик не должны зависеть от сети."""
    texts = ["a", "b"]
    raw = '{"1": {"w": "law", "r": "TR", "s": 5}, "2": {"w": "-", "r": "NO"}}'
    out = _parse_judge(raw, 2)
    assert out == [("TR", 5), ("NO", None)], out
    print("bench_providers selfcheck: ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
