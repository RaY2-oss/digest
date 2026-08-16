# -*- coding: utf-8 -*-
"""
bench2.py — второй раунд, с поправками по итогам первого.

Что проверяем дополнительно:
  qwen3 + /no_think — в первом раунде Qwen сломал JSON именно там, где ушёл в
      длинное рассуждение (2914 токенов). Qwen3 умеет отключать режим
      размышлений директивой в промпте; для строгого формата это должно
      помочь, а заодно втрое сократить время.
  gemma3 q8_0     — те же 12B весов, но почти без потерь квантования. Способ
      поднять качество, не увеличивая модель: на 16 ГБ 12B@q8 помещается,
      а 27B не помещается ни в каком вменяемом кванте.
"""

import json
import re
import sqlite3
import sys
import time

import requests

sys.path.insert(0, "/opt/digest")
from sunday_processor_mmr import _SYSTEM_PROMPT_RETELL  # noqa: E402

OLLAMA = "http://10.13.13.3:11434/v1/chat/completions"

# (метка, модель, добавка к системному промпту)
VARIANTS = [
    ("qwen3-nothink", "qwen3:14b", "\n/no_think"),
    ("gemma3-q4", "gemma3:12b", ""),
    ("gemma3-q8", "gemma3:12b-it-q8_0", ""),
]
N_ARTICLES = 5

CYRILLIC_LEAKS = ["Меб", "Тенмак", "Йок", "Тюбитак", "Ишкур", "Эрасмус"]
# Грубые русские ошибки, замеченные в первом раунде. Список заведомо неполный —
# это индикатор, а не грамматическая проверка.
RU_ERRORS = ["Турские", "трилетн", "университета Университет", "Университета Эгейского университета"]


def fetch(limit):
    con = sqlite3.connect("/opt/digest/data/digest.db")
    con.row_factory = sqlite3.Row
    cur = con.execute(
        "SELECT url, title, text FROM articles "
        "WHERE text IS NOT NULL AND length(text) > 1500 "
        "ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cur]
    con.close()
    return rows


def ask(model, system, user, timeout=600):
    t0 = time.time()
    r = requests.post(OLLAMA, json={
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.1,
    }, timeout=timeout)
    dt = time.time() - t0
    r.raise_for_status()
    b = r.json()
    c = b["choices"][0]["message"]["content"]
    if "</think>" in c:
        c = c.rsplit("</think>", 1)[-1]
    return c.strip(), dt, (b.get("usage") or {}).get("completion_tokens", 0)


def score(text):
    out = {}
    try:
        m = re.search(r"\{.*\}", text, re.S)
        data = json.loads(m.group(0)) if m else None
    except (ValueError, AttributeError):
        data = None
    out["json"] = isinstance(data, dict)
    s = data.get("summary", "") if isinstance(data, dict) else ""
    if not s:
        s = text
    out["sentences"] = len(re.findall(r"[.!?](?:\s|$)", s))
    out["in_range"] = 3 <= out["sentences"] <= 4
    out["one_para"] = "\n" not in s.strip()
    out["latin_ok"] = not any(b in s for b in CYRILLIC_LEAKS)
    out["ru_errors"] = sum(1 for e in RU_ERRORS if e in s)
    # Цифры в тексте — прокси фактической насыщенности: пересказ, сохранивший
    # конкретику (сколько команд, сколько заявок), полезнее обтекаемого.
    out["digits"] = len(re.findall(r"\d", s))
    out["chars"] = len(s)
    out["summary"] = s
    return out


def main():
    arts = fetch(N_ARTICLES)
    print(f"статей: {len(arts)}\n")
    agg = {v[0]: {"t": 0.0, "json": 0, "range": 0, "para": 0,
                  "err": 0, "digits": 0, "n": 0} for v in VARIANTS}

    for i, a in enumerate(arts, 1):
        user = f"TITLE: {a['title']}\n\nTEXT:\n{a['text'][:6000]}"
        print("=" * 72)
        print(f"[{i}] {a['title'][:88]}")
        for label, model, extra in VARIANTS:
            try:
                text, dt, tok = ask(model, _SYSTEM_PROMPT_RETELL + extra, user)
            except Exception as exc:  # noqa: BLE001
                print(f"  {label:<15} ОШИБКА {type(exc).__name__}: {str(exc)[:80]}")
                continue
            s = score(text)
            g = agg[label]
            g["t"] += dt; g["n"] += 1
            g["json"] += int(s["json"]); g["range"] += int(s["in_range"])
            g["para"] += int(s["one_para"]); g["err"] += s["ru_errors"]
            g["digits"] += s["digits"]
            flags = []
            if not s["json"]:
                flags.append("JSON СЛОМАН")
            if not s["in_range"]:
                flags.append(f"{s['sentences']} предл.")
            if not s["one_para"]:
                flags.append("абзац разбит")
            if s["ru_errors"]:
                flags.append(f"{s['ru_errors']} груб.ошибок")
            mark = ("  [" + ", ".join(flags) + "]") if flags else "  [ok]"
            print(f"  {label:<15} {dt:5.1f}с {tok:5d}ток  цифр:{s['digits']:3d}{mark}")
        print()

    print("=" * 72)
    print(f"{'вариант':<15} {'сек':>6} {'JSON':>7} {'формат':>7} {'абзац':>7} {'ошибки':>7} {'цифры':>7}")
    for label, _, _ in VARIANTS:
        g = agg[label]
        n = max(g["n"], 1)
        print(f"{label:<15} {g['t']/n:6.1f} {g['json']}/{n:>5} {g['range']}/{n:>5} "
              f"{g['para']}/{n:>5} {g['err']:>7} {g['digits']/n:7.1f}")


if __name__ == "__main__":
    main()
