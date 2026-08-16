# -*- coding: utf-8 -*-
"""Проверка компактности: новый промпт просит 380-550 знаков вместо «3-4
предложения». Меряем на реальных статьях, что получилось по факту и не
потерялись ли детали — числа и имена собственные должны остаться."""

import json
import re
import sqlite3
import sys

import requests

sys.path.insert(0, "/opt/digest")
from sunday_processor_mmr import _SYSTEM_PROMPT_RETELL, _validate_summary_fast  # noqa

URL = "http://10.13.13.3:11434/v1/chat/completions"
MODEL = "gemma3:12b-it-q8_0"
N = 5


def ask(user):
    r = requests.post(URL, json={
        "model": MODEL,
        "messages": [{"role": "system", "content": _SYSTEM_PROMPT_RETELL},
                     {"role": "user", "content": user}],
        "temperature": 0.1,
    }, timeout=300)
    r.raise_for_status()
    c = r.json()["choices"][0]["message"]["content"]
    if "</think>" in c:
        c = c.rsplit("</think>", 1)[-1]
    m = re.search(r"\{.*\}", c, re.S)
    return json.loads(m.group(0)) if m else None


con = sqlite3.connect("/opt/digest/data/digest.db")
rows = con.execute("SELECT title, text FROM articles WHERE text IS NOT NULL "
                   "AND length(text)>1500 ORDER BY id DESC LIMIT ?", (N,)).fetchall()

lens, ok_n = [], 0
print(f"{'знаков':>7} {'цифр':>5} {'валид':>6}  начало пересказа")
for title, text in rows:
    user = f"TITLE: {title}\n\nTEXT:\n{text[:6000]}"
    try:
        d = ask(user)
    except Exception as e:  # noqa: BLE001
        print(f"ошибка: {type(e).__name__}")
        continue
    if not d or "summary" not in d:
        print("не разобрался JSON")
        continue
    s = d["summary"]
    ok, why = _validate_summary_fast(d, user)
    ok_n += int(ok)
    lens.append(len(s))
    digits = len(re.findall(r"\d", s))
    print(f"{len(s):>7} {digits:>5} {'да' if ok else 'НЕТ':>6}  {s[:105]}")
    if not ok:
        print(f"        причина: {why}")

if lens:
    print(f"\nсредняя длина {sum(lens)/len(lens):.0f} знаков "
          f"(min {min(lens)}, max {max(lens)}), в целевом коридоре 380-550: "
          f"{sum(1 for x in lens if 380 <= x <= 550)}/{len(lens)}, "
          f"валидных {ok_n}/{len(lens)}")
