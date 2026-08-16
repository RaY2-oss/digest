# -*- coding: utf-8 -*-
"""
bench_local.py — сравнение локальных моделей на РЕАЛЬНОЙ задаче дайджеста.

Синтетический "напиши стихотворение" тут бесполезен: важно, справляется ли
модель именно с _SYSTEM_PROMPT_RETELL — пересказ на русском с глоссами,
латиницей для аббревиатур и жёстким форматом JSON. Поэтому берём настоящий
промпт проекта и настоящие статьи из базы.

Оценивается не только "красиво ли", а измеримое:
  - отдала ли валидный JSON нужной формы (иначе пайплайн её ответ выбросит);
  - уложилась ли в 3-4 предложения одним абзацем;
  - не насыпала ли кириллицей там, где промпт требует латиницу;
  - сколько секунд заняла — при равном качестве быстрее лучше.
"""

import json
import re
import sqlite3
import sys
import time

import requests

sys.path.insert(0, "/opt/digest")
from sunday_processor_mmr import _SYSTEM_PROMPT_RETELL  # noqa: E402

# Через туннель, а не с самого сервера: это ровно тот путь, которым пойдут
# боевые вызовы, вместе с задержкой обфусцированного канала.
OLLAMA = "http://10.13.13.3:11434/v1/chat/completions"
MODELS = ["qwen3:14b", "gemma3:12b"]
N_ARTICLES = 3

# Аббревиатуры, которые промпт требует оставлять латиницей. Их кириллический
# двойник — прямое нарушение, и его видно регуляркой.
CYRILLIC_LEAKS = ["Меб", "Тенмак", "Йок", "Тюбитак", "Ишкур", "Эрасмус"]


def fetch_articles(limit):
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
    body = r.json()
    content = body["choices"][0]["message"]["content"]
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[-1]
    usage = body.get("usage") or {}
    return content.strip(), dt, usage


def score(text):
    """Измеримые признаки соответствия промпту."""
    out = {}
    try:
        m = re.search(r"\{.*\}", text, re.S)
        data = json.loads(m.group(0)) if m else None
    except (ValueError, AttributeError):
        data = None
    out["json"] = data is not None
    summary = (data or {}).get("summary", "") if isinstance(data, dict) else ""
    if not summary:
        summary = text
    out["sentences"] = len(re.findall(r"[.!?](?:\s|$)", summary))
    out["in_range"] = 3 <= out["sentences"] <= 4
    out["one_para"] = "\n" not in summary.strip()
    out["latin_ok"] = not any(bad in summary for bad in CYRILLIC_LEAKS)
    # Латиница внутри русского текста — признак непереведённого куска, но
    # аббревиатуры латиницей промптом как раз ТРЕБУЮТСЯ, поэтому считаем только
    # длинные слова строчными буквами: они на аббревиатуру не похожи.
    out["untranslated"] = len(re.findall(r"\b[a-z]{5,}\b", summary))
    out["chars"] = len(summary)
    out["summary"] = summary
    return out


def main():
    arts = fetch_articles(N_ARTICLES)
    print(f"статей для теста: {len(arts)}\n")
    totals = {m: {"time": 0.0, "json": 0, "range": 0, "latin": 0} for m in MODELS}

    for i, a in enumerate(arts, 1):
        user = f"TITLE: {a['title']}\n\nTEXT:\n{a['text'][:6000]}"
        print("=" * 70)
        print(f"[{i}] {a['title'][:90]}")
        print(f"    {a['url'][:90]}")
        for model in MODELS:
            try:
                text, dt, usage = ask(model, _SYSTEM_PROMPT_RETELL, user)
            except Exception as exc:  # noqa: BLE001
                print(f"  {model:<14} ОШИБКА: {type(exc).__name__}: {exc}")
                continue
            s = score(text)
            totals[model]["time"] += dt
            totals[model]["json"] += int(s["json"])
            totals[model]["range"] += int(s["in_range"])
            totals[model]["latin"] += int(s["latin_ok"])
            tok = usage.get("completion_tokens", 0)
            tps = tok / dt if dt and tok else 0
            print(f"\n  --- {model} — {dt:.1f}с, {tok} токенов, {tps:.1f} т/с ---")
            print(f"  JSON:{'да' if s['json'] else 'НЕТ'}  "
                  f"предложений:{s['sentences']}{'' if s['in_range'] else ' (вне 3-4!)'}  "
                  f"абзац:{'один' if s['one_para'] else 'РАЗБИТ'}  "
                  f"латиница:{'ок' if s['latin_ok'] else 'НАРУШЕНА'}  "
                  f"непереведённых:{s['untranslated']}")
            print(f"  {s['summary'][:700]}")
        print()

    print("=" * 70)
    print("ИТОГО")
    n = len(arts)
    for m in MODELS:
        t = totals[m]
        print(f"  {m:<14} среднее {t['time']/n:6.1f}с   "
              f"JSON {t['json']}/{n}   формат 3-4 предл. {t['range']}/{n}   "
              f"латиница {t['latin']}/{n}")


if __name__ == "__main__":
    main()
