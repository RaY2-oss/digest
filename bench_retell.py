# -*- coding: utf-8 -*-
"""bench_retell.py — полный промпт пересказа против урезанного.

Метод из аудита: прогнать один набор сюжетов двумя редакциями системного
промпта и сравнить по срабатываниям сторожей. У метода есть граница, которую
надо назвать сразу: **правило без сторожа не проверяется никак**. В промпте
около сорока правил, сторожей четырнадцать. «Не калькируй английский
синтаксис» или «yerli значит отечественный» ничем не измеряются — снять их
можно только на глаз, а на глаз мы и так умеем.

Поэтому урезается не «вдвое», а ровно то, за чем стоит сторож, ни разу не
сработавший за неделю (замер по logs/processor.log*, 401 отказ за 8 суток):

    правило                                  сторож                 сработал
    «RUSSIAN, не украинский/болгарский»       _cyrillic_share             0
                                              _foreign_cyrillic_share     0
    надпись на объекте с глоссой              _quote_not_in_source        1
    обоснование бюджета длины (40 слов)       слишком длинный             0

Не трогаем правила, чьи сторожа стреляют: translit_guard (44), согласование
(43), смешение скриптов (30), число не из источника (18), аббревиатуры (15),
заголовок-обрубок (13). И не трогаем «заголовок без сказуемого»: сторож нулевой,
но и правило, и сторож заведены 14.08.2026 — двух суток мало, чтобы называть
ноль результатом.

Про мощность замера. Частота срабатывания у отдельного сторожа 1–3% попыток;
на сорока сюжетах это ноль или один случай, и различить редакции по одному
сторожу нельзя. Различима только СУММА отказов (около 31% попыток) и
распределение длин — длина есть у каждого ответа, а не у одного из тридцати.

Запуск:
    python bench_retell.py --n 40 --model qwen3:8b
"""
import argparse
import collections
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

OLLAMA = os.environ.get("OLLAMA_URL", "http://10.13.13.3:11431/api/chat")

# Куски системного промпта, которые снимает урезанная редакция. Ключ —
# что снимается, значение — точный текст из _SYSTEM_PROMPT_RETELL.
_CUTS = {
    "не-украинский": (
        "RUSSIAN means Russian specifically, NOT Ukrainian, Belarusian, "
        "Bulgarian or any \\\nother Cyrillic language. Never mirror the "
        "language of the source article.\n"),
    "надпись-на-объекте": (
        "- Text that will physically READ in the source language — an "
        "inscription, a \\\nsign, a badge, a slogan — keeps its original "
        "spelling, with a Russian gloss \\\nright after it: надпись «Bahçe "
        "Görevlisi» («работник по уходу за территорией»). \\\nPutting a "
        "Russian phrase in the quotation marks instead invents a quote that "
        "\\\nnobody will see on the object.\n"),
    "обоснование-длины": (
        " This is a hard budget, not a suggestion — count the \\\nwords before "
        "you answer and cut until you are inside it. For reference, this \\\n"
        "very rule is about 40 words long. Sentence count does not matter: use "
        "three \\\nshort sentences or five, whichever fits the facts — the "
        "reader's effort tracks \\\ntotal length, not the number of full "
        "stops."),
}


def prompts():
    """-> {'full': текст, 'lean': текст}. Урезание проверяется, а не верится:
    если кусок в промпте не найден дословно, замер падает здесь, а не выдаёт
    два одинаковых промпта под разными именами."""
    import sunday_processor_mmr as P
    full = P._SYSTEM_PROMPT_RETELL
    lean = full
    for name, chunk in _CUTS.items():
        plain = chunk.replace("\\\n", "")
        if plain not in lean:
            raise SystemExit("Кусок «%s» в промпте не найден — правило "
                             "переписали, поправь _CUTS." % name)
        lean = lean.replace(plain, "")
    return {"full": full, "lean": lean}


def ask(model, system, user, schema, timeout=900):
    """Тот же путь, что у боевого пересказа: родной /api/chat со схемой."""
    for _ in range(20):
        r = requests.post(OLLAMA, json={
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False, "think": False, "format": schema,
            "options": {"temperature": 0.1, "num_ctx": 16384},
        }, timeout=timeout)
        if r.status_code == 503:
            try:
                wait = float(r.json().get("retry_after_s", 30))
            except Exception:  # noqa: BLE001
                wait = 30.0
            time.sleep(min(wait, 60))
            continue
        r.raise_for_status()
        return r.json()["message"]["content"]
    raise RuntimeError("карта занята дольше, чем стоит ждать")


def sample(n):
    """n сюжетов, которые пайплайн и правда пересказывал бы: верх каждой
    корзины по важности. Случайная выборка из корзины дала бы промпты, до
    которых пересказ никогда не доходит."""
    import bench
    import sunday_processor_mmr as P
    basket, issues = bench.load()
    P._ARCHIVE = bench._archive_emb(issues)
    arts = bench._articles(basket)
    out = []
    per = -(-n // len(P.QUOTA))
    for bucket in P.QUOTA:
        sub = [a for a in arts if (a[4] if a[4] in P.QUOTA else "MIX") == bucket]
        stories = P._group_stories(sub)
        E = np.vstack([a[2] for a in stories]).astype(np.float32)
        for i in np.argsort(-P._importance(stories, E))[:per]:
            out.append(stories[i])
    return out[:n]


def run_arm(model, system, stories, workers=3):
    """-> (список причин отказа или None, список длин пересказа в словах)."""
    import sunday_processor_mmr as P

    def one(story):
        user, _urls = P._retell_sources(P._versions(story))
        if not user:
            return None, None
        try:
            raw = ask(model, system, user, P._RETELL_SCHEMA)
        except Exception as exc:  # noqa: BLE001
            return "запрос упал: %s" % type(exc).__name__, None
        result = P._parse_llm_json(raw)
        if result is None:
            return "не удалось распарсить JSON", None
        ok, reason = P._validate_summary_fast(result, user)
        words = len(str(result.get("summary", "")).split())
        return (None if ok else reason), words

    with ThreadPoolExecutor(max_workers=workers) as ex:
        pairs = list(ex.map(one, stories))
    return [p[0] for p in pairs], [p[1] for p in pairs if p[1]]


def _rule(reason):
    """Причина отказа -> короткое имя сторожа."""
    if reason is None:
        return "прошло"
    for needle, name in (
            ("слово источника кириллицей", "translit_guard"),
            ("нарушено согласование", "согласование"),
            ("смешение скриптов", "смешение скриптов"),
            ("число не из источника", "число не из источника"),
            ("аббревиатура источника", "аббревиатура кириллицей"),
            ("заголовок-обрубок", "заголовок-обрубок"),
            ("заголовок без сказуемого", "заголовок без сказуемого"),
            ("пересказ оборван", "оборван"),
            ("пересказ не на русском", "не на русском"),
            ("пересказ на другом кириллическом", "другой кириллический"),
            ("надпись не из источника", "надпись не из источника"),
            ("принятое написание подменено", "экзоним"),
            ("«изучение» без объекта", "изучение без объекта"),
            ("третий алфавит", "третий алфавит"),
            ("пересказ слишком длинный", "слишком длинный"),
            ("не удалось распарсить", "невалидный JSON")):
        if reason.startswith(needle):
            return name
    return reason[:40]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--out", default=os.path.join(BASE, "bench",
                                                  "retell_runs.json"))
    a = ap.parse_args()

    variants = prompts()
    stories = sample(a.n)
    print("сюжетов: %d, модель %s" % (len(stories), a.model))
    for k, v in variants.items():
        print("  промпт %-5s %d знаков" % (k, len(v)))

    report = {}
    for name, system in variants.items():
        t0 = time.time()
        reasons, words = run_arm(a.model, system, stories)
        hits = collections.Counter(_rule(r) for r in reasons)
        passed = hits.pop("прошло", 0)
        report[name] = {
            "chars": len(system), "n": len(reasons), "passed": passed,
            "rejected": len(reasons) - passed, "by_rule": dict(hits),
            "words_median": float(np.median(words)) if words else None,
            "words_p90": float(np.percentile(words, 90)) if words else None,
            "words_over_120": sum(1 for w in words if w > 120),
            "seconds": round(time.time() - t0),
        }
        r = report[name]
        print("\n=== %s (%ds)" % (name, r["seconds"]))
        print("  прошло %d/%d, забраковано %d" % (r["passed"], r["n"],
                                                  r["rejected"]))
        for rule, cnt in sorted(hits.items(), key=lambda kv: -kv[1]):
            print("    %2d  %s" % (cnt, rule))
        print("  длина: медиана %.0f слов, 90-й процентиль %.0f, длиннее 120: %d"
              % (r["words_median"] or 0, r["words_p90"] or 0,
                 r["words_over_120"]))

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"model": a.model, "report": report}, fh,
                  ensure_ascii=False, indent=1)
    print("\n-> %s" % a.out)


def _selfcheck():
    v = prompts()
    assert len(v["lean"]) < len(v["full"]), (len(v["lean"]), len(v["full"]))
    assert "Ukrainian" not in v["lean"] and "Ukrainian" in v["full"]
    assert "Bahçe Görevlisi" not in v["lean"]
    assert "90–120 WORDS" in v["lean"], "бюджет длины снимать нельзя"
    assert "TÜBİTAK" in v["lean"], "правила со стреляющими сторожами остаются"
    assert _rule(None) == "прошло"
    assert _rule("нарушено согласование: «х»") == "согласование"
    assert _rule("что-то новое") == "что-то новое"
    print("bench_retell: ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
