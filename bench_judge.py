# -*- coding: utf-8 -*-
"""bench_judge.py — калибровка шкалы масштаба у судьи, на локальной модели.

Зачем локальная. Замер требует сотен вызовов (две редакции промпта × два
прогона на самосогласованность × батчи), а облачная квота суточная и уходит
на настоящие прогоны. Локальная модель отвечает на тот же вопрос, который
здесь и задаётся: «меняет ли редакция промпта РАСПРЕДЕЛЕНИЕ оценок». Это
свойство рубрики, а не провайдера. Абсолютные числа переносить на облачного
судью нельзя, сдвиг — можно.

Что сравнивается:

    v1 — промпт как в daily_collector: три деления, ответ «TR3» одной строкой,
         доля («в десятке не больше двух троек»), порядок статей как пришёл;
    v2 — пять делений с якорными примерами вместо перечня признаков, ответ
         разнесён на поля {"r":..,"s":..}, перед цифрой одно слово «что
         изменилось», порядок батча перемешан.

Метрики:

    top_share   — доля верхнего разряда. У v1 на живом потоке 0.373;
    dist        — всё распределение: у трёхбалльной шкалы верхний разряд
                  занимает треть корпуса, и это её единственная беда;
    self_agree  — согласие модели с самой собой на том же батче при другом
                  порядке. Ниже него никакая правка весов не различима;
    vs_prod     — согласие с оценкой боевого судьи (колонка articles.scale).

Запуск:
    python bench_judge.py --n 120 --model qwen3.5:9b
    python bench_judge.py --apply v2      # переписать scale в bench/basket.npz
"""
import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

# Порт 11431 — слот digest у gpu-broker (Radeon, приоритет 1), а не общий
# 11434 «всё, что не назвалось»: тот идёт только по свободной карте и на
# занятой отвечает 503, из-за чего замер молча превращался в 30 пропусков.
OLLAMA = os.environ.get("OLLAMA_URL", "http://10.13.13.3:11431/api/chat")
BATCH = 10          # как LLM_FILTER_BATCH в daily_collector
TEXT_CHARS = 2000   # как JUDGE_TEXT_CHARS


# ---------------------------------------------------------------------------
# Редакция v2: пять делений, якоря, «что изменилось» перед цифрой
# ---------------------------------------------------------------------------
# Признаки судья читает как чек-лист: нашёл министерство — поставил верхний
# разряд. Якорь так прочитать нельзя, с ним статью можно только сравнить.
# Примеры взяты из корзины 09.08–16.08.2026 и размечены по вопросу самой
# рубрики («что было бы в стране иначе»), а не по тому, что поставил боевой
# судья: наследовать его же смещение — значит откалибровать шкалу по нему.
_SCALE_V2 = (
    "Then rate how much the event CHANGES the country's science, education or "
    "youth system, on a scale of 1 to 5. Compare the article to these anchors "
    "instead of looking for keywords:\n"
    "\n"
    "  5 — the rules of the system are different afterwards.\n"
    "      · a law on student amnesty is published in the Official Gazette and "
    "takes effect;\n"
    "      · a national curriculum reform is adopted for all schools.\n"
    "  4 — a country-wide programme or body is created or funded, but the "
    "existing rules stay.\n"
    "      · a state AI-ecosystem call for proposals opens nationwide funding;\n"
    "      · an intergovernmental education agreement is signed with concrete "
    "obligations.\n"
    "  3 — one institution or one sector really changes, with money or "
    "obligations behind it.\n"
    "      · a university opens a new laboratory with advanced equipment;\n"
    "      · universities in one city shorten the academic year by a month.\n"
    "  2 — a change is announced or prepared, or numbers of a regular process "
    "are published.\n"
    "      · state-order places for a group of specialities are announced;\n"
    "      · a ministry says an exam will be revised next year.\n"
    "  1 — nothing in the system changes.\n"
    "      · a minister visits a province and is met at the airport;\n"
    "      · a company presents a book, a forum is held, a contest names its "
    "winners.\n"
    "\n"
    "A ministry, a minister or a national company as the ACTOR does not by "
    "itself raise the level, and neither does the fact that something happens "
    "across the whole country. Ask what would be DIFFERENT in the country if "
    "this news had not happened.\n"
)

_USER_V2 = (
    "Below are {n} articles, each preceded by its number in square brackets.\n"
    "Reply ONLY with a JSON object, no extra text. For every article number "
    'give an object with three fields IN THIS ORDER: '
    '{{"w": "<one word or short phrase: what changed>", "r": "<region>", '
    '"s": <scale 1-5>}}.\n'
    'For a rejected article reply {{"w": "-", "r": "NO"}} with no "s".\n'
    'Example: {{"1": {{"w": "law adopted", "r": "TR", "s": 5}}, '
    '"2": {{"w": "-", "r": "NO"}}}}\n'
    "Every number from 1 to {n} must appear exactly once.\n\n"
    "{articles}"
)


# Та же пятибалльная шкала, но признаками вместо якорей — как в v1, только
# делений пять. Нужна ровно для одного вопроса: распределение выровняли якоря
# или само число делений? Без неё v2 меняет четыре вещи разом и непонятно, за
# какую платить сопровождением (якоря надо пересматривать, когда меняется
# повестка; лишнее деление — нет).
_SCALE_V2A = (
    "Then rate how much the event CHANGES the country's science, education or "
    "youth system, on a scale of 1 to 5:\n"
    "  5 - a law or reform adopted, a national programme launched, an "
    "intergovernmental agreement signed, a new national university or "
    "laboratory opened, national statistics or a country-wide ranking "
    "published;\n"
    "  4 - a country-wide programme or body is created or funded while the "
    "existing rules stay;\n"
    "  3 - a real change inside ONE institution or ONE sector, with money or "
    "obligations behind it;\n"
    "  2 - such a change is announced or prepared, or the numbers of a regular "
    "process are published;\n"
    "  1 - nothing in the system changes: a meeting, a visit, a forum, a "
    "competition or its results, a corporate or charitable initiative, and any "
    "seasonal or annual routine.\n"
    "\n"
    "A ministry, a minister or a national company as the ACTOR does not by "
    "itself raise the level, and neither does the fact that something happens "
    "across the whole country. Ask what would be DIFFERENT in the country if "
    "this news had not happened.\n"
)


def prompts(variant):
    """-> (system, user_template, число делений шкалы)."""
    import daily_collector as D
    if variant == "v1":
        return D._JUDGE_SYSTEM, D._JUDGE_USER_TMPL, 3
    # Тело промпта — то же самое: T1/T2 и список отказов замером не оспорены,
    # спор идёт только о шкале. Меняем ровно её, иначе непонятно, что подействовало.
    head = D._JUDGE_SYSTEM.split("Right after the region, append ONE digit")[0]
    tail = "The article may be in any language — judge by content, not language."
    scale = {"v2": _SCALE_V2, "v2a": _SCALE_V2A}[variant]
    return head + scale + "\n" + tail, _USER_V2, 5


# ---------------------------------------------------------------------------
_VERDICT_RE = re.compile(r"(NO|TR|CA|SC|MIX)\s*([1-5])?$")


def parse(content, n, variant):
    """-> список (регион, масштаб 1..K) длины n; None = решения нет."""
    m = re.search(r"\{.*\}", content or "", re.DOTALL)
    if not m:
        return [None] * n
    try:
        data = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return [None] * n
    if not isinstance(data, dict):
        return [None] * n
    out = []
    for i in range(1, n + 1):
        v = data.get(str(i), data.get(i))
        if variant != "v1" and isinstance(v, dict):
            r = str(v.get("r", "")).strip().upper()
            s = v.get("s")
            if r not in ("NO", "TR", "CA", "SC", "MIX"):
                out.append(None)
            elif r == "NO" or not isinstance(s, (int, float)):
                out.append((r, None))
            else:
                out.append((r, int(s)))
            continue
        mm = _VERDICT_RE.match(str(v).strip().upper()) if v is not None else None
        out.append((mm.group(1), int(mm.group(2)) if mm.group(2) else None)
                   if mm else None)
    return out


def ask(model, system, user, timeout=900):
    """Один запрос к брокеру. 503 — карта занята более важной работой; брокер
    сам говорит, через сколько вернуться, и ждать надо именно столько.

    Родной /api/chat, а не OpenAI-совместимый: только через него задаются
    num_ctx и think. Через совместимый обе локальные модели на батче из девяти
    статей молча пересказывали их вместо ответа JSON — инструкция терялась.
    Облачного судью это не касается, там контекст другой; здесь правится
    стенд, а не промпт.
    """
    for _ in range(20):
        r = requests.post(OLLAMA, json={
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "format": "json",
            # Рассуждение выключено НАРОЧНО у обеих редакций: иначе внутренняя
            # цепочка подменит собой поле «что изменилось», ради которого v2 и
            # затевалась, и сравнение перестанет что-либо значить.
            "think": False,
            "options": {"temperature": 0.0, "num_ctx": 32768},
        }, timeout=timeout)
        if r.status_code == 503:
            try:
                wait = float(r.json().get("retry_after_s", 30))
            except Exception:  # noqa: BLE001
                wait = 30.0
            time.sleep(min(wait, 60))
            continue
        r.raise_for_status()
        c = r.json()["message"]["content"]
        return c.rsplit("</think>", 1)[-1].strip()
    raise RuntimeError("карта занята дольше, чем стоит ждать")


def judge(model, variant, texts, seed=0, workers=3):
    """Оценить все тексты батчами. seed != 0 перемешивает порядок внутри батча.

    Перемешивание — не украшение: позиция статьи в батче сама по себе двигает
    ответ, и без него самосогласованность мерила бы кэш, а не суждение.
    """
    system, tmpl, _k = prompts(variant)
    idx = list(range(len(texts)))
    if seed:
        rnd = random.Random(seed)
        rnd.shuffle(idx)
    chunks = [idx[i:i + BATCH] for i in range(0, len(idx), BATCH)]

    raws = []

    def one(chunk):
        block = "".join(
            "[%d] %s\n\n" % (j + 1, texts[k].replace("\n", " ")[:TEXT_CHARS])
            for j, k in enumerate(chunk))
        for attempt in range(3):
            try:
                raw = ask(model, system,
                          tmpl.format(n=len(chunk), articles=block))
                out = parse(raw, len(chunk), variant)
                # Сырой ответ нужен только там, где разбор что-то потерял:
                # «без решения: 5» без него — число, о котором нечего сказать.
                if any(v is None for v in out):
                    raws.append(raw[:1200])
                return out
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    print("  батч упал: %s" % exc, file=sys.stderr)
                    return [None] * len(chunk)
                time.sleep(2)

    out = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for chunk, res in zip(chunks, ex.map(one, chunks)):
            for k, v in zip(chunk, res):
                out[k] = v
    return out, raws


# ---------------------------------------------------------------------------
def _norm(s, k):
    """Разряд 1..k -> [0,1]. Обе шкалы сравнимы только после этого."""
    return None if s is None else (s - 1) / (k - 1)


def stats(verdicts, k, prod):
    """Распределение, доля верха и согласие с боевым судьёй."""
    sc = [v[1] for v in verdicts if v and v[1] is not None]
    res = {"decided": len(sc), "none": sum(1 for v in verdicts if v is None)}
    res["dist"] = {g: sc.count(g) for g in range(1, k + 1)}
    res["top_share"] = sc.count(k) / len(sc) if sc else float("nan")
    pairs = [(_norm(v[1], k), _norm(p, 3))
             for v, p in zip(verdicts, prod)
             if v and v[1] is not None and p is not None and p > 0]
    if len(pairs) > 2:
        a = np.array([x for x, _ in pairs])
        b = np.array([y for _, y in pairs])
        res["vs_prod_mae"] = float(np.abs(a - b).mean())
        res["vs_prod_spearman"] = (
            float(np.corrcoef(np.argsort(np.argsort(a)),
                              np.argsort(np.argsort(b)))[0, 1])
            if a.std() and b.std() else float("nan"))
    return res


def self_agree(v1, v2, k):
    """Согласие двух прогонов: доля совпавших разрядов и средний разрыв."""
    pairs = [(a[1], b[1]) for a, b in zip(v1, v2)
             if a and b and a[1] is not None and b[1] is not None]
    if not pairs:
        return float("nan"), float("nan")
    exact = sum(1 for a, b in pairs if a == b) / len(pairs)
    gap = float(np.mean([abs(_norm(a, k) - _norm(b, k)) for a, b in pairs]))
    return exact, gap


def sample(n, seed=7):
    """n статей корзины с оценкой боевого судьи, поровну из каждого разряда."""
    b = dict(np.load(os.path.join(BASE, "bench", "basket.npz"),
                     allow_pickle=True))
    rnd = random.Random(seed)
    picked = []
    for g in (1, 2, 3):
        pool = [i for i in range(len(b["scale"])) if b["scale"][i] == g
                and len(str(b["text"][i])) > 400]
        picked += rnd.sample(pool, min(n // 3, len(pool)))
    rnd.shuffle(picked)
    texts = ["%s\n%s" % (b["title"][i], b["text"][i]) for i in picked]
    return picked, texts, [int(b["scale"][i]) for i in picked]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--model", default="qwen3.5:9b")
    ap.add_argument("--variants", default="v1,v2,v2a")
    ap.add_argument("--out", default=os.path.join(BASE, "bench", "judge_runs.json"))
    a = ap.parse_args()

    idx, texts, prod = sample(a.n)
    print("статей: %d (боевые оценки 1/2/3: %d/%d/%d), модель %s"
          % (len(texts), prod.count(1), prod.count(2), prod.count(3), a.model))

    runs, report = {}, {}
    for variant in a.variants.split(","):
        k = prompts(variant)[2]
        t0 = time.time()
        r1, raw1 = judge(a.model, variant, texts, seed=0)
        r2, _raw2 = judge(a.model, variant, texts, seed=101)
        dt = time.time() - t0
        s = stats(r1, k, prod)
        s["self_exact"], s["self_gap"] = self_agree(r1, r2, k)
        s["seconds"] = round(dt)
        s["levels"] = k
        report[variant] = s
        runs[variant] = {"verdicts": [list(v) if v else None for v in r1],
                         "unparsed": raw1}
        print("\n=== %s (делений %d, %ds)" % (variant, k, dt))
        print("  распределение: %s" % s["dist"])
        print("  доля верхнего разряда: %.3f   без решения: %d"
              % (s["top_share"], s["none"]))
        print("  сам с собой: точно %.3f, средний разрыв %.3f"
              % (s["self_exact"], s["self_gap"]))
        if "vs_prod_mae" in s:
            print("  против боевого судьи: MAE %.3f, Spearman %+.3f"
                  % (s["vs_prod_mae"], s["vs_prod_spearman"]))

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"model": a.model, "urls": [str(i) for i in idx],
                   "prod": prod, "report": report, "runs": runs},
                  fh, ensure_ascii=False, indent=1)
    print("\n-> %s" % a.out)


def _selfcheck():
    v = parse('{"1": {"w":"law", "r":"TR", "s":5}, "2": {"w":"-","r":"NO"}}',
              2, "v2")
    assert v == [("TR", 5), ("NO", None)], v
    assert parse('{"1":"TR3","2":"NO"}', 2, "v1") == [("TR", 3), ("NO", None)]
    assert parse("мусор", 2, "v1") == [None, None]
    assert _norm(1, 5) == 0.0 and _norm(5, 5) == 1.0 and _norm(3, 3) == 1.0
    e, g = self_agree([("TR", 3), ("TR", 1)], [("TR", 3), ("TR", 2)], 3)
    assert abs(e - 0.5) < 1e-9 and abs(g - 0.25) < 1e-9, (e, g)
    print("bench_judge: ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
