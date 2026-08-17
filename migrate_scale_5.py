# -*- coding: utf-8 -*-
"""Разовая миграция шкалы масштаба 1..3 -> 1..5 (17.08.2026).

С этого дня судья ставит пять делений вместо трёх (daily_collector.
_JUDGE_SYSTEM). Строки, оценённые до перехода, надо переписать: иначе цифра 3
в одной строке означает «верх старой шкалы», а в соседней — «середину новой»,
и по строке не отличить. Соответствие взято по якорям рубрики:

    3 (страна другая после события) -> 5
    2 (изменилось одно заведение)   -> 3
    1 (не изменилось ничего)        -> 1

Разряды 2 и 4 новой шкалы старым строкам не достаются — их в трёхбалльной
шкале не существовало, придумывать за судью нечего.

Запускается один раз, сама себя стопорит: если в базе уже есть 4 или 5,
переход состоялся и повторный прогон только испортил бы данные. Файл живёт
до следующей уборки статей (7 суток), после неё в базе не остаётся ни одной
строки старой шкалы и скрипт можно удалить.

Тем же преобразованием переписывается фикстура стенда bench/basket.npz: она
заморожена на неделе 09–16.08.2026 и оценена старой шкалой, а нормирует
оценки тот же _SCALE_VALUE. Числа базовой линии от этого не двигаются —
0.0/0.5/1.0 остаются 0.0/0.5/1.0, меняется только цифра под ними.
"""
import os
import sqlite3
import sys

import numpy as np

import config

BASKET = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "bench", "basket.npz")


def fixture():
    if not os.path.exists(BASKET):
        return
    b = dict(np.load(BASKET, allow_pickle=True))   # своя же фикстура, см. bench.py
    if b["scale"].max() > 3:
        print("фикстура стенда уже на пятибалльной шкале")
        return
    b["scale"] = np.array([{2: 3, 3: 5}.get(int(s), int(s)) for s in b["scale"]],
                          dtype=np.int8)
    np.savez_compressed(BASKET, **b)
    print("фикстура стенда переписана: %s" % BASKET)


def main():
    fixture()
    conn = sqlite3.connect(config.DB_PATH)
    top = conn.execute("SELECT MAX(scale) FROM articles").fetchone()[0]
    if top is None:
        print("оценённых строк нет")
        return
    if top > 3:
        print("в базе уже пятибалльная шкала, миграция не нужна")
        return
    n = conn.execute(
        "UPDATE articles SET scale = CASE scale WHEN 3 THEN 5 WHEN 2 THEN 3 "
        "ELSE scale END WHERE scale IS NOT NULL").rowcount
    conn.commit()
    dist = conn.execute("SELECT scale, COUNT(*) FROM articles "
                        "WHERE scale IS NOT NULL GROUP BY 1").fetchall()
    print("переписано строк: %d, распределение %s" % (n, dist))


if __name__ == "__main__":
    sys.exit(main())
