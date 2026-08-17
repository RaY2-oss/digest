# -*- coding: utf-8 -*-
"""Разбор ответа судьи и раскладка перемешанного батча обратно.

Обе вещи ломаются молча и одинаково дорого: вердикт достаётся чужой статье.
Формат ответа менялся дважды (строка "TR3" -> объект {"w","r","s"}), а с
17.08.2026 батч ещё и уезжает к судье в перемешанном порядке.

    python test_judge_parse.py
"""
import daily_collector as D


def test_parse():
    cases = [
        ('{"1": {"w": "law", "r": "TR", "s": 5}}', ("TR", 5)),
        ('{"1": {"w": "-", "r": "NO"}}', ("NO", None)),
        ('{"1": {"r": "ca", "s": "3"}}', ("CA", 3)),      # цифра строкой
        ('{"1": {"r": "SC", "s": 4.0}}', ("SC", 4)),      # и дробью
        ('{"1": {"r": "SC", "s": 9}}', ("SC", 5)),        # вне шкалы -> край
        ('{"1": {"r": "TR", "s": "нет"}}', ("TR", None)),
        ('{"1": {"r": "zz", "s": 2}}', None),             # региона нет
        ('{"1": "TR3"}', ("TR", 3)),                      # форма до 17.08.2026
        ("не json вовсе", None),
    ]
    for raw, want in cases:
        got = D._parse_judge(raw, 1)
        got = got[0] if got else None
        assert got == want, (raw, got, want)


def test_shuffle_roundtrip():
    """Судья отвечает по позиции в батче — вердикт обязан вернуться к своей
    статье, а не к той, что стояла на этом месте в промпте."""
    texts = ["статья %d" % i for i in range(10)]
    seen = {}

    def fake_call(system, user, ref_url=None):
        # Достаём из промпта, какая статья оказалась на какой позиции, и
        # отвечаем масштабом, равным её собственному номеру + 1.
        out = {}
        for line in user.splitlines():
            if line.startswith("["):
                pos, text = line.split("]", 1)
                num = int(text.strip().split()[-1])
                out[pos[1:]] = {"w": "-", "r": "TR", "s": num % 5 + 1}
                seen[num] = int(pos[1:])
        return __import__("json").dumps(out)

    orig, D._call_openrouter_raw = D._call_openrouter_raw, fake_call
    try:
        got = D._judge_call(texts)
    finally:
        D._call_openrouter_raw = orig
    assert len(seen) == len(texts), seen
    for i, v in enumerate(got):
        assert v == ("TR", i % 5 + 1), (i, v)


if __name__ == "__main__":
    test_parse()
    test_shuffle_roundtrip()
    print("judge parse + shuffle: ok")
