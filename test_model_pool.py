# -*- coding: utf-8 -*-
"""Проверка поведения LLM-стадии при недоступных моделях.

Захардкоженного фолбэк-списка нет, поэтому пул может остаться пустым.
Проверяем, что это деградация, а не падение — и, главное, что молчание
провайдеров НЕ превращается в отказ по статье.

Запуск: ./venv/bin/python test_model_pool.py
"""

import daily_collector as dc
import model_rotation as mr


class _Boom(Exception):
    pass


def test_pool_stays_empty_and_api_not_hammered():
    """Отказ /models -> пустой пул, и повторно API в этом процессе не дёргаем."""
    calls = []

    def fake_get(*a, **kw):
        calls.append(1)
        raise _Boom("openrouter down")

    real_get = mr.requests.get
    mr.requests.get = fake_get
    try:
        mr._free_models_pool = []
        mr._pool_refresh_failed = False
        mr._provider_dead_until = {}

        mr._init_models_pool()
        assert mr._free_models_pool == [], mr._free_models_pool
        assert len(calls) == 1, calls

        # Второй вызов не должен идти в сеть: иначе ~20 вызовов подряд
        # повисли бы по 45 с каждый.
        mr._init_models_pool()
        assert len(calls) == 1, calls
    finally:
        mr.requests.get = real_get
        mr._free_models_pool = []
        mr._pool_refresh_failed = False
        mr._provider_dead_until = {}
    print("model_rotation: пустой пул, /models дёрнут 1 раз — OK")


def test_silence_goes_to_pending_not_reject():
    """Молчание всех провайдеров -> None (pending), а НЕ отказ по статье.

    Регрессия на прогон week_swap 21.07: старый код в этом месте возвращал
    [False]*len(chunk) и ничего не логировал, из-за чего 6006 кандидатов из
    6006 стали "отклонёнными" просто потому, что кончились суточные лимиты.
    """
    real = dc._call_openrouter_raw
    dc._call_openrouter_raw = lambda *a, **kw: None
    try:
        got = dc.judge_parallel(["t1", "t2", "t3"])
        assert got == [None, None, None], got
    finally:
        dc._call_openrouter_raw = real
    print("judge_parallel: молчание -> pending, не отказ — OK")


def test_partial_answer_only_missing_goes_pending():
    """Модель ответила не на все номера: недостающие в pending, прочие приняты."""
    real = dc._call_openrouter_raw
    dc._call_openrouter_raw = lambda *a, **kw: '{"1": "CA", "3": "NO"}'
    try:
        got = dc.judge_parallel(["a", "b", "c"])
        assert got == ["CA", None, "NO"], got
    finally:
        dc._call_openrouter_raw = real
    print("judge_parallel: частичный ответ -> в pending только пропущенные — OK")


def test_object_format_cannot_shift_indices():
    """Ключи-номера не дают вердикту достаться чужой статье."""
    assert dc._parse_judge('{"3":"TR","1":"NO","2":"MIX"}', 3) == ["NO", "MIX", "TR"]
    assert dc._parse_judge("вообще не json", 2) is None
    assert dc._parse_judge('{"1":"BOGUS","2":"SC"}', 2) == [None, "SC"]
    # Позиционный массив раньше молча съезжал; теперь он просто не разбирается.
    assert dc._parse_judge('["NO","SC"]', 2) is None
    print("_parse_judge: формат-объект защищает от съезда индексов — OK")


def demo():
    test_pool_stays_empty_and_api_not_hammered()
    test_silence_goes_to_pending_not_reject()
    test_partial_answer_only_missing_goes_pending()
    test_object_format_cannot_shift_indices()


if __name__ == "__main__":
    demo()
    print("ok")
