# -*- coding: utf-8 -*-
"""Проверка поведения при недоступном списке моделей OpenRouter.

Захардкоженного фолбэк-списка нет, поэтому пул может остаться пустым.
Проверяем, что это деградация, а не падение.

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

        mr._init_models_pool()
        assert mr._free_models_pool == [], mr._free_models_pool
        assert len(calls) == 1, calls

        # Второй вызов не должен идти в сеть: иначе ~20 вызовов подряд повисли бы
        # по 45 с каждый.
        mr._init_models_pool()
        assert len(calls) == 1, calls

        # Пустой пул -> честный None, без исключения.
        assert mr._call_openrouter_raw("sys", "user") is None
        assert len(calls) == 1, calls
    finally:
        mr.requests.get = real_get
        mr._free_models_pool = []
        mr._pool_refresh_failed = False
    print("model_rotation: пустой пул -> None, /models дёрнут 1 раз — OK")


def test_empty_pool_does_not_crash_executor():
    """Пустой пул не должен доходить до ThreadPoolExecutor(max_workers=0)."""
    chunks = [["a", "b"], ["c"]]

    def call_fn(model, chunk):  # не должен вызываться
        raise AssertionError("call_fn не должен вызываться при пустом пуле")

    got = dc._run_batches_parallel(
        chunks, [], call_fn=call_fn, empty_result_fn=lambda c: [False] * len(c)
    )
    assert got == [[False, False], [False]], got

    got = dc._run_batches_parallel(
        chunks, [], call_fn=call_fn, empty_result_fn=lambda c: ["MIX"] * len(c)
    )
    assert got == [["MIX", "MIX"], ["MIX"]], got
    print("daily_collector: пустой пул -> дефолты без падения — OK")


def test_build_pool_returns_existing_on_failure():
    """При отказе /models возвращается existing (возможно пустой), не фолбэк."""

    def fake_get(*a, **kw):
        raise _Boom("openrouter down")

    real_get = dc.requests.get
    dc.requests.get = fake_get
    try:
        assert dc._build_pool([]) == []
        assert dc._build_pool(["x:free"], force=True) == ["x:free"]
    finally:
        dc.requests.get = real_get
    print("daily_collector._build_pool: отказ -> existing — OK")


def demo():
    test_pool_stays_empty_and_api_not_hammered()
    test_empty_pool_does_not_crash_executor()
    test_build_pool_returns_existing_on_failure()


if __name__ == "__main__":
    demo()
    print("ok")
