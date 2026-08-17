# -*- coding: utf-8 -*-
"""
test_llm7_fallback.py — бесплатный шлюз стоит МЕЖДУ исчерпанным облаком и
локальной моделью, а не вместо облака.

Порядок здесь и есть вся суть правки: пока у облака остались суточные лимиты,
качество там выше, и уход на llm7 был бы потерей на ровном месте. Проверяем
обе стороны — что не зовётся раньше времени и что зовётся раньше локальной.

Сеть трогает только последняя проверка (живой ли шлюз): её можно пропустить,
она помечена в выводе отдельно.
"""

import os
import sys
import tempfile

os.environ["QUOTA_STATE_PATH"] = os.path.join(tempfile.mkdtemp(), "q.json")
sys.path.insert(0, "/opt/digest")

import model_rotation as mr  # noqa: E402
import quota  # noqa: E402


def _exhaust(tags):
    quota._state = quota._blank()
    quota._or_limit_cached = 1
    quota.note_request("OpenRouter")
    for t in tags:
        if t != "OpenRouter":
            quota.note_response(t, type("R", (), {
                "status_code": 429,
                "headers": {"Retry-After": "3600"},
            })())


def _trace(monkey_local="локальная", monkey_llm7="шлюз"):
    """Подменяет обе ступени на метки и возвращает (журнал вызовов, откат)."""
    calls = []
    orig_llm7, orig_local = mr._llm7_attempt, mr._local_attempt

    def fake_llm7(*a, **kw):
        calls.append("llm7")
        return monkey_llm7

    def fake_local(*a, **kw):
        calls.append("local")
        return monkey_local

    mr._llm7_attempt, mr._local_attempt = fake_llm7, fake_local

    def undo():
        mr._llm7_attempt, mr._local_attempt = orig_llm7, orig_local

    return calls, undo


def test_pool_is_configured():
    """Ступень включена и первой стоит модель, победившая в замере."""
    assert mr.LLM7_MODELS, "пул пуст — ступень выключена"
    assert mr.LLM7_MODELS[0] == "gemini-3.1-flash-lite", mr.LLM7_MODELS


def test_llm7_before_local_when_cloud_is_out():
    """Облако выбрано -> сначала шлюз, локальная не трогается."""
    tags = mr._ring_tags()
    assert tags, "не настроен ни один облачный ключ"
    _exhaust(tags)
    calls, undo = _trace()
    try:
        # Провайдеры отвечают None: кольцо проходит круг и упирается в ветку
        # «все облачные исчерпаны».
        orig = mr._provider_call
        mr._provider_call = lambda *a, **kw: None
        try:
            out = mr._call_openrouter_raw("s", "u", "probe")
        finally:
            mr._provider_call = orig
    finally:
        undo()
    assert out == "шлюз", out
    assert calls == ["llm7"], calls


def test_local_still_reached_if_gateway_silent():
    """Шлюз не ответил — локальная остаётся на месте."""
    tags = mr._ring_tags()
    _exhaust(tags)
    calls, undo = _trace(monkey_llm7=None)
    try:
        orig = mr._provider_call
        mr._provider_call = lambda *a, **kw: None
        try:
            out = mr._call_openrouter_raw("s", "u", "probe")
        finally:
            mr._provider_call = orig
    finally:
        undo()
    assert out == "локальная", out
    assert calls == ["llm7", "local"], calls


def test_live_gateway_answers():
    """СЕТЬ: анонимный тариф отвечает без ключа."""
    out = mr._llm7_attempt("Ответь ровно одним словом на русском, без пояснений.",
                           "Столица Франции?", "probe")
    assert out is not None, "шлюз не ответил ни одной моделью пула"
    assert "ариж" in out, "неожиданный ответ: %s" % out[:120]
    print("      ответ шлюза: %s" % out.strip()[:80])


def main():
    fails = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print("  ok  %s" % name)
        except AssertionError as e:
            fails += 1
            print("FAIL  %s: %s" % (name, e))
        except Exception as e:  # noqa: BLE001
            fails += 1
            print("ERR   %s: %s: %s" % (name, type(e).__name__, e))
    print("OK" if not fails else "%d провалов" % fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
