# -*- coding: utf-8 -*-
"""
test_provider_cooldown.py — самопроверка circuit-breaker'а model_rotation.

Ловит регрессию, из-за которой дайджест уезжал с 1-5 статьями из 20: частотный
429 выбивал живого провайдера из ротации фиксированным кулдауном 600с (дольше
всего прогона), а когда так выбивало всех трёх, _call_openrouter_raw начинал
возвращать None за микросекунды — и _process_slot сжигал резерв корзины.

Запуск: venv/bin/python3 test_provider_cooldown.py
"""

import time

import model_rotation as mr


def _reset():
    mr._provider_dead_until.clear()
    mr._provider_strikes.clear()
    mr._retry_after.clear()
    mr._wait_spent = 0.0
    mr._rr_start = 0


def _fake_providers(script):
    """script: tag -> список ответов по вызовам ("текст" или None)."""
    calls = {tag: 0 for tag in mr._PROVIDERS}

    def _provider_call(tag, system_prompt, user_message, ref_url):
        i = calls[tag]
        calls[tag] += 1
        answers = script.get(tag, [])
        return answers[i] if i < len(answers) else answers[-1] if answers else None

    mr._provider_call = _provider_call
    mr._has_key = lambda tag: True
    return calls


def test_transient_429_does_not_blackout_the_run():
    """Первый провал пула = короткий кулдаун: частотный лимит успевает отпустить."""
    _reset()
    _fake_providers({})
    mr._mark_dead("Groq")
    left = mr._provider_dead_until["Groq"] - time.time()
    assert left <= mr.COOLDOWN_STEPS[0] + 1, left
    assert left < 600, f"первый 429 не должен выбивать провайдера на весь прогон: {left}с"


def test_cooldown_escalates_then_success_resets():
    """Подряд идущие провалы удлиняют кулдаун (суточный лимит), успех обнуляет."""
    _reset()
    _fake_providers({})
    steps = []
    for _ in range(4):
        mr._mark_dead("OpenRouter")
        steps.append(round(mr._provider_dead_until["OpenRouter"] - time.time()))
    assert steps[0] < steps[1] < steps[2], steps
    assert steps[2] == steps[3] == mr.COOLDOWN_STEPS[-1], steps

    mr._mark_alive("OpenRouter")
    assert not mr._in_cooldown("OpenRouter")
    mr._mark_dead("OpenRouter")
    assert round(mr._provider_dead_until["OpenRouter"] - time.time()) == mr.COOLDOWN_STEPS[0]


def test_retry_after_wins_when_longer_than_step():
    _reset()
    _fake_providers({})
    mr._retry_after["Google"] = 240.0
    mr._mark_dead("Google")
    assert round(mr._provider_dead_until["Google"] - time.time()) == 240


def test_retry_after_takes_the_earliest_hint_of_the_pool():
    """Лимиты у Groq/Google по-модельные: ждём оживления первой модели, не последней."""
    _reset()
    _fake_providers({})

    class Resp:
        def __init__(self, secs):
            self.headers = {"Retry-After": str(secs)}

    for secs in (839, 120, 600):        # обход пула: три 429 с разными подсказками
        mr._note_retry_after("Groq", Resp(secs))
    assert mr._retry_after["Groq"] == 120.0, mr._retry_after

    mr._mark_dead("Groq")
    assert round(mr._provider_dead_until["Groq"] - time.time()) == 120


def test_note_retry_after_survives_odd_responses():
    """Путь обработки ошибки не должен превращать 429 в исключение."""
    _reset()

    class NoHeaders:
        pass

    class HttpDate:
        headers = {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}

    mr._note_retry_after("Groq", NoHeaders())
    mr._note_retry_after("Groq", HttpDate())
    assert "Groq" not in mr._retry_after


def test_ring_waits_for_revival_instead_of_failing_instantly():
    """Ключевая регрессия: круг провалился -> ждём оживления, а не отдаём None."""
    _reset()
    # Все трое молчат на первом круге, Groq отвечает на втором.
    _fake_providers({
        "OpenRouter": [None],
        "Groq": [None, "ok-from-groq"],
        "Google": [None],
    })
    mr.COOLDOWN_STEPS = (1, 2, 3)          # чтобы тест не спал минуту
    started = time.time()
    out = mr._call_openrouter_raw("sys", "usr", ref_url="test")
    elapsed = time.time() - started
    assert out == "ok-from-groq", out
    assert elapsed >= 1, f"вызов не ждал оживления, вернулся за {elapsed:.3f}с"
    assert not mr.providers_exhausted()


def test_ring_gives_up_when_wait_budget_is_spent():
    """Если провайдеры лежат по-настоящему — прогон не висит, а честно сдаётся."""
    _reset()
    _fake_providers({"OpenRouter": [None], "Groq": [None], "Google": [None]})
    mr.COOLDOWN_STEPS = (1, 1, 1)
    mr._wait_spent = mr.RING_WAIT_BUDGET   # бюджет израсходован ранее в прогоне
    started = time.time()
    out = mr._call_openrouter_raw("sys", "usr", ref_url="test")
    assert out is None
    assert time.time() - started < 1, "исчерпанный бюджет должен падать сразу"
    assert mr.providers_exhausted(), "вызывающий код должен узнать, что LLM недоступна"


if __name__ == "__main__":
    _orig_steps = mr.COOLDOWN_STEPS
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            mr.COOLDOWN_STEPS = _orig_steps
            fn()
            print(f"ok  {name}")
    print("\nвсе проверки пройдены")
