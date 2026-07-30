# -*- coding: utf-8 -*-
"""
test_provider_cooldown.py — самопроверка ротации провайдеров в model_rotation.

Ловит регрессию, из-за которой дайджест уезжал с 1-5 статьями из 20: частотный
429 выбивал живого провайдера из ротации кулдауном (сначала фиксированные 600с,
потом эскалация с 60с), а когда так выбивало всех трёх, _call_openrouter_raw
начинал возвращать None за микросекунды — и _process_slot сжигал резерв корзины.
Своего кулдауна теперь нет вообще: провал пула не блокирует провайдера, отсидка
бывает только по Retry-After, который провайдер прислал сам.

Запуск: venv/bin/python3 test_provider_cooldown.py
"""

import time

import model_rotation as mr


def _reset():
    mr._provider_dead_until.clear()
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


def test_pool_failure_keeps_provider_in_rotation():
    """Провал пула без Retry-After не блокирует провайдера ни на секунду."""
    _reset()
    _fake_providers({})
    for _ in range(4):                  # даже подряд идущие провалы не копятся
        mr._mark_dead("Groq")
        assert not mr._in_cooldown("Groq"), mr._provider_dead_until
    assert "Groq" not in mr._provider_dead_until


def test_retry_after_is_the_only_cooldown():
    _reset()
    _fake_providers({})
    mr._retry_after["Google"] = 240.0
    mr._mark_dead("Google")
    assert round(mr._provider_dead_until["Google"] - time.time()) == 240
    assert mr._in_cooldown("Google")

    mr._mark_alive("Google")
    assert not mr._in_cooldown("Google")
    mr._mark_dead("Google")             # подсказка израсходована — снова без отсидки
    assert not mr._in_cooldown("Google")


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


def test_ring_retries_provider_instead_of_skipping_it():
    """Ключевая регрессия: провалившийся провайдер пробуется на следующем круге."""
    _reset()
    # Все трое молчат на первом круге, Groq отвечает на втором.
    calls = _fake_providers({
        "OpenRouter": [None],
        "Groq": [None, "ok-from-groq"],
        "Google": [None],
    })
    mr.RING_RETRY_PAUSE = 0.01             # чтобы тест не спал полминуты
    started = time.time()
    out = mr._call_openrouter_raw("sys", "usr", ref_url="test")
    elapsed = time.time() - started
    assert out == "ok-from-groq", out
    assert calls["Groq"] == 2, f"Groq не пробовали снова: {calls}"
    assert elapsed >= 0.01, f"вызов пошёл на второй круг без паузы, {elapsed:.3f}с"
    assert not mr.providers_exhausted()


def test_ring_gives_up_when_wait_budget_is_spent():
    """Если провайдеры лежат по-настоящему — прогон не висит, а честно сдаётся."""
    _reset()
    _fake_providers({"OpenRouter": [None], "Groq": [None], "Google": [None]})
    mr._wait_spent = mr.RING_WAIT_BUDGET   # бюджет израсходован ранее в прогоне
    started = time.time()
    out = mr._call_openrouter_raw("sys", "usr", ref_url="test")
    assert out is None
    assert time.time() - started < 1, "исчерпанный бюджет должен падать сразу"
    assert mr.providers_exhausted(), "вызывающий код должен узнать, что LLM недоступна"


if __name__ == "__main__":
    _orig_pause = mr.RING_RETRY_PAUSE
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            mr.RING_RETRY_PAUSE = _orig_pause
            fn()
            print(f"ok  {name}")
    print("\nвсе проверки пройдены")
