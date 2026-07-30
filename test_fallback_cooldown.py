# -*- coding: utf-8 -*-
"""Проверка ротации: исчерпанный OpenRouter НЕ выпадает из кольца — следующий
вызов снова идёт в его пул. Пропускаем только того, кто сам прислал Retry-After."""
import config, model_rotation as m


def _isolate(retry_after=None):
    """Один model в пуле, без сети на /models, фолбэки off. Возвращает счётчик
    запросов к OpenRouter."""
    calls = {"or": 0}

    class Resp:
        status_code = 429
        headers = {"Retry-After": str(retry_after)} if retry_after else {}

        def json(self):  # noqa: D401
            return {"error": "rate limit"}

    def fake_post(url, **kw):
        if "openrouter" in url:
            calls["or"] += 1
        return Resp()

    m._free_models_pool = ["dummy-model:free"]
    m._pool_refresh_failed = False
    m._batches_processed = 0
    m._provider_dead_until.clear()
    m._retry_after.clear()
    m._wait_spent = m.RING_WAIT_BUDGET  # без бюджета ожидания кольцо падает сразу
    m.requests.post = fake_post
    m.time.sleep = lambda *a, **k: None
    config.OPENROUTER_API_KEY = "x"
    config.GROQ_API_KEY = ""
    config.GOOGLE_API_KEY = ""
    config.PROXIES = None
    return calls


def test_429_without_retry_after_keeps_provider_in_rotation():
    calls = _isolate()

    # 1-й вызов: обходит пул (1 модель, 429) и ничего не блокирует
    assert m._call_openrouter_raw("s", "u", "url1") is None
    assert calls["or"] == 1, calls
    assert not m._in_cooldown("OpenRouter")

    # 2-й вызов: провайдер снова в ротации, пул дёргается заново
    assert m._call_openrouter_raw("s", "u", "url2") is None
    assert calls["or"] == 2, f"провайдер выпал из ротации без Retry-After: {calls}"


def test_retry_after_skips_the_pool():
    calls = _isolate(retry_after=600)

    assert m._call_openrouter_raw("s", "u", "url1") is None
    assert calls["or"] == 1, calls
    assert m._in_cooldown("OpenRouter"), "Retry-After обязан уводить провайдера в отсидку"

    # пока отсидка не вышла — пул пропущен, повторно не дёргаем
    assert m._call_openrouter_raw("s", "u", "url2") is None
    assert calls["or"] == 1, f"OpenRouter дёрнут во время Retry-After: {calls}"


def test_weak_blocklist():
    """Блок-лист режет слабые тиры, но не задевает сильные (регрессия: 'mini' в 'gemini')."""
    weak = ["gemini-flash-lite-latest", "llama-3.1-8b-instant", "gpt-4o-mini", "gemma-3-1b"]
    strong = ["gemini-pro-latest", "gemini-flash-latest", "llama-3.3-70b-versatile",
              "openai/gpt-oss-120b", "gemini-2.5-pro"]
    for w in weak:
        assert m._is_weak(w), f"должен блокироваться: {w}"
    for s in strong:
        assert not m._is_weak(s), f"НЕ должен блокироваться: {s}"


if __name__ == "__main__":
    test_429_without_retry_after_keeps_provider_in_rotation()
    print("OK: провал пула не выбивает OpenRouter из ротации")
    test_retry_after_skips_the_pool()
    print("OK: Retry-After пропускает исчерпанный OpenRouter")
    test_weak_blocklist()
    print("OK: блок-лист слабых моделей (и 'gemini' не ложно-срабатывает)")
