# -*- coding: utf-8 -*-
"""Проверка circuit-breaker: исчерпанный OpenRouter не дёргается повторно
в течение кулдауна — следующие вызовы идут сразу к фолбэкам."""
import config, model_rotation as m


def test_openrouter_cooldown_skips_pool():
    calls = {"or": 0}

    class Resp:
        status_code = 429
        def json(self):  # noqa: D401
            return {"error": "rate limit"}

    def fake_post(url, **kw):
        if "openrouter" in url:
            calls["or"] += 1
        return Resp()

    # изоляция: один model в пуле, без сети на /models, паузы мгновенны, фолбэки off
    m._free_models_pool = ["dummy-model:free"]
    m._pool_refresh_failed = False
    m._batches_processed = 0
    m._provider_dead_until.clear()
    m._provider_strikes.clear()
    m._wait_spent = m.RING_WAIT_BUDGET  # без бюджета ожидания кольцо падает сразу
    m.requests.post = fake_post
    m.time.sleep = lambda *a, **k: None
    config.OPENROUTER_API_KEY = "x"
    config.GROQ_API_KEY = ""
    config.GOOGLE_API_KEY = ""
    config.PROXIES = None

    # 1-й вызов: обходит пул (1 модель, 429), помечает OpenRouter мёртвым
    assert m._call_openrouter_raw("s", "u", "url1") is None
    assert calls["or"] == 1, calls
    assert m._in_cooldown("OpenRouter")

    # 2-й вызов: кулдаун -> пул пропущен, OpenRouter повторно НЕ дёргается
    assert m._call_openrouter_raw("s", "u", "url2") is None
    assert calls["or"] == 1, f"OpenRouter дёрнут повторно в кулдауне: {calls}"


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
    test_openrouter_cooldown_skips_pool()
    print("OK: circuit-breaker пропускает исчерпанный OpenRouter")
    test_weak_blocklist()
    print("OK: блок-лист слабых моделей (и 'gemini' не ложно-срабатывает)")
