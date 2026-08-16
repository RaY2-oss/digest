# -*- coding: utf-8 -*-
"""
test_quota.py — учёт остатков лимитов.

Главное, что здесь проверяется: локальная модель включается ТОЛЬКО когда
суточные лимиты выбраны у всех трёх облачных провайдеров. Частотный 429
(короткий Retry-After) не должен уводить вызовы на локальную — она слабее,
и уходить на неё, пока облако живо, значит терять качество дайджеста.
"""

import os
import sys
import tempfile
import time

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("GOOGLE_API_KEY", "test-key")

_tmp = tempfile.mkdtemp()
os.environ["QUOTA_STATE_PATH"] = os.path.join(_tmp, "quota_state.json")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quota  # noqa: E402

TAGS = ["OpenRouter", "Groq", "Google"]


class FakeResp:
    def __init__(self, status_code=200, headers=None):
        self.status_code = status_code
        self.headers = headers or {}


def _reset():
    quota._state = quota._blank()
    quota._or_limit_cached = quota.PAID_TIER_DAILY


def test_fresh_state_nothing_exhausted():
    _reset()
    assert not quota.exhausted("OpenRouter")
    assert not quota.all_cloud_exhausted(TAGS)


def test_openrouter_counts_its_own_requests():
    """OpenRouter не отдаёт остаток :free-запросов, считаем сами."""
    _reset()
    start = quota.remaining("OpenRouter")
    assert start == quota.PAID_TIER_DAILY
    quota.note_request("OpenRouter")
    quota.note_request("OpenRouter")
    assert quota.remaining("OpenRouter") == start - 2


def test_openrouter_exhausts_at_daily_cap():
    _reset()
    quota._or_limit_cached = 3
    for _ in range(3):
        quota.note_request("OpenRouter")
    assert quota.remaining("OpenRouter") == 0
    assert quota.exhausted("OpenRouter")


def test_groq_remaining_read_from_headers():
    """Groq — единственный, кто отдаёт остаток в каждом ответе."""
    _reset()
    quota.note_response("Groq", FakeResp(200, {"x-ratelimit-remaining-requests": "7"}))
    assert quota.remaining("Groq") == 7
    quota.note_response("Groq", FakeResp(200, {"x-ratelimit-remaining-requests": "0"}))
    assert quota.remaining("Groq") == 0
    assert quota.exhausted("Groq")


def test_short_retry_after_is_not_exhaustion():
    """Частотный 429 отпускает за секунды — это НЕ повод считать лимит выбранным."""
    _reset()
    quota.note_response("Google", FakeResp(429, {"Retry-After": "30"}))
    assert not quota.exhausted("Google"), "короткий Retry-After принят за суточный лимит"


def test_long_retry_after_is_exhaustion():
    _reset()
    quota.note_response("Google", FakeResp(429, {"Retry-After": "3600"}))
    assert quota.exhausted("Google")


def test_local_only_when_all_three_are_out():
    """Ради этого всё и написано."""
    _reset()
    quota._or_limit_cached = 1

    quota.note_response("Google", FakeResp(429, {"Retry-After": "3600"}))
    assert not quota.all_cloud_exhausted(TAGS), "ушли на локальную, когда жив OpenRouter"

    quota.note_response("Groq", FakeResp(429, {"x-ratelimit-remaining-requests": "0",
                                               "Retry-After": "3600"}))
    assert not quota.all_cloud_exhausted(TAGS), "ушли на локальную, когда жив OpenRouter"

    quota.note_request("OpenRouter")
    assert quota.all_cloud_exhausted(TAGS), "все три исчерпаны, а локальная не включилась"


def test_success_clears_block():
    _reset()
    quota.note_response("Groq", FakeResp(429, {"Retry-After": "3600"}))
    assert quota.exhausted("Groq")
    quota.note_success("Groq")
    assert not quota.exhausted("Groq")


def test_state_survives_process_restart():
    """cron стартует новый процесс каждые 6 часов — счётчик обязан пережить."""
    _reset()
    quota.note_request("OpenRouter")
    quota.note_request("OpenRouter")
    used_before = quota._prov("OpenRouter")["used"]

    quota._state = None  # имитируем новый процесс
    assert quota._prov("OpenRouter")["used"] == used_before


def test_state_resets_on_new_utc_day():
    _reset()
    quota.note_request("OpenRouter")
    quota._state["date"] = "2000-01-01"
    quota._save()
    quota._state = None
    assert quota._prov("OpenRouter")["used"] == 0


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERR   {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
