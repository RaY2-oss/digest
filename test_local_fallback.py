# -*- coding: utf-8 -*-
"""
test_local_fallback.py — сквозная проверка: доходит ли вызов до домашнего
сервера, когда облако кончилось, и НЕ доходит, пока не кончилось.

Отличается от test_quota.py тем, что бьёт по настоящей модели через туннель,
а не по заглушке: проверяется вся цепочка — quota -> кольцо -> _local_attempt
-> AmneziaWG -> Ollama.
"""

import os
import sys
import tempfile

os.environ["QUOTA_STATE_PATH"] = os.path.join(tempfile.mkdtemp(), "q.json")
sys.path.insert(0, "/opt/digest")

import config  # noqa: E402
import model_rotation as mr  # noqa: E402
import quota  # noqa: E402

SYS = "Ответь ровно одним словом на русском языке, без пояснений."
USR = "Столица Франции?"


def test_local_endpoint_alive():
    """Домашний сервер отвечает через туннель."""
    out = mr._local_attempt(SYS, USR, "probe")
    assert out is not None, "локальная модель недоступна с VPS"
    assert "ариж" in out, f"неожиданный ответ: {out[:120]}"
    print(f"      ответ локальной модели: {out.strip()[:80]}")


def test_cloud_alive_keeps_local_untouched():
    """Пока облако живо, локальная не зовётся."""
    quota._state = quota._blank()
    tags = mr._ring_tags()
    assert tags, "не настроен ни один облачный ключ"
    assert not quota.all_cloud_exhausted(tags), \
        "локальная включилась бы при живом облаке"


def test_all_exhausted_opens_local():
    """Когда все три выбраны — путь на локальную открыт."""
    quota._state = quota._blank()
    quota._or_limit_cached = 1
    tags = mr._ring_tags()
    quota.note_request("OpenRouter")
    for t in tags:
        if t != "OpenRouter":
            quota.note_response(t, type("R", (), {
                "status_code": 429,
                "headers": {"Retry-After": "3600"},
            })())
    assert quota.all_cloud_exhausted(tags), quota.snapshot(tags)


def main():
    fails = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"  ok  {name}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"ERR   {name}: {type(e).__name__}: {e}")
    print("OK" if not fails else f"{fails} провалов")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
