# -*- coding: utf-8 -*-
"""Проверка, что секреты из config.py не доезжают до хендлеров логирования.

Запуск: python3 /opt/digest/test_secret_redaction.py
Падает, если фабрика записей перестала чистить токен — а именно так он и
утекал в bot_listener.log через строку исключения requests.
"""
import io
import logging
import os
import sys

FAKE_TOKEN = "123456789:AAFakeTokenForTestOnly_0123456789abc"
os.environ["TELEGRAM_BOT_TOKEN"] = FAKE_TOKEN
os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-fake-key-for-test"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402  (импорт после подмены окружения — он ставит фабрику)

assert config.TELEGRAM_BOT_TOKEN == FAKE_TOKEN, "тест не подхватил токен из окружения"

stream = io.StringIO()
handler = logging.StreamHandler(stream)
log = logging.getLogger("redaction_test")
log.addHandler(handler)
log.setLevel(logging.INFO)
log.propagate = False

# Тот самый путь утечки: %s-подстановка строки исключения с полным URL.
exc = Exception(
    "HTTPSConnectionPool(host='api.telegram.org', port=443): Max retries exceeded "
    f"with url: /bot{FAKE_TOKEN}/getUpdates?timeout=30"
)
log.error("Ошибка polling: %s", exc)
log.error("ключ в аргументе: %s", config.OPENROUTER_API_KEY)
log.info("обычное сообщение без секретов")

out = stream.getvalue()
assert FAKE_TOKEN not in out, f"токен утёк в лог: {out}"
assert config.OPENROUTER_API_KEY not in out, f"ключ OpenRouter утёк в лог: {out}"
assert "***REDACTED***" in out, f"редакция не сработала: {out}"
assert "обычное сообщение без секретов" in out, "фабрика сломала обычные записи"
assert "api.telegram.org" in out, "фабрика съела полезную часть сообщения"

print("OK: секреты вычищаются из записей логов")
