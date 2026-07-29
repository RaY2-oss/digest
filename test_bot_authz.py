# -*- coding: utf-8 -*-
"""Проверка, что /rundigest и /resetlock доступны только ALLOWED_TG_USER_IDS.

Запуск: python3 /opt/digest/test_bot_authz.py
Падает, если чужой user_id снова сможет поднять sunday_processor_mmr.py или
снять лок — то есть если проверка отправителя опять потеряется.
"""
import os
import sys

os.environ["TELEGRAM_BOT_TOKEN"] = "123456789:AAFakeTokenForTestOnly_0123456789abc"
os.environ["ALLOWED_TG_USER_IDS"] = "111"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import telegram_bot_listener as bot  # noqa: E402

OWNER, STRANGER = 111, 999
assert config.ALLOWED_TG_USER_IDS == [OWNER]

started, unlocked, sent = [], [], []
bot.run_processor = lambda chat_id: started.append(chat_id)
bot.send_message = lambda chat_id, text: sent.append(text)
bot.users_store.register_user = lambda chat_id: None
bot.os.path.exists = lambda path: unlocked.append(path) or True
bot.os.remove = lambda path: unlocked.append(path)
# Поток бы стартовал асинхронно — зовём цель синхронно, чтобы проверить факт вызова.
bot.threading.Thread = lambda target, args, daemon: type(
    "T", (), {"start": lambda self: target(*args)}
)()


def update(user_id, text):
    return {"message": {"text": text, "chat": {"id": 42}, "from": {"id": user_id}}}


bot.handle_update(update(STRANGER, "/rundigest"))
assert started == [], f"чужой user_id запустил обработку: {started}"
assert any("владельцу" in t for t in sent), f"нет отказа чужому: {sent}"

unlocked.clear()
bot.handle_update(update(STRANGER, "/resetlock"))
assert unlocked == [], f"чужой user_id снял лок: {unlocked}"

# Отсутствующий from.id (например, channel_post) тоже не владелец.
bot.handle_update({"message": {"text": "/rundigest", "chat": {"id": 42}}})
assert started == [], f"апдейт без from.id прошёл авторизацию: {started}"

bot.handle_update(update(OWNER, "/rundigest"))
assert started == [42], f"владелец не смог запустить обработку: {started}"

# Пустой allowlist -> прежнее поведение (не запираем владельца при пустом .env).
config.ALLOWED_TG_USER_IDS = []
assert bot.may_run_commands(STRANGER) is True

print("OK: команды бота закрыты для чужих user_id")
