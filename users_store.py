# -*- coding: utf-8 -*-
"""
users_store.py — реестр Telegram chat_id, кому рассылать воскресный дайджест.

Пользователь регистрируется автоматически при первом сообщении боту
(см. telegram_bot_listener.handle_update). Хранилище — плоский JSON-файл
со списком chat_id; _lock защищает read-modify-write от гонки между
потоками бота (все вызовы дешёвые и редкие, поэтому без БД).
"""
import json
import os
import threading

USERS_PATH = "/opt/digest/users.json"
_lock = threading.Lock()


def _read() -> set:
    """Возвращает множество зарегистрированных chat_id. Пустое множество, если файла нет или он повреждён."""
    if not os.path.exists(USERS_PATH):
        return set()
    try:
        with open(USERS_PATH, "r", encoding="utf-8") as f:
            return set(json.load(f))
    except Exception:
        return set()


def register_user(chat_id):
    """Добавляет chat_id в реестр, если его там ещё нет. Идемпотентно."""
    with _lock:
        users = _read()
        if chat_id not in users:
            users.add(chat_id)
            with open(USERS_PATH, "w", encoding="utf-8") as f:
                json.dump(list(users), f)


def get_all_users():
    """Список всех chat_id для рассылки (используется в telegram_sender.send_document)."""
    with _lock:
        return list(_read())
