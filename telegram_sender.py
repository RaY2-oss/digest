# -*- coding: utf-8 -*-
"""
telegram_sender.py — отправка готового .docx в Telegram методом sendDocument.

Запрос идёт через прокси Xray (PROXIES). Ошибки логируются в
/opt/digest/logs/errors.log; функция возвращает True/False и не бросает
исключений наружу (чтобы sunday_processor не падал из-за Telegram).
"""

import logging
import os
import time

import requests

import config

import users_store


def _get_logger():
    os.makedirs(config.LOG_DIR, exist_ok=True)
    logger = logging.getLogger("telegram_sender")
    if not logger.handlers:
        handler = logging.FileHandler(
            os.path.join(config.LOG_DIR, "errors.log"), encoding="utf-8"
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def send_document(file_path, chat_id=None):
    """
    chat_id=None -> broadcast всем зарегистрированным пользователям (воскресный cron).
    chat_id=<int> -> отправка только указанному получателю (ручной /rundigest).
    Возвращает True, если хотя бы одна отправка успешна.
    """
    log = _get_logger()

    if not config.TELEGRAM_BOT_TOKEN:
        log.error("Не задан TELEGRAM_BOT_TOKEN в config.py")
        return False

    if not os.path.exists(file_path):
        log.error("Файл для отправки не найден: %s", file_path)
        return False

    targets = [chat_id] if chat_id is not None else users_store.get_all_users()
    if not targets:
        log.error("Список получателей пуст")
        return False

    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendDocument"
    caption = f"\U0001F4C4 Дайджест ИВ РАН — {time.strftime('%d.%m.%Y')}"
    ok_any = False

    for target in targets:
        try:
            with open(file_path, "rb") as f:
                files = {"document": (os.path.basename(file_path), f)}
                data = {"chat_id": str(target), "caption": caption}
                resp = requests.post(
                    url, data=data, files=files,
                    proxies=config.PROXIES, timeout=60,
                )
            resp.raise_for_status()
            body = resp.json()
            if body.get("ok"):
                ok_any = True
            else:
                log.error("Telegram вернул ошибку для chat_id=%s: %s", target, body)
        except Exception as exc:
            log.error("Ошибка отправки в Telegram (target=%s, %s): %s", target, file_path, exc)

    return ok_any
