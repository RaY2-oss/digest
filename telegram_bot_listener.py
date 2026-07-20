# -*- coding: utf-8 -*-
"""
telegram_bot_listener.py — long-polling бот для ручного запуска
sunday_processor_mmr.py по команде /rundigest.

Запускается как systemd-сервис (постоянно активен), слушает getUpdates,
запускает обработку в отдельном потоке, чтобы не блокировать polling.
"""

import logging
import os
import subprocess
import threading
import time

import requests

import config
import users_store

# Файловый (не in-memory) лок: sunday_processor_mmr.py запускается как
# отдельный subprocess, а не в потоке этого сервиса, поэтому обычный
# threading.Lock его бы не увидел.
LOCK_PATH = "/tmp/sunday_processor.lock"
POLL_TIMEOUT = 30  # long-polling таймаут для getUpdates, сек

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(config.LOG_DIR, "bot_listener.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("bot_listener")

API_URL = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}"


def send_message(chat_id, text):
    try:
        requests.post(
            f"{API_URL}/sendMessage",
            data={"chat_id": chat_id, "text": text},
            proxies=config.PROXIES,
            timeout=15,
        )
    except Exception as exc:
        log.error("Не удалось отправить сообщение: %s", exc)


def is_running():
    return os.path.exists(LOCK_PATH)


VENV_PYTHON = os.path.join(os.path.dirname(__file__), "venv", "bin", "python3")


def run_processor(chat_id):
    if is_running():
        send_message(chat_id, "\u23f3 Обработка уже запущена, дождитесь завершения.")
        return

    open(LOCK_PATH, "w").close()
    send_message(chat_id, "\U0001f680 Запускаю формирование дайджеста...")
    log.info("Ручной запуск sunday_processor_mmr.py инициирован из чата %s", chat_id)

    env = os.environ.copy()
    env["TARGET_CHAT_ID"] = str(chat_id)

    try:
        result = subprocess.run(
            [VENV_PYTHON, os.path.join(os.path.dirname(__file__), "sunday_processor_mmr.py")],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode == 0:
            send_message(chat_id, "\u2705 Дайджест сформирован и отправлен.")
        else:
            log.error("sunday_processor_mmr завершился с ошибкой: %s", result.stderr[-2000:])
            send_message(chat_id, "\u274c Ошибка при формировании дайджеста. Проверьте logs/processor.log.")
    except Exception as exc:
        log.error("Исключение при запуске: %s", exc)
        send_message(chat_id, f"\u274c Непредвиденная ошибка: {exc}")
    finally:
        if os.path.exists(LOCK_PATH):
            os.remove(LOCK_PATH)


def handle_update(update):
    """Регистрирует отправителя и обрабатывает одну команду из апдейта.
    /rundigest  — запускает обработку в фоновом потоке (не блокирует polling).
    /resetlock  — снимает LOCK_PATH вручную (на случай, если предыдущий
                  запуск завис и не снял лок сам).
    Обе команды доступны всем пользователям бота."""
    message = update.get("message", {})
    text = message.get("text", "")
    chat_id = message.get("chat", {}).get("id")

    if chat_id is not None:
        users_store.register_user(chat_id)

    if not text:
        return

    if text.startswith("/rundigest"):
        threading.Thread(target=run_processor, args=(chat_id,), daemon=True).start()

    elif text.startswith("/resetlock"):
        if os.path.exists(LOCK_PATH):
            os.remove(LOCK_PATH)
            send_message(chat_id, "\U0001f513 Блокировка снята.")
        else:
            send_message(chat_id, "Блокировки не найдено.")


def main():
    """Бесконечный long-polling цикл getUpdates. offset = update_id последнего
    обработанного апдейта + 1 — так Telegram не присылает его повторно."""
    log.info("=== Бот-листенер запущен ===")
    offset = None
    while True:
        try:
            params = {"timeout": POLL_TIMEOUT}
            if offset:
                params["offset"] = offset
            resp = requests.get(
                f"{API_URL}/getUpdates",
                params=params,
                proxies=config.PROXIES,
                timeout=POLL_TIMEOUT + 10,
            )
            resp.raise_for_status()
            updates = resp.json().get("result", [])
            for upd in updates:
                offset = upd["update_id"] + 1
                handle_update(upd)
        except Exception as exc:
            log.error("Ошибка polling: %s", exc)
            time.sleep(5)


if __name__ == "__main__":
    main()
