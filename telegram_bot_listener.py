# -*- coding: utf-8 -*-
"""
telegram_bot_listener.py — long-polling бот: ручной запуск дайджеста и выдача
уже готовых.

Запускается как systemd-сервис (постоянно активен), слушает getUpdates,
запускает обработку в отдельном потоке, чтобы не блокировать polling.

Команды:
    /rundigest  — собрать дайджест сейчас; ход прогона показывается ПРАВКОЙ
                  одного и того же сообщения, а не лентой новых
    /digests    — последние готовые дайджесты кнопками, каждый со временем МСК
    /last       — самый свежий готовый дайджест
    /status     — на каком этапе прогон и когда собран последний дайджест
    /resetlock  — снять зависшую блокировку
    /help       — то же меню

Время везде московское: сервер живёт по UTC, а читатель — нет, и «дайджест от
09.08» без часа не отличить от вчерашнего, если прогонов за день было два.
"""

import json
import logging
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

import config
import users_store

# Файловый (не in-memory) лок: sunday_processor_mmr.py запускается как
# отдельный subprocess, а не в потоке этого сервиса, поэтому обычный
# threading.Lock его бы не увидел.
LOCK_PATH = "/tmp/sunday_processor.lock"
POLL_TIMEOUT = 30  # long-polling таймаут для getUpdates, сек

MSK = ZoneInfo("Europe/Moscow")
ARCHIVE_MAX = 8       # столько прошлых дайджестов показываем кнопками
EDIT_EVERY = 3.0      # с: Telegram режет частые правки одного сообщения
# Клеток в полосе — по одной на сюжет. Фиксированная ширина читалась как
# второй счётчик: на «сюжет 3 из 20» полоса из двенадцати клеток выглядит как
# «из 12». Потолок — чтобы строка не переносилась на телефоне.
BAR_MAX = 20

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


def _api(method, data=None, files=None, timeout=15):
    """Вызов Bot API. Возвращает result или None — наружу не бросает ничего:
    бот, падающий из-за отказа Telegram, перестаёт отвечать и на всё остальное."""
    try:
        resp = requests.post(f"{API_URL}/{method}", data=data, files=files,
                             proxies=config.PROXIES, timeout=timeout)
        body = resp.json()
        if not body.get("ok"):
            log.error("%s -> %s", method, body.get("description"))
            return None
        return body.get("result")
    except Exception as exc:
        log.error("%s не удался: %s", method, exc)
        return None


def send_message(chat_id, text, keyboard=None):
    """Возвращает message_id — по нему сообщение потом правится."""
    data = {"chat_id": chat_id, "text": text}
    if keyboard:
        data["reply_markup"] = json.dumps({"inline_keyboard": keyboard})
    res = _api("sendMessage", data)
    return (res or {}).get("message_id")


def edit_message(chat_id, message_id, text, keyboard=None):
    if message_id is None:
        return
    data = {"chat_id": chat_id, "message_id": message_id, "text": text}
    if keyboard:
        data["reply_markup"] = json.dumps({"inline_keyboard": keyboard})
    _api("editMessageText", data)


def send_file(chat_id, path, caption):
    with open(path, "rb") as f:
        return _api("sendDocument",
                    data={"chat_id": chat_id, "caption": caption},
                    files={"document": (os.path.basename(path), f)},
                    timeout=60)


def is_running():
    return os.path.exists(LOCK_PATH)


def _lock_write(started, stage):
    """Лок заодно несёт состояние прогона.

    /status обрабатывается в потоке polling, а прогон идёт в своём — общей
    памяти между ними нет по той же причине, по какой сам лок файловый.
    """
    try:
        with open(LOCK_PATH, "w", encoding="utf-8") as f:
            f.write("%d\n%s" % (started, stage))
    except OSError:
        pass


def _lock_read():
    """(время старта, этап) или (None, None) — для пустого лока с прошлых версий."""
    try:
        with open(LOCK_PATH, encoding="utf-8") as f:
            started, _, stage = f.read().partition("\n")
        return float(started), stage
    except (OSError, ValueError):
        return None, None


VENV_PYTHON = os.path.join(os.path.dirname(__file__), "venv", "bin", "python3")


# ---------------------------------------------------------------------------
# Готовые дайджесты
# ---------------------------------------------------------------------------
def msk(ts) -> str:
    """UNIX-время -> «09.08.2026 11:22 МСК»."""
    return datetime.fromtimestamp(ts, MSK).strftime("%d.%m.%Y %H:%M МСК")


def archive(limit=ARCHIVE_MAX):
    """Готовые .docx, свежие сверху: [(имя файла, время сборки), ...].

    Время берётся у файла, а не из имени: имя несёт только дату, а прогонов за
    день бывает несколько (ручной /rundigest поверх воскресного cron), и по
    одной дате их не различить.
    """
    try:
        names = [n for n in os.listdir(config.OUTPUT_DIR) if n.endswith(".docx")]
    except OSError:
        return []
    out = [(n, os.path.getmtime(os.path.join(config.OUTPUT_DIR, n))) for n in names]
    out.sort(key=lambda x: -x[1])
    return out[:limit]


def _archive_path(name):
    """Имя из кнопки -> путь к файлу, если такой дайджест действительно есть.

    Команды бота открыты всем (см. handle_update), а callback_data приходит от
    клиента и может быть любой строкой, поэтому имя не склеивается с путём, а
    ищется среди реально готовых файлов — иначе «../../etc/passwd» отдавался бы
    как дайджест.
    """
    if name in {n for n, _ in archive(limit=10 ** 6)}:
        return os.path.join(config.OUTPUT_DIR, name)
    return None


def _archive_keyboard():
    return [[{"text": "📄 %s" % msk(ts), "callback_data": "d:%s" % name}]
            for name, ts in archive()]


def send_archive_list(chat_id):
    items = archive()
    if not items:
        send_message(chat_id, "Готовых дайджестов пока нет.")
        return
    send_message(chat_id,
                 "📚 Последние дайджесты — выберите, какой прислать:",
                 _archive_keyboard())


def send_archive_item(chat_id, name):
    path = _archive_path(name)
    if not path:
        send_message(chat_id, "Такого дайджеста уже нет.")
        return
    send_file(chat_id, path, "📄 Дайджест ИВ РАН — %s" % msk(os.path.getmtime(path)))


# ---------------------------------------------------------------------------
# Прогон с показом хода
# ---------------------------------------------------------------------------
_STAGE = re.compile(r"^DIGEST_STAGE=(.+)$")
_ARTICLES = re.compile(r"^DIGEST_ARTICLES=(\d+)/(\d+)$", re.M)
_OF = re.compile(r"(\d+) из (\d+)")


def _articles_note(stdout):
    """Дописывает к "✅ отправлен" реальное число статей. Недобор до плана виден
    по самому числу — предупреждения с диагнозом "провайдеры исчерпаны" тут нет
    намеренно: причина бывает и другой, а лог всё равно смотрят по факту."""
    m = _ARTICLES.search(stdout or "")
    if not m:
        return ""
    return f" Статей: {m.group(1)}."


def _bar(done, total):
    width = min(BAR_MAX, max(1, total))
    filled = max(0, min(width, round(width * done / max(1, total))))
    return "▰" * filled + "▱" * (width - filled)


def _elapsed(started):
    mins = int((time.time() - started) // 60)
    return "прошло %d мин" % mins if mins else "только начали"


def progress_text(stage, started, counts=None):
    """Текст сообщения о ходе прогона.

    Полоса появляется на первом же «N из M» и дальше не убирается: пересказ
    занимает почти всё время прогона, и полоса, исчезнувшая на следующем
    этапе, читается как сброс, а не как «эта часть позади».
    """
    lines = ["⚙️ Собираю дайджест", "", "▸ %s" % stage]
    if counts:
        lines.append("%s %d/%d" % (_bar(*counts), counts[0], counts[1]))
    lines.append("")
    lines.append(_elapsed(started))
    return "\n".join(lines)


class Progress(threading.Thread):
    """Правит одно сообщение о ходе прогона, пока идёт subprocess.

    Отдельным потоком, а не правкой по приходу этапа: между сюжетами проходят
    минуты, и throttle «не чаще EDIT_EVERY» ронял как раз последнюю правку
    перед такой паузой — сообщение молча показывало предыдущий сюжет и стоящий
    счётчик минут. Здесь наоборот: поток шлёт всё, что изменилось, и ничего,
    что не изменилось, так что минуты тикают сами, а лишних правок нет.
    """

    def __init__(self, chat_id, msg_id, started):
        super().__init__(daemon=True)
        self.chat_id, self.msg_id, self.started = chat_id, msg_id, started
        self.stage = "запускаюсь"
        self.counts = None
        self._sent = None
        # Не _stop: так называется внутренний метод threading.Thread, и join()
        # падает на нём с «'Event' object is not callable».
        self._over = threading.Event()

    def set_stage(self, stage):
        """Вызывается из потока, читающего stdout прогона."""
        self.stage = stage
        m = _OF.search(stage)
        if m:
            self.counts = (int(m.group(1)), int(m.group(2)))
        _lock_write(self.started, stage)

    def run(self):
        while not self._over.wait(EDIT_EVERY):
            text = progress_text(self.stage, self.started, self.counts)
            if text != self._sent:
                self._sent = text
                edit_message(self.chat_id, self.msg_id, text)

    def finish(self, text):
        self._over.set()
        self.join(timeout=EDIT_EVERY + 5)
        edit_message(self.chat_id, self.msg_id, text)


def run_processor(chat_id):
    if is_running():
        send_message(chat_id, "⏳ Прогон уже идёт — дождитесь его или снимите "
                              "блокировку командой /resetlock.")
        return

    started = time.time()
    _lock_write(started, "запускаюсь")
    msg_id = send_message(chat_id, progress_text("запускаюсь", started))
    log.info("Ручной запуск sunday_processor_mmr.py инициирован из чата %s", chat_id)

    env = os.environ.copy()
    env["TARGET_CHAT_ID"] = str(chat_id)

    prog = Progress(chat_id, msg_id, started)
    prog.start()
    try:
        # stderr сведён в stdout намеренно: труба на 64 КБ, и никем не читаемый
        # stderr рано или поздно упрётся в неё и подвесит прогон целиком.
        proc = subprocess.Popen(
            [VENV_PYTHON, os.path.join(os.path.dirname(__file__), "sunday_processor_mmr.py")],
            cwd=os.path.dirname(__file__),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=env,
        )
        out = []
        for line in proc.stdout:
            out.append(line)
            m = _STAGE.match(line.strip())
            if m:
                prog.set_stage(m.group(1))
        code = proc.wait()
        stdout = "".join(out)

        mins = max(1, int((time.time() - started) // 60))
        if code == 0:
            prog.finish("✅ Дайджест собран и отправлен.%s Заняло %d мин.\n"
                        "Файл — /last, прошлые — /digests."
                        % (_articles_note(stdout), mins))
        else:
            log.error("sunday_processor_mmr завершился с ошибкой: %s", stdout[-2000:])
            prog.finish("❌ Прогон оборвался на этапе «%s». "
                        "Смотрите logs/processor.log." % prog.stage)
    except Exception as exc:
        log.error("Исключение при запуске: %s", exc)
        prog.finish("❌ Непредвиденная ошибка: %s" % exc)
    finally:
        if os.path.exists(LOCK_PATH):
            os.remove(LOCK_PATH)


# ---------------------------------------------------------------------------
# Команды
# ---------------------------------------------------------------------------
_MENU = (
    "🗞 Дайджест ИВ РАН — наука, образование и молодёжная политика\n"
    "Турция, Центральная Азия, Южный Кавказ.\n\n"
    "/rundigest — собрать дайджест сейчас (5–20 минут)\n"
    "/last — прислать самый свежий готовый\n"
    "/digests — выбрать из последних готовых\n"
    "/status — что происходит прямо сейчас\n"
    "/resetlock — снять зависшую блокировку\n\n"
    "Воскресный дайджест приходит сам, вечером — сразу после последнего "
    "за сутки сбора новостей (около 21:30 МСК)."
)


def send_status(chat_id):
    """Что происходит сейчас. Этап знаем только у прогонов, поднятых ботом:
    воскресный cron запускает обработчик мимо лока, и для него это просто
    «прогон не идёт» — дайджест придёт сам."""
    items = archive()
    last = "последний готовый: %s" % msk(items[0][1]) if items else "готовых пока нет"
    if is_running():
        started, stage = _lock_read()
        head = "⚙️ Идёт прогон: %s" % (stage or "запускается")
        head += " (%s)." % _elapsed(started) if started else "."
    else:
        head = "💤 Прогон не идёт."
    send_message(chat_id, "%s\n%s\nВсего сохранено дайджестов: %d."
                 % (head, last, len(archive(limit=10 ** 6))))


def handle_update(update):
    """Регистрирует отправителя и обрабатывает один апдейт — команду или
    нажатие кнопки архива.

    Команды открыты всем — так решил владелец (2026-07-29). Цена: любой, кто
    нашёл бота, может поднять тяжёлый subprocess и потратить платные
    LLM-кредиты, а /resetlock снимает лок и позволяет запустить их параллельно.
    Ограничение по user_id было в коммите f944abe, если понадобится вернуть."""
    callback = update.get("callback_query")
    if callback:
        chat_id = callback.get("message", {}).get("chat", {}).get("id")
        data = callback.get("data", "")
        _api("answerCallbackQuery", {"callback_query_id": callback.get("id")})
        if chat_id is not None and data.startswith("d:"):
            send_archive_item(chat_id, data[2:])
        return

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
            send_message(chat_id, "🔓 Блокировка снята.")
        else:
            send_message(chat_id, "Блокировки не найдено.")

    elif text.startswith("/digests") or text.startswith("/archive"):
        send_archive_list(chat_id)

    elif text.startswith("/last"):
        items = archive(limit=1)
        if items:
            send_archive_item(chat_id, items[0][0])
        else:
            send_message(chat_id, "Готовых дайджестов пока нет. /rundigest соберёт новый.")

    elif text.startswith("/status"):
        send_status(chat_id)

    elif text.startswith("/start") or text.startswith("/help"):
        send_message(chat_id, _MENU)

    elif text.startswith("/"):
        send_message(chat_id, "Такой команды у меня нет.\n\n" + _MENU)


def main():
    """Бесконечный long-polling цикл getUpdates. offset = update_id последнего
    обработанного апдейта + 1 — так Telegram не присылает его повторно."""
    log.info("=== Бот-листенер запущен ===")
    # Меню команд в клиенте: без него команды знает только тот, кому их
    # показали. Ставится при каждом старте — Telegram хранит список у себя, и
    # правка _MENU в коде иначе до пользователя не доедет.
    _api("setMyCommands", {"commands": json.dumps([
        {"command": c, "description": d} for c, d in (
            ("rundigest", "собрать дайджест сейчас"),
            ("last", "прислать самый свежий готовый"),
            ("digests", "выбрать из последних готовых"),
            ("status", "что происходит прямо сейчас"),
            ("resetlock", "снять зависшую блокировку"),
            ("help", "меню"))])})
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


def _selfcheck():
    """python3 telegram_bot_listener.py --selfcheck — без сети и без Telegram."""
    assert _bar(0, 20) == "▱" * 20
    assert _bar(3, 20) == "▰" * 3 + "▱" * 17, "клетка на сюжет, а не 12 на всё"
    assert _bar(20, 20) == "▰" * 20
    assert _bar(1, 1) == "▰"

    # Полоса берётся из counts и живёт дальше этапа, где встретилось «N из M».
    now = time.time()
    assert "▰" not in progress_text("Читаю статьи за неделю", now)
    assert "3/20" in progress_text("Пересказываю сюжет 3 из 20", now, (3, 20))
    assert "20/20" in progress_text("Свожу дубликаты", now, (20, 20))

    # Минуты меняют текст сами — на этом держится heartbeat: поток шлёт правку
    # только при изменении, и без тикающих минут сообщение бы застыло.
    assert progress_text("x", now) != progress_text("x", now - 61)

    started = int(now)
    _lock_write(started, "Пересказываю сюжет 7 из 20")
    assert _lock_read() == (float(started), "Пересказываю сюжет 7 из 20")
    open(LOCK_PATH, "w").close()          # лок прошлых версий — пустой
    assert _lock_read() == (None, None)
    os.remove(LOCK_PATH)
    assert _lock_read() == (None, None)   # лока нет вовсе

    print("selfcheck ok")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
