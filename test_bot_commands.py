# -*- coding: utf-8 -*-
"""
test_bot_commands.py — команды бота: архив готовых дайджестов и живой прогресс.

Проверяется то, что ломается молча:
  * ход прогона ПРАВИТ одно сообщение, а не сыплет ленту новых (иначе Telegram
    начнёт резать по лимиту ~1 сообщение в секунду в чат);
  * имя файла из кнопки ищется среди готовых дайджестов, а не склеивается с
    путём: команды бота открыты всем, и callback_data приходит от клиента;
  * время везде московское — сервер живёт по UTC.

Сеть и subprocess подменяются: настоящих запросов к Telegram тест не делает.

Запуск: venv/bin/python test_bot_commands.py
"""
import json
import os
import sys
import tempfile
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import telegram_bot_listener as tbl


class FakeResp:
    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body

    def raise_for_status(self):
        pass


class FakeAPI:
    """Считает вызовы Bot API вместо того, чтобы их делать."""

    def __init__(self):
        self.calls = []
        self.next_id = 100

    def post(self, url, data=None, files=None, proxies=None, timeout=None):
        method = url.rsplit("/", 1)[-1]
        self.calls.append((method, data or {}, files))
        self.next_id += 1
        return FakeResp({"ok": True, "result": {"message_id": self.next_id}})

    def of(self, method):
        return [d for m, d, _ in self.calls if m == method]


def _fake_net():
    api = FakeAPI()
    tbl.requests = types.SimpleNamespace(post=api.post, get=None)
    # Иначе тестовые chat_id попали бы в реальный список подписчиков, и
    # воскресный broadcast начал бы слать дайджест в чат №1.
    tbl.users_store = types.SimpleNamespace(register_user=lambda chat_id: None)
    return api


def _with_output(files):
    """Каталог с готовыми дайджестами: [(имя, mtime), ...]."""
    tmp = tempfile.mkdtemp()
    for name, mtime in files:
        path = os.path.join(tmp, name)
        with open(path, "wb") as f:
            f.write(b"docx")
        os.utime(path, (mtime, mtime))
    config.OUTPUT_DIR = tmp
    return tmp


# 09.08.2026 11:11 МСК и 08.08.2026 11:23 МСК
T1, T2 = 1786263060, 1786177380


def test_archive_is_newest_first_and_labelled_in_msk():
    _with_output([("digest_2026-08-08.docx", T2), ("digest_2026-08-09.docx", T1)])
    api = _fake_net()

    tbl.handle_update({"message": {"text": "/digests", "chat": {"id": 1}}})

    (msg,) = api.of("sendMessage")
    rows = json.loads(msg["reply_markup"])["inline_keyboard"]
    # Свежий сверху, и в подписи есть час: две сборки одного дня иначе не
    # различить.
    assert [b[0]["text"].split(" ", 1)[1] for b in rows] == [
        "09.08.2026 11:11 МСК", "08.08.2026 11:23 МСК"], rows
    assert rows[0][0]["callback_data"] == "d:digest_2026-08-09.docx"
    # Telegram режет callback_data по 64 байтам — молча, вместе с кнопкой.
    for row in rows:
        assert len(row[0]["callback_data"].encode()) <= 64, row


def test_button_sends_that_very_file():
    _with_output([("digest_2026-08-09.docx", T1)])
    api = _fake_net()

    tbl.handle_update({"callback_query": {
        "id": "42", "data": "d:digest_2026-08-09.docx",
        "message": {"chat": {"id": 1}}}})

    assert api.of("answerCallbackQuery"), "кнопка обязана перестать «крутиться»"
    (doc,) = api.of("sendDocument")
    assert "09.08.2026 11:11 МСК" in doc["caption"], doc["caption"]


def test_made_up_name_gets_nothing():
    """callback_data приходит от клиента и может быть любой строкой."""
    _with_output([("digest_2026-08-09.docx", T1)])
    api = _fake_net()

    for evil in ("d:../../etc/passwd", "d:", "d:digest_2000-01-01.docx"):
        tbl.handle_update({"callback_query": {
            "id": "1", "data": evil, "message": {"chat": {"id": 1}}}})

    assert not api.of("sendDocument"), api.calls
    assert len(api.of("sendMessage")) == 3


def test_last_sends_the_freshest():
    _with_output([("digest_2026-08-08.docx", T2), ("digest_2026-08-09.docx", T1)])
    api = _fake_net()

    tbl.handle_update({"message": {"text": "/last", "chat": {"id": 1}}})

    (doc,) = api.of("sendDocument")
    assert doc["caption"].endswith("09.08.2026 11:11 МСК"), doc["caption"]


def _run_with_stdout(paced, every=0.1):
    """Прогон на подставном subprocess: [(пауза перед строкой, строка), ...]."""
    class FakeProc:
        stdout = (time.sleep(p) or line for p, line in paced)

        def wait(self):
            return 0

    real_popen, tbl.subprocess.Popen = tbl.subprocess.Popen, \
        lambda *a, **kw: FakeProc()
    real_every, tbl.EDIT_EVERY = tbl.EDIT_EVERY, every
    try:
        tbl.run_processor(chat_id=7)
    finally:
        tbl.subprocess.Popen = real_popen
        tbl.EDIT_EVERY = real_every


def test_progress_edits_one_message_instead_of_flooding():
    """Главное в пункте: сообщение о запуске меняется по ходу прогона.

    ПОСЛЕДНЕЕ слово всегда за итогом прогона, иначе в чате навсегда останется
    «пересказываю сюжет 7 из 20» под уже отправленным дайджестом.
    """
    _with_output([])
    api = _fake_net()

    _run_with_stdout([(0.0, "DIGEST_STAGE=Читаю статьи за неделю\n"),
                      (0.0, "какой-то шум в stdout\n"),
                      (0.15, "DIGEST_STAGE=Пересказываю сюжет 1 из 20\n"),
                      (0.15, "DIGEST_ARTICLES=18/20\n"),
                      (0.15, "DIGEST_STAGE=Отправляю\n")])

    assert len(api.of("sendMessage")) == 1, "новых сообщений быть не должно"
    edits = api.of("editMessageText")
    assert [e["message_id"] for e in edits] == [101] * len(edits), edits
    texts = [e["text"] for e in edits]
    assert any("Читаю статьи" in t for t in texts), texts
    assert any("1/20" in t and "▰" in t for t in texts), texts
    assert "какой-то шум" not in " ".join(texts)
    assert "✅" in texts[-1] and "Статей: 18" in texts[-1], texts[-1]
    assert not os.path.exists(tbl.LOCK_PATH), "лок обязан сняться"


def test_stage_is_never_stale():
    """09.08.2026: читатель увидел «Пересказываю сюжет N» и решил, что полоса
    врёт. Правка по приходу этапа прорежалась «не чаще EDIT_EVERY» — и роняла
    как раз последнюю правку перед долгой паузой: сюжет пересказывался минуты,
    а в чате всё это время стоял предыдущий. Правит теперь отдельный поток, и
    сообщение доходит до последнего, что реально идёт."""
    _with_output([])
    api = _fake_net()

    _run_with_stdout([(0.0, "DIGEST_STAGE=Пересказываю сюжет 1 из 20\n"),
                      (0.01, "DIGEST_STAGE=Пересказываю сюжет 2 из 20\n"),
                      (0.4, "DIGEST_STAGE=Свожу дубликаты\n"),
                      (0.4, "сведение идёт\n")])

    texts = [e["text"] for e in api.of("editMessageText")]
    assert any("сюжет 2 из 20" in t for t in texts), texts
    # Полоса не исчезает после пересказа: пропавшая читается как сброс.
    assert any("Свожу дубликаты" in t and "2/20" in t for t in texts), texts


def test_bar_has_one_cell_per_story():
    """«Шкала из 12 штук, а не из 20» — фиксированная ширина читалась вторым
    счётчиком, спорящим с подписью."""
    assert tbl._bar(3, 20) == "▰" * 3 + "▱" * 17
    assert tbl._bar(0, 20) == "▱" * 20
    assert tbl._bar(20, 20) == "▰" * 20
    assert tbl._bar(1, 1) == "▰"


def test_status_tells_the_stage_of_a_running_digest():
    """«Что происходит прямо сейчас» — это этап, а не «прогон идёт». Этап живёт
    в файле лока: /status приходит в поток polling, а прогон идёт в своём."""
    _with_output([])
    api = _fake_net()
    tbl._lock_write(time.time() - 300, "Пересказываю сюжет 7 из 20")
    try:
        tbl.handle_update({"message": {"text": "/status", "chat": {"id": 1}}})
    finally:
        os.remove(tbl.LOCK_PATH)
    text = api.of("sendMessage")[0]["text"]
    assert "сюжет 7 из 20" in text and "5 мин" in text, text


def test_unknown_command_gets_an_answer():
    """Молчание в ответ на команду читается как «бот умер»."""
    _with_output([])
    api = _fake_net()
    tbl.handle_update({"message": {"text": "/degests", "chat": {"id": 1}}})
    text = api.of("sendMessage")[0]["text"]
    assert "нет" in text and "/digests" in text, text


def test_busy_run_does_not_start_a_second_one():
    _with_output([])
    api = _fake_net()
    open(tbl.LOCK_PATH, "w").close()
    try:
        tbl.run_processor(chat_id=7)
    finally:
        os.remove(tbl.LOCK_PATH)
    assert "/resetlock" in api.of("sendMessage")[0]["text"]
    assert not api.of("editMessageText")


if __name__ == "__main__":
    _out = config.OUTPUT_DIR
    try:
        test_archive_is_newest_first_and_labelled_in_msk()
        test_button_sends_that_very_file()
        test_made_up_name_gets_nothing()
        test_last_sends_the_freshest()
        test_progress_edits_one_message_instead_of_flooding()
        test_stage_is_never_stale()
        test_bar_has_one_cell_per_story()
        test_status_tells_the_stage_of_a_running_digest()
        test_unknown_command_gets_an_answer()
        test_busy_run_does_not_start_a_second_one()
    finally:
        config.OUTPUT_DIR = _out
    print("ok")
