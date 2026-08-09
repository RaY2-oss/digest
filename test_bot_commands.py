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


def test_progress_edits_one_message_instead_of_flooding():
    """Главное в пункте: сообщение о запуске меняется по ходу прогона.

    Этапы идут пачками, поэтому правки прорежены по времени (EDIT_EVERY) — но
    ПОСЛЕДНЕЕ слово всегда за итогом прогона, иначе в чате навсегда останется
    «пересказываю сюжет 7 из 20» под уже отправленным дайджестом.
    """
    _with_output([])
    api = _fake_net()

    lines = ["DIGEST_STAGE=Читаю статьи за неделю\n",
             "какой-то шум в stdout\n",
             "DIGEST_STAGE=Пересказываю сюжет 1 из 20\n",
             "DIGEST_STAGE=Пересказываю сюжет 2 из 20\n",
             "DIGEST_ARTICLES=18/20\n",
             "DIGEST_STAGE=Отправляю\n"]

    class FakeProc:
        stdout = iter(lines)

        def wait(self):
            return 0

    real_popen, tbl.subprocess.Popen = tbl.subprocess.Popen, \
        lambda *a, **kw: FakeProc()
    real_every, tbl.EDIT_EVERY = tbl.EDIT_EVERY, 0.0
    try:
        tbl.run_processor(chat_id=7)
    finally:
        tbl.subprocess.Popen = real_popen
        tbl.EDIT_EVERY = real_every

    assert len(api.of("sendMessage")) == 1, "новых сообщений быть не должно"
    edits = api.of("editMessageText")
    assert [e["message_id"] for e in edits] == [101] * len(edits), edits
    assert "Читаю статьи" in edits[0]["text"]
    assert "▰" in edits[1]["text"] and "1/20" in edits[1]["text"]
    assert "какой-то шум" not in " ".join(e["text"] for e in edits)
    assert "✅" in edits[-1]["text"] and "Статей: 18" in edits[-1]["text"]
    assert not os.path.exists(tbl.LOCK_PATH), "лок обязан сняться"


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
        test_busy_run_does_not_start_a_second_one()
    finally:
        config.OUTPUT_DIR = _out
    print("ok")
