# -*- coding: utf-8 -*-
"""event_date.py — дата СОБЫТИЯ, а не публикации, и потолок в неделю.

Дайджест до сих пор датировался публикацией: у сюжета бралась самая ранняя из
дат его перепечаток (_story_date). Это ближайшее к событию, что даёт корпус,
но не само событие: саммит прошёл 29 июля, а первая попавшая к нам заметка о
нём вышла 2 августа, и в документе стояло 02.08.

Даты события нет ни в GKG (там время выгрузки), ни в разметке страницы (там
время публикации) — она есть только в самом тексте: «29 Temmuz'da düzenlenen
zirve», «мероприятие прошло 26 июля». Значит, её кто-то должен ПРОЧИТАТЬ.

Читает её та же LLM, которая и так пересказывает сюжет (лишнего вызова нет,
поле в том же JSON). А здесь — сторож, который её проверяет, потому что
правило в промпте соблюдается вероятностно, а дата в шапке пункта выглядит
одинаково убедительно и когда она верна, и когда выдумана:

    1. дата обязана попадать в [сегодня − WINDOW, дата первой публикации].
       Позже публикации — это анонс («приём заявок до 1 сентября»), а не дата
       события; раньше окна — сюжет не нашей недели.
    2. дата обязана БУКВАЛЬНО стоять в тексте источника: «29 Temmuz»,
       «26 июля», «29.07.2026». Без этого проверить её нечем.

Почему не разбором текста вместо LLM. Замер 09.08.2026 на недельном корпусе
(1403 статьи, 725 сюжетов): строгий поиск «число + месяц» находит дату у 22–35 %
сюжетов и меняет дату сюжета у 4–9 %, но примерно половина этих изменений
неверна — regex не отличает дату события от фона. «her geçen gün» («с каждым
днём») он читает как «вчера», «1 Ağustos itibarıyla» («по состоянию на») — как
день события. Отличает их смысл, а не форма, поэтому дату выбирает модель, а
таблица месяцев остаётся здесь только сторожем: она отвечает на вопрос «стоит
ли такая дата в тексте», а не «та ли это дата».

Таблицы названий месяцев берутся из локалей dateparser (турецкий, русский,
английский, казахский, азербайджанский, узбекский, грузинский, армянский,
киргизский) — руками их писать незачем, пакет уже стоит в зависимостях.
"""
import re
from datetime import date, timedelta

from dateparser.languages.loader import default_loader

WINDOW = 7          # дайджест недельный: дата старше — не наша новость

_LOCALES = ("tr", "ru", "en", "kk", "az", "uz", "ka", "hy", "ky")
_MONTH_KEYS = ("january february march april may june july august september "
               "october november december").split()


def _norm(w: str) -> str:
    """Турецкая i без точки — отдельная буква, и `.lower()` разводит «Mayıs» и
    «MAYIS» по разным ключам. Сводим их к одному."""
    return (w.lower().replace("ı", "i").replace("İ", "i")
            .replace("̇", ""))


def _months():
    out = {}
    for code in _LOCALES:
        info = default_loader.get_locale(code).info
        for i, key in enumerate(_MONTH_KEYS, 1):
            for name in info.get(key, []):
                if len(name) >= 3:          # «янв.» и «Oca» ловим уже суффиксом
                    out[_norm(name)] = i
    return out


MONTHS = _months()

# «29 Temmuz'da», «26 июля», «1 August 2026». Суффикс склеен с названием
# месяца в турецком и казахском, поэтому хвост слова просто съедается.
_DMY = re.compile(r"\b(\d{1,2})\s*[-–]?\s*(?:го\b|nji\b|inci\b)?\s*(%s)\w*"
                  r"(?:\s*,?\s*(\d{4}))?"
                  % "|".join(sorted((re.escape(m) for m in MONTHS),
                                    key=len, reverse=True)),
                  re.IGNORECASE | re.UNICODE)
_NUM = re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b")
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def parse(raw) -> date | None:
    """'2026-08-04' → date. Мусор и None → None, без исключения."""
    if not raw or not isinstance(raw, str):
        return None
    m = _ISO.search(raw.strip())
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def mentioned(text: str, day: date) -> bool:
    """Стоит ли эта дата в тексте явно — в любом из разбираемых языков.

    Год в новостях почти всегда опущен («29 Temmuz»), поэтому по умолчанию
    сверяются число и месяц; если год написан, он обязан совпасть.
    """
    if not text:
        return False
    for m in _DMY.finditer(text):
        y = m.group(3)
        if (int(m.group(1)) == day.day and MONTHS[_norm(m.group(2))] == day.month
                and (y is None or int(y) == day.year)):
            return True
    for m in _NUM.finditer(text):
        if (int(m.group(1)), int(m.group(2)), int(m.group(3))) == (
                day.day, day.month, day.year):
            return True
    for m in _ISO.finditer(text):
        if (int(m.group(1)), int(m.group(2)), int(m.group(3))) == (
                day.year, day.month, day.day):
            return True
    return False


def clamp(iso: str, today: date) -> str:
    """Дата пункта дайджеста: не старше недели и не из будущего.

    Окно выборки статей это и так обеспечивает (`publish_date >= -7 days`), но
    оно живёт в SQL воскресного прогона, а печатается дата здесь. Пустая строка
    значит «даты нет» — пункт печатается без неё, как и раньше.
    """
    d = parse(iso)
    if d is None or d > today or d < today - timedelta(days=WINDOW):
        return ""
    return d.isoformat()


def choose(raw, pub_iso: str, source: str, today: date) -> tuple[str, str]:
    """Дата события от LLM → (дата для дайджеста, причина отказа).

    pub_iso — дата первой публикации сюжета (она же запасной вариант).
    source  — текст промпта: ровно то, что модель читала, и ничто иное.
    """
    d = parse(raw)
    if d is None:
        return clamp(pub_iso, today), "" if raw in (None, "", "null") else "не дата: %r" % (raw,)

    pub = parse(pub_iso)
    if pub is not None and d > pub:
        return clamp(pub_iso, today), "%s позже публикации %s" % (d, pub)
    if d > today or d < today - timedelta(days=WINDOW):
        return clamp(pub_iso, today), "%s вне недельного окна" % d
    if not mentioned(source, d):
        return clamp(pub_iso, today), "%s в тексте не упомянута" % d
    return d.isoformat(), ""


def _selfcheck():
    today = date(2026, 8, 9)
    src = ("Article text:\nTürkiye'nin yapay zeka vizyonu, 4 Ağustos'ta Ankara'da "
           "düzenlenen zirve ile konusuldu. Kazakstanda 6 августа прошла олимпиада, "
           "а 03.08.2026 объявлены итоги.")

    assert parse("2026-08-04") == date(2026, 8, 4)
    assert parse(None) is None and parse("вчера") is None and parse("2026-13-40") is None

    # Дата стоит в тексте — на трёх языках и в трёх написаниях.
    assert mentioned(src, date(2026, 8, 4))
    assert mentioned(src, date(2026, 8, 6))
    assert mentioned(src, date(2026, 8, 3))
    assert not mentioned(src, date(2026, 8, 5))
    assert not mentioned(src, date(2026, 7, 29))
    # Написанный год обязан совпасть: «29 Temmuz 2025» — не про эту неделю.
    assert not mentioned("zirve 29 Temmuz 2025 tarihinde yapildi", date(2026, 7, 29))
    # Турецкая i без точки: MAYIS и Mayıs — один месяц.
    assert mentioned("MAYIS ayinda 5 MAYIS gunu", date(2026, 5, 5))

    # Потолок в неделю: старьё и будущее не печатаются.
    assert clamp("2026-08-04", today) == "2026-08-04"
    assert clamp("2026-08-02", today) == "2026-08-02"      # ровно неделя — ещё наша
    assert clamp("2026-08-01", today) == ""
    assert clamp("2026-08-10", today) == ""
    assert clamp("", today) == "" and clamp(None, today) == ""

    # Ради чего всё и делалось: событие 4 августа, первая публикация 7-го.
    assert choose("2026-08-04", "2026-08-07", src, today) == ("2026-08-04", "")
    # Модель промолчала — остаётся публикация.
    assert choose(None, "2026-08-07", src, today) == ("2026-08-07", "")
    # Анонс: дата позже публикации — это «когда будет», а не «когда было».
    d, why = choose("2026-08-09", "2026-08-07", src, today)
    assert d == "2026-08-07" and "позже публикации" in why, (d, why)
    # Выдуманная дата: в окне, раньше публикации, но в тексте её нет.
    d, why = choose("2026-08-05", "2026-08-07", src, today)
    assert d == "2026-08-07" and "не упомянута" in why, (d, why)
    # Дата старше окна не проходит, даже если она в тексте: пункт остаётся
    # (опубликован-то он на этой неделе), но датируется публикацией.
    old = "Meeting took place on 25 July 2026."
    d, why = choose("2026-07-25", "2026-08-07", old, today)
    assert d == "2026-08-07" and "вне недельного окна" in why, (d, why)
    # Публикация старше окна: печатать нечего, а не печатать старьё.
    assert choose(None, "2026-07-20", src, today) == ("", "")

    print("event_date selfcheck ok")


if __name__ == "__main__":
    _selfcheck()
