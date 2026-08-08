# -*- coding: utf-8 -*-
"""Спасение текста: сторож заглушек, лестница способов и общий бюджет.

Ни браузер, ни архив здесь не поднимаются и не опрашиваются — они стоят
секунд и живут в сети, а проверять надо не их, а то, что вокруг: заглушку
антибота не пускать в статьи, разбор повторять на полноту, архив спрашивать
раньше браузера, а бюджет тратить на открытые страницы, не на удавшиеся.

Запуск: ./venv/bin/python test_browser_rescue.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import archive_fetch
import config
import daily_collector as dc

# Настоящие ответы щитов, снятые camoufox 08.08.2026 с bursadabugun.com и
# memurlar.net: обе страницы длиннее MIN_TEXT_LENGTH, и без сторожа обе
# уходили в статьи.
SHIELD_BURSA = """<html><head><title>www.bursadabugun.com</title></head><body>
<p>Performing security verification</p>
<p>This website uses a security service to protect against malicious bots.
This page is displayed while the website verifies you are not a bot. This
website uses a security service to protect against malicious bots. This page
is displayed while the website verifies you are not a bot.</p></body></html>"""

SHIELD_CF = """<html><head><title>Sorry, you have been blocked</title></head><body>
<p>This website is using a security service to protect itself from online
attacks. The action you just performed triggered the security solution. There
are several actions that could trigger this block including submitting a
certain word or phrase, a SQL command or malformed data. You can email the
site owner to let them know you were blocked.</p></body></html>"""

ARTICLE = """<html><head><title>Новый ускоритель в Аммане</title></head><body>
<article><p>%s</p></article></body></html>""" % ("Иорданский центр SESAME "
    "запустил новую линию синхротронного излучения; работать на ней смогут "
    "группы из восьми стран-участниц. " * 6)


def check_antibot():
    for name, html in (("bursadabugun", SHIELD_BURSA), ("cloudflare", SHIELD_CF)):
        text, title, _ = dc.extract_page(html, "https://example.org/a")
        assert text is None, "заглушка %s прошла как статья: %r" % (name, text)
    text, title, _ = dc.extract_page(ARTICLE, "https://example.org/b")
    assert text and len(text) >= config.MIN_TEXT_LENGTH, \
        "сторож съел настоящую статью: %r" % (text,)
    print("  сторож: обе заглушки отбиты, статья на %d символов прошла" % len(text))


def check_budget():
    """Бюджет — на прогон, а не на запрос, и платится за открытые страницы.

    Считать его по спасённым значило бы: щит не пустил — открываем ещё одну,
    и трудная страна съедает браузер целиком.
    """
    budget = [5]
    opened = [dc.take_for_browser(s, budget)
              for s in (["a", "b", "c"], ["d", "e", "f", "g"])]
    assert opened == [["a", "b", "c"], ["d", "e"]], opened
    assert budget[0] == 0, budget
    # Третий запрос при исчерпанном бюджете не должен уходить в минус и не
    # должен, отрезав пустоту, поднимать браузер.
    assert dc.take_for_browser(["h"], budget) == [] and budget[0] == 0, budget
    print("  бюджет: 5 страниц на два запроса разошлись как 3 + 2, добавки нет")


def check_recall_retry():
    """Пустой разбор на точность обязан переспрашиваться на полноту.

    Разбор на точность отрезает всё, в чём не уверен, и на непривычной вёрстке
    оставляет пустоту — на замере из ста «пустых» адресов повтор на полноту
    добавлял десять статей. Проверяем не trafilatura, а порядок: второй заход
    случается тогда и только тогда, когда от первого нет текста выше порога.
    """
    calls, real = [], dc._pull

    def spy(html, url, recall):
        calls.append(recall)
        return ("з" * 40, "т") if recall is False else ("с" * 900, "т")

    dc._pull = spy
    try:
        text, _, _ = dc.extract_page("<html></html>", "https://example.org/a")
        assert calls == [False, True], calls
        assert len(text) == 900, "взят короткий разбор вместо полного"

        calls.clear()
        dc._pull = lambda html, url, recall: (calls.append(recall)
                                              or ("д" * 900, "т"))
        dc.extract_page("<html></html>", "https://example.org/b")
        assert calls == [False], "полный разбор дёрнули без нужды: %r" % (calls,)
    finally:
        dc._pull = real
    print("  разбор: короткий результат переспрошен на полноту, длинный — нет")


def check_archive_before_browser():
    """Архив отвечает раньше браузера, и браузеру достаётся только остаток.

    Архив дешевле браузера впятеро и берёт вдвое больше (замер в
    archive_fetch), поэтому бюджет браузера обязан тратиться на то, чего в
    архиве не нашлось.
    """
    have = {"https://a.example/1": ARTICLE.encode("utf-8")}
    asked, real = [], archive_fetch.snapshot
    archive_fetch.snapshot = lambda u, **kw: (asked.append(u) or have.get(u))
    try:
        hard = ["https://a.example/1", "https://b.example/2"]
        got = dc.rescue_from_archive(hard)
        assert asked == hard, asked
        assert list(got) == ["https://a.example/1"], got
        assert got["https://a.example/1"][0].startswith("Иорданский"), got
        left = [u for u in hard if u not in got]
        assert left == ["https://b.example/2"], left
    finally:
        archive_fetch.snapshot = real
    assert dc.rescue_from_archive([]) == {}
    print("  лестница: архив взял своё, браузеру остался только остаток")


def check_pass_through():
    """Пустой список браузер не поднимает — прогон без неудач не платит."""
    assert dc.rescue_with_browser([]) == {}
    print("  без неудач браузер не поднимается")


if __name__ == "__main__":
    check_antibot()
    check_recall_retry()
    check_archive_before_browser()
    check_budget()
    check_pass_through()
    print("спасение текста: ок")
