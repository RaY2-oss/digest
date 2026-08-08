# -*- coding: utf-8 -*-
"""Спасательный проход браузером: сторож заглушек и общий бюджет.

Браузер здесь не поднимается — он стоит пятнадцать секунд на страницу, а
проверять надо не его, а то, что вокруг: заглушку антибота не пускать в
статьи, а бюджет тратить на открытые страницы, не на удавшиеся.

Запуск: ./venv/bin/python test_browser_rescue.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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


def check_pass_through():
    """Пустой список браузер не поднимает — прогон без неудач не платит."""
    assert dc.rescue_with_browser([]) == {}
    print("  без неудач браузер не поднимается")


if __name__ == "__main__":
    check_antibot()
    check_budget()
    check_pass_through()
    print("спасение браузером: ок")
