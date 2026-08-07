# -*- coding: utf-8 -*-
"""
test_retell_russian.py — пересказ пишется сразу по-русски, без плеча перевода.

Раньше LLM писала по-английски, а на русский переводила локальная
opus-mt-tc-big-en-zle, и это плечо ломало текст: «здания будут усилены
землетрясением», «Искур. Гоув. tr», предложения по девяносто слов. Плечо снято.

Здесь то, что при таком переходе ломается молча. Приёмка вывода стоит ПЕРЕД
слотом: если она начнёт браковать нормальный русский, слоты будут пустеть один
за другим, а в логе будет написано «резерв корзины исчерпан» — про язык там не
будет ни слова.

Запуск: venv/bin/python test_retell_russian.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import sunday_processor_mmr as spm

_OK = "В Турции Министерство национального образования наймёт 30 000 охранников"


def test_english_output_is_rejected():
    """Ради чего проверка и стоит: модель сорвалась на язык источника."""
    ok, why = spm._validate_summary_fast({
        "title": "Turkey to hire school guards",
        "summary": "The Ministry of National Education announced the plan.",
    })
    assert not ok and "не на русском" in why, why


def test_russian_output_passes():
    ok, why = spm._validate_summary_fast({"title": _OK, "summary": _OK + "."})
    assert ok, why


def test_latin_organisation_names_do_not_fail_the_check():
    """Промпт САМ велит оставлять İŞKUR и TÜBİTAK латиницей. Порог, который
    считает такой пересказ нерусским, выкосил бы именно те статьи, ради которых
    дайджест и собирается."""
    summary = ("Приём заявок идёт через İŞKUR и TÜBİTAK по программе Erasmus+ "
               "в Yıldız Teknik Üniversitesi и Nazarbayev University.")
    ok, why = spm._validate_summary_fast({"title": _OK, "summary": summary})
    assert ok, why
    assert spm._cyrillic_share(summary) < 0.7, "пример перестал быть латинским"


def test_ukrainian_output_is_rejected():
    """Пункт 20 дайджеста от 07.08.2026 — дословно, как он ушёл читателю.
    Доля кириллицы у него 0.93, то есть верхний порог его пропускает и будет
    пропускать: язык здесь виден только по буквам, которых нет в русском."""
    title = ("20. 04.08.2026 — В Тбілісі та Астані: підготовка збройної "
             "команди з ШІ до міжнародної олімпіади")
    summary = ("У місті Тбілісі та Астані розгорнуто інтенсивну підготовку до "
               "міжнародної олімпіади з штучного інтелекту (IOAI 2026), де "
               "збройна команда з ШІ з Грузіниї, створена за підтримки Тбілісі "
               "та БТУ, змагатиметься у Астані 2–8 серпня.")
    assert spm._cyrillic_share(f"{title} {summary}") > spm._MIN_CYRILLIC
    ok, why = spm._validate_summary_fast({"title": title, "summary": summary})
    assert not ok and "кириллическом" in why, why


def test_one_foreign_name_does_not_cost_a_slot():
    """Порог не нулевой: перевыбор стоит слота, а одно имя в исходном написании
    посреди русского текста — это ещё не смена языка."""
    ok, why = spm._validate_summary_fast({
        "title": _OK,
        "summary": ("Соглашение подписали в Астане; со стороны Киева документ "
                    "визировал Зеленський, о чём сообщила пресс-служба "
                    "министерства образования Казахстана."),
    })
    assert ok, why


def test_mixed_script_inside_a_word_still_fails():
    """Сторож смешения скриптов пережил переход: «Турцuя» с латинской u внутри —
    это поломка, а не имя собственное. Ловится именно буква ВНУТРИ слова:
    целиком латинское İŞKUR рядом с русским текстом законно (см. выше)."""
    ok, why = spm._validate_summary_fast(
        {"title": _OK, "summary": "В Турцuи открыли лабораторию."})
    assert not ok and "смешение" in why, why


def test_cyrillic_share_ignores_digits_and_punctuation():
    assert spm._cyrillic_share("2026 — 30 000 (45 ₺)") == 0.0
    assert spm._cyrillic_share("") == 0.0
    assert spm._cyrillic_share("слово") == 1.0


def test_glossary_block_does_not_eat_the_article_budget():
    """Словарь едет отдельным куском перед статьями: если он попадёт внутрь
    бюджета символов, длинные сюжеты начнут молча терять хвост."""
    versions = [("https://o%d.com/1" % i, "z" * 10_000) for i in range(4)]
    src = spm._retell_sources(versions)
    assert src.count("z") <= config.RETELL_CHARS_TOTAL, src.count("z")
    assert src.count("----------") == 3, "разделитель версий сбился"


def test_translation_leg_is_gone():
    """Плечо перевода снято намеренно. Если оно вернётся импортом «чтобы было»,
    воскресный прогон снова потянет 300 МБ модели и снова начнёт портить текст."""
    assert not hasattr(spm, "_translate_to_russian")
    assert "translate_ru" not in sys.modules or "translate_ru" not in dir(spm)


if __name__ == "__main__":
    test_english_output_is_rejected()
    test_russian_output_passes()
    test_latin_organisation_names_do_not_fail_the_check()
    test_ukrainian_output_is_rejected()
    test_one_foreign_name_does_not_cost_a_slot()
    test_mixed_script_inside_a_word_still_fails()
    test_cyrillic_share_ignores_digits_and_punctuation()
    test_glossary_block_does_not_eat_the_article_budget()
    test_translation_leg_is_gone()
    print("ok")
