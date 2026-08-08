# -*- coding: utf-8 -*-
"""
test_retell_merge.py — пересказ сюжета по НЕСКОЛЬКИМ перепечаткам сразу.

Раньше в промпт уходила ровно одна статья, а её дубликаты с других сайтов
_process_slot просто выбрасывал (прогон 26.07: 49 отброшенных). Детали, которые
сохранила только одна редакция — цифры, имена, цитаты, — терялись вместе с ними.
Теперь версии сюжета склеиваются в один промпт под общим потолком по символам,
и все использованные ссылки идут в документ.

Запуск: venv/bin/python test_retell_merge.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import sunday_processor_mmr as spm
import word_generator


def _v(url, n):
    return (url, "z" * n)


def test_single_version_budget_unchanged():
    """Одна версия получает столько же символов, сколько получала раньше
    единственная статья, — переход не должен дорожать на обычной новости."""
    src = spm._retell_sources([_v("https://a.com/1", 10_000)])
    assert src.count("z") == config.RETELL_CHARS_PER_VERSION, src.count("z")
    assert "https://a.com/1" in src


def test_several_versions_share_the_budget():
    versions = [_v(f"https://o{i}.com/1", 10_000) for i in range(4)]
    src = spm._retell_sources(versions)
    assert src.count("z") <= config.RETELL_CHARS_TOTAL, src.count("z")
    for i in range(4):
        assert f"https://o{i}.com/1" in src
    assert src.count("----------") == 3, "версии обязаны быть разделены"


def test_extra_versions_are_dropped_not_squeezed():
    """Потолок по числу версий важнее полноты: десять обрывков по 600 символов
    модель пересказывает хуже, чем четыре внятных куска."""
    versions = [_v(f"https://o{i}.com/1", 5_000) for i in range(10)]
    src = spm._retell_sources(versions)
    assert "https://o3.com/1" in src
    assert "https://o4.com/1" not in src
    assert src.count("z") <= config.RETELL_CHARS_TOTAL


def test_empty_text_does_not_break_the_prompt():
    src = spm._retell_sources([("https://a.com/1", None), ("https://b.com/1", "тело")])
    assert "https://a.com/1" in src and "тело" in src


def test_docx_lists_every_source_url(tmp="/tmp"):
    """Все ссылки, попавшие в промпт, обязаны быть в документе — иначе пересказ
    нечем проверить."""
    old_out = config.OUTPUT_DIR
    config.OUTPUT_DIR = tmp
    try:
        path = word_generator.build_digest([{
            "title": "Заголовок", "summary": "Пересказ.",
            "url": "https://a.com/1",
            "urls": ["https://a.com/1", "https://b.com/1", "https://c.com/1"],
            "publish_date": "2026-07-27",
        }])
        from docx import Document
        text = "\n".join(p.text for p in Document(path).paragraphs)
        for u in ("https://a.com/1", "https://b.com/1", "https://c.com/1"):
            assert u in text, f"{u} потерян в документе"
    finally:
        config.OUTPUT_DIR = old_out


def test_docx_falls_back_to_single_url():
    """Старый формат словаря (без "urls") обязан рисоваться как раньше."""
    old_out = config.OUTPUT_DIR
    config.OUTPUT_DIR = "/tmp"
    try:
        path = word_generator.build_digest([{
            "title": "Заголовок", "summary": "Пересказ.",
            "url": "https://a.com/1", "publish_date": "2026-07-27",
        }])
        from docx import Document
        text = "\n".join(p.text for p in Document(path).paragraphs)
        assert "URL: https://a.com/1" in text
    finally:
        config.OUTPUT_DIR = old_out


def test_story_date_is_the_earliest_reprint():
    """Дату сюжету даёт первая его перепечатка, а не представитель отбора.

    Представителя выбирает MMR, и это запросто третья по счёту публикация:
    датировать событие ею — значит показать в дайджесте день, когда сюжет
    перепечатали, вместо дня, когда о нём написали.
    """
    rep = ("u", "t", None, "2026-08-06", "TR", 0.0, "TR1")
    assert spm._story_date(rep, "нет") == "2026-08-06"

    story = rep + ([("a", "", None, "2026-08-08", "TR", 0.0, "TR1"),
                    ("b", "", None, "2026-08-04", "TR", 0.0, "TR1"),
                    # у части статей publish_date не определился — такие в
                    # счёт не идут, иначе пустая строка выиграла бы min()
                    ("c", "", None, "", "TR", 0.0, "TR1")],)
    assert spm._story_date(story, "нет") == "2026-08-04"

    blank = ("u", "t", None, "", "TR", 0.0, "TR1",
             [("a", "", None, "", "TR", 0.0, "TR1")])
    assert spm._story_date(blank, "2026-08-01") == "2026-08-01"


if __name__ == "__main__":
    test_story_date_is_the_earliest_reprint()
    test_single_version_budget_unchanged()
    test_several_versions_share_the_budget()
    test_extra_versions_are_dropped_not_squeezed()
    test_empty_text_does_not_break_the_prompt()
    test_docx_lists_every_source_url()
    test_docx_falls_back_to_single_url()
    print("ok")
