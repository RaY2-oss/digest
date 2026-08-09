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


def test_third_script_is_rejected():
    """Оба случая с боевого выхлопа. Арабский — замер 07.08.2026, слово
    «организовал» модель собрала наполовину из исходного письма. Армянский
    ушёл читателю в пункте 20 дайджеста от 25.07.2026. Прежний сторож ловил
    только пару кириллица+латиница и оба пропустил."""
    ok, why = spm._validate_summary_fast({
        "title": _OK,
        "summary": "Обучающий этап نظمил Министерство энергетики Турции.",
    })
    assert not ok and "третий алфавит" in why, why

    ok, why = spm._validate_summary_fast({
        "title": _OK,
        "summary": "Правительство Армении передаст участок фонду "
                   "«Գյումրիի տեղեկատվական տեխնոլոգիաների կենտրոն».",
    })
    assert not ok and "третий алфавит" in why, why


def test_greek_letters_stay_allowed():
    """Научный текст: β-распад и α-частицы — не поломка письма."""
    ok, why = spm._validate_summary_fast({
        "title": _OK,
        "summary": "В лаборатории измерили β-распад и поток α-частиц.",
    })
    assert ok, why


def test_truncated_summary_is_rejected():
    """Потолок токенов ставит провайдер, и обрыв доезжает до читателя как
    валидный JSON. Замер по 448 отгруженным пересказам: без концевой точки нет
    ни одного, так что проверка не стоит ни одного слота."""
    ok, why = spm._validate_summary_fast({
        "title": _OK,
        "summary": "Норма гарантирует сохранение личных выплат контрактных преподавателей",
    })
    assert not ok and "оборван" in why, why

    ok, why = spm._validate_summary_fast(
        {"title": _OK, "summary": "Работу примут осенью (по данным TÜBİTAK)."})
    assert ok, why


def test_cyrillic_share_ignores_digits_and_punctuation():
    assert spm._cyrillic_share("2026 — 30 000 (45 ₺)") == 0.0
    assert spm._cyrillic_share("") == 0.0
    assert spm._cyrillic_share("слово") == 1.0


def test_abbreviation_glosses_reach_the_prompt():
    """Модель не может расшифровать TENMAK, если ей не сказать, что это. Пояснения
    едут четвёртой колонкой glossary_abbr.tsv; тест смотрит на весь путь до
    промпта, потому что молчаливая потеря колонки выглядит ровно как «модель
    поленилась»."""
    src, _ = spm._retell_sources([("https://x.example/1",
                                "TENMAK ve NÜKEN, LGS ve YKS hakkında açıklama yaptı.")])
    for term, gloss in [("TENMAK", "энергетик"), ("LGS", "лицей"), ("YKS", "вуз")]:
        line = next((l for l in src.splitlines() if l.strip().startswith('"%s"' % term.lower())), "")
        assert gloss in line, "нет пояснения для %s: %r" % (term, line)


def test_glossary_block_does_not_eat_the_article_budget():
    """Словарь едет отдельным куском перед статьями: если он попадёт внутрь
    бюджета символов, длинные сюжеты начнут молча терять хвост."""
    stem = "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()
    versions = [("https://o%d.com/1" % i,
                 " ".join(" ".join("%s%d" % (w, i * 1000 + k) for w in stem) + "."
                          for k in range(200)))
                for i in range(4)]
    src, _ = spm._retell_sources(versions)
    body = "".join(src.split("Article text:\n")[1:])
    assert len(body) <= config.RETELL_CHARS_TOTAL * 1.05, len(body)
    assert src.count("----------") == 3, "разделитель версий сбился"


def test_source_word_in_cyrillic_is_rejected():
    """09.08.2026 читателю ушло «В Турции увеличен контэнджан стипендиальной
    программы TENMAK», днём раньше — «контентжант». `kontenjan` по-турецки
    «квота мест», обычное нарицательное слово, и прежние шесть сторожей такой
    пункт пропускали: JSON валиден, текст русский, доля кириллицы в норме."""
    source = ("Article text:\nEnerji Bakanligi TENMAK burs programinda "
              "kontenjani 300 den 500 e cikardi.")
    for word in ("контэнджан", "контентжант"):
        ok, why = spm._validate_summary_fast(
            {"title": _OK, "summary": "В Турции увеличен %s программы TENMAK." % word},
            source)
        assert not ok and "кириллицей вместо перевода" in why, (word, why)

    ok, why = spm._validate_summary_fast(
        {"title": _OK, "summary": "В Турции увеличена квота стипендиальной программы."},
        source)
    assert ok, why


def test_borrowed_words_are_not_transliteration():
    """Порог держится на том, что слово уже живёт в русском: «хакатон» пришёл
    из чужого языка ровно так же, но пишется по-русски всеми, а «контэнджан» —
    никем. Если эта половина проверки отвалится, дайджест начнёт браковать
    половину своей же лексики."""
    ok, why = spm._validate_summary_fast(
        {"title": _OK,
         "summary": "В Анкаре прошёл хакатон для студентов университета."},
        "Article text:\nAnkarada universite ogrencileri icin hackathon duzenlendi.")
    assert ok, why


def test_guard_stays_quiet_without_the_source():
    """Сторож сравнивает с текстом источника и без него молчит, а не гадает:
    вызов из тестов и любой будущий вызов без промпта не должны падать."""
    ok, why = spm._validate_summary_fast(
        {"title": _OK, "summary": "В Турции увеличен контэнджан программы."})
    assert ok, why


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
    test_third_script_is_rejected()
    test_greek_letters_stay_allowed()
    test_truncated_summary_is_rejected()
    test_abbreviation_glosses_reach_the_prompt()
    test_cyrillic_share_ignores_digits_and_punctuation()
    test_glossary_block_does_not_eat_the_article_budget()
    test_source_word_in_cyrillic_is_rejected()
    test_borrowed_words_are_not_transliteration()
    test_guard_stays_quiet_without_the_source()
    test_translation_leg_is_gone()
    print("ok")
