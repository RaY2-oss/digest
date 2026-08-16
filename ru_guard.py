"""ru_guard.py — морфологические стражи русского текста дайджеста.

Три проверки, каждая выросла из того, что уже было напечатано в выпуске
14.08.2026 и дошло до читателя.

1. `mixed_script` — «технопарк в ОДTÜ»: «ОД» кириллицей, «TÜ» латиницей.
   Прежняя регулярка в `sunday_processor_mmr` требовала, чтобы чужая буква
   стояла МЕЖДУ двумя своими (`[а-я][a-z][а-я]`), и переход алфавита на
   границе слова не видела вовсе. Здесь условие другое: буквенный отрезок
   без дефисов и цифр не может содержать буквы двух алфавитов сразу.
   Дефис разделяет отрезки намеренно — «AI-платформа» и «ИИ-хаб» законны.

2. `title_without_predicate` — «В Эрзинджане и Бингёльском регионе»:
   заголовок без сказуемого не сообщает о событии ничего. Прежний страж
   требовал только три значимых слова, и такое проходило.

3. `bad_agreement` — «правила программы студенческого амнистия»:
   прилагательное не согласовано с существительным.

Замер на выпуске 14.08.2026 (18 заголовков, 18 абзацев пересказа):
заголовки — 2 срабатывания по сказуемому и 1 по согласованию, все три
настоящие, ложных нет; тексты — 1 настоящее («Новый модель»), ложных нет.

Правило согласования намеренно узкое. Без отсечек (знак препинания между
словами, числительное перед парой, причастие вместо прилагательного,
слово короче трёх букв) на тех же 18 абзацах было 9 срабатываний, из них
настоящих 2. Отсечки убрали ровно ложные: «два магматических резервуара»
— числительное, «не прошедшие регистрацию» — причастие, «несовершеннолетних
Дети» — разные предложения, «высшее и» — союз, разобранный как
существительное.
"""
import re
import threading

_morph_obj = None
_morph_lock = threading.Lock()


def _morph():
    """MorphAnalyzer грузится ~секунду и держит словарь в памяти — один на процесс."""
    global _morph_obj
    if _morph_obj is None:
        with _morph_lock:
            if _morph_obj is None:
                import pymorphy3
                _morph_obj = pymorphy3.MorphAnalyzer()
    return _morph_obj


# Буквенный отрезок: только буквы, без цифр, дефисов и подчёркиваний.
_LETTERS = re.compile(r"[^\W\d_]+", re.UNICODE)
# Латиница вместе с расширенной: Ü, ş, ğ, İ, ı — турецкие буквы приходят
# в текст вместе с названиями и должны считаться латиницей, а не «третьим».
_LATIN = re.compile(r"[A-Za-zÀ-ɏ]")
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")

_WORDS_RU = re.compile(r"[А-Яа-яЁё-]+")
_PREDICATE_POS = {"VERB", "INFN", "PRTS"}
_NUMERAL_POS = {"NUMR", "NUMB"}
_MIN_AGREEMENT_WORD = 3


def mixed_script(text: str) -> str | None:
    """Слово, внутри которого смешаны кириллица и латиница, либо None."""
    for m in _LETTERS.finditer(text or ""):
        word = m.group()
        if _LATIN.search(word) and _CYRILLIC.search(word):
            return word
    return None


def title_without_predicate(title: str) -> bool:
    """True, если в заголовке нет ни глагола, ни инфинитива, ни краткого причастия."""
    morph = _morph()
    for word in _WORDS_RU.findall(title or ""):
        if len(word) < 2:
            continue
        # Хватает трёх разборов: pymorphy сортирует их по убыванию вероятности,
        # и глагол, стоящий четвёртым вариантом, — это уже не сказуемое, а
        # совпадение форм.
        for parse in morph.parse(word)[:3]:
            if str(parse.tag.POS) in _PREDICATE_POS:
                return False
    return True


def bad_agreement(text: str) -> list[str]:
    """Пары «прилагательное + существительное», которые не согласуются ни в одном разборе."""
    morph = _morph()
    found = []
    # Разрыв по знакам препинания: соседство через точку или скобку — это
    # разные синтаксические места, согласовываться там нечему.
    for sentence in re.split(r"[.!?;:()«»\"—–]+", text or ""):
        words = re.findall(r"[А-Яа-яЁё-]+|[0-9]+", sentence)
        for i, (first, second) in enumerate(zip(words, words[1:])):
            if first.isdigit() or second.isdigit():
                continue
            if len(first) < _MIN_AGREEMENT_WORD or len(second) < _MIN_AGREEMENT_WORD:
                continue
            # «два магматических резервуара» — при числительном существительное
            # стоит в форме, которая с прилагательным и не должна совпадать.
            previous = words[i - 1] if i else ""
            if previous.isdigit() or any(str(p.tag.POS) in _NUMERAL_POS
                                         for p in morph.parse(previous)[:2]):
                continue
            # Причастие управляет дополнением, а не согласуется с ним:
            # «не прошедшие регистрацию».
            if any(str(p.tag.POS) == "PRTF" for p in morph.parse(first)):
                continue
            # Местоимённые прилагательные исключены: «который», «этот», «свой»
            # согласуются с далёким словом, а не с соседним. «порядок, по
            # которым подростки-нарушители направляются» — «которым» смотрит на
            # «порядок», и пара с «подростки» ложная. Замер 14.08.2026: на живом
            # прогоне это срабатывание сожгло все три попытки слота.
            adjectives = [p for p in morph.parse(first)
                          if p.tag.POS == "ADJF" and "Apro" not in p.tag
                          and "Anph" not in p.tag]
            nouns = [p for p in morph.parse(second) if p.tag.POS == "NOUN"]
            if not adjectives or not nouns:
                continue
            agrees = any(
                a.tag.case == n.tag.case
                and a.tag.number == n.tag.number
                and (a.tag.number == "plur" or a.tag.gender == n.tag.gender)
                for a in adjectives for n in nouns
            )
            if not agrees:
                found.append(f"{first} {second}")
    return found


# ---------------------------------------------------------------------------
# Смысловые стражи: аббревиатура, переписанная кириллицей, и число из воздуха
# ---------------------------------------------------------------------------

# Кириллица -> латиница, огрублённо: нужна не транслитерация для чтения, а
# сопоставимая форма. Сравниваются всё равно только согласные.
_CYR2LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}
_VOWELS = set("aeiouäöüıèéêàâûôî")
# Аббревиатура в источнике: заглавная латиница, от двух букв. TÜBİTAK, YÖK, MEB.
_LATIN_ABBR = re.compile(r"\b[A-ZÀ-Þ][A-ZÀ-Þ0-9]{1,7}\b")
_CYR_TOKEN = re.compile(r"[А-Яа-яЁё]{2,10}")
# Порог частотности. Ноль здесь не годится: «юк» стоит в wordfreq с zipf 2.61,
# «меб» с 1.83 — это шум корпуса, а не русские слова, и на пороге «частота > 0»
# страж пропускал ровно тот случай, ради которого написан. У настоящих русских
# слов zipf от 3 и выше («университет» 4.54). Вторая половина условия —
# совпадение скелета с аббревиатурой ИЗ ЭТОЙ статьи — и держит точность.
_ABBR_MAX_ZIPF = 3.0


_STUDY_PLACE = re.compile(
    r"\bизучени[еяюийя]\w*\s+(?:в|во|на)\s+"
    r"(?:вуз\w*|университет\w*|школ\w*|колледж\w*|институт\w*|академи\w*|лице\w*)\b",
    re.IGNORECASE)


def study_without_object(text: str) -> str | None:
    """«Изучение» требует объекта: изучают ЧТО-ТО. «Изучение в вузах» не значит
    ничего — нужное слово «обучение» или «учёба».

    16.08.2026 читателю ушло «право на отсрочку призыва для изучения в вузах».
    Ошибка тихая: JSON валиден, текст русский, алфавит один, согласование в
    порядке — все прежние сторожа такой пункт пропускают.

    Правило намеренно узкое: «изучение» с ПРЕДЛОГОМ МЕСТА сразу после него.
    Законные употребления всегда идут с объектом («изучение искусственного
    интеллекта», «изучение истории Казахстана», «платформа изучения языков»)
    и под правило не попадают. Замер по всем 626 отгруженным пунктам за
    июль-август: одно срабатывание, и оно же — тот самый брак. Ложных нет.

    Списка паронимов в корпусе ru-text нет (проверено), поэтому пара живёт
    здесь закрытым правилом, а не общей проверкой: расширять её «по аналогии»
    нельзя, иначе сторож начнёт браковать нормальный текст.
    """
    m = _STUDY_PLACE.search(text or "")
    return m.group(0) if m else None


def _skeleton(word: str) -> str:
    """Согласный скелет слова в латинице: «ЮК» и YÖK оба дают «yk»."""
    lat = "".join(_CYR2LAT.get(ch, ch) for ch in word.lower())
    return "".join(ch for ch in lat if ch.isalpha() and ch not in _VOWELS)


def transliterated_abbreviations(text: str, source: str) -> list[tuple[str, str]]:
    """Латинские аббревиатуры источника, переписанные в тексте кириллицей.

    Промпт требует оставлять YÖK, MEB, TÜBİTAK латиницей, но модель регулярно
    пишет «ЮК» и «Меб» — в выпуске 14.08.2026 напечатано «Совет по высшему
    образованию (ЮК)». Формально всё чисто: JSON валиден, алфавит один,
    кириллицы достаточно, так что ни один прежний страж этого не видел.

    Признак — конъюнкция, как и в translit_guard: слово не встречается в
    русском языке И его согласный скелет совпадает со скелетом заглавной
    латинской аббревиатуры из статьи, которую модель читала. По отдельности
    ни одна половина не годится: скелет «мб» совпадёт со случайным MB, а
    «несуществующих в русском» слов в любом тексте хватает.
    """
    if not source:
        return []
    from wordfreq import zipf_frequency
    abbrs = {}
    for m in _LATIN_ABBR.finditer(source):
        abbrs.setdefault(_skeleton(m.group()), m.group())
    if not abbrs:
        return []
    found = []
    seen = set()
    for m in _CYR_TOKEN.finditer(text or ""):
        word = m.group()
        if word.lower() in seen:
            continue
        if zipf_frequency(word.lower(), "ru") >= _ABBR_MAX_ZIPF:
            continue
        skel = _skeleton(word)
        if len(skel) < 2:
            continue
        if skel in abbrs:
            seen.add(word.lower())
            found.append((word, abbrs[skel]))
    return found


_NUMBER = re.compile(r"\d[\d\u00a0\u202f .,]*\d|\d")


def _norm_number(raw: str) -> str:
    """«89 457», «89.457», «89 457» -> «89457». Разделители у источника и у
    модели разные, сравнивать их бессмысленно."""
    return re.sub(r"[^\d]", "", raw)


def numbers_not_in_source(text: str, source: str, min_digits: int = 3) -> list[str]:
    """Числа пересказа, которых нет в прочитанном моделью тексте.

    Считаются только числа от min_digits цифр: двузначные («12–15 лет»,
    «81 регион») модель законно получает из словесных форм и из арифметики
    вроде «30 bin» -> «30 тыс.», и на них проверка шумит. Крупные же цифры —
    суммы, количества мест, бюджеты — модель либо взяла из статьи, либо
    выдумала, и второе в дайджест попадать не должно.
    """
    if not source:
        return []
    have = {_norm_number(m.group()) for m in _NUMBER.finditer(source)}
    # Год публикации и текущий год модель законно берёт из даты, а не из тела.
    missing = []
    for m in _NUMBER.finditer(text or ""):
        num = _norm_number(m.group())
        if len(num) < min_digits or num in have:
            continue
        if len(num) == 4 and num.startswith(("19", "20")):
            continue
        if num not in missing:
            missing.append(m.group().strip())
    return missing


def _self_check():
    # Смешение алфавитов: настоящий случай из выпуска и законные соседи.
    assert mixed_script("технопарк в ОДTÜ") == "ОДTÜ"
    assert mixed_script("Сovet") == "Сovet"          # латинская C в русском слове
    assert mixed_script("AI-платформы и ИИ-хаб") is None
    assert mixed_script("UNICEF Zhastary Hub на базе КазНУ") is None
    assert mixed_script("TÜBİTAK и İŞKUR") is None

    # Сказуемое в заголовке.
    assert title_without_predicate("В Эрзинджане и Бингёльском регионе")
    assert title_without_predicate("В Хуросонском районе Таджикистана")
    assert not title_without_predicate("В Турции ужесточены меры по защите детей")
    assert not title_without_predicate("В Алматы прошёл молодёжный форум")
    assert not title_without_predicate("В Казахстане запускают центр данных")

    # Согласование.
    assert bad_agreement("правила программы студенческого амнистия")
    assert bad_agreement("Новый модель представлена в Астане")
    assert not bad_agreement("правила программы студенческой амнистии")
    assert not bad_agreement("под землёй обнаружены два магматических резервуара")
    assert not bad_agreement("студенты, не прошедшие регистрацию")
    assert not bad_agreement("В 2026/2027 учебном году выделено 89 457 грантов")
    # Аббревиатура, переписанная кириллицей.
    src = "YÖK usul ve esaslari belirledi, MEB ile birlikte"
    assert transliterated_abbreviations("Совет по высшему образованию (ЮК) утвердил", src) == [("ЮК", "YÖK")]
    assert transliterated_abbreviations("Совет по высшему образованию (YÖK) утвердил", src) == []
    assert transliterated_abbreviations("Министерство образования Турции", src) == []

    # Числа из воздуха.
    assert numbers_not_in_source("выделено 89 457 грантов", "were 89.457 grants given") == []
    assert numbers_not_in_source("выделено 89 457 грантов", "were 12 000 grants given") == ["89 457"]
    assert numbers_not_in_source("в 2027 году", "launch planned soon") == []
    assert numbers_not_in_source("81 регион", "in 81 provinces") == []

    print("ru_guard: все проверки прошли")


if __name__ == "__main__":
    _self_check()
