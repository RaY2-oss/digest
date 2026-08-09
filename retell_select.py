# -*- coding: utf-8 -*-
"""retell_select.py — что из версий сюжета уходит в промпт пересказа.

Раньше `_retell_sources` резал КАЖДУЮ версию по первым N символов и склеивал
куски в один промпт. Идея была правильная («разные редакции сохраняют разные
детали»), исполнение — нет: перепечатки начинаются одинаково, поэтому четыре
куска по 1500 символов — это один и тот же лид, набранный четыре раза, а
детали, ради которых версии и брались, лежат ниже среза и до модели не
доезжают.

Здесь вместо обрезки отбор, как в gdelt_rss/overview.py: статья разбирается на
предложения, вес слов считается по самому сюжету, и жадно набираются
предложения — сначала по одному от РАЗНЫХ изданий, потом добор, и каждое
следующее обязано меньше чем на четверть состоять из уже сказанных оборотов.
Бюджет промпта тот же, но занят он разными фактами, а не повтором лида.

Замер на боевой базе 09.08.2026 (244 многоверсионных сюжета, бюджет 6000
символов): обрезка отдаёт модели 265 разных слов и 434 разных трёхсловных
оборота на сюжет, отбор — 313 и 515 при МЕНЬШЕМ промпте (4220 символов против
5403). То есть плюс пятая часть фактов и минус пятая часть токенов сразу.
Изданий в промпте 3.25 против 2.88, и на самое многословное из них приходится
45% текста, а не 60%, как при отборе без круговых кругов.

Почему шинглы, а не эмбеддинги: вопрос здесь не «про то же ли это» (про то же,
иначе версии не попали бы в один сюжет), а «есть ли новые формулировки». На
него отвечает лексика. Косинус двух пересказов одного события даст 0.95
независимо от того, добавила вторая редакция факты или нет.

Язык не разбирается вообще. На входе бывает турецкий, русский, английский и
казахский, и весь разбор — одна регулярка плюс счёт слов; ничего, что зависело
бы от алфавита или морфологии, здесь нет.
"""
import re

WORD = re.compile(r"[^\W_]+", re.UNICODE)
_URL = re.compile(r"https?://\S+|www\.\S+")
# Разрыв предложения: точка/восклицание/вопрос перед пробелом либо перевод
# строки. Двоеточие НЕ разрывает: турецкие подзаголовки и подводки к цитате
# набраны через него («Bakan Bayraktar: "..."»), и разрыв отрезал бы реплику
# от того, кто её произнёс.
_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")

MIN_WORDS = 5           # слов С БУКВАМИ; строка «5 Kontenjan» предложением не является
MAX_CHARS = 400         # длиннее — обычно неразобранный список или врезка
MIN_WORD_SHARE = 0.5    # доля буквенных токенов: ниже — цифровая каша из таблицы
DUP = 0.28              # доля общих оборотов, после которой предложение — повтор
SHINGLE = 4

# Конец предложения. Строка без него — не предложение: подзаголовок вроде
# «EN YÜKSEK KONTENJAN ELEKTRİK ALANINDA» или строка таблицы «Elektrik
# Mühendisliği: 80 Kontenjan». Слов с буквами в такой строке хватает, и по
# длине она проходит, а сказано в ней ничего. Правило снимает 14% кандидатов
# по всей базе и заодно обрубок, которым кончается срез по depth.
_ENDS = re.compile(r"[.!?…][\"»'’”)\]]*$")


def _words(s):
    return [w.lower() for w in WORD.findall(s)]


def _shingles(s, n=SHINGLE):
    w = _words(s)
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def sentences(text, depth):
    """Текст статьи → предложения, годные в промпт. depth — сколько символов
    статьи вообще рассматриваем: ниже него у новостных сайтов идут блоки
    «читайте также» и подвал, а полезного текста там уже нет."""
    out, loose = [], []
    for part in _SPLIT.split((text or "")[:depth]):
        s = _URL.sub("", part)
        s = re.sub(r"\s+", " ", s).strip(" -—•*\t\"'“”")
        if len(s) > MAX_CHARS:
            continue
        toks = WORD.findall(s)
        letters = [t for t in toks if any(c.isalpha() for c in t)]
        if len(letters) < MIN_WORDS or len(letters) < len(toks) * MIN_WORD_SHARE:
            continue
        (out if _ENDS.search(s) else loose).append(s)
    # У 18 статей из 1488 текст извлёкся вообще без точек. Отдать по ним пустоту
    # значит выбросить версию из промпта целиком, поэтому там правило снимается.
    return out or loose


def _tf_idf(sents):
    """Веса слов по корпусу из самих предложений сюжета.

    В новостном сюжете повторяется как раз суть, а не вода, поэтому вес растёт
    с частотой по предложениям — но до середины: слово, стоящее вообще везде,
    сюжет не характеризует, как и слово из одного предложения (имя
    корреспондента, город подписи).
    """
    df = {}
    for s in sents:
        for w in set(_words(s)):
            df[w] = df.get(w, 0) + 1
    n = len(sents)
    return {w: (c / n) * (1.0 - c / (n + 1.0)) for w, c in df.items()}


def _score(sent, weight):
    w = set(_words(sent))
    if not w:
        return 0.0
    # Делим на корень размера, а не на размер: иначе побеждают обрывки в три
    # слова, у которых вся длина — одно ключевое слово.
    return sum(weight.get(x, 0.0) for x in w) / (len(w) ** 0.5)


def _candidates(versions, depth):
    """(url, место в статье, предложение) по всем версиям.

    Внутри ОДНОЙ версии почти-повторы снимаются: у турецких изданий статья
    часто кончается таблицей вида «Makine Mühendisliği: 57 kontenjan», и
    двадцать таких строк раздували бы вес слова «kontenjan» до уровня сути
    сюжета. МЕЖДУ версиями повторы, наоборот, оставляем — это и есть сигнал
    центральности: что повторили все, то и главное.
    """
    cand = []
    for url, text in versions:
        seen = set()
        for pos, s in enumerate(sentences(text, depth)):
            sh = _shingles(s)
            if sh and len(sh & seen) > len(sh) * DUP:
                continue
            seen |= sh
            cand.append((url, pos, s))
    return cand


def pick(versions, budget, depth):
    """[(url, текст)] версий сюжета → {url: [предложения]} для промпта.

    Порядок источников решает вызывающий; внутри источника предложения идут в
    порядке статьи. Источник, не давший ни одного предложения, в ответе
    отсутствует — подписывать в промпте нечего.
    """
    cand = _candidates(versions, depth)
    if not cand:
        return {}

    weight = _tf_idf([c[2] for c in cand])
    # Лид новости несёт факт, дальше идут обстоятельства.
    scored = sorted(((_score(s, weight) * (1.3 if pos == 0 else 1.1 if pos == 1 else 1.0),
                      url, pos, s) for url, pos, s in cand), key=lambda x: -x[0])

    out, seen, total, used = [], set(), 0, {}
    # Круговой обход: за круг каждое издание отдаёт не больше одного
    # предложения. Иначе первое же издание съедает бюджет целиком — у турецких
    # сайтов статья кончается таблицей распределения мест, и двадцать её строк
    # набирают вес не хуже сути, после чего промпт «по шестнадцати изданиям»
    # оказывается пересказом одного. Когда у остальных предложения кончились,
    # круги продолжаются и одно издание добирает остаток: сюжет из одной версии
    # занимает весь бюджет, как и раньше.
    rounds = 0
    while True:
        rounds += 1
        added = False
        for _w, url, pos, s in scored:
            if used.get(url, 0) >= rounds or total + len(s) > budget:
                continue
            sh = _shingles(s)
            if not sh or len(sh & seen) > len(sh) * DUP:
                continue
            out.append((url, pos, s))
            seen |= sh
            total += len(s)
            used[url] = used.get(url, 0) + 1
            added = True
        if not added:
            break

    grouped = {}
    for url, _pos, s in sorted(out, key=lambda x: x[1]):
        grouped.setdefault(url, []).append(s)
    return grouped


def _selfcheck():
    lead = ("Enerji Bakanligi TENMAK Arastirma Burs Programinda kontenjani "
            "300 den 500 e cikardi.")
    echo = ("Bakanlik acikladi TENMAK Arastirma Burs Programinda kontenjani "
            "300 den 500 e yukseltti.")
    extra = ("Basvurular 1 Eylul ile 15 Eylul tarihleri arasinda alinacak ve "
             "burs aylik 13750 lira olacak.")

    # Пять перепечаток одного и того же дают одно предложение, а не пять.
    assert sum(len(v) for v in pick([("u%d" % i, lead) for i in range(5)], 6000, 6000).values()) == 1

    # Разные факты попадают оба, повтор — нет: из трёх изданий в промпте два,
    # причём издание с новым фактом обязательно.
    got = pick([("a", lead), ("b", echo), ("c", extra)], 6000, 6000)
    assert set(got) == {"c"} | ({"a"} if "a" in got else {"b"}), got
    assert len(got) == 2, got

    # Первый круг — по одному от разных изданий: издание, у которого есть свой
    # факт, попадает в промпт, даже когда обе строки соседа центральнее.
    other = ("Programa bu yil ilk kez basvuran ogrenci sayisi otuz bini asti "
             "diye duyurdu bakanlik.")
    got = pick([("a", " ".join([lead, extra])), ("b", other)], 6000, 6000)
    assert "b" in got, got

    # Бюджет соблюдается.
    got = pick([("a", " ".join([lead, echo, extra]))], len(lead) + 5, 6000)
    assert sum(len(s) for v in got.values() for s in v) <= len(lead) + 5, got

    # Таблица внутри одной статьи не раздувает вес своего слова: из двадцати
    # строк «X: 5 kontenjan» в кандидаты проходит одна-две, а не двадцать.
    table = "\n".join("Bolum %d Muhendisligi icin ayrilan kontenjan 5 kisidir." % i
                      for i in range(20))
    assert len(_candidates([("a", table)], 6000)) <= 3, _candidates([("a", table)], 6000)

    # Цифровая каша и обрывки предложениями не считаются.
    assert sentences("5 Kontenjan\nMekatronik: 5\n2026 2027 60 30", 6000) == []
    assert sentences("Kisa cumle.", 6000) == []
    # Строка таблицы и подзаголовок отсеиваются точкой, которой у них нет, —
    # но только пока в статье есть хоть одно настоящее предложение.
    table = ("Bolum ve kontenjan dagilimi soyle\n"
             "Elektrik Elektronik Muhendisligi Elektrik Muhendisligi: 80 Kontenjan\n"
             "Cevre Muhendisligi ve Iklim Bilimi Muhendisligi: 15 Kontenjan\n" + lead)
    assert sentences(table, 6000) == [lead], sentences(table, 6000)
    assert len(sentences(table.replace(lead, ""), 6000)) == 3
    # Адрес из текста снимается, сама фраза остаётся.
    s = sentences("Basvurular https://girisimci.tenmak.gov.tr/burs adresinden "
                  "eylul ayinda alinacaktir.", 6000)
    assert len(s) == 1 and "http" not in s[0], s
    # Двоеточие не отрывает реплику от того, кто её произнёс.
    s = sentences("Bakan Bayraktar: Genclerimizin gosterdigi ilgiye cok "
                  "sevindik dedi.", 6000)
    assert len(s) == 1, s
    # Глубина отсекает подвал сайта.
    assert sentences(" " * 200 + lead, 100) == []

    print("retell_select selfcheck ok")


if __name__ == "__main__":
    _selfcheck()
