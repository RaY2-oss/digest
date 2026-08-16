# -*- coding: utf-8 -*-
"""issue_archive.py — что уже было напечатано в прошлых выпусках.

Отдельный модуль, потому что этим знанием пользуются трое и по-разному:
отбор (фактор новизны в _importance), стенд (метрики repeat_share и stale) и
глазами человек. Держать разбор .docx в каждом из них — три расходящиеся
копии одного регулярного выражения.

Единица здесь — ВЫПУСК, а не файл. sunday_processor_mmr стоит в cron на
воскресенье (`0 18 * * 0`), а в output/ копятся ещё и проверочные прогоны
будних дней: 30 файлов на 6 выпусков. Два прогона соседних дней делят шесть
дней окна из семи и совпадают почти целиком по построению — считать их
разными выпусками значит мерить не отбор, а календарь.

Эмбеддинги считаются один раз и лежат в data/issue_archive.npz. Кэш
пересобирается, когда меняется набор файлов или их время правки: сотня
пунктов на прогоне отбора — это лишняя минута на CPU без AVX.
"""
import datetime
import os
import re

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(BASE, "output")
CACHE = os.path.join(BASE, "data", "issue_archive.npz")

# «12. 15.08.2026 — Заголовок пункта»
_ITEM_RE = re.compile(r"^(\d+)\.\s+(\d{2}\.\d{2}\.\d{4})\s+—\s+(.+)$")
_DOC_RE = re.compile(r"^digest_(\d{4}-\d{2}-\d{2})(?:-(\d+))?\.docx$")


def parse_issue(path):
    """.docx выпуска -> [{'title':.., 'summary':.., 'urls':[..], 'date':..}]."""
    import docx
    items, cur = [], None
    for p in docx.Document(path).paragraphs:
        t = p.text.strip()
        m = _ITEM_RE.match(t)
        if m:
            if cur:
                items.append(cur)
            cur = {"date": m.group(2), "title": m.group(3), "summary": "",
                   "urls": []}
        elif cur is not None and t.startswith("URL:"):
            cur["urls"] = [u.strip() for u in t[4:].split(";") if u.strip()]
        elif cur is not None and t and not cur["summary"]:
            cur["summary"] = t
    if cur:
        items.append(cur)
    return items


def files(output_dir=OUTPUT):
    """[(дата, путь)] — по одному файлу на дату, последний прогон дня.

    digest_2026-08-16-6.docx новее, чем digest_2026-08-16.docx: суффикс
    ставится при повторной сборке в тот же день, и читателю ушёл последний.
    """
    out = {}
    for name in sorted(os.listdir(output_dir)):
        m = _DOC_RE.match(name)
        if m:
            rank, date = int(m.group(2) or 0), m.group(1)
            if date not in out or rank > out[date][0]:
                out[date] = (rank, name)
    return [(d, os.path.join(output_dir, n)) for d, (_, n) in sorted(out.items())]


def weekly(dates):
    """Множество дат настоящих выпусков: тот же день недели, что у последнего."""
    days = sorted({datetime.date.fromisoformat(str(d)) for d in dates})
    if not days:
        return set()
    wd = days[-1].weekday()
    return {d.isoformat() for d in days if d.weekday() == wd}


def _stamp(path):
    return "%s:%d" % (os.path.basename(path), os.path.getmtime(path))


def load(only_weekly=True, before=None, output_dir=OUTPUT, cache=CACHE):
    """-> dict с ключами issue_date, title, summary, emb (нормированные строки).

    only_weekly — отбросить проверочные прогоны.
    before      — оставить строго более ранние выпуски. Нужен отбору: свежий
                  выпуск собран из той же корзины, и оставить его значит
                  сравнить отбор с его же результатом.
    """
    pairs = files(output_dir)
    old = None
    if os.path.exists(cache):
        try:
            # allow_pickle — под строковые колонки СОБСТВЕННОГО кэша, который
            # пишет _build() строкой ниже. Чужой .npz сюда не попадает.
            old = dict(np.load(cache, allow_pickle=True))
        except Exception:  # noqa: BLE001
            old = None
    data = _build(pairs, old)
    if data is not old:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, **data)

    dates = data["issue_date"]
    # Сначала отсечь по дате, потом искать день недели: weekly() опирается на
    # ПОСЛЕДНЮЮ дату, и проверочный прогон в понедельник иначе объявил бы
    # понедельник днём выпуска и выбросил все воскресные.
    past = [str(d) for d in dates if before is None or str(d) < str(before)]
    keep = weekly(past) if only_weekly else set(past)
    idx = [i for i in range(len(dates)) if str(dates[i]) in keep]
    E = data["emb"][idx] if idx else np.zeros((0, data["emb"].shape[1]),
                                              np.float32)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-10) if len(E) else E
    return {"issue_date": dates[idx], "title": data["title"][idx],
            "summary": data["summary"][idx], "emb": E}


def _build(pairs, old):
    """Досчитать эмбеддинги только новых файлов, старые взять из кэша.

    Полная пересборка стоит семь минут на этом процессоре (586 пунктов, e5 на
    onnxruntime без AVX), а каждый прогон добавляет к архиву один файл. Ключ —
    имя файла и время правки: пересобранный в тот же день выпуск считается
    новым, потому что содержимое у него другое.
    """
    have = {}
    if old is not None and "stamp" in old:
        for k, st in enumerate(old["stamp"]):
            have.setdefault(str(st), []).append(k)

    cols = {"stamp": [], "issue_date": [], "title": [], "summary": []}
    keep_rows, new_texts, new_rows = [], [], []
    for date, path in pairs:
        st = _stamp(path)
        if st in have:
            for k in have[st]:
                keep_rows.append(k)
                cols["stamp"].append(st)
                cols["issue_date"].append(str(old["issue_date"][k]))
                cols["title"].append(str(old["title"][k]))
                cols["summary"].append(str(old["summary"][k]))
                new_rows.append(None)
            continue
        for it in parse_issue(path):
            cols["stamp"].append(st)
            cols["issue_date"].append(date)
            cols["title"].append(it["title"])
            cols["summary"].append(it["summary"])
            keep_rows.append(None)
            # Тот же префикс и та же склейка, что в
            # sunday_processor_mmr._summary_embedding: e5 без префикса живёт в
            # другом углу пространства, и мешать две системы координат нельзя.
            new_texts.append("passage: %s. %s" % (it["title"], it["summary"]))
            new_rows.append(len(new_texts) - 1)

    if not new_texts and old is not None and len(keep_rows) == len(old["stamp"]):
        return old
    if new_texts:
        import embedder
        fresh = np.asarray(embedder.get_engine().encode(new_texts, batch_size=16),
                           np.float32)
    dim = old["emb"].shape[1] if old is not None and len(old["emb"]) else 384
    emb = np.zeros((len(cols["stamp"]), dim), np.float32)
    for row, (k, j) in enumerate(zip(keep_rows, new_rows)):
        emb[row] = old["emb"][k] if k is not None else fresh[j]
    return {"stamp": np.array(cols["stamp"], object),
            "issue_date": np.array(cols["issue_date"], object),
            "title": np.array(cols["title"], object),
            "summary": np.array(cols["summary"], object),
            "emb": emb}


# Ниже этого косинуса к прежним выпускам сюжет считается новым, и фактор о нём
# молчит. Порог измерен по корзине 09.08–16.08 (1172 статьи против 99 пунктов
# пяти выпусков): медиана близости 0.820, 90-й процентиль 0.857, 95-й — 0.878,
# 99-й — 0.910.
#
# Ставить его по хвосту (0.90, 20 статей) оказалось нельзя, и это главный урок
# замера: у одного повторённого события ВОСЕМЬ версий, и они размазаны по
# 0.888..0.963. Порог 0.90 штрафовал верхние семь, а восьмая, ровно тот же
# сюжет по-армянски, оставалась ниже порога и въезжала в выпуск на
# освободившееся место. Порог обязан лежать НИЖЕ самой слабой версии кластера,
# иначе фактор не убирает повтор, а тасует его версии.
#
# 0.88 — 54 статьи из 1172 (4.6%), всё ещё хвост, и весь кластер Firebird
# внутри. Ниже начинается не хвост, а наклон всей шкалы: 0.86 даёт 101 статью,
# 0.84 — уже 284.
NOVELTY_FLOOR = 0.88


def novelty(embeddings, archive_emb, floor=None):
    """Новизна относительно прежних выпусков: 1 = не было, 0 = точный повтор.

    Не «1 − косинус». Косинус тела статьи к пункту выпуска у всей корзины
    лежит между 0.78 и 0.96, и линейный фактор из него двигал бы важность у
    ВСЕХ сюжетов сразу, ничего не различая: перебор веса (bench.py sweep)
    показал ноль изменений в отборе при весах до 0.10 и порчу выбора при 0.30.
    Поэтому фактор нейтрален (1.0) ниже floor и линейно падает до нуля к
    полному совпадению — он трогает хвост, а не всю шкалу.

    Пустой архив -> единицы: первый выпуск ничего не повторяет по определению,
    и фактор в нём обязан быть нейтральным, а не нулевым (ноль опустил бы
    ВСЕ сюжеты одинаково, то есть просто сдвинул шкалу).
    """
    floor = NOVELTY_FLOOR if floor is None else floor
    E = np.asarray(embeddings, np.float32)
    if not len(E):
        return np.zeros(0, np.float64)
    if archive_emb is None or not len(archive_emb):
        return np.ones(len(E), np.float64)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-10)
    top = (E @ archive_emb.T).max(axis=1)
    return np.clip((1.0 - top) / max(1.0 - floor, 1e-9), 0.0, 1.0).astype(
        np.float64)


def _selfcheck():
    assert _ITEM_RE.match("12. 15.08.2026 — Заголовок").group(3) == "Заголовок"
    assert not _ITEM_RE.match("URL: https://x")
    assert _DOC_RE.match("digest_2026-08-16-6.docx").groups() == ("2026-08-16", "6")

    # Только настоящие выпуски: у последней даты воскресенье, будни отпадают.
    assert weekly(["2026-08-02", "2026-08-05", "2026-08-09", "2026-08-16"]) == {
        "2026-08-02", "2026-08-09", "2026-08-16"}
    assert weekly([]) == set()

    a = np.array([[1.0, 0.0], [0.0, 1.0]], np.float32)
    n = novelty(np.array([[1.0, 0.0], [0.7, 0.7]], np.float32), a)
    assert abs(n[0]) < 1e-6, n            # точный повтор -> новизна 0
    assert n[1] == 1.0, n                 # 0.707 ниже порога -> фактор молчит
    # Ровно на пороге фактор ещё нейтрален, за ним падает линейно. Порог
    # задан явно: тест не должен переезжать вместе с настройкой.
    half = np.array([[0.95, np.sqrt(1 - 0.95 ** 2)]], np.float32)
    assert abs(novelty(half, a, floor=0.90)[0] - 0.5) < 0.02
    at = np.array([[0.89, np.sqrt(1 - 0.89 ** 2)]], np.float32)
    assert novelty(at, a, floor=0.90)[0] == 1.0
    assert np.allclose(novelty(a, np.zeros((0, 2), np.float32)), [1.0, 1.0])
    assert len(novelty(np.zeros((0, 2), np.float32), a)) == 0

    # Кэш покрывает все файлы -> ни одного обращения к эмбеддеру. Проверяется
    # тождеством объекта: пересборка вернула бы новый dict.
    pairs = files()
    if pairs:
        stamps = [_stamp(p) for _d, p in pairs]
        old = {"stamp": np.array(stamps, object),
               "issue_date": np.array([d for d, _p in pairs], object),
               "title": np.array([""] * len(pairs), object),
               "summary": np.array([""] * len(pairs), object),
               "emb": np.zeros((len(pairs), 384), np.float32)}
        assert _build(pairs, old) is old
    print("issue_archive: ok")


if __name__ == "__main__":
    _selfcheck()
