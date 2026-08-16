# -*- coding: utf-8 -*-
"""bench.py — регрессионный стенд релевантности.

Зачем. До сих пор каждый замер был разовым скриптом под один вопрос: поменял
промпт судьи — узнаёшь результат через неделю и на глаз. Здесь набор
заморожен, метрики названы, база записана в bench/BASELINE.md. Любая правка
ранжирования сравнивается с ней, а не с воспоминанием.

Чего в наборе НЕТ и почему. «26 архивных недель» собрать не из чего:
cleanup_old_articles.py удаляет статьи старше семи дней, поэтому корзина
существует ровно одна — текущая. Журнал вердиктов живёт дольше (44 740
записей), но в нём нет ни текста, ни масштаба, только вердикт. Отгруженные
.docx — это выход, а не вход: по ним видно, что напечатано, и не видно, из
чего выбирали. Отсюда три источника набора, а не 26 прогонов:

    1. basket   — замороженный снимок недельной корзины (снят 16.08.2026, до
                  очередной чистки). Единственное место, где есть и статьи, и
                  эмбеддинги, и оценки судьи одновременно;
    2. issues   — отгруженные выпуски из output/*.docx. Файлов 30, выпусков 6:
                  sunday_processor стоит в cron на воскресенье, остальное —
                  проверочные прогоны будних дней (см. _weekly). Дают
                  единственную некруговую метрику: повтор сюжета от выпуска к
                  выпуску;
    3. labels   — ручная разметка владельца, bench/labels.tsv. Пока файла нет,
                  nDCG@7 не считается и НЕ подменяется суррогатом: любой
                  автоматический «эталон» здесь берётся из тех же факторов,
                  которые и оцениваются, то есть меряет сам себя.

Метрики. Всё, кроме nDCG@7, считается без разметки и без вызовов LLM:

    scale_top_share   — доля верхнего разряда масштаба (замер аудита: 0.37 при
                        требовании промпта «в десятке не больше двух»);
    top_scale_stories — сколько сюжетов корзины делят верхний масштаб. Это и
                        есть «различает почти ничего», выраженное числом;
    gap_at_cut        — отрыв последнего взятого в квоту от первого
                        отброшенного. Чем меньше, тем произвольнее граница;
    mean_importance   — средняя важность выпуска, с квотами и без;
    stale             — средний максимальный косинус отобранных сюжетов к
                        пунктам ПРЕЖНИХ выпусков. Абсолютное значение ничего
                        не значит (сравниваются тело статьи и пункт выпуска),
                        значит его сдвиг между вариантами отбора;
    repeat_share      — доля пунктов выпуска, повторяющих пункт прежнего
                        выпуска (косинус ≥ REPEAT_COS, тот же порог, которым
                        внутри выпуска ловятся дубликаты);
    mmr_overlap       — совпадение выбора MMR с чистым ранжированием;
    ndcg@7            — только при наличии bench/labels.tsv.

Запуск:
    python bench.py build      # заморозить набор (перезаписывает bench/)
    python bench.py run        # посчитать метрики, обновить bench/BASELINE.md
    python bench.py labels     # выписать 200 статей на разметку владельцу
    python bench.py selfcheck
"""
import os
import re
import sqlite3
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(BASE, "bench")
BASKET = os.path.join(BENCH, "basket.npz")
ISSUES = os.path.join(BENCH, "issues.npz")
LABELS = os.path.join(BENCH, "labels.tsv")
LABEL_TASK = os.path.join(BENCH, "label_task.tsv")
BASELINE = os.path.join(BENCH, "BASELINE.md")

# Тот же порог, которым sunday_processor_mmr ловит дубликат внутри выпуска.
# Он выбран замером по 5822 парам пунктов (99-й процентиль 0.881), и брать для
# межвыпускного повтора другое число значило бы мерить двумя линейками.
REPEAT_COS = 0.90


# ---------------------------------------------------------------------------
# 1. Заморозка набора
# ---------------------------------------------------------------------------
def _freeze_basket(conn):
    rows = conn.execute(
        "SELECT url, text, embedding, publish_date, region_bucket, entities, "
        "scale, title FROM articles WHERE embedding IS NOT NULL"
    ).fetchall()
    keep = [r for r in rows if r[2] and len(r[2]) // 4 == 384]
    return {
        "url": np.array([r[0] for r in keep], dtype=object),
        "text": np.array([r[1] or "" for r in keep], dtype=object),
        "emb": np.vstack([np.frombuffer(r[2], dtype=np.float32) for r in keep]),
        "pub_date": np.array([r[3] or "" for r in keep], dtype=object),
        "bucket": np.array([r[4] or "MIX" for r in keep], dtype=object),
        "entities": np.array([r[5] or "" for r in keep], dtype=object),
        "scale": np.array([-1 if r[6] is None else r[6] for r in keep], dtype=np.int8),
        "title": np.array([r[7] or "" for r in keep], dtype=object),
    }


# «12. 15.08.2026 — Заголовок пункта»
_ITEM_RE = re.compile(r"^(\d+)\.\s+(\d{2}\.\d{2}\.\d{4})\s+—\s+(.+)$")
_DOC_RE = re.compile(r"^digest_(\d{4}-\d{2}-\d{2})(?:-(\d+))?\.docx$")


def _parse_issue(path):
    """.docx выпуска -> [{'title':.., 'summary':.., 'urls':[..], 'date':..}]."""
    import docx
    pars = [p.text.strip() for p in docx.Document(path).paragraphs]
    items, cur = [], None
    for t in pars:
        m = _ITEM_RE.match(t)
        if m:
            if cur:
                items.append(cur)
            cur = {"date": m.group(2), "title": m.group(3), "summary": "", "urls": []}
        elif cur is not None and t.startswith("URL:"):
            cur["urls"] = [u.strip() for u in t[4:].split(";") if u.strip()]
        elif cur is not None and t and not cur["summary"]:
            cur["summary"] = t
    if cur:
        items.append(cur)
    return items


def _issue_files():
    """По одному .docx на дату: повторный прогон того же дня — не новый выпуск.

    Берётся ПОСЛЕДНИЙ файл даты (digest_2026-08-16-6.docx новее, чем
    digest_2026-08-16.docx): именно он ушёл читателю.
    """
    out = {}
    for name in os.listdir(os.path.join(BASE, "output")):
        m = _DOC_RE.match(name)
        if m:
            rank = int(m.group(2) or 0)
            date = m.group(1)
            if date not in out or rank > out[date][0]:
                out[date] = (rank, name)
    return [(d, os.path.join(BASE, "output", n)) for d, (_, n) in sorted(out.items())]


def _weekly(dates):
    """Из всех дат — только настоящие выпуски.

    sunday_processor_mmr стоит в cron на воскресенье (`0 18 * * 0`), а в
    output/ лежат ещё и проверочные прогоны будних дней: 30 файлов на 6
    выпусков. Читатель получил воскресные, и повтор считать надо по ним —
    два прогона соседних дней делят шесть дней окна из семи и совпадают почти
    целиком по построению, а не от плохого отбора.
    """
    import datetime
    days = sorted({datetime.date.fromisoformat(str(d)) for d in dates})
    if not days:
        return set()
    wd = days[-1].weekday()
    return {d.isoformat() for d in days if d.weekday() == wd}


def _freeze_issues():
    import embedder
    eng = embedder.get_engine()
    dates, titles, summaries, texts = [], [], [], []
    for date, path in _issue_files():
        for it in _parse_issue(path):
            dates.append(date)
            titles.append(it["title"])
            summaries.append(it["summary"])
            # Тот же префикс и та же склейка, что в _summary_embedding:
            # иначе архив и живой пункт окажутся в разных углах пространства.
            texts.append("passage: %s. %s" % (it["title"], it["summary"]))
    emb = eng.encode(texts, batch_size=16, convert_to_numpy=True)
    return {
        "issue_date": np.array(dates, dtype=object),
        "title": np.array(titles, dtype=object),
        "summary": np.array(summaries, dtype=object),
        "emb": np.asarray(emb, dtype=np.float32),
    }


def build():
    os.makedirs(BENCH, exist_ok=True)
    sys.path.insert(0, BASE)
    import config
    conn = sqlite3.connect(config.DB_PATH)
    try:
        b = _freeze_basket(conn)
    finally:
        conn.close()
    np.savez_compressed(BASKET, **b)
    print("basket: %d статей -> %s" % (len(b["url"]), BASKET))

    i = _freeze_issues()
    np.savez_compressed(ISSUES, **i)
    print("issues: %d пунктов в %d выпусках -> %s"
          % (len(i["title"]), len(set(i["issue_date"])), ISSUES))


def load():
    # allow_pickle нужен только под строковые колонки собственного снимка:
    # оба файла делает build() из локальной базы и лежат они в репозитории.
    # Стороннего .npz сюда не приходит.
    b = dict(np.load(BASKET, allow_pickle=True))
    i = dict(np.load(ISSUES, allow_pickle=True))
    return b, i


# ---------------------------------------------------------------------------
# 2. Метрики
# ---------------------------------------------------------------------------
def _articles(basket):
    """Снимок -> список кортежей в формате load_week_articles()."""
    sys.path.insert(0, BASE)
    import config
    import entities
    ent_df = entities.document_freq(basket["entities"])
    known = sum(1 for v in ent_df.values() if v >= config.ENTITY_MIN_DF)
    if not known:
        ent_df = None
    out = []
    for k in range(len(basket["url"])):
        prom = 0.0 if ent_df is None else entities.weight(
            entities.prominent(basket["entities"][k], ent_df, config.ENTITY_MIN_DF),
            config.ENTITY_FULL_AT)
        sc = int(basket["scale"][k])
        out.append((str(basket["url"][k]), str(basket["text"][k]),
                    basket["emb"][k], str(basket["pub_date"][k]),
                    str(basket["bucket"][k]), prom, None if sc < 0 else sc))
    return out


def _ndcg_at_k(gains, k):
    """nDCG@k по вектору выигрышей в порядке выдачи."""
    g = np.asarray(gains, dtype=np.float64)[:k]
    disc = 1.0 / np.log2(np.arange(2, len(g) + 2))
    ideal = np.sort(np.asarray(gains, dtype=np.float64))[::-1][:k]
    idcg = float((ideal * disc[:len(ideal)]).sum())
    return float((g * disc).sum() / idcg) if idcg > 0 else float("nan")


def _read_labels():
    """bench/labels.tsv -> {url: оценка 0..3}. Нет файла -> None."""
    if not os.path.exists(LABELS):
        return None
    out = {}
    with open(LABELS, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or not parts[1].strip().isdigit():
                continue
            out[parts[0].strip()] = int(parts[1])
    return out or None


def repeat_share(issues, cos=REPEAT_COS):
    """Доля пунктов, повторяющих пункт ЛЮБОГО прежнего выпуска.

    Сравнение только назад по времени: выпуск не может повторить будущий.
    Внутри одного выпуска дубликаты уже ловит сам пайплайн, поэтому пункты
    той же даты в счёт не идут — иначе метрика мерила бы чужую работу.

    Считается по настоящим выпускам (см. _weekly): проверочные прогоны будних
    дней дают повтор по построению.
    """
    E = issues["emb"]
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-10)
    dates = issues["issue_date"]
    keep = _weekly(dates)
    order = [i for i in np.argsort(dates, kind="stable") if str(dates[i]) in keep]
    sim = E @ E.T
    hits, total, examples = 0, 0, []
    for pos, k in enumerate(order):
        earlier = [j for j in order[:pos] if dates[j] < dates[k]]
        if not earlier:
            continue
        total += 1
        s = sim[k, earlier]
        j = int(np.argmax(s))
        if s[j] >= cos:
            hits += 1
            if len(examples) < 5:
                examples.append((float(s[j]), str(dates[k]),
                                 str(issues["title"][k]),
                                 str(dates[earlier[j]]),
                                 str(issues["title"][earlier[j]])))
    return (hits / total if total else float("nan")), total, examples


def _archive_emb(issues):
    """Эмбеддинги пунктов ПРЕЖНИХ выпусков — без самого свежего.

    Свежий выпуск собран из той самой корзины, которую стенд и ранжирует:
    оставить его в архиве значит сравнить отбор с его же результатом.
    """
    dates = issues["issue_date"]
    weekly = _weekly(dates)
    if not weekly:
        return np.zeros((0, issues["emb"].shape[1]), np.float32)
    last = max(weekly)
    keep = [i for i in range(len(dates))
            if str(dates[i]) in weekly and str(dates[i]) < last]
    E = issues["emb"][keep]
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-10)


def _stale(stories, issues):
    """Средний максимальный косинус отобранных сюжетов к прежним выпускам.

    Порога нет: сравниваются тело статьи и пункт выпуска — тексты разной
    природы, и абсолютное число тут ничего не значит. Значит его СДВИГ между
    вариантами отбора: чем ниже, тем меньше выпуск повторяет прошлые.
    """
    A = _archive_emb(issues)
    if not len(stories) or not len(A):
        return float("nan")
    S = np.vstack([a[2] for a in stories]).astype(np.float32)
    S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-10)
    return float((S @ A.T).max(axis=1).mean())


def measure(basket, issues, quota=None, lam=None, labels=None):
    """Все метрики стенда одним проходом. Возвращает плоский dict."""
    sys.path.insert(0, BASE)
    import sunday_processor_mmr as P

    quota = dict(P.QUOTA if quota is None else quota)
    lam = P.MMR_LAMBDA if lam is None else lam
    arts = _articles(basket)
    labels = _read_labels() if labels is None else labels

    res = {"buckets": {}, "n_articles": len(arts)}
    sc_all = [a[6] for a in arts if a[6] is not None]
    res["scale_top_share"] = (sum(1 for s in sc_all if s == 3) / len(sc_all)
                              if sc_all else float("nan"))

    picked_all, imp_all = [], []
    for bucket, q in quota.items():
        sub = [a for a in arts if (a[4] if a[4] in quota else "MIX") == bucket]
        if not sub:
            continue
        stories = P._group_stories(sub)
        E = np.vstack([a[2] for a in stories]).astype(np.float32)
        imp = P._importance(stories, E)
        order = np.argsort(-imp)
        pure = [stories[i] for i in order[:q]]
        mmr = P._mmr_pick(stories, q, lam=lam)

        # Разрешение на срезе: чем меньше отрыв последнего взятого от первого
        # отброшенного, тем произвольнее граница. Порога тут нет намеренно —
        # число сравнивается с базовой линией, а не с выдуманной константой.
        cut_i = min(q, len(order)) - 1
        gap = (float(imp[order[cut_i]] - imp[order[cut_i + 1]])
               if cut_i + 1 < len(order) else float("nan"))
        raw_scale = [P._story_scale(a) for a in stories]
        top_sc = max((s for s in raw_scale if s is not None), default=None)

        b = {
            "n_articles": len(sub), "n_stories": len(stories), "quota": q,
            "mean_importance": float(imp[order[:q]].mean()),
            "gap_at_cut": gap,
            "top_scale_stories": (sum(1 for s in raw_scale if s == top_sc)
                                  if top_sc is not None else 0),
            "stale": _stale(pure, issues),
            "mmr_overlap": len({a[0] for a in pure} & {a[0] for a in mmr}),
        }
        if labels:
            gains = [labels.get(stories[i][0], 0) for i in order]
            if any(g for g in gains):
                b["ndcg@7"] = _ndcg_at_k(gains, 7)
        res["buckets"][bucket] = b
        picked_all.extend(pure)
        imp_all.append((imp, order, q, stories))

    if picked_all:
        # Средняя важность выпуска считается по важности внутри своей корзины:
        # это то же число, которым отбор и пользуется.
        vals = [float(imp[i]) for imp, order, q, _st in imp_all for i in order[:q]]
        res["mean_importance"] = float(np.mean(vals))
        free = np.concatenate([imp for imp, _o, _q, _s in imp_all])
        n_slots = sum(q for _i, _o, q, _s in imp_all)
        res["mean_importance_noquota"] = float(np.sort(free)[::-1][:n_slots].mean())
        res["quota_cost"] = res["mean_importance_noquota"] - res["mean_importance"]

    rs, n_cmp, ex = repeat_share(issues)
    res["repeat_share"] = rs
    res["repeat_compared"] = n_cmp
    res["repeat_examples"] = ex
    res["n_issues"] = len(_weekly(issues["issue_date"]))
    res["n_issue_items"] = len(issues["title"])
    res["labels"] = len(labels) if labels else 0
    return res


def _fmt(res):
    L = []
    L.append("корзина: %d статей | выпусков: %d (%d пунктов) | ручных меток: %d"
             % (res["n_articles"], res["n_issues"], res["n_issue_items"],
                res["labels"]))
    L.append("scale_top_share = %.3f   (доля троек среди оценённых статей)"
             % res["scale_top_share"])
    L.append("mean_importance = %.4f  без квот %.4f  цена квоты %.4f"
             % (res["mean_importance"], res["mean_importance_noquota"],
                res["quota_cost"]))
    L.append("repeat_share    = %.3f   (%d пунктов сравнивались с прошлыми)"
             % (res["repeat_share"], res["repeat_compared"]))
    L.append("")
    L.append("| корзина | статей | сюжетов | квота | важность | отрыв на срезе |"
             " сюжетов с верхним масштабом | близость к прежним | MMR∩важность |"
             " nDCG@7 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for k, b in res["buckets"].items():
        L.append("| %s | %d | %d | %d | %.4f | %.4f | %d | %.3f | %d/%d | %s |"
                 % (k, b["n_articles"], b["n_stories"], b["quota"],
                    b["mean_importance"], b["gap_at_cut"],
                    b["top_scale_stories"], b["stale"],
                    b["mmr_overlap"], b["quota"],
                    ("%.3f" % b["ndcg@7"]) if "ndcg@7" in b else "—"))
    if res["repeat_examples"]:
        L.append("")
        L.append("Повторы (косинус, выпуск, пункт ← прежний выпуск, пункт):")
        for s, d1, t1, d2, t2 in res["repeat_examples"]:
            L.append("- %.3f  %s «%.60s» ← %s «%.60s»" % (s, d1, t1, d2, t2))
    return "\n".join(L)


def run():
    b, i = load()
    res = measure(b, i)
    text = _fmt(res)
    print(text)
    import datetime
    with open(BASELINE, "w", encoding="utf-8") as fh:
        fh.write("# Базовая линия релевантности\n\n"
                 "Снято %s командой `python bench.py run`.\n"
                 "Набор заморожен в `bench/basket.npz` и `bench/issues.npz`;\n"
                 "что в нём есть и чего нет — в шапке `bench.py`.\n\n"
                 % datetime.date.today().strftime("%d.%m.%Y"))
        fh.write(text + "\n")
    print("\n-> %s" % BASELINE)


# ---------------------------------------------------------------------------
# 3. Задание на ручную разметку
# ---------------------------------------------------------------------------
def labels_task(n=200):
    """200 статей владельцу на разметку — не случайных, а трудных.

    Случайная выборка из корзины уйдёт в очевидное: 3/4 статей отбор не
    рассматривает всерьёз, и метки на них не отличат ни один вариант
    ранжирования от другого. Берём окрестность решения: сюжеты, стоящие в
    своей корзине вокруг квотного среза, плюс верхушку и явный низ для
    привязки шкалы.
    """
    sys.path.insert(0, BASE)
    import sunday_processor_mmr as P
    basket, _ = load()
    arts = _articles(basket)
    title_of = dict(zip((str(u) for u in basket["url"]),
                        (str(t) for t in basket["title"])))
    rows = []
    for bucket, q in P.QUOTA.items():
        sub = [a for a in arts if (a[4] if a[4] in P.QUOTA else "MIX") == bucket]
        if not sub:
            continue
        stories = P._group_stories(sub)
        E = np.vstack([a[2] for a in stories]).astype(np.float32)
        order = np.argsort(-P._importance(stories, E))
        share = n // len(P.QUOTA)
        # Окно вокруг среза: треть выше квоты (чтобы шкала имела верх), две
        # трети ниже — там и стоит вопрос, кого пускать.
        lo = max(0, min(q - share // 3, len(order) - share))
        for i in order[lo:lo + share]:
            a = stories[i]
            head = "%s | %s" % (title_of.get(a[0], ""), str(a[1] or "")[:260])
            rows.append((a[0], bucket,
                         head.replace("\t", " ").replace("\n", " ")))
    os.makedirs(BENCH, exist_ok=True)
    with open(LABEL_TASK, "w", encoding="utf-8") as fh:
        fh.write("# url\tоценка\tкорзина\tначало текста\n")
        fh.write("# оценка: 3 — обязан быть в выпуске, 2 — уместен,\n")
        fh.write("#         1 — слабо, 0 — не наша тема или рутина\n")
        fh.write("# заполнить второй столбец и сохранить как bench/labels.tsv\n")
        for url, bucket, head in rows:
            fh.write("%s\t\t%s\t%s\n" % (url, bucket, head))
    print("%d строк -> %s" % (len(rows), LABEL_TASK))


# ---------------------------------------------------------------------------
def _selfcheck():
    assert abs(_ndcg_at_k([3, 2, 1], 3) - 1.0) < 1e-9
    assert _ndcg_at_k([1, 2, 3], 3) < 1.0
    assert _ndcg_at_k([0, 0, 0], 3) != _ndcg_at_k([0, 0, 0], 3) or True  # nan
    assert np.isnan(_ndcg_at_k([0, 0], 2))

    m = _ITEM_RE.match("12. 15.08.2026 — Заголовок")
    assert m and m.group(2) == "15.08.2026" and m.group(3) == "Заголовок"
    assert not _ITEM_RE.match("URL: https://x")

    # Повтор ловится только назад по времени и только между разными выпусками.
    v = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], np.float32)
    iss = {"emb": v, "issue_date": np.array(["2026-01-04", "2026-01-11",
                                             "2026-01-18"], dtype=object),
           "title": np.array(["a", "a2", "b"], dtype=object)}
    share, total, _ex = repeat_share(iss)
    assert total == 2 and abs(share - 0.5) < 1e-9, (share, total)

    # Проверочные прогоны будних дней в счёт не идут.
    iss2 = dict(iss, issue_date=np.array(["2026-01-04", "2026-01-05",
                                          "2026-01-18"], dtype=object))
    _s2, total2, _e2 = repeat_share(iss2)
    assert total2 == 1, total2
    # Свежий выпуск из архива исключён: остаются два прежних из трёх.
    assert len(_archive_emb(iss)) == 2
    print("bench: ok")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"build": build, "run": run, "labels": labels_task,
     "selfcheck": _selfcheck}[cmd]()
