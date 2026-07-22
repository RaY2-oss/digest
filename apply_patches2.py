# -*- coding: utf-8 -*-
"""apply_patches2.py — докстринг пайплайна и удаление осиротевшего кода."""
import io
import os
import sys

BASE = "/opt/digest"
errors = []


def rd(n):
    with io.open(os.path.join(BASE, n), encoding="utf-8") as f:
        return f.read()


def wr(n, t):
    with io.open(os.path.join(BASE, n), "w", encoding="utf-8") as f:
        f.write(t)


def sub(src, old, new, tag):
    if src.count(old) != 1:
        errors.append("%s: вхождений %d, ожидалось 1" % (tag, src.count(old)))
        return src
    return src.replace(old, new)


OLD_DOC = '''    1. fetch_gdelt()      — сканирует 15-минутные дампы GKG (EN + переводной
                             поток), фильтрует по themes/locations, отдаёт
                             набор URL-кандидатов с датой из GKG.
    2. fetch_and_extract() — скачивает страницу, извлекает текст/заголовок
                             (trafilatura) и дату публикации (htmldate).
    3. is_non_event_article() — быстрый regex-отсев не-новостных материалов
                             (интервью/колонки/мнения на нескольких языках).
    4. llm_filter_parallel()  — строгая тематическая релевантность через LLM:
                             тема должна быть ГЛАВНЫМ предметом статьи (не
                             случайным упоминанием) и новость — национального
                             или крупного масштаба, не местячковой (батчами,
                             параллельно по нескольким моделям).
    5. classify_region_parallel() — региональная метка (TR/CA/SC/MIX) через LLM.
    6. flush_batch()       — считает embedding (sentence-transformers) и
                             пишет батч в БД.
'''

NEW_DOC = '''    0. seen_store          — очередь pending разбирается ДО новых дампов, а
                             URL с окончательным вердиктом (accepted/rejected/
                             short/non_event) повторно не скачивается. Раньше
                             проверка шла по таблице articles, где лежат только
                             ПРИНЯТЫЕ статьи, и 61.7% работы прогона уходило
                             на повторную обработку уже отклонённых URL.
    1. fetch_gdelt()      — сканирует 15-минутные дампы GKG (EN + переводной
                             поток); gkg_filter отбирает строки по СВЯЗИ темы
                             и страны через символьные офсеты полей V2.
    2. fetch_and_extract() — скачивает страницу, извлекает текст/заголовок
                             (trafilatura) и дату публикации (htmldate).
    3. is_non_event_article() — быстрый regex-отсев не-новостных материалов
                             (интервью/колонки/мнения на нескольких языках).
    4. embed_texts()       — embedding считается ОДИН раз на кандидата и
                             переиспользуется на стадиях 5, 6 и 8.
    5. cluster_duplicates() — синдикация одного сюжета схлопывается в кластер,
                             в LLM уходит один представитель; в БД пишутся все.
    6. prefilter           — локальный классификатор-дистиллят отбрасывает
                             заведомый мусор без запроса к LLM (если обучен).
    7. judge_parallel()    — ОДИН LLM-вызов отдаёт и релевантность, и регион
                             (TR/CA/SC/MIX). Нет ответа -> pending, не отказ.
    8. flush_batch()       — пишет принятые статьи с готовым embedding в БД.
'''

OLD_FN = '''def url_exists(conn, url):
    return conn.execute("SELECT 1 FROM articles WHERE url = ? LIMIT 1", (url,)).fetchone() is not None


'''

dc = rd("daily_collector.py")
dc = sub(dc, OLD_DOC, NEW_DOC, "DC/docstring")
dc = sub(dc, OLD_FN, "", "DC/url_exists")

if errors:
    print("НЕ ПРИМЕНЕНО:")
    for e in errors:
        print("  -", e)
    sys.exit(1)

wr("daily_collector.py", dc)
print("Патч 2 применён")
