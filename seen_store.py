# -*- coding: utf-8 -*-
"""seen_store.py — журнал уже обработанных URL.

Зачем. url_exists() смотрел в таблицу articles, куда попадают ТОЛЬКО принятые
статьи. Отклонённая нигде не отмечалась, поэтому тот же URL возвращался в
24-часовом окне GKG и заново скачивался и заново уходил в LLM. Замер по логу:
51380 Accept-событий на 19662 уникальных URL — 61.7% работы дублировалось.

Вердикты:
    accepted   — прошла LLM, лежит в articles;
    rejected   — LLM явно ответила "нет";
    short      — текст короче MIN_TEXT_LENGTH или не извлёкся;
    non_event  — отсеяна regex-правилом (интервью/колонка/аналитика);
    pending    — решения НЕТ: провайдеры молчали или ответ не разобрался.

Окончательны первые четыре. pending — это очередь, которую следующий прогон
разбирает ДО сканирования новых дампов. Именно поэтому исчерпанный лимит или
обрыв сети не превращают статью в вечный чёрный список: раньше такой сбой
молча помечал батч как отклонённый (прогон week_swap 21.07: 6006 кандидатов,
0 принятых), и статья спасалась только тем, что URL возвращался в окне GKG —
то есть ровно за счёт той переобработки, которую эта таблица убирает.

Эмбеддинг сохраняется для accepted/rejected — это готовая размеченная
выборка для train_prefilter.py. Для pending не сохраняем: текст всё равно
не хранится, следующий прогон скачает и посчитает заново.
"""
import sqlite3

FINAL = ("accepted", "rejected", "short", "non_event")

DDL = """
CREATE TABLE IF NOT EXISTS seen_urls (
    url         TEXT PRIMARY KEY,
    query_index INTEGER,
    first_seen  TEXT,
    last_seen   TEXT,
    verdict     TEXT NOT NULL,
    attempts    INTEGER NOT NULL DEFAULT 1,
    embedding   BLOB
);
CREATE INDEX IF NOT EXISTS idx_seen_verdict ON seen_urls(verdict, query_index);
CREATE INDEX IF NOT EXISTS idx_seen_lastseen ON seen_urls(last_seen);
"""

# ponytail: чанк по 500 — упор в лимит SQLITE_MAX_VARIABLE_NUMBER (999),
# без вычисления точного лимита рантайма; запас двукратный.
_CHUNK = 500


def ensure(conn) -> None:
    conn.executescript(DDL)
    conn.commit()


def final_urls(conn, urls) -> set:
    """Подмножество urls, по которым решение уже принято окончательно."""
    urls = list(urls)
    out = set()
    for i in range(0, len(urls), _CHUNK):
        chunk = urls[i:i + _CHUNK]
        q = ("SELECT url FROM seen_urls WHERE verdict IN (%s) AND url IN (%s)"
             % (",".join("?" * len(FINAL)), ",".join("?" * len(chunk))))
        out.update(r[0] for r in conn.execute(q, FINAL + tuple(chunk)))
    return out


def pending_urls(conn, query_index) -> list:
    """Очередь нерешённых, самые старые и наименее пробованные — первыми."""
    return [r[0] for r in conn.execute(
        "SELECT url FROM seen_urls WHERE verdict='pending' AND query_index=? "
        "ORDER BY attempts ASC, first_seen ASC", (query_index,))]


def mark(conn, rows, day) -> None:
    """rows: iterable из (url, query_index, verdict, embedding|None)."""
    rows = list(rows)
    if not rows:
        return
    conn.executemany(
        "INSERT INTO seen_urls (url,query_index,first_seen,last_seen,verdict,attempts,embedding) "
        "VALUES (?,?,?,?,?,1,?) "
        "ON CONFLICT(url) DO UPDATE SET "
        "  last_seen = excluded.last_seen, "
        "  verdict   = excluded.verdict, "
        "  attempts  = seen_urls.attempts + 1, "
        "  embedding = COALESCE(excluded.embedding, seen_urls.embedding)",
        [(u, qi, day, day, v, (e.tobytes() if e is not None else None))
         for u, qi, v, e in rows])
    conn.commit()


def prune(conn, keep_days) -> int:
    """Чистка журнала. Строки с эмбеддингом — обучающая выборка, поэтому
    удаляются только служебные вердикты без вектора."""
    cur = conn.execute(
        "DELETE FROM seen_urls WHERE embedding IS NULL "
        "AND last_seen < date('now', ?)", (f"-{int(keep_days)} days",))
    conn.commit()
    return cur.rowcount


def stats(conn) -> dict:
    return dict(conn.execute("SELECT verdict, COUNT(*) FROM seen_urls GROUP BY 1"))


def label_counts(conn) -> tuple:
    """(положительных, отрицательных) с сохранённым эмбеддингом."""
    row = conn.execute(
        "SELECT SUM(verdict='accepted'), SUM(verdict='rejected') FROM seen_urls "
        "WHERE embedding IS NOT NULL").fetchone()
    return (row[0] or 0, row[1] or 0)
