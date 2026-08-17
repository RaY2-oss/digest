# -*- coding: utf-8 -*-
"""
init_db.py — одноразовая инициализация базы данных SQLite.

Запускать один раз перед первым стартом коллектора:
    /opt/digest/venv/bin/python /opt/digest/init_db.py

Создаёт таблицу articles. Повторный запуск безопасен
(CREATE TABLE IF NOT EXISTS ничего не портит).

Схема таблицы articles:
    id           INTEGER PRIMARY KEY  — автоинкремент;
    query_index  INTEGER              — из какого запроса (0 / 1 / 2);
    url          TEXT UNIQUE          — ссылка (ключ дедупликации / кэша);
    fetch_date   DATE                 — дата загрузки (YYYY-MM-DD);
    title        TEXT                 — заголовок из GDELT;
    text         TEXT                 — полный текст статьи (trafilatura);
    embedding    BLOB                 — numpy float32 вектор (384 dim) в bytes;
    entities     TEXT                 — персоны/организации из GKG (entities.py);
    scale        INTEGER              — 1..5, национальный масштаб события от
                                        судьи-LLM (NULL = не оценивался).
"""

import os
import sqlite3

import config
import seen_store

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS articles (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    query_index   INTEGER NOT NULL,
    url           TEXT UNIQUE NOT NULL,
    fetch_date    DATE NOT NULL,
    publish_date  DATE,
    title         TEXT,
    text          TEXT,
    embedding     BLOB,
    region_bucket TEXT,
    language      TEXT,
    entities      TEXT,
    scale         INTEGER
);

-- Индекс по дате ускоряет выборку "за последние 7 дней" и очистку старых строк.
CREATE INDEX IF NOT EXISTS idx_articles_fetch_date ON articles(fetch_date);
"""


MIGRATIONS = [
    ("language", "TEXT"),
    ("entities", "TEXT"),   # субъекты из GKG V1Persons/V1Organizations, см. entities.py
    ("scale", "INTEGER"),   # 1..5 — национальный масштаб от судьи, см. daily_collector
]


def migrate(conn):
    """Добавляет отсутствующие колонки в уже существующую таблицу articles."""
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(articles)")}
    for col_name, col_type in MIGRATIONS:
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE articles ADD COLUMN {col_name} {col_type}")
            print(f"[MIGRATE] Добавлена колонка: {col_name} {col_type}")


def main():
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    try:
        conn.executescript(SCHEMA_SQL)
        migrate(conn)
        seen_store.ensure(conn)
        conn.commit()
        print(f"[OK] База инициализирована: {config.DB_PATH}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
