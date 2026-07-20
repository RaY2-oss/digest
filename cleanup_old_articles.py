# -*- coding: utf-8 -*-
"""
cleanup_old_articles.py — ежедневная независимая чистка БД от статей старше
недели.

Зачем отдельный скрипт, а не только cleanup_old() в sunday_processor_mmr.py:
там очистка выполняется ПОСЛЕДНИМ шагом, после успешной отправки дайджеста —
если формирование дайджеста упадёт или выборка окажется пустой, БД вообще не
почистится. Здесь — самостоятельный cron, не зависящий от остального пайплайна.

Запускается по cron раз в сутки (см. crontab), вне окон daily_collector.py
(00/06/12/18) и sunday_processor_mmr.py (вс 08:00), чтобы не конкурировать
за БД.
"""
import sqlite3
import sys

import config


def main():
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    try:
        cur = conn.execute("DELETE FROM articles WHERE fetch_date < date('now', '-7 days')")
        deleted = cur.rowcount
        conn.commit()
        conn.execute("VACUUM")
        print(f"[cleanup_old_articles] Удалено строк старше 7 дней: {deleted}")
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
