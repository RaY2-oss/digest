#!/bin/bash
# week_swap_run.sh — пересборка базы за неделю в ОТДЕЛЬНЫЙ файл с подменой
# боевой базы только после успешного прогона.
#
# Почему так. Прошлый прогон 23.07 умер в 00:48 сразу после загрузки модели:
# он стартовал в 00:01, через минуту после cron-прогона, и два процесса с
# эмбеддинг-моделью (~1.1 ГБ каждый) не помещались в 3.9 ГБ при уже занятом
# свопе. Плюс сам процесс висел на сессии SSH. Здесь оба источника сбоя сняты:
# cron на время сборки отключается и восстанавливается в любом случае (trap),
# а запускать скрипт нужно через setsid, чтобы он пережил разрыв соединения.
set -u

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$BASE/venv/bin/python"
LIVE_DB="$($PY -c "import sys; sys.path.insert(0,'$BASE'); import config; print(config.DB_PATH)")"
NEW_DB="${LIVE_DB%.db}_week.db"
STAMP="$(date +%Y%m%d_%H%M%S)"
CRON_BACKUP="$BASE/logs/crontab_before_week_$STAMP.txt"

say() { echo "[$(date '+%F %T')] $*"; }

# --- cron восстанавливается при любом исходе, включая падение и Ctrl-C ---
restore_cron() {
    if [ -f "$CRON_BACKUP" ]; then
        crontab "$CRON_BACKUP" && say "cron восстановлен из $CRON_BACKUP"
    fi
}
trap restore_cron EXIT INT TERM

say "=== Недельная пересборка ==="
say "боевая база : $LIVE_DB"
say "новая база  : $NEW_DB"

crontab -l > "$CRON_BACKUP" 2>/dev/null || : > "$CRON_BACKUP"
crontab -l 2>/dev/null | sed 's#^\([^#].*daily_collector\.py\)#\#WEEKSWAP \1#' | crontab -
say "cron daily_collector приостановлен на время сборки"

rm -f "$NEW_DB"
DIGEST_DB_PATH="$NEW_DB" "$PY" "$BASE/init_db.py" || { say "init_db упал"; exit 1; }

say "старт коллектора в недельном режиме (672 тика x 2 потока x 2 запроса)"
DIGEST_DB_PATH="$NEW_DB" "$PY" "$BASE/daily_collector.py" --week
RC=$?
say "коллектор завершился с кодом $RC"
[ "$RC" -ne 0 ] && { say "ОТМЕНА: ненулевой код возврата, боевая база не тронута"; exit 1; }

# --- Порог приёмки выводится из самой боевой базы, а не задан числом:
#     недельная сборка обязана дать не меньше, чем лучший день боевой. ---
read -r NEW_N GATE < <($PY - "$NEW_DB" "$LIVE_DB" << 'PYEOF'
import sqlite3, sys
def q(path, sql, default=0):
    try:
        conn = sqlite3.connect(path)
        v = conn.execute(sql).fetchone()[0]
        conn.close()
        return v or default
    except Exception:
        return default
new_n = q(sys.argv[1], "SELECT COUNT(*) FROM articles")
gate  = q(sys.argv[2], "SELECT COALESCE(MAX(c),0) FROM (SELECT COUNT(*) c FROM articles GROUP BY fetch_date)")
print(new_n, gate)
PYEOF
)
say "статей в новой базе: $NEW_N | порог приёмки (лучший день боевой): $GATE"
if [ "$NEW_N" -le 0 ] || [ "$NEW_N" -lt "$GATE" ]; then
    say "ОТМЕНА: новая база не прошла порог. Файл оставлен для разбора: $NEW_DB"
    exit 1
fi

# --- Журнал вердиктов боевой базы переносится в новую: это накопленная
#     разметка для предфильтра. Вердикты свежего прогона приоритетнее,
#     поэтому OR IGNORE — дописываются только отсутствующие URL. ---
DIGEST_DB_PATH="$NEW_DB" "$PY" - "$LIVE_DB" << 'PYEOF'
import sqlite3, sys
sys.path.insert(0, "/opt/digest")
import config, seen_store
conn = sqlite3.connect(config.DB_PATH)
seen_store.ensure(conn)
conn.execute("ATTACH DATABASE ? AS live", (sys.argv[1],))
n = conn.execute("""
INSERT OR IGNORE INTO seen_urls (url, query_index, first_seen, last_seen, verdict, attempts, embedding)
SELECT url, query_index, first_seen, last_seen, verdict, attempts, embedding FROM live.seen_urls
""").rowcount
conn.commit()
print("[merge] перенесено вердиктов из боевой базы: %d" % n)
print("[merge] журнал новой базы: %s" % seen_store.stats(conn))
print("[merge] разметка (принято, отклонено): %s" % (seen_store.label_counts(conn),))
conn.close()
PYEOF

BACKUP="${LIVE_DB%.db}_before_week_$STAMP.db"
cp -p "$LIVE_DB" "$BACKUP" && say "боевая база сохранена: $BACKUP"
mv "$NEW_DB" "$LIVE_DB" && say "ПОДМЕНА ВЫПОЛНЕНА: новая база на месте боевой"
say "=== Готово. Откат: cp '$BACKUP' '$LIVE_DB' ==="
