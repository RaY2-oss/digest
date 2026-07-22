# -*- coding: utf-8 -*-
"""Замер селективности GKG-предфильтра: текущий вариант vs строгие варианты."""
import io, sys, zipfile, collections
from datetime import datetime, timedelta, timezone
import pandas as pd, requests

sys.path.insert(0, "/opt/digest")
import config

USECOLS = [1, 3, 4, 7, 9]
NAMES = ["date", "source", "url", "themes", "locations"]
EN = "http://data.gdeltproject.org/gdeltv2/{ts}.gkg.csv.zip"
TL = "http://data.gdeltproject.org/gdeltv2/{ts}.translation.gkg.csv.zip"

N_TICKS = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def ticks(n):
    now = datetime.now(timezone.utc)
    a = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
    return [(a - timedelta(minutes=15 * i)).strftime("%Y%m%d%H%M%S") for i in range(1, n + 1)]


def load(ts, tl):
    try:
        r = requests.get((TL if tl else EN).format(ts=ts), timeout=60)
        if r.status_code != 200:
            return None
        with zipfile.ZipFile(io.BytesIO(r.content)) as z, z.open(z.namelist()[0]) as f:
            return pd.read_csv(f, sep="\t", header=None, usecols=USECOLS, names=NAMES,
                               on_bad_lines="skip", low_memory=False, dtype=str,
                               encoding_errors="replace")
    except Exception as e:
        print("  err", ts, tl, e)
        return None


def loc_codes(s):
    """V1Locations: 'type#name#CC#ADM1#lat#lon#featureid;...' -> список кодов стран."""
    out = []
    for ent in (s or "").split(";"):
        p = ent.split("#")
        if len(p) >= 3 and len(p[2]) == 2:
            out.append(p[2])
    return out


def analyse(df, cfg, stats):
    if df is None or df.empty:
        return
    themes_set = set(t.upper() for t in cfg["themes"])
    locs_set = set(cfg["locations"])
    th = df["themes"].fillna("")
    lo = df["locations"].fillna("")

    # --- вариант A: текущий (подстрочный contains) ---
    cur = (th.str.contains("|".join(cfg["themes"]), case=False, na=False)
           & lo.str.contains("|".join(f"#{c}#" for c in locs_set), na=False))
    stats["A_current"] += int(cur.sum())

    idx = df.index[cur]
    for i in idx:
        toks = set(t.strip().upper() for t in th[i].split(";") if t.strip())
        codes = loc_codes(lo[i])
        cnt = collections.Counter(codes)
        hit_themes = toks & themes_set
        n_target = sum(v for k, v in cnt.items() if k in locs_set)
        total = sum(cnt.values()) or 1
        share = n_target / total
        top1 = cnt.most_common(1)[0][0] if cnt else ""

        if hit_themes:
            stats["B_exact_theme"] += 1
            if share >= 0.30:
                stats["C_exact+share30"] += 1
            if top1 in locs_set:
                stats["D_exact+top1"] += 1
                if len(hit_themes) >= 2:
                    stats["E_exact+top1+2themes"] += 1
                if share >= 0.30:
                    stats["F_exact+top1+share30"] += 1


for qi, cfg in enumerate(config.QUERIES_GKG):
    stats = collections.Counter()
    rows = 0
    for ts in ticks(N_TICKS):
        for tl in (False, True):
            df = load(ts, tl)
            if df is not None:
                rows += len(df)
                analyse(df, cfg, stats)
    print(f"\n=== QUERY #{qi} | просканировано строк GKG: {rows} | тиков: {N_TICKS} ===")
    base = stats["A_current"] or 1
    for k in ["A_current", "B_exact_theme", "C_exact+share30", "D_exact+top1",
              "E_exact+top1+2themes", "F_exact+top1+share30"]:
        print(f"  {k:26s} {stats[k]:6d}   ({100*stats[k]/base:5.1f}% от текущего)")
