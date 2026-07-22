# -*- coding: utf-8 -*-
"""Сравнение старого и нового правила отбора строк GKG на живых дампах.
LLM не вызывается, страницы не скачиваются — только фильтрация."""
import sys
sys.path.insert(0, "/opt/digest")
import numpy as np
import config
import daily_collector as dc
import gkg_filter

N_TICKS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
ticks = dc.gkg_timestamps(hours=24)[-N_TICKS:]


def old_rule(df, cfg):
    """Прежнее правило: ПОДСТРОКА по темам И присутствие кода страны."""
    themes = df["v1t"].fillna("")
    locs = df["v1l"].fillna("")
    m = (themes.str.contains("|".join(cfg["themes"]), case=False, na=False)
         & locs.str.contains("|".join("#%s#" % c for c in cfg["locations"]), na=False))
    return set(df.loc[m, "url"].dropna())


for qi, cfg in enumerate(config.QUERIES_GKG):
    old_urls, new_urls, stats = set(), set(), {}
    for ts in ticks:
        for tr in (False, True):
            df = dc.fetch_gkg_file(ts, translation=tr)
            if df is None or df.empty:
                continue
            old_urls |= old_rule(df, cfg)
            new_urls |= {u for u, _ in gkg_filter.select(df, cfg, stats)}

    o, n = len(old_urls), len(new_urls)
    print("\n=== Query #%d | %d тиков x 2 потока ===" % (qi, N_TICKS))
    print("  строк просмотрено      : %d" % stats.get("rows", 0))
    print("  старое правило (URL)   : %d" % o)
    print("  новое правило  (URL)   : %d  (%.1f%% от старого)"
          % (n, 100.0 * n / o if o else 0.0))
    print("  откат на V1 (нет полей): %d" % stats.get("v1_fallback", 0))
    lost = new_urls - old_urls
    if lost:
        print("  ВНИМАНИЕ: новых, которых не было у старого: %d" % len(lost))
    gaps = stats.get("gaps")
    if gaps:
        p = np.percentile(gaps, [50, 75, 90, 95, 99])
        print("  разрыв тема-локация    : p50=%.0f p75=%.0f p90=%.0f p95=%.0f p99=%.0f" % tuple(p))
    sh = stats.get("shares")
    if sh:
        p = np.percentile(sh, [10, 25, 50, 75])
        print("  доля целевой страны    : p10=%.2f p25=%.2f p50=%.2f p75=%.2f" % tuple(p))
    dropped = sorted(old_urls - new_urls)
    print("  примеры отсечённого:")
    for u in dropped[:6]:
        print("    -", u[:118])
