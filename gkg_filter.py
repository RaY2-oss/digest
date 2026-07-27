# -*- coding: utf-8 -*-
"""gkg_filter.py — отбор URL из дампа GKG по СВЯЗИ темы и локации.

Старое правило: "в статье встречается нужная тема И встречается нужная
страна". Связи между условиями нет. Колонки V1Themes/V1Locations — два
независимых плоских списка на весь документ, каждая страна ровно один раз,
без позиции и без частоты. Эстонская статья, где "образование" в первом
абзаце, а Казахстан в двадцатом, проходила. Отсюда в языковой статистике
Query #1: et=115, zh-cn=58, de=52, ko=20 при 972 кандидатах.

Колонки V2 несут символьные офсеты в ОДИН И ТОТ ЖЕ очищенный текст GDELT:

    V2EnhancedThemes     THEME,offset;THEME,offset;...
    V2EnhancedLocations  type#name#CC#ADM1#ADM2#lat#lon#featureid#offset;...

Значит расстояние между упоминанием темы и упоминанием страны считается
напрямую, и "тема обсуждается в контексте этой страны" превращается в
"минимальное расстояние меньше N символов". Плюс страна повторяется на
каждое упоминание, поэтому доступна и её доля среди всех упоминаний.

# ponytail: N и порог доли — общие для всех тем и стран, без поправки на
# язык и длину статьи. Грубо, но обе величины логируются каждым прогоном
# (см. daily_collector), так что калибруются по факту, а не на глаз.
"""
import re
from datetime import datetime

import entities

_V1_MIN_PARTS = 3   # type#name#CC — минимум, чтобы достать код страны
_V2_MIN_PARTS = 9   # ...#featureid#offset — офсет последним полем


def _txt(cell):
    """Пустая ячейка приходит из pandas как float('nan'), а НЕ как "" — и
    `nan or ""` возвращает nan, потому что NaN истинен. Приводим явно."""
    return cell if isinstance(cell, str) else ""


def parse_themes(cell) -> dict:
    """V2EnhancedThemes -> {THEME: [offset, ...]}"""
    out = {}
    for ent in _txt(cell).split(";"):
        if not ent:
            continue
        name, _, off = ent.rpartition(",")
        off = off.strip()
        if not name or not off.isdigit():
            continue
        out.setdefault(name.strip().upper(), []).append(int(off))
    return out


def parse_locations(cell) -> dict:
    """V2EnhancedLocations -> {CC: [offset, ...]} (повтор на каждое упоминание)"""
    out = {}
    for ent in _txt(cell).split(";"):
        p = ent.split("#")
        if len(p) < _V2_MIN_PARTS:
            continue
        cc, off = p[2].strip().upper(), p[-1].strip()
        if len(cc) == 2 and off.isdigit():
            out.setdefault(cc, []).append(int(off))
    return out


def min_gap(a, b):
    """Минимальное |x-y| между двумя наборами офсетов, слиянием за O(n+m)."""
    if not a or not b:
        return None
    a, b = sorted(a), sorted(b)
    i = j = 0
    best = abs(a[0] - b[0])
    while i < len(a) and j < len(b):
        d = abs(a[i] - b[j])
        if d < best:
            best = d
        if a[i] < b[j]:
            i += 1
        else:
            j += 1
    return best


def judge_v2(v2t, v2l, themes, locations, max_gap, min_share):
    """Решение по одной строке на V2-полях.

    -> (ok, gap, share) либо None, если V2-полей нет и решать не на чем
    (вызывающий код откатывается на judge_v1)."""
    # Откат на V1 только если строка ВООБЩЕ не несёт полей V2. Если поля
    # есть, но в них нет локаций (статья без географии) или нет тем — это
    # содержательный отказ, а не отсутствие данных: уводить такую строку на
    # слабое V1-правило значило бы возвращать тот самый шум.
    if not _txt(v2t).strip() and not _txt(v2l).strip():
        return None
    th = parse_themes(v2t)
    lo = parse_locations(v2l)
    if not th or not lo:
        return (False, None, None)

    hits = []
    for t in themes:
        if t in th:
            hits.extend(th[t])
    if not hits:
        return (False, None, None)

    target = {c: lo[c] for c in locations if c in lo}
    if not target:
        return (False, None, None)

    total = sum(len(v) for v in lo.values())
    share = (sum(len(v) for v in target.values()) / total) if total else 0.0
    if min_share is not None and share < min_share:
        return (False, None, share)

    gaps = [g for g in (min_gap(hits, v) for v in target.values()) if g is not None]
    gap = min(gaps) if gaps else None
    if max_gap is not None and (gap is None or gap > max_gap):
        return (False, gap, share)
    return (True, gap, share)


def judge_v1(v1t, v1l, themes, locations) -> bool:
    """Откат для записей с пустыми V2-полями: ТОЧНОЕ совпадение темы вместо
    подстроки (contains("SCIENCE") ловил WB_1462_SCIENCE_TECHNOLOGY_AND_
    INNOVATION и десяток соседей) плюс присутствие страны. Слабее связи по
    офсетам, но заметно строже прежнего."""
    got = {t.strip().upper() for t in _txt(v1t).split(";") if t.strip()}
    if got.isdisjoint(themes):
        return False
    codes = set()
    for ent in _txt(v1l).split(";"):
        p = ent.split("#")
        if len(p) >= _V1_MIN_PARTS and len(p[2].strip()) == 2:
            codes.add(p[2].strip().upper())
    return not codes.isdisjoint(locations)


def _gkg_date(raw):
    try:
        return datetime.strptime(str(raw or "")[:8], "%Y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        return None


def select(df, cfg, stats=None):
    """df — дамп GKG с колонками url,date,source,v1t,v2t,v1l,v2l (+v1p,v1o).
    cfg — элемент config.QUERIES_GKG. -> [(url, gkg_date, entities), ...]

    entities — субъекты статьи из V1Persons/V1Organizations, склеенные
    entities.parse (см. entities.py: политический вес против локального шума).
    Колонок v1p/v1o в df может не быть — тогда строка просто пустая.

    stats — необязательный dict-счётчик; заполняется для лога, чтобы пороги
    max_theme_loc_gap / min_country_share подбирались по наблюдаемым
    распределениям, а не на глаз."""
    if df is None or df.empty:
        return []

    themes = {t.strip().upper() for t in cfg.get("themes", [])}
    locations = {c.strip().upper() for c in cfg.get("locations", [])}
    max_gap = cfg.get("max_theme_loc_gap")
    min_share = cfg.get("min_country_share")
    extra = [e.lower() for e in (cfg.get("extra_terms") or [])]

    # Дешёвый ВЕКТОРНЫЙ отсев перед построчным разбором. Строка заведомо не
    # пройдёт, если в её V2-полях нет ни одного целевого имени темы или ни
    # одного кода страны. Проверка по подстроке — надмножество точного
    # правила, поэтому ложных пропусков не даёт, а до дорогого разбора
    # доходят единицы процентов строк. Без этого недельный прогон тратил
    # ~4.5 часа на одно только сканирование 2.5 млн строк GKG.
    if len(df) and themes and locations:
        v2t_s = df["v2t"].fillna("") if "v2t" in df.columns else None
        v2l_s = df["v2l"].fillna("") if "v2l" in df.columns else None
        if v2t_s is not None and v2l_s is not None:
            t_pat = "|".join(re.escape(t) for t in themes)
            l_pat = "|".join(re.escape("#%s#" % c) for c in locations)
            hit = (v2t_s.str.contains(t_pat, regex=True, na=False)
                   & v2l_s.str.contains(l_pat, regex=True, na=False))
            # Строки вообще без полей V2 идут дальше — их рассудит judge_v1.
            no_v2 = (v2t_s.str.len() == 0) & (v2l_s.str.len() == 0)
            df = df[hit | no_v2]
            if df.empty:
                return []

    out = {}
    for row in df.itertuples(index=False):
        url = getattr(row, "url", None)
        if not isinstance(url, str) or not url.startswith("http") or url in out:
            continue

        if extra:
            hay = (str(getattr(row, "source", "") or "") + " " + url).lower()
            if not any(e in hay for e in extra):
                continue

        verdict = judge_v2(getattr(row, "v2t", ""), getattr(row, "v2l", ""),
                           themes, locations, max_gap, min_share)
        if verdict is None:
            ok, gap, share = judge_v1(getattr(row, "v1t", ""),
                                      getattr(row, "v1l", ""),
                                      themes, locations), None, None
            if stats is not None:
                stats["v1_fallback"] = stats.get("v1_fallback", 0) + 1
        else:
            ok, gap, share = verdict

        if stats is not None:
            stats["rows"] = stats.get("rows", 0) + 1
            stats["passed" if ok else "dropped"] = \
                stats.get("passed" if ok else "dropped", 0) + 1
            if ok and gap is not None:
                stats.setdefault("gaps", []).append(gap)
            if ok and share is not None:
                stats.setdefault("shares", []).append(share)

        if ok:
            out[url] = (_gkg_date(getattr(row, "date", "")),
                        entities.parse(getattr(row, "v1p", ""),
                                       getattr(row, "v1o", "")))
    return [(u, d, e) for u, (d, e) in out.items()]
