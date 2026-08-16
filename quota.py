# -*- coding: utf-8 -*-
"""
quota.py — сколько ещё можно спросить у каждого облачного провайдера.

Зачем это существует. Кольцо в model_rotation переключает провайдеров по факту
отказа, но 429 бывает двух природ: частотный (Groq/Google, отпускает за десятки
секунд) и суточный (OpenRouter ":free" — до полуночи UTC). Локальная модель на
домашнем сервере слабее любой облачной, поэтому включать её имеет смысл ТОЛЬКО
когда суточные лимиты выбраны у всех трёх. Уводить на неё вызовы из-за
частотного 429 — терять качество дайджеста там, где облако ещё живо.

Честного "осталось N запросов" не отдаёт никто:

  OpenRouter — /api/v1/key показывает ДЕНЬГИ, а ":free"-модели стоят $0, так что
               usage_daily по ним всегда 0. Запросы считаем сами. Потолок берём
               оттуда же: is_free_tier=true → 50/сут, false → 1000/сут (разовая
               покупка $10 поднимает пол навсегда).
  Groq       — единственный, кто кладёт x-ratelimit-remaining-requests в
               заголовки КАЖДОГО ответа. Читаем и верим.
  Google     — не отдаёт ничего. Судим только по 429 и его Retry-After.

Состояние переживает процессы: daily_collector стартует из cron каждые 6 часов
новым процессом, счётчик в памяти обнулялся бы при каждом запуске.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone

import requests

import config

log = logging.getLogger("quota")

STATE_PATH = os.environ.get(
    "QUOTA_STATE_PATH", os.path.join(os.path.dirname(__file__), "data", "quota_state.json")
)

# Retry-After короче этого — частотный лимит, провайдер вернётся сам.
# Длиннее — считаем суточным: столько ждать дороже, чем сходить на локальную.
# 300с выбрано по наблюдаемым значениям: частотные отсидки у Groq/Google
# укладываются в десятки секунд, суточные измеряются часами.
SUSTAINED_THRESHOLD = 300

# Потолки суточных запросов OpenRouter к ":free"-моделям.
FREE_TIER_DAILY = 50
PAID_TIER_DAILY = 1000

_LOCK = threading.RLock()
_state = None
_or_limit_cached = None


def _today():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _blank():
    return {"date": _today(), "providers": {}}


def _load():
    """Читает состояние с диска, сбрасывая его при смене UTC-суток.

    Битый файл — не повод падать: счётчик восстановим, а прогон дайджеста
    важнее точности учёта. Начинаем с чистого листа."""
    global _state
    if _state is not None and _state.get("date") == _today():
        return _state
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("date") != _today():
            data = _blank()
    except (OSError, ValueError):
        data = _blank()
    _state = data
    return _state


def _save():
    try:
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        tmp = STATE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_state, fh, ensure_ascii=False)
        os.replace(tmp, STATE_PATH)
    except OSError as exc:
        log.warning("не удалось сохранить состояние квот: %s", exc)


def _prov(tag):
    st = _load()
    return st["providers"].setdefault(tag, {"used": 0, "remaining": None, "blocked_until": 0})


def openrouter_daily_limit():
    """Потолок суточных запросов к ":free". Один раз за процесс — значение
    меняется только при покупке кредитов, опрашивать чаще незачем."""
    global _or_limit_cached
    if _or_limit_cached is not None:
        return _or_limit_cached
    _or_limit_cached = PAID_TIER_DAILY
    try:
        resp = requests.get(
            "https://openrouter.ai/api/v1/key",
            headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"},
            proxies=config.PROXIES,
            timeout=15,
        )
        if resp.status_code == 200:
            free = resp.json().get("data", {}).get("is_free_tier", True)
            _or_limit_cached = FREE_TIER_DAILY if free else PAID_TIER_DAILY
            log.info("OpenRouter суточный потолок :free = %d", _or_limit_cached)
    except Exception as exc:  # noqa: BLE001
        log.warning("не удалось узнать тир OpenRouter (%s), берём %d",
                    exc, _or_limit_cached)
    return _or_limit_cached


def note_request(tag):
    """Запрос ушёл. Считаем ДО ответа: неудачные попытки тоже съедают квоту."""
    with _LOCK:
        p = _prov(tag)
        p["used"] += 1
        _save()


def note_response(tag, resp):
    """Разбирает ответ: остаток из заголовков и признак исчерпания при 429."""
    headers = getattr(resp, "headers", None) or {}
    status = getattr(resp, "status_code", None)

    with _LOCK:
        p = _prov(tag)

        # Groq кладёт остаток в каждый ответ; у остальных заголовка нет.
        raw = headers.get("x-ratelimit-remaining-requests")
        if raw is not None:
            try:
                p["remaining"] = int(float(raw))
            except (TypeError, ValueError):
                pass

        # OpenRouter отдаёт X-RateLimit-* только в ошибках лимита.
        raw = headers.get("X-RateLimit-Remaining")
        if raw is not None:
            try:
                p["remaining"] = int(float(raw))
            except (TypeError, ValueError):
                pass

        if status == 429:
            delay = 0.0
            for key in ("Retry-After", "X-RateLimit-Reset"):
                val = headers.get(key)
                if not val:
                    continue
                try:
                    delay = max(delay, float(val))
                except (TypeError, ValueError):
                    continue
            if delay >= SUSTAINED_THRESHOLD:
                p["blocked_until"] = time.time() + delay
                log.warning("%s: суточный лимит, до %s", tag,
                            datetime.fromtimestamp(p["blocked_until"], timezone.utc).isoformat())
            elif p.get("remaining") == 0:
                # Остаток честно нулевой, а Retry-After короткий или его нет:
                # верим цифре, а не длительности.
                p["blocked_until"] = time.time() + SUSTAINED_THRESHOLD
        _save()


def note_success(tag):
    """Успех снимает пометку исчерпания — провайдер явно жив."""
    with _LOCK:
        p = _prov(tag)
        p["blocked_until"] = 0
        _save()


def remaining(tag):
    """Сколько запросов осталось, или None если провайдер не даёт цифру.

    Для OpenRouter считаем сами: потолок минус потраченное за сутки."""
    with _LOCK:
        p = _prov(tag)
        if tag == "OpenRouter":
            return max(0, openrouter_daily_limit() - p["used"])
        return p.get("remaining")


def exhausted(tag):
    """True, если у провайдера на сегодня уже нечего просить."""
    with _LOCK:
        p = _prov(tag)
        if p.get("blocked_until", 0) > time.time():
            return True
        left = remaining(tag)
        return left is not None and left <= 0


def all_cloud_exhausted(tags):
    """True, когда исчерпаны ВСЕ переданные облачные провайдеры.

    Это единственное условие, при котором стоит уходить на локальную модель."""
    if not tags:
        return True
    return all(exhausted(t) for t in tags)


def snapshot(tags):
    """Человекочитаемая сводка для логов: что где осталось."""
    out = []
    for t in tags:
        left = remaining(t)
        mark = "исчерпан" if exhausted(t) else (
            f"осталось {left}" if left is not None else "остаток неизвестен")
        out.append(f"{t}: {mark}")
    return "; ".join(out)
