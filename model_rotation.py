# -*- coding: utf-8 -*-
"""
model_rotation.py — вызов OpenRouter с ротацией бесплатных ("":free"") моделей.

Используется sunday_processor_mmr.py для всех LLM-вызовов (пересказ статьи,
проверка темы, постобработка дайджеста) через единственную публичную функцию
_call_openrouter_raw(): она принимает system-промпт и user-сообщение явно —
никакого скрытого глобального состояния для промптов, вызывающий код сам
решает, что отправить модели.

Пул моделей (_free_models_pool) и счётчик вызовов (_batches_processed) —
глобальные и переживают несколько вызовов _call_openrouter_raw в рамках
одного процесса: sunday_processor_mmr вызывает функцию ~20+ раз подряд
(пересказ + проверки для каждой статьи дайджеста), и держать TCP/пул моделей
"тёплыми" между вызовами дешевле, чем инициализировать их каждый раз заново.
"""

import logging
import os
import threading
import time

import requests

import config

log = logging.getLogger("model_rotation")

# --- Глобальное состояние пула (переживает несколько вызовов подряд) ---
_free_models_pool = []
_batches_processed = 0

# daily_collector зовёт _call_openrouter_raw из нескольких потоков сразу,
# поэтому мутации пула идут под замком: без него два потока могли пройти
# проверку `if x in pool` одновременно, и второй remove() падал с ValueError.
# Сам HTTP-запрос остаётся ВНЕ замка, иначе параллельность батчей потерялась бы.
_POOL_LOCK = threading.RLock()


def _demote(model_id):
    """Неудачная модель уходит в конец пула."""
    with _POOL_LOCK:
        if model_id in _free_models_pool:
            _free_models_pool.remove(model_id)
            _free_models_pool.append(model_id)


def _promote(model_id):
    """Успешная модель встаёт в начало пула."""
    with _POOL_LOCK:
        if model_id in _free_models_pool:
            _free_models_pool.remove(model_id)
        _free_models_pool.insert(0, model_id)

POOL_REFRESH_INTERVAL = 10    # обновлять список моделей раз в N успешных вызовов
CLASSIFY_TIMEOUT = 45         # таймаут одного вызова chat/completions, сек
POOL_SIZE = 5                 # сколько ":free"-моделей держим в пуле одновременно
# BETWEEN_MODEL_PAUSE убран намеренно: 429 у ":free"-моделей почти всегда
# означает исчерпанный СУТОЧНЫЙ лимит, и пауза в секундах его не лечит — она
# лишь удлиняет прогон и тратит время впустую. Следующая модель пробуется
# сразу, а исчерпанный провайдер целиком уходит в кулдаун (PROVIDER_COOLDOWN).

# Переопределяется через .env (config._load_dotenv отработал при import config выше),
# чтобы пустить вызовы через headroom на 127.0.0.1:8787. Дефолт — прямой OpenRouter,
# поэтому без записи в .env поведение не меняется.
# Список моделей (MODELS_URL) намеренно остаётся прямым: это не chat/completions,
# сжимать там нечего, а лишний хоп ломал бы обновление пула при падении прокси.
OPENROUTER_API_URL = os.environ.get(
    "OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions"
)
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Платные/квотные фолбэки: когда весь :free-пул OpenRouter отдал 429/5xx (типично —
# исчерпан суточный лимит free-models-per-day), дожимаем через Groq, затем Google.
# У каждого свой /models: тянем список, оставляем чат-модели, сортируем "лучшие
# вперёд" и пробуем по очереди — без захардкоженных имён (они у провайдеров
# уезжают в 404). Включается только при заданном ключе провайдера.
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
# Gemini отдаёт OpenAI-совместимый chat/completions (ключ как Bearer),
# а список моделей — по нативному эндпоинту с ?key=.
GOOGLE_API_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
GOOGLE_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
# Groq и Google — free-tier аккаунты (лимит по частоте, не по деньгам): их /models
# не содержат платных моделей для такого ключа, поэтому фильтровать "бесплатные"
# нечего — в отличие от OpenRouter, где это делает суффикс ":free".
FALLBACK_POOL_SIZE = 5

# Чёрный список заведомо слабых моделей: если id содержит один из этих
# (регистронезависимых) фрагментов — в пул не берём. Слабая модель = плохая
# грамматика/куцые заголовки, лучше пропустить провайдера, чем отгрузить брак.
# У API нет сигнала "качества", поэтому фильтр по имени тира. Правь под себя.
# ponytail: подстрочный блок-лист — грубо; если пул зачистится в ноль, провайдер
# просто пропустится (кольцо уйдёт к следующему).
WEAK_MODEL_BLOCKLIST = ("lite", "-8b", "8b-", "-4b", "-3b", "-1b", "-mini", "-small", "nano")


def _is_weak(mid: str) -> bool:
    low = mid.lower()
    return any(bad in low for bad in WEAK_MODEL_BLOCKLIST)

# None = список ещё не тянули; [] = тянули, пусто/ошибка (повторно не дёргаем).
_groq_pool = None
_google_pool = None

# Circuit-breaker: когда весь пул провайдера упал (типично — исчерпан суточный
# лимит), помечаем его "мёртвым" на PROVIDER_COOLDOWN секунд и следующие вызовы
# в этом же прогоне не тратят на него ~50с (5 моделей × пауза 10с), а идут сразу
# к рабочему. Состояние в памяти процесса: новый cron-запуск пробует всех заново.
# ponytail: фиксированный кулдаун — грубо (суточный лимит живёт часами, 5xx —
# секунды); хватает на один прогон дайджеста, тонкая настройка не нужна.
PROVIDER_COOLDOWN = 600
_provider_dead_until = {}  # tag -> epoch, до которого провайдер считается исчерпанным


def _in_cooldown(tag: str) -> bool:
    return _provider_dead_until.get(tag, 0) > time.time()


def _mark_dead(tag: str) -> None:
    _provider_dead_until[tag] = time.time() + PROVIDER_COOLDOWN
    log.warning("%s помечен исчерпанным на %dс — пропускаем до восстановления", tag, PROVIDER_COOLDOWN)

# Захардкоженного списка моделей нет намеренно: ":free"-модели у OpenRouter
# регулярно уезжают в платные (gemma-3-27b-it стала отдавать 404 "use the paid
# slug"), и зашитый фолбэк тихо гниёт — выглядит как страховка, а на деле
# гарантированно проваливает вызовы. Не собрали пул — возвращаем None, весь
# вызывающий код это уже умеет.
_pool_refresh_failed = False


# ---------------------------------------------------------------------------
# Инициализация / обновление пула моделей
# ---------------------------------------------------------------------------
def _init_models_pool(force_refresh=False):
    """
    Тянет список моделей с OpenRouter, оставляет только ":free",
    сортирует по context_length по убыванию и берёт топ-POOL_SIZE.
    При ошибке запроса пул остаётся пустым.
    """
    global _free_models_pool, _pool_refresh_failed

    if _free_models_pool and not force_refresh:
        return

    if _pool_refresh_failed:
        # Список уже не отдался в этом процессе. Без раннего выхода мы дёргали бы
        # /models на каждом из ~20+ вызовов подряд, по CLASSIFY_TIMEOUT (45 с)
        # на попытку — прогон висел бы четверть часа, чтобы всё равно вернуть
        # None. Следующий запуск cron стартует новый процесс и попробует снова.
        return

    try:
        resp = requests.get(
            OPENROUTER_MODELS_URL,
            headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"},
            proxies=config.PROXIES,
            timeout=CLASSIFY_TIMEOUT,
        )
        resp.raise_for_status()
        models = resp.json().get("data", [])

        free = []
        for m in models:
            mid = m.get("id", "")
            if not mid.endswith(":free"):
                continue
            ctx = m.get("context_length") or 0
            free.append((mid, ctx))

        free.sort(key=lambda x: x[1], reverse=True)
        _free_models_pool = [mid for mid, _ in free[:POOL_SIZE]]

        if not _free_models_pool:
            raise ValueError("Список ':free' моделей пуст")

        _pool_refresh_failed = False
        log.info("Пул моделей обновлён: %s", _free_models_pool)
    except Exception as exc:  # noqa: BLE001
        log.warning("Не удалось получить список моделей (%s). Пул пуст.", exc)
        _free_models_pool = []
        _pool_refresh_failed = True


# ---------------------------------------------------------------------------
# Публичная функция: один LLM-запрос с ротацией моделей при сбоях
# ---------------------------------------------------------------------------
def _openrouter_attempt(system_prompt: str, user_message: str, ref_url: str = "") -> str | None:
    """
    Отправляет (system_prompt, user_message) очередной модели из пула.
    При 429/5xx или сетевой ошибке пессимизирует модель (сдвигает в конец
    пула) и пробует следующую; при успехе — сдвигает модель в начало пула.
    Возвращает текст ответа модели или None, если весь пул провалился.
    """
    global _free_models_pool, _batches_processed

    force = _batches_processed > 0 and _batches_processed % POOL_REFRESH_INTERVAL == 0
    _init_models_pool(force_refresh=force)

    for model_id in list(_free_models_pool):
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            "temperature": 0.1,
        }
        headers = {
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        need_pause = False
        try:
            resp = requests.post(
                OPENROUTER_API_URL,
                json=payload,
                headers=headers,
                proxies=config.PROXIES,
                timeout=CLASSIFY_TIMEOUT,
            )
        except Exception as exc:
            log.warning("_call_openrouter_raw сетевая ошибка %s: %s", model_id, exc)
            # Сетевая ошибка — сразу следующая модель, без паузы.
            _demote(model_id)
            continue

        if resp.status_code == 200:
            try:
                body = resp.json()
                choices = body.get("choices")
                if not choices:
                    # HTTP 200, но без "choices" — так OpenRouter пробрасывает
                    # ошибки апстрим-провайдера (перегрузка/модерация/сбой
                    # самой модели). Пул сам восстановится: модель уйдёт в
                    # конец пула, попробуем следующую.
                    raise ValueError(body.get("error", body))
                content = choices[0]["message"]["content"]
                _promote(model_id)
                _batches_processed += 1
                return content.strip()
            except Exception as exc:
                log.warning(
                    "_call_openrouter_raw не удалось разобрать ответ %s: %s",
                    model_id, exc,
                )
        elif resp.status_code == 429 or 500 <= resp.status_code < 600:
            log.warning("%d от %s -> смена модели", resp.status_code, model_id)
            need_pause = True
        else:
            log.warning("%d от %s -> смена модели", resp.status_code, model_id)

        _demote(model_id)  # пессимизация: неудачная модель уходит в конец пула
        if need_pause:
            log.debug("Смена модели после %s без паузы", model_id)

    return None  # весь :free-пул провалился; кольцо решит, что дальше


# ---------------------------------------------------------------------------
# Публичная функция: один LLM-запрос с round-robin по провайдерам.
# Провайдеры образуют кольцо OpenRouter -> Groq -> Google -> (снова OpenRouter).
# Стартовая точка вращается между вызовами, поэтому нагрузка размазывается по
# всем трём (суммарной суточной квоты хватает дольше), а исчерпанный провайдер
# пропускается по кулдауну. За один вызов делаем не более одного круга.
# ---------------------------------------------------------------------------
_PROVIDERS = ["OpenRouter", "Groq", "Google"]
_rr_start = 0


def _call_openrouter_raw(system_prompt: str, user_message: str, ref_url: str = "") -> str | None:
    global _rr_start
    n = len(_PROVIDERS)
    for i in range(n):
        tag = _PROVIDERS[(_rr_start + i) % n]
        if _in_cooldown(tag):
            continue
        out = _provider_call(tag, system_prompt, user_message, ref_url)
        if out is not None:
            _rr_start = (_rr_start + i + 1) % n  # следующий вызов — со следующего провайдера
            log.info("%s отдал ответ для %s", tag, ref_url)
            return out
        _mark_dead(tag)  # весь пул провайдера провалился
    log.error("_call_openrouter_raw: все провайдеры исчерпаны для %s", ref_url)
    return None


def _provider_call(tag: str, system_prompt: str, user_message: str, ref_url: str) -> str | None:
    if tag == "OpenRouter":
        return _openrouter_attempt(system_prompt, user_message, ref_url)
    if tag == "Groq":
        if not config.GROQ_API_KEY:
            return None
        return _try_pool(_groq_models(), GROQ_API_URL, config.GROQ_API_KEY,
                         system_prompt, user_message, "Groq")
    if tag == "Google":
        if not config.GOOGLE_API_KEY:
            return None
        return _try_pool(_google_models(), GOOGLE_API_URL, config.GOOGLE_API_KEY,
                         system_prompt, user_message, "Google")
    return None


def _openai_chat(url: str, api_key: str, model_id: str,
                 system_prompt: str, user_message: str) -> str:
    """Один OpenAI-совместимый chat/completions запрос. Бросает при не-200."""
    resp = requests.post(
        url,
        json={
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            "temperature": 0.1,
        },
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        proxies=config.PROXIES,
        timeout=CLASSIFY_TIMEOUT,
    )
    if resp.status_code != 200:
        raise ValueError(f"HTTP {resp.status_code}")
    content = resp.json()["choices"][0]["message"]["content"]
    # reasoning-модели (qwen, glm…) оборачивают ответ в <think>…</think> прямо в
    # content — берём хвост после последнего закрывающего тега, парсер ждёт чистый текст.
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[-1]
    return content.strip()


def _try_pool(models, url, api_key, system_prompt, user_message, tag) -> str | None:
    """Пробуем модели пула по очереди; битую/занятую (404/429/5xx) пропускаем."""
    for mid in models:
        try:
            return _openai_chat(url, api_key, mid, system_prompt, user_message)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s %s -> %s, следующая модель", tag, mid, exc)
    return None


def _groq_models():
    """Пул лучших чат-моделей Groq (кэш на процесс). [] если список не отдался."""
    global _groq_pool
    if _groq_pool is not None:
        return _groq_pool
    _groq_pool = []
    try:
        resp = requests.get(
            GROQ_MODELS_URL,
            headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
            proxies=config.PROXIES, timeout=CLASSIFY_TIMEOUT,
        )
        resp.raise_for_status()
        chat = []
        for m in resp.json().get("data", []):
            mid = m.get("id", "")
            if not m.get("active", True):
                continue
            # у Groq в списке ещё аудио/модерация — не для чата
            if any(x in mid for x in ("whisper", "tts", "guard")):
                continue
            if _is_weak(mid):
                continue
            chat.append((m.get("context_window") or 0, mid))
        # ponytail: "лучшая" == наибольший контекст — грубая эвристика, без сигнала
        # качества из API; ротация всё равно перебирает пул при сбое.
        chat.sort(reverse=True)
        _groq_pool = [mid for _, mid in chat[:FALLBACK_POOL_SIZE]]
        log.info("Groq-пул: %s", _groq_pool)
    except Exception as exc:  # noqa: BLE001
        log.warning("Groq: не удалось получить список моделей: %s", exc)
    return _groq_pool


def _google_models():
    """Пул лучших gemini-моделей (кэш на процесс). [] если список не отдался."""
    global _google_pool
    if _google_pool is not None:
        return _google_pool
    _google_pool = []
    try:
        resp = requests.get(
            GOOGLE_MODELS_URL,
            params={"key": config.GOOGLE_API_KEY},
            proxies=config.PROXIES, timeout=CLASSIFY_TIMEOUT,
        )
        resp.raise_for_status()
        chat = []
        for m in resp.json().get("models", []):
            if "generateContent" not in m.get("supportedGenerationMethods", []):
                continue
            mid = m.get("name", "").split("/", 1)[-1]
            # только текстовые gemini-* (отсекает lyria/robotics/embedding/imagen…)
            if not mid.startswith("gemini-") or "robotics" in mid:
                continue
            if _is_weak(mid):
                continue
            ctx = m.get("inputTokenLimit") or 0
            # tie-break при равном контексте: стабильные -latest вперёд, preview назад
            chat.append(((ctx, mid.endswith("-latest"), "preview" not in mid), mid))
        chat.sort(reverse=True)
        _google_pool = [mid for _, mid in chat[:FALLBACK_POOL_SIZE]]
        log.info("Google-пул: %s", _google_pool)
    except Exception as exc:  # noqa: BLE001
        log.warning("Google: не удалось получить список моделей: %s", exc)
    return _google_pool


