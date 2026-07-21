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
import time

import requests

import config

log = logging.getLogger("model_rotation")

# --- Глобальное состояние пула (переживает несколько вызовов подряд) ---
_free_models_pool = []
_batches_processed = 0

POOL_REFRESH_INTERVAL = 10    # обновлять список моделей раз в N успешных вызовов
CLASSIFY_TIMEOUT = 45         # таймаут одного вызова chat/completions, сек
POOL_SIZE = 5                 # сколько ":free"-моделей держим в пуле одновременно
BETWEEN_MODEL_PAUSE = 10      # пауза перед сменой модели при 429/5xx, сек

# Переопределяется через .env (config._load_dotenv отработал при import config выше),
# чтобы пустить вызовы через headroom на 127.0.0.1:8787. Дефолт — прямой OpenRouter,
# поэтому без записи в .env поведение не меняется.
# Список моделей (MODELS_URL) намеренно остаётся прямым: это не chat/completions,
# сжимать там нечего, а лишний хоп ломал бы обновление пула при падении прокси.
OPENROUTER_API_URL = os.environ.get(
    "OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions"
)
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Используются, если запрос списка моделей у OpenRouter не удался.
FALLBACK_MODELS = [
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-3.1-8b-instruct:free",
]


# ---------------------------------------------------------------------------
# Инициализация / обновление пула моделей
# ---------------------------------------------------------------------------
def _init_models_pool(force_refresh=False):
    """
    Тянет список моделей с OpenRouter, оставляет только ":free",
    сортирует по context_length по убыванию и берёт топ-POOL_SIZE.
    При ошибке запроса откатывается на FALLBACK_MODELS.
    """
    global _free_models_pool

    if _free_models_pool and not force_refresh:
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

        log.info("Пул моделей обновлён: %s", _free_models_pool)
    except Exception as exc:  # noqa: BLE001
        log.warning("Не удалось получить список моделей (%s). Fallback.", exc)
        _free_models_pool = list(FALLBACK_MODELS)


# ---------------------------------------------------------------------------
# Публичная функция: один LLM-запрос с ротацией моделей при сбоях
# ---------------------------------------------------------------------------
def _call_openrouter_raw(system_prompt: str, user_message: str, ref_url: str = "") -> str | None:
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
            if model_id in _free_models_pool:
                _free_models_pool.remove(model_id)
                _free_models_pool.append(model_id)
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
                if model_id in _free_models_pool:
                    _free_models_pool.remove(model_id)
                _free_models_pool.insert(0, model_id)
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

        # Пессимизация: неудачная модель уходит в конец пула.
        if model_id in _free_models_pool:
            _free_models_pool.remove(model_id)
            _free_models_pool.append(model_id)

        if need_pause:
            log.info(
                "Пауза %ds перед переходом к следующей модели (после %s)",
                BETWEEN_MODEL_PAUSE, model_id,
            )
            time.sleep(BETWEEN_MODEL_PAUSE)

    log.error("_call_openrouter_raw: все модели провалились для %s", ref_url)
    return None
