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
import quota

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
# BETWEEN_MODEL_PAUSE убран намеренно: у OpenRouter ":free" 429 означает
# исчерпанный СУТОЧНЫЙ лимит, и пауза в секундах его не лечит. Следующая модель
# пробуется сразу, а провалившийся пул просто отдаёт ход следующему провайдеру
# в кольце — пауза, если она нужна, берётся на уровне кольца (RING_RETRY_PAUSE).

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

# Бесплатная ступень перед своей моделью: шлюз llm7.io. Ключ не нужен вообще —
# анонимный тариф отдаёт часть моделей без регистрации (docs.llm7.io/limits:
# 1 запрос в секунду, 10 в минуту, 60 в час, 500 тыс. токенов в сутки; почта в
# dash.llm7.io поднимает до 40/мин и 1 млн токенов). Батч судьи весит ~24 тыс.
# знаков, суточная работа — около 40 батчей, то есть ~360 тыс. токенов: в
# анонимный лимит помещается, в лимит с почтой — с запасом.
#
# В кольцо round-robin НЕ входит по той же причине, что и локальная модель:
# пока у облака остались суточные лимиты, качество там выше. Включается там же,
# где локальная, — по quota.all_cloud_exhausted(), и пробуется ПЕРЕД ней.
#
# Замер 17.08.2026 (bench_providers.py, 90 статей корзины, батчи по 10, отсидка
# Retry-After как в кольце): обе модели разобрали 9 батчей из 9 и решили 90
# статей из 90; регион совпал с боевым судьёй в 0.97-0.98 принятых, средняя
# ошибка масштаба 0.92 (gemini-3.1-flash-lite) и 1.20 (gpt-oss:20b) при
# медиане 2.4 и 18.9 с. Порядок пула отсюда: быстрая и более точная первой.
# Без отсидки Retry-After gpt-oss отдавал 4 батча из 9 — на 2000 max_tokens
# рассуждение съедало ответ целиком; в кольце max_tokens не задаётся вовсе.
#
# Провайдер сторонний и моделей не хостит: это шлюз. Отсюда и место в цепочке —
# не «дешёвая замена облаку», а то, что стоит между исчерпанной квотой и 1.7B
# на домашней карте.
LLM7_API_URL = os.environ.get("LLM7_API_URL",
                              "https://api.llm7.io/v1/chat/completions")
LLM7_MODELS = [m.strip() for m in os.environ.get(
    "LLM7_MODELS", "gemini-3.1-flash-lite,gpt-oss:20b,deepseek-v3").split(",")
    if m.strip()]
# Анонимному тарифу ключ не нужен, но заголовок Authorization уходит всегда:
# _openai_chat общий для всех провайдеров, а llm7 на любую строку отвечает так
# же, как на её отсутствие (проверено 17.08.2026). Пустой LLM7_MODELS выключает
# ступень целиком.
LLM7_API_KEY = os.environ.get("LLM7_API_KEY", "anonymous")

# Последний рубеж: своя модель на домашнем сервере (Ollama, OpenAI-совместимый
# /v1/chat/completions по VPN). В кольцо round-robin НЕ входит намеренно —
# она слабее любой облачной, и отдавать ей вызовы, пока у облака остались
# суточные лимиты, значит терять качество дайджеста на ровном месте.
# Включается только когда quota.all_cloud_exhausted() говорит, что выбраны все
# три. Пустой LOCAL_LLM_URL полностью выключает эту ветку.
LOCAL_LLM_URL = os.environ.get("LOCAL_LLM_URL", "")
LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "qwen3:14b-q5_K_M")
# Локальная 14B на 16 ГБ отвечает заметно медленнее облачной: таймаут щедрее.
LOCAL_TIMEOUT = int(os.environ.get("LOCAL_LLM_TIMEOUT", "300"))

# Окно контекста локальной модели.
#
# Ollama поднимает модель со своим умолчанием в 4096 токенов, и это не «мало»,
# а тихо ломающе. Промпт пересказа — системный (~2.1 тыс. токенов) плюс до
# шести версий сюжета — весит 4-6 тыс. Переполнение Ollama не отвергает и
# ничем не сообщает: она срезает промпт примерно ВДВОЕ И ОТ НАЧАЛА. Начало —
# это весь системный промпт: требование отвечать JSON'ом, требование писать
# по-русски, правила написания имён. Модель оставалась с одними текстами
# статей и добросовестно пересказывала их маркдауном на языке источника.
#
# Замер 15.08.2026: 61% ответов «JSON не распарсился», 234 обращения к модели
# ради 20 пунктов, 88% времени прогона впустую, полтора-два часа на дайджест.
#
# 16384 покрывает и пересказ, и постобработку — та шлёт весь дайджест одним
# куском и столько же генерирует обратно. На карте это 5.8 ГБ: влезает целиком,
# проверено 16.08.2026. Растить дальше без нужды не стоит — это VRAM на общей
# карте, где ещё считают sciwriter и quantlab.
LOCAL_NUM_CTX = int(os.environ.get("LOCAL_LLM_NUM_CTX", "16384"))

# qwen3 — гибридная reasoning-модель: без явного выключения она тратит на
# <think> больше токенов, чем на сам ответ. Замер 16.08.2026 на 12 реальных
# сюжетах: выключение ускорило пересказ с 32.7 до 14.9 с при том же числе
# годных ответов (7 из 12). Включать обратно есть смысл, только если появится
# модель, которой размышление реально помогает — тогда LOCAL_LLM_THINK=1.
LOCAL_THINK = os.environ.get("LOCAL_LLM_THINK", "0") not in ("0", "", "false", "no")

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


def _strip_think(content: str) -> str:
    """reasoning-модели (qwen, glm…) оборачивают ответ в <think>…</think> прямо в
    content — берём хвост после последнего закрывающего тега, парсер ждёт чистый
    текст. Нужно на ВСЕХ путях: у OpenRouter в пуле те же reasoning-модели, и без
    зачистки их ответ падал в _parse_llm_json, а вызывающий код списывал это на
    «модель не справилась» и жёг ретрай."""
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[-1]
    return content.strip()

# None = список ещё не тянули; [] = тянули, пусто/ошибка (повторно не дёргаем).
_groq_pool = None
_google_pool = None

# Провайдер, у которого весь пул только что провалился, из кольца НЕ выбывает:
# следующий вызов пробует его снова. Своего кулдауна у нас нет намеренно. 429
# бывает двух природ — частотный (Groq/Google, отпускает за десятки секунд) и
# суточный (OpenRouter ":free", до полуночи UTC) — отличить их по ответу
# единообразно нельзя, а любая наша догадка о длине отсидки выбивала из ротации
# живого провайдера: сначала фиксированные 600с, потом эскалация с 60с — и в
# обоих случаях дайджест уезжал с 1-5 статьями из 20. Провал пула стоит один
# обход моделей, а не остаток прогона.
#
# Единственная отсидка — по Retry-After: это не догадка, а прямое указание
# провайдера, стучаться раньше бессмысленно.
_provider_dead_until = {}  # tag -> epoch из Retry-After, до которого не стучимся
_retry_after = {}          # tag -> Retry-After последнего 429, сек


def _in_cooldown(tag: str) -> bool:
    return _provider_dead_until.get(tag, 0) > time.time()


def _note_retry_after(tag: str, resp) -> None:
    """Запоминает Retry-After ответа: провайдер сам сказал, когда возвращаться.
    За обход пула заголовков прилетает несколько (у Groq/Google лимиты
    по-модельные), и берём МИНИМУМ: провайдера стоит дёрнуть снова, как только
    оживёт первая его модель, а не последняя.
    Живёт на пути обработки ошибки, поэтому не бросает: отсутствующий или
    нечисловой заголовок (HTTP-date) — не повод превращать 429 в исключение."""
    headers = getattr(resp, "headers", None) or {}
    raw = headers.get("Retry-After")
    if not raw:
        return
    try:
        hint = max(0.0, float(raw))
    except (TypeError, ValueError):
        return  # HTTP-date вместо секунд — редкий формат, шага кулдауна хватит
    _retry_after[tag] = min(_retry_after.get(tag, hint), hint)


def _mark_dead(tag: str) -> None:
    """Весь пул провайдера провалился. Отсидку даём только если он сам её попросил
    через Retry-After; иначе провайдер остаётся в кольце и пробуется снова."""
    delay = _retry_after.pop(tag, 0)
    if not delay:
        log.warning("%s: весь пул провалился — остаётся в ротации", tag)
        return
    _provider_dead_until[tag] = time.time() + delay
    log.warning("%s просит Retry-After %dс — пропускаем до восстановления", tag, delay)


def _mark_alive(tag: str) -> None:
    """Успех стирает подсказки предыдущего провала."""
    _retry_after.pop(tag, None)
    _provider_dead_until.pop(tag, None)

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
        # Считаем ДО отправки: у OpenRouter неудачные попытки тоже съедают
        # суточную квоту ":free", так что списывать только успехи нельзя.
        quota.note_request("OpenRouter")
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

        quota.note_response("OpenRouter", resp)

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
                return _strip_think(content)
            except Exception as exc:
                log.warning(
                    "_call_openrouter_raw не удалось разобрать ответ %s: %s",
                    model_id, exc,
                )
        elif resp.status_code == 429 or 500 <= resp.status_code < 600:
            log.warning("%d от %s -> смена модели", resp.status_code, model_id)
            _note_retry_after("OpenRouter", resp)
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
# всем трём (суммарной суточной квоты хватает дольше). Провайдеры из кольца не
# выбывают: пропускаем только того, кто прислал Retry-After и ещё не отсидел.
# ---------------------------------------------------------------------------
_PROVIDERS = ["OpenRouter", "Groq", "Google"]
_rr_start = 0

# Когда круг пройден и не ответил никто, вызов ЖДЁТ и пробует снова, а не
# возвращает None мгновенно. Без ожидания частотный 429 превращался в пустой дайджест:
# _process_slot не отличает "LLM недоступна" от "статья не подошла", получал None
# за микросекунды и за пару секунд сжигал весь резерв корзины, оставляя слоты
# пустыми — а прогон отчитывался как успешный. Ожидание ограничено и на один
# вызов, и суммарно на процесс: если провайдеры лежат по-настоящему, прогон
# закончится с честным "LLM недоступна", а не повиснет до утра.
RING_MAX_WAIT = 120       # сколько один вызов готов ждать оживления, сек
# Пауза между кругами, когда никто не ответил и Retry-After никто не прислал:
# провайдер не заблокирован, но идти тем же кругом в ту же микросекунду
# бессмысленно — частотный лимит отпускает за десятки секунд.
RING_RETRY_PAUSE = 30
# Бюджет калибруется по реальным Retry-After, а не по круглому числу: Groq
# 2026-07-29 просил 839с. Бюджет должен вмещать хотя бы одну такую отсидку с
# запасом, иначе прогон сдаётся за полминуты до оживления провайдера. 1800с
# держат прогон в пределах ~45 минут (обычный — около 18). Если провайдерам
# нужно больше — статей на дайджест сегодня всё равно не хватит, и честный
# отказ лучше долгого ожидания.
RING_WAIT_BUDGET = 1800   # суммарный бюджет ожидания на весь процесс, сек
_wait_spent = 0.0


def _has_key(tag: str) -> bool:
    if tag == "Groq":
        return bool(config.GROQ_API_KEY)
    if tag == "Google":
        return bool(config.GOOGLE_API_KEY)
    return bool(config.OPENROUTER_API_KEY)


def _ring_tags() -> list[str]:
    """Провайдеры с настроенным ключом: без ключа в кольцо не берём, иначе каждый
    круг тратил бы на него итерацию и сбивал расчёт времени ожидания."""
    return [t for t in _PROVIDERS if _has_key(t)]


def providers_exhausted() -> bool:
    """True, когда бюджет ожидания израсходован — круги проваливались суммарно
    RING_WAIT_BUDGET секунд, и дальше LLM-вызовы падают мгновенно. Вызывающий код
    проверяет это, чтобы не сжигать резерв корзины на заведомо провальных вызовах.
    Кулдаун провайдеров тут больше не смотрим: его нет, а бюджет тратится только
    на кругах, где не ответил никто."""
    tags = _ring_tags()
    if not tags:
        # Ключей нет вовсе — но ни локальная модель, ни llm7 ключа и не требуют.
        return not (LOCAL_LLM_URL or LLM7_MODELS)
    if _wait_spent >= RING_WAIT_BUDGET:
        # Бюджет ожидания сожжён. Сама по себе настроенная локальная модель это
        # не отменяет: путь к ней открывается ТОЛЬКО через all_cloud_exhausted,
        # то есть когда выбраны суточные лимиты. Круги, проваленные по другой
        # причине (упавшие модели, сеть), к ней не ведут — и тогда для
        # вызывающего кода это действительно "всё".
        return not ((LOCAL_LLM_URL or LLM7_MODELS)
                    and quota.all_cloud_exhausted(tags))
    return False


def _call_openrouter_raw(system_prompt: str, user_message: str, ref_url: str = "",
                         prefer_local: bool = False,
                         schema: dict | None = None) -> str | None:
    """prefer_local=True — сначала своя модель, облако только если она не ответила.

    Две разные политики для двух разных задач, и разница тут не в экономии:

      Классификация (daily_collector) — тысячи коротких вызовов за прогон.
      Ошибка в одном стоит одной статьи, зато объём такой, что локальная карта
      стала бы узким местом. Идём в облако, локальная — аварийный резерв.

      Пересказ и постобработка дайджеста (sunday_processor_mmr) — десятки
      вызовов раз в неделю, и каждый попадает в итоговый .docx, который читает
      человек. Здесь важнее предсказуемость: своя модель не кончается по
      лимиту посреди сборки и даёт стабильное качество прогон за прогоном,
      тогда как облачный пул каждый раз подсовывает другую ":free"-модель.
    """
    global _rr_start, _wait_spent

    if prefer_local and LOCAL_LLM_URL:
        out = _local_attempt(system_prompt, user_message, ref_url, schema)
        if out is not None:
            return out
        log.warning("Локальная модель не ответила — уходим в облако (%s)", ref_url)

    tags = _ring_tags()
    if not tags:
        # Ключей нет вовсе, но ступени без ключа остаются: сначала бесплатный
        # шлюз, потом своя модель. Так проект собирается на чистой машине, где
        # в .env ещё ничего не вписано.
        out = _llm7_attempt(system_prompt, user_message, ref_url)
        if out is not None:
            return out
        if LOCAL_LLM_URL and not prefer_local:
            return _local_attempt(system_prompt, user_message, ref_url, schema)
        log.error("_call_openrouter_raw: не задан ни один ключ провайдера")
        return None

    n = len(tags)
    while True:
        for i in range(n):
            tag = tags[(_rr_start + i) % n]
            if _in_cooldown(tag):
                continue
            out = _provider_call(tag, system_prompt, user_message, ref_url)
            if out is not None:
                _rr_start = (_rr_start + i + 1) % n  # следующий вызов — со следующего провайдера
                _mark_alive(tag)
                quota.note_success(tag)
                log.info("%s отдал ответ для %s", tag, ref_url)
                return out
            _mark_dead(tag)  # весь пул провайдера провалился

        # Круг пройден без ответа. Прежде чем ждать — проверяем, не выбраны ли
        # СУТОЧНЫЕ лимиты у всех трёх сразу. Если да, ждать бессмысленно (они до
        # полуночи UTC), и вызов уходит на локальную модель. Частотный 429 сюда
        # не приводит: quota различает их по длине Retry-After, и на короткой
        # отсидке мы по-прежнему ждём облако — оно даёт лучшее качество.
        if quota.all_cloud_exhausted(tags):
            log.warning("Все облачные исчерпаны (%s) — уходим на бесплатные",
                        quota.snapshot(tags))
            out = _llm7_attempt(system_prompt, user_message, ref_url)
            if out is not None:
                return out
            out = _local_attempt(system_prompt, user_message, ref_url, schema)
            if out is not None:
                return out

        # Провайдеры остаются в ротации — ждём либо до ближайшего Retry-After,
        # либо RING_RETRY_PAUSE, и идём на следующий круг.
        soonest = min(_provider_dead_until.get(t, 0) for t in tags)
        wait = min(max(soonest - time.time(), RING_RETRY_PAUSE),
                   RING_MAX_WAIT, RING_WAIT_BUDGET - _wait_spent)
        if wait <= 0:
            log.error("_call_openrouter_raw: все провайдеры исчерпаны для %s", ref_url)
            return None
        log.warning("Круг без ответа — ждём %.0fс и пробуем снова (%s)", wait, ref_url)
        time.sleep(wait)
        _wait_spent += wait


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


def _local_native_url() -> str:
    """LOCAL_LLM_URL исторически указывает на OpenAI-совместимый
    /v1/chat/completions. Родной эндпоинт Ollama лежит на том же хосте.

    Ходим именно в родной, потому что через /v1 НЕ проходят три параметра, и
    каждый из них здесь обязателен: options.num_ctx (иначе окно 4096 и молчаливая
    обрезка промпта), format (JSON-схема) и think. Это не прихоть Ollama, а
    известное ограничение её OpenAI-совместимого слоя."""
    base = LOCAL_LLM_URL.split("/v1/")[0].rstrip("/")
    return base + "/api/chat"


def _llm7_attempt(system_prompt: str, user_message: str,
                  ref_url: str = "") -> str | None:
    """Бесплатный шлюз llm7.io. Пул пробуется по очереди, как у Groq и Google.

    Частотный лимит здесь жёстче облачного (10 запросов в минуту анонимно), и
    ответ на него приходит с Retry-After: _openai_chat отдаёт его в
    _note_retry_after, а кольцо само решает, ждать или идти дальше."""
    if not LLM7_MODELS:
        return None
    return _try_pool(LLM7_MODELS, LLM7_API_URL, LLM7_API_KEY,
                     system_prompt, user_message, "LLM7")


def _local_attempt(system_prompt: str, user_message: str, ref_url: str = "",
                   schema: dict | None = None) -> str | None:
    """Своя модель на домашнем сервере.

    Ключа не требует (сервер доступен только по VPN), таймаут щедрее облачного:
    8B на одной карте отвечает медленнее, но альтернатива здесь не "быстрее", а
    "никак".

    schema — JSON-схема ожидаемого ответа. Ollama строит по ней грамматику и
    физически не может выдать невалидный JSON. Это принципиально надёжнее
    просьбы «верни только JSON» в промпте: просьбу модель забывает ровно тогда,
    когда промпт длинный, то есть в самых важных сюжетах."""
    if not LOCAL_LLM_URL:
        log.warning("LOCAL_LLM_URL не задан — локального резерва нет")
        return None
    payload = {
        "model": LOCAL_LLM_MODEL,
        "stream": False,
        "think": LOCAL_THINK,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "options": {"temperature": 0.1, "num_ctx": LOCAL_NUM_CTX},
    }
    if schema:
        payload["format"] = schema
    try:
        resp = requests.post(
            _local_native_url(),
            json=payload,
            headers={"Content-Type": "application/json"},
            # Домашний сервер в VPN: прокси тут не нужен и только мешал бы.
            proxies=None,
            timeout=LOCAL_TIMEOUT,
        )
        if resp.status_code != 200:
            log.warning("Локальная модель: HTTP %d", resp.status_code)
            return None
        data = resp.json()
        used = data.get("prompt_eval_count", 0) or 0
        made = data.get("eval_count", 0) or 0
        # Тихая обрезка промпта — та самая ошибка, ради которой всё и правилось.
        # Молчать о ней второй раз нельзя: она не падает, а портит текст.
        if used + made >= LOCAL_NUM_CTX:
            log.warning("Локальная модель: промпт %d + ответ %d упёрлись в окно %d "
                        "для %s — Ollama срежет контекст молча, поднимите "
                        "LOCAL_LLM_NUM_CTX", used, made, LOCAL_NUM_CTX, ref_url)
        content = data["message"]["content"]
        log.info("Локальная модель отдала ответ для %s (промпт %d, ответ %d)",
                 ref_url, used, made)
        return _strip_think(content)
    except Exception as exc:  # noqa: BLE001
        log.warning("Локальная модель недоступна (%s)", exc)
        return None


def _openai_chat(url: str, api_key: str, model_id: str,
                 system_prompt: str, user_message: str, tag: str = "") -> str:
    """Один OpenAI-совместимый chat/completions запрос. Бросает при не-200."""
    if tag:
        quota.note_request(tag)
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
    if tag:
        # Groq кладёт остаток лимита в заголовки КАЖДОГО ответа — единственный
        # провайдер, дающий честную цифру без ожидания отказа.
        quota.note_response(tag, resp)
    if resp.status_code != 200:
        if tag:
            _note_retry_after(tag, resp)
        raise ValueError(f"HTTP {resp.status_code}")
    return _strip_think(resp.json()["choices"][0]["message"]["content"])


def _try_pool(models, url, api_key, system_prompt, user_message, tag) -> str | None:
    """Пробуем модели пула по очереди; битую/занятую (404/429/5xx) пропускаем."""
    for mid in models:
        try:
            return _openai_chat(url, api_key, mid, system_prompt, user_message, tag)
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


