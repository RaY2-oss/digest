# -*- coding: utf-8 -*-
"""
config.py — центральный файл конфигурации проекта дайджеста.

ЗДЕСЬ хранятся:
  - фильтры сбора статей из GDELT GKG (QUERIES_GKG), используются daily_collector.py;
  - пути к БД, логам, выходным файлам;
  - прокси Xray (socks5h);
  - секреты OpenRouter и Telegram (заполняются вручную ниже).

ВАЖНО про socks5h: суффикс "h" означает, что DNS-резолвинг
выполняется на стороне прокси (Xray), а не локально. Это обязательно
для корректного обхода блокировок — иначе домены резолвятся на сервере
и трафик частично идёт мимо прокси.
"""

import os


def _load_dotenv(path: str) -> None:
    """Минимальный парсер .env без внешних зависимостей: строки вида KEY=VALUE.
    Уже заданные переменные окружения не перетираются (os.environ.setdefault),
    поэтому значение можно переопределить через окружение процесса/systemd."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


_load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ---------------------------------------------------------------------------
# Прокси Xray (SOCKS5 с удалённым DNS). Отдельный inbound порт 10800.
# Порт 10443 — VLESS/REALITY backend nginx, SOCKS5 не поддерживает.
# ---------------------------------------------------------------------------
PROXIES = None  # прямой доступ, WARP недоступен из AlexHost

# ---------------------------------------------------------------------------
# Пути проекта. Вычисляются от расположения этого файла, поэтому остаются
# абсолютными (нужно для cron), но не привязаны к /opt/digest — проект можно
# развернуть в любом каталоге.
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "digest.db")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# ---------------------------------------------------------------------------
# Модель эмбеддингов (скачается автоматически в ~/.cache/huggingface/ при
# первом запуске, ~90 МБ). Размерность вектора = 384 (float32).
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384

# Порог отсева слишком коротких статей (символов).
MIN_TEXT_LENGTH = 300

# Итоговый размер дайджеста: sunday_processor_mmr.py отбирает эту сумму
# статей по квотам QUOTA = {"TR": 7, "CA": 7, "SC": 6} (+ добор из MIX при дефиците).
N_CLUSTERS = 20

# ---------------------------------------------------------------------------
# СЕКРЕТЫ — ЗАПОЛНИТЬ ВРУЧНУЮ.
# ---------------------------------------------------------------------------
# 1) OPENROUTER_API_KEY  — задаётся в /opt/digest/.env (не в этом файле).
#    Как получить:
#      - Зайти на https://openrouter.ai/ , нажать Sign In (можно через Google).
#      - Открыть https://openrouter.ai/keys , нажать "Create Key".
#      - Скопировать ключ вида "sk-or-v1-..." в .env строкой OPENROUTER_API_KEY=...
#        Бесплатные модели (:free) доступны без пополнения баланса (суточные лимиты).
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# 1b) GROQ_API_KEY — платный фолбэк, когда суточный лимит :free-моделей
#     OpenRouter исчерпан (429 free-models-per-day). Ключ вида "gsk_..." с
#     https://console.groq.com/keys , в .env строкой GROQ_API_KEY=...
#     Пусто -> фолбэк выключен, поведение как раньше (вернём None).
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# 1c) GOOGLE_API_KEY — следующий фолбэк после Groq (Google AI Studio / Gemini).
#     Ключ с https://aistudio.google.com/apikey , в .env строкой GOOGLE_API_KEY=...
#     Пусто -> этот фолбэк выключен.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# 2) TELEGRAM_BOT_TOKEN  — задаётся в /opt/digest/.env (не в этом файле).
#    Как создать бота:
#      - В Telegram открыть чат с @BotFather.
#      - Команда /newbot -> задать имя (например "IV RAN Digest")
#        и username (должен заканчиваться на "bot", напр. iv_ran_digest_bot).
#      - BotFather пришлёт токен вида "123456789:AAExxxxxxxxxxxxxxxxxxxxxxxx".
#        Скопировать его в .env строкой TELEGRAM_BOT_TOKEN=...
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# Список Telegram user_id, которым разрешён ручной запуск /rundigest
ALLOWED_TG_USER_IDS = [
    1161066987  # ты
]

# 3) TELEGRAM_CHAT_ID
#    Как получить chat_id личного чата / группы / канала:
#      Вариант А (личный чат):
#        - Написать боту любое сообщение.
#        - Открыть в браузере (можно через прокси):
#          https://api.telegram.org/bot<ТОКЕН>/getUpdates
#        - Найти "chat":{"id": 123456789 ...} — это chat_id (для себя это
#          положительное число).
#      Вариант Б (группа):
#        - Добавить бота в группу, отправить сообщение.
#        - В getUpdates chat_id группы будет ОТРИЦАТЕЛЬНЫМ (напр. -1001234567890).
#      Вариант В (канал):
#        - Сделать бота администратором канала.
#        - Опубликовать пост, посмотреть getUpdates -> "channel_post" -> chat_id
#          (для каналов формата -100...).
#    chat_id можно хранить как строку.
TELEGRAM_CHAT_ID = "1161066987"

# ---------------------------------------------------------------------------
# Фильтры сбора статей из GDELT GKG (15-минутные дампы). Индекс в списке =
# query_index в БД. daily_collector.py прогоняет каждый фильтр по themes
# (GDELT GKG theme-коды) и locations (двухбуквенные FIPS-коды стран:
# TU=Турция, KZ=Казахстан, UZ=Узбекистан, TX=Туркменистан, KG=Киргизия,
# TI=Таджикистан, GG=Грузия, AM=Армения, AJ=Азербайджан).
# ---------------------------------------------------------------------------
QUERIES_GKG = [
    {   # Запрос 0 — наука, образование, молодёжь/студенты в Турции, ЦА и ЮК
        #
        # ВАЖНО: раньше здесь были ещё TAX_FNCACT_STUDENT/TEACHER/PROFESSOR/
        # RESEARCHER/CHILD/CHILDREN. Это НЕ тематические теги, а теги типа
        # актора у GDELT — они срабатывают на любое упоминание "учителя"
        # или "студента" в тексте (например, жертва в криминальной хронике),
        # никак не проверяя, что статья ДЕЙСТВИТЕЛЬНО о науке/образовании.
        # Из-за этого в БД попадала масса нерелевантного мусора ещё на этапе
        # дешёвого пред-фильтра, до полноценной LLM-проверки. Убраны —
        # реальную релевантность отсеивает LLM-фильтр (см. daily_collector.py).
        "themes": [
            "EDUCATION",
            "WB_470_EDUCATION",
            "SOC_POINTSOFINTEREST_SCHOOL",
            "SOC_POINTSOFINTEREST_UNIVERSITY",
            "SOC_POINTSOFINTEREST_COLLEGE",
            "SCIENCE",
        ],
        "locations": ["TU", "KZ", "UZ", "TX", "KG", "TI", "GG", "AM", "AJ"],
    },
    {   # Запрос 1 — действия внешних акторов в ЦА и ЮК
        "themes": [
            "EDUCATION",
            "WB_470_EDUCATION",
            "SCIENCE",
            "WB_831_GOVERNANCE",
        ],
        "locations": ["KZ", "UZ", "TX", "KG", "TI", "GG", "AM", "AJ"],
    },
]

