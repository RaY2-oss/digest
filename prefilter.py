# -*- coding: utf-8 -*-
"""prefilter.py — локальный отсев кандидатов перед обращением к LLM.

Принцип: LLM уже приняла тысячи решений, каждое из них — готовый размеченный
пример (текст -> да/нет). Локальная модель обучается ПОВТОРЯТЬ эти решения
(дистилляция дорогого судьи в дешёвого). Обучение — в train_prefilter.py,
здесь только применение.

Порог один, нижний: p < lo — отбрасываем без LLM, всё остальное уходит в LLM.
Верхнего порога намеренно нет. Он экономил бы мало (положительных всего ~13%
потока), но лишал бы статью региональной метки TR/CA/SC/MIX, которую ставит
тот же самый LLM-вызов. Так что классификатор режет только заведомый мусор.

lo лежит в артефакте рядом с весами и пересчитывается при каждом
переобучении под заданную config.PREFILTER_TARGET_RECALL, поэтому числа
порога в коде нет.

Пока артефакта нет — is_ready() False, и пайплайн работает как раньше.
"""
import logging
import os

import numpy as np

log = logging.getLogger("collector")

_bundle = None
_tried = False


def _load(path):
    global _bundle, _tried
    if _tried:
        return _bundle
    _tried = True
    if not os.path.exists(path):
        log.info("Предфильтр: артефакта нет (%s) — стадия пропускается", path)
        return None
    try:
        import joblib
        b = joblib.load(path)
        _bundle = b
        log.info("Предфильтр загружен: обучен на %d примерах, порог lo=%.4f, "
                 "ожидаемая полнота %.1f%%",
                 b.get("n_train", 0), b["lo"], 100 * b.get("recall", float("nan")))
    except Exception as exc:
        log.warning("Предфильтр не загрузился (%s) — стадия пропускается", exc)
    return _bundle


def is_ready(path) -> bool:
    return _load(path) is not None


def drop_mask(path, embeddings):
    """-> булев массив: True = отбросить, не тратя LLM-запрос."""
    n = len(embeddings)
    b = _load(path)
    if b is None or n == 0:
        return np.zeros(n, dtype=bool)
    X = np.asarray(embeddings, dtype=np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    try:
        p = b["clf"].predict_proba(X)[:, 1]
    except Exception as exc:
        log.warning("Предфильтр упал на предсказании (%s) — пропускаем всех в LLM", exc)
        return np.zeros(n, dtype=bool)
    return p < b["lo"]
