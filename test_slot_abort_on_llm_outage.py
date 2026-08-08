# -*- coding: utf-8 -*-
"""
test_slot_abort_on_llm_outage.py — при мёртвых LLM-провайдерах слот НЕ перебирает
весь резерв корзины.

Регрессия 2026-07-29: _process_slot не отличал "LLM недоступна" от "статья не
подошла". Провайдеры уходили в кулдаун, _call_openrouter_raw начинал отдавать
None за микросекунды, и один слот прожёвывал до MAX_RESERVE_SCAN кандидатов за
пару секунд — забирая заодно кандидатов у следующих слотов (processed_urls общий).
Дайджест уезжал с 1-5 статьями из 20 и отчитывался как успешный.

Запуск: venv/bin/python3 test_slot_abort_on_llm_outage.py
"""

import numpy as np

import model_rotation as mr
import sunday_processor_mmr as sp


def _dead_ring():
    """Провайдеры исчерпаны, бюджет ожидания израсходован: вызовы падают сразу."""
    mr._provider_dead_until.clear()
    mr._retry_after.clear()
    mr._wait_spent = mr.RING_WAIT_BUDGET
    for tag in mr._PROVIDERS:
        mr._provider_dead_until[tag] = mr.time.time() + 3600
    mr._has_key = lambda tag: True
    assert mr.providers_exhausted()


def _reserve(n):
    """n кандидатов-сюжетов, попарно непохожих — дубликат-фильтр их не срежет.

    Формат тот же, что отдаёт _group_stories: статья плюс восьмым полем
    список её версий."""
    out = []
    for i in range(n):
        emb = np.zeros(8, dtype=np.float32)
        emb[i % 8] = 1.0
        art = (f"https://example.com/a{i}", f"text {i}", emb, "2026-07-29",
               "TR", 0.0, None)
        out.append(art + ([art],))
    return out


def test_slot_stops_at_first_candidate_when_llm_is_down():
    _dead_ring()
    calls = {"n": 0}
    orig = sp._call_openrouter_raw

    def counting(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)

    sp._call_openrouter_raw = counting
    try:
        reserve = _reserve(40)
        first = reserve[0]
        result = sp._process_slot(
            primary=first,
            regional_reserves={"TR": reserve[1:]},
            processed_urls=set(),
            accepted_embeddings=[],
            slot_label="[1/20]",
        )
    finally:
        sp._call_openrouter_raw = orig

    assert result is None, "мёртвая LLM не может дать пересказ"
    # Один кандидат = MAX_RETRIES попыток пересказа, и всё: LLM-проверок темы
    # больше нет (были ещё +1 pre-check на кандидата и +1 на каждую попытку).
    # Раньше тут прожёвывался весь резерв (~40 кандидатов × те же вызовы).
    budget = sp.MAX_RETRIES
    assert calls["n"] <= budget, (
        f"слот сжёг резерв вместо остановки: {calls['n']} вызовов, ожидалось <= {budget}"
    )


def test_slot_still_scans_reserve_when_llm_is_alive():
    """Обратная сторона: пока LLM жива, отказ по статье по-прежнему берёт следующую."""
    mr._provider_dead_until.clear()
    mr._wait_spent = 0.0
    mr._has_key = lambda tag: True
    assert not mr.providers_exhausted()

    seen = []
    sp._call_openrouter_raw = lambda *a, **kw: None  # ответа нет, но провайдеры живы
    orig_retell = sp._retell_article

    def tracking(url, versions, pub_date, label):
        seen.append(url)
        return None

    sp._retell_article = tracking
    try:
        reserve = _reserve(6)
        first = reserve[0]
        result = sp._process_slot(
            primary=first,
            regional_reserves={"TR": reserve[1:]},
            processed_urls=set(),
            accepted_embeddings=[],
            slot_label="[1/20]",
        )
    finally:
        sp._retell_article = orig_retell

    assert result is None
    assert len(seen) == 6, f"живая LLM: резерв должен перебираться целиком, а не {len(seen)}"


if __name__ == "__main__":
    test_slot_stops_at_first_candidate_when_llm_is_down()
    print("ok  слот останавливается, когда LLM недоступна")
    test_slot_still_scans_reserve_when_llm_is_alive()
    print("ok  живая LLM по-прежнему перебирает резерв")
    print("\nвсе проверки пройдены")
