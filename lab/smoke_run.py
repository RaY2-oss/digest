# -*- coding: utf-8 -*-
"""Полный прогон пайплайна на 3 тиках GKG вместо 96 — проверка всех стадий
включая реальный вызов LLM. Пишет в боевую БД, как обычный прогон."""
import sys
sys.path.insert(0, "/opt/digest")
import daily_collector as dc

_real = dc.gkg_timestamps
dc.gkg_timestamps = lambda hours=24: _real(hours=24)[-3:]
dc.main()
