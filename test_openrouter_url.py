# -*- coding: utf-8 -*-
"""Проверка, что URL chat/completions берётся из окружения, а без него — прямой OpenRouter.

Запуск: ./venv/bin/python test_openrouter_url.py

Каждый случай — отдельный процесс: URL вычисляется на уровне модуля, поэтому
подменять os.environ после import бессмысленно.
"""

import os
import subprocess
import sys

DIRECT = "https://openrouter.ai/api/v1/chat/completions"
VIA_PROXY = "http://127.0.0.1:8787/v1/chat/completions"

CASES = [
    ("model_rotation", "OPENROUTER_API_URL"),
    ("daily_collector", "_OR_API"),
]


def _url(module: str, attr: str, override: str | None) -> str:
    env = dict(os.environ)
    env.pop("OPENROUTER_API_URL", None)
    if override:
        env["OPENROUTER_API_URL"] = override
    out = subprocess.run(
        [sys.executable, "-c", f"import {module}; print({module}.{attr})"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip().splitlines()[-1]


def demo() -> None:
    for module, attr in CASES:
        # config._load_dotenv использует setdefault, поэтому заданная извне
        # переменная выигрывает у .env — это и проверяем.
        assert _url(module, attr, VIA_PROXY) == VIA_PROXY, module
        print(f"{module}.{attr}: override OK")

    # Дефолт проверяем, только если .env не переопределяет URL глобально.
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    dotenv_sets_url = os.path.exists(env_path) and any(
        line.strip().startswith("OPENROUTER_API_URL=")
        for line in open(env_path, encoding="utf-8")
    )
    if dotenv_sets_url:
        print("default: пропущено — OPENROUTER_API_URL задан в .env")
        return
    for module, attr in CASES:
        assert _url(module, attr, None) == DIRECT, module
        print(f"{module}.{attr}: default OK")


if __name__ == "__main__":
    demo()
    print("ok")
