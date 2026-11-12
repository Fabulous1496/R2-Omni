"""Read evaluator credentials without embedding account details in artifacts."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_api_credentials(path: Path | None = None, pair: int = 1) -> tuple[str, str]:
    key = os.environ.get("OPENAI_API_KEY")
    base = os.environ.get("OPENAI_BASE_URL")
    if key:
        return (base or "https://api.openai.com/v1").rstrip("/"), key
    if path is None:
        raise ValueError("Set OPENAI_API_KEY (and optional OPENAI_BASE_URL), or pass --api-file.")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if pair < 1 or len(lines) % 2 or len(lines) < pair * 2:
        raise ValueError(f"API pair {pair} is unavailable or the credentials file is malformed")
    selected_base, selected_key = lines[2 * (pair - 1):2 * pair]
    return (base or selected_base).rstrip("/"), selected_key
