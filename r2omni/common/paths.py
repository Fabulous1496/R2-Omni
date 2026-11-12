"""Portable defaults shared by installed runners and source checkouts."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_workspace(value: str | Path | None = None) -> Path:
    return Path(value or os.environ.get("R2OMNI_WORKSPACE") or project_root()).expanduser().resolve()


def resolve_third_party(name: str, workspace: str | Path | None = None) -> Path:
    """Prefer dependencies installed with this checkout; retain legacy layouts."""
    named_variable = {"KIVI": "KIVI_ROOT", "OTT": "OTT_ROOT", "OScaR-KV-Quant": "OSCAR_ROOT"}.get(name)
    if named_variable and os.environ.get(named_variable):
        return Path(os.environ[named_variable]).expanduser().resolve()
    dependency_root = os.environ.get("R2OMNI_THIRD_PARTY_DIR")
    if dependency_root:
        return Path(dependency_root).expanduser().resolve() / name
    local = project_root() / "third_party" / name
    legacy = resolve_workspace(workspace) / "Baselines" / name
    return local if local.exists() or not legacy.exists() else legacy
