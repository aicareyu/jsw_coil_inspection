from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"

_CACHE: dict[str, Any] | None = None


def load_config(path: Path | None = None) -> dict[str, Any]:
    global _CACHE
    cfg_path = Path(path) if path else CONFIG_PATH
    if _CACHE is not None and path is None:
        return _CACHE
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing configuration file: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if path is None:
        _CACHE = data
    return data


def project_root() -> Path:
    return ROOT


def resolve_path(rel: str | Path) -> Path:
    path = Path(rel)
    return path if path.is_absolute() else ROOT / path
