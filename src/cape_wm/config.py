from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config with an optional relative ``base_config`` chain."""

    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError("top-level config must be a mapping")
    base_reference = payload.pop("base_config", None)
    if base_reference is None:
        return payload
    base_path = (source.parent / base_reference).resolve()
    if base_path == source:
        raise ValueError("a config cannot inherit from itself")
    return _merge(load_config(base_path), payload)
