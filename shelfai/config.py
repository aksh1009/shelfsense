"""Configuration: configs/default.yaml (or $SHELFAI_CONFIG), overridable by environment variables.

Environment overrides use the pattern SHELFAI_<SECTION>_<KEY> for nested keys and SHELFAI_<KEY>
for top-level keys, e.g.
    SHELFAI_DEVICE=cpu
    SHELFAI_GROUPING_DISTANCE_THRESHOLD=0.15
    SHELFAI_SERVICES_DETECTOR_URL=http://detector:8001
"""
from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def _coerce(value: str, like):
    """Convert an environment string to the type of the existing config value."""
    if isinstance(like, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(like, int) and not isinstance(like, bool):
        return int(value)
    if isinstance(like, float):
        return float(value)
    return value


def _apply_env(cfg: dict) -> dict:
    for key, val in list(cfg.items()):
        if isinstance(val, dict):
            for sub, subval in list(val.items()):
                env = os.environ.get(f"SHELFAI_{key}_{sub}".upper())
                if env is not None:
                    val[sub] = _coerce(env, subval)
        else:
            env = os.environ.get(f"SHELFAI_{key}".upper())
            if env is not None:
                cfg[key] = _coerce(env, val)
    return cfg


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> dict:
    path = Path(path or os.environ.get("SHELFAI_CONFIG") or DEFAULT_CONFIG)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = _apply_env(cfg)
    for section, values in (overrides or {}).items():
        if isinstance(values, dict):
            cfg.setdefault(section, {}).update(values)
        else:
            cfg[section] = values
    # resolve project-relative paths
    for key in ("models_dir", "output_dir"):
        p = Path(cfg[key])
        cfg[key] = str(p if p.is_absolute() else PROJECT_ROOT / p)
    return cfg


@lru_cache(maxsize=1)
def get_config() -> dict:
    return load_config()


def resolve_device(device: str) -> str:
    """'auto' -> 'cuda' if available else 'cpu'; '0' -> 'cuda:0'."""
    device = str(device).strip().lower()
    if device == "cpu":
        return "cpu"
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    if device == "auto":
        return "cuda" if has_cuda else "cpu"
    if device.isdigit():
        return f"cuda:{device}" if has_cuda else "cpu"
    if device.startswith("cuda"):
        return device if has_cuda else "cpu"
    return device


def deep_copy(cfg: dict) -> dict:
    return copy.deepcopy(cfg)
