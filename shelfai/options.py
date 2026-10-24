"""Per-request method options: validation, ranges and defaults. The same rules apply to the API and the web page."""
from __future__ import annotations

from .errors import InvalidRequest

GROUPING_METHODS = {
    "agglomerative": "Visual clusters (agglomerative)",
    "hdbscan": "Visual clusters (HDBSCAN)",
    "ocr_weighted": "Visual + OCR text (weighted)",
}
RANGES = {  # name: (min, max)
    "conf": (0.01, 0.95), "iou": (0.1, 0.95), "containment": (0.3, 0.99),
    "threshold": (0.01, 1.0), "text_weight": (0.0, 1.0), "min_cluster_size": (2, 10),
}


def _num(raw: dict, key: str, default, cast=float):
    v = raw.get(key, default)
    try:
        v = cast(v)
    except (TypeError, ValueError) as e:
        raise InvalidRequest(f"'{key}' must be a number.") from e
    lo, hi = RANGES[key]
    if not (lo <= v <= hi):
        raise InvalidRequest(f"'{key}' must be between {lo} and {hi} (got {v}).")
    return v


def _choice(raw: dict, key: str, default, allowed):
    v = str(raw.get(key, default)).strip().lower()
    if v not in allowed:
        raise InvalidRequest(f"'{key}' must be one of: {', '.join(allowed)} (got '{v}').")
    return v


def parse_detection(raw: dict, cfg: dict) -> dict:
    d = cfg["detection"]["defaults"]
    cont = raw.get("containment", d["containment"])
    containment = 0.0 if cont in (0, "0", "off", "false", False, None, "") else _num({"containment": cont}, "containment", 0)
    return {
        "detector": _choice(raw, "detector", d["detector"], list(cfg["detection"]["models"])),
        "conf": _num(raw, "conf", d["conf"]),
        "iou": _num(raw, "iou", d["iou"]),
        "containment": containment,
    }


def parse_grouping(raw: dict, cfg: dict) -> dict:
    d = cfg["grouping"]["defaults"]
    return {
        "method": _choice(raw, "method", d["method"], list(GROUPING_METHODS)),
        "embedder": _choice(raw, "embedder", d["embedder"], list(cfg["grouping"]["embedders"])),
        "threshold": _num(raw, "threshold", d["threshold"]),
        "text_weight": _num(raw, "text_weight", d["text_weight"]),
        "min_cluster_size": _num(raw, "min_cluster_size", d["min_cluster_size"], int),
        "no_cache": str(raw.get("no_cache", "false")).lower() in {"1", "true", "yes", "on"},  # also re-run OCR
    }


def parse_options(raw: dict | None, cfg: dict) -> dict:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise InvalidRequest("'options' must be a JSON object.")
    return {"detection": parse_detection(raw, cfg), "grouping": parse_grouping(raw, cfg),  # grouping carries no_cache
            "show_removed": str(raw.get("show_removed", "false")).lower() in {"1", "true", "yes", "on"},
            "no_cache": str(raw.get("no_cache", "false")).lower() in {"1", "true", "yes", "on"}}
