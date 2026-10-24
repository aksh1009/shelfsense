#!/usr/bin/env python3
"""Download and test-load models into ./models, so the first analysis isn't slow.

    python scripts/download_models.py                 # everything (3 detectors, 3 embedders, OCR)
    python scripts/download_models.py --skip-ocr      # without EasyOCR
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from shelfai.config import load_config, resolve_device  # noqa: E402
from shelfai.detection import DetectorPool  # noqa: E402
from shelfai.grouping import GroupingEngine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-ocr", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    device = args.device or cfg["device"]
    print(f"device: {resolve_device(device)} | models dir: {cfg['models_dir']}")
    pool = DetectorPool(cfg["detection"], cfg["models_dir"], device)
    for name in cfg["detection"]["models"]:
        _, ms = pool.get(name); print(f"  detector {name:10s} ready ({ms / 1000:.0f} s)")
    engine = GroupingEngine(cfg["grouping"], cfg["models_dir"], device)
    for name in cfg["grouping"]["embedders"]:
        _, ms = engine.embedder(name); print(f"  embedder {name:10s} ready ({ms / 1000:.0f} s)")
    if not args.skip_ocr:
        _, ms = engine.ocr(); print(f"  ocr      easyocr    ready ({ms / 1000:.0f} s)")
    print("All models downloaded and working.")


if __name__ == "__main__":
    main()
