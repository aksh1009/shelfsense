#!/usr/bin/env python3
"""Latency and throughput of the running API under concurrent load, for one chosen method.

    python scripts/benchmark.py                          # 20 sample images, concurrency 1, 2, 4
    python scripts/benchmark.py --concurrency 1 4 8 --requests 40

Reports per-stage server timings (p50 / p95), end-to-end latency and throughput (images/s) per concurrency
level, and writes outputs/benchmark.json. Run it on the hardware you want to report, and state that hardware.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
STAGES = ["decode", "detect", "group", "visualize", "save", "total"]
OPTIONS = "{}"


def pct(vals, q):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(round(q / 100 * (len(vals) - 1))))] if vals else 0.0


def one(url, path):
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        r = requests.post(f"{url}/api/v1/analyze", files={"image": (path.name, f, "image/jpeg")},
                          data={"options": OPTIONS}, timeout=600)
    wall = (time.perf_counter() - t0) * 1000
    body = r.json()
    return wall, body.get("timings_ms", {}), r.status_code == 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--options", default="{}",
                    help='method options as JSON, e.g. \'{"detector": "yolo11s", "method": "ocr_weighted", "embedder": "clip"}\' '
                         "(anything not given uses configs/default.yaml)")
    ap.add_argument("--images", default=str(ROOT / "sample_images"))
    ap.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--requests", type=int, default=20, help="requests per concurrency level")
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()
    try:
        requests.get(f"{args.url}/api/v1/health", timeout=5)
    except requests.ConnectionError:
        raise SystemExit(f"The app isn't running at {args.url}. Start it in another terminal: python scripts/run_local.py")
    global OPTIONS
    OPTIONS = json.dumps({**json.loads(args.options), "no_cache": True})  # measure real detection every time
    paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    health = requests.get(f"{args.url}/api/v1/health", timeout=10).json()
    for p in paths[:args.warmup]:
        one(args.url, p)

    report = {"machine": platform.platform(), "python": platform.python_version(), "health": health, "levels": []}
    for c in args.concurrency:
        jobs = [paths[i % len(paths)] for i in range(args.requests)]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=c) as ex:
            results = list(ex.map(lambda p: one(args.url, p), jobs))
        elapsed = time.perf_counter() - t0
        ok = [r for r in results if r[2]]
        level = {"concurrency": c, "requests": len(jobs), "ok": len(ok), "throughput_img_per_s": round(len(ok) / elapsed, 2),
                 "wall_ms": {"p50": round(pct([r[0] for r in ok], 50), 1), "p95": round(pct([r[0] for r in ok], 95), 1)},
                 "server_ms": {s: {"p50": round(pct([r[1].get(s, 0) for r in ok], 50), 1),
                                   "p95": round(pct([r[1].get(s, 0) for r in ok], 95), 1)} for s in STAGES}}
        report["levels"].append(level)
        print(f"concurrency {c}: {level['throughput_img_per_s']} img/s | end-to-end p50 {level['wall_ms']['p50']} ms, "
              f"p95 {level['wall_ms']['p95']} ms | detect p50 {level['server_ms']['detect']['p50']} ms, "
              f"group p50 {level['server_ms']['group']['p50']} ms")
    out = ROOT / "outputs" / "benchmark.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
