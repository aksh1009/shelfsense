#!/usr/bin/env python3
"""Send every sample image through the running API and save the outputs (the deliverable images).

    python scripts/run_samples.py                      # uses http://localhost:8000 and ./sample_images
    python scripts/run_samples.py --images path/to/photos --out outputs/samples

Writes <out>/<image>_visualization.jpg and <out>/<image>_result.json, plus <out>/summary.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--options", default="{}",
                    help='method options as JSON, e.g. \'{"detector": "yolo11s", "method": "ocr_weighted", "embedder": "clip"}\' '
                         "(anything not given uses configs/default.yaml)")
    ap.add_argument("--images", default=str(ROOT / "sample_images"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "samples"))
    args = ap.parse_args()
    try:
        requests.get(f"{args.url}/api/v1/health", timeout=5)
    except requests.ConnectionError:
        raise SystemExit(f"The app isn't running at {args.url}. Start it in another terminal: python scripts/run_local.py")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not paths:
        raise SystemExit(f"No images in {args.images}")
    rows = []
    for p in paths:
        t0 = time.perf_counter()
        with open(p, "rb") as f:
            r = requests.post(f"{args.url}/api/v1/analyze", files={"image": (p.name, f, "image/jpeg")},
                              data={"options": json.dumps({**json.loads(args.options), "no_cache": True})},
                              timeout=600)  # no_cache: measure real detection time for every photo
        wall = (time.perf_counter() - t0) * 1000
        body = r.json()
        if r.status_code != 200:
            print(f"  FAILED {p.name}: {body.get('error', {}).get('message')}")
            continue
        stem = p.stem.replace(" ", "_")
        vis = requests.get(f"{args.url}{body['visualization']['url']}", timeout=60)
        (out / f"{stem}_visualization.jpg").write_bytes(vis.content)
        (out / f"{stem}_result.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
        s, t = body["summary"], body["timings_ms"]
        rows.append(dict(image=p.name, products=s["num_products"], groups=s["num_groups"], singletons=s["num_singletons"],
                         detect_ms=t["detect"], group_ms=t["group"], total_server_ms=t["total"], total_wall_ms=round(wall, 1)))
        print(f"  {p.name[:50]:50s} {s['num_products']:4d} products -> {s['num_groups']:3d} groups + "
              f"{s['num_singletons']:3d} singles   {t['total']:7.0f} ms")
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved {len(rows)} visualizations + JSON to {out}")


if __name__ == "__main__":
    main()
