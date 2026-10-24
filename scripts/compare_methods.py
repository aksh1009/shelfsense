#!/usr/bin/env python3
"""Run the alternative detectors and grouping methods on the same photos, through the running app.

The app itself (python scripts/run_local.py) always runs YOLO11n + CLIP. This script sends the same photos with
per-request overrides, so every alternative runs through the same services, and saves the results side by side.

    python scripts/run_local.py                         # terminal 1: start the app as usual
    python scripts/compare_methods.py                   # terminal 2: every method on every sample photo
    python scripts/compare_methods.py --images path/to/photos --methods clip dinov2 ocr
    python scripts/compare_methods.py --list            # show the method names

Writes outputs/methods/<method>/<photo>_visualization.jpg and _result.json for each method, plus
outputs/methods/latency.csv and outputs/methods/report.md (median and 95th-percentile time per stage).

The first run of each alternative model downloads it (YOLO11s, YOLOv8l, SigLIP2, DINOv2, EasyOCR: about 2 GB in
total) and loads it into the running services; that loading time is excluded from the latency numbers.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent

# name: (description, options sent to the API). Thresholds are the values used in the write-up.
METHODS = {
    "clip":     ("YOLO11n + CLIP + agglomerative (the app's pipeline)",
                 {"detector": "yolo11n", "method": "agglomerative", "embedder": "clip", "threshold": 0.20}),
    "yolo11s":  ("YOLO11s + CLIP + agglomerative",
                 {"detector": "yolo11s", "method": "agglomerative", "embedder": "clip", "threshold": 0.20}),
    "yolov8l":  ("YOLOv8l + CLIP + agglomerative",
                 {"detector": "yolov8l", "method": "agglomerative", "embedder": "clip", "threshold": 0.20}),
    "dinov2":   ("YOLO11n + DINOv2 + agglomerative",
                 {"detector": "yolo11n", "method": "agglomerative", "embedder": "dinov2", "threshold": 0.40}),
    "siglip2":  ("YOLO11n + SigLIP2 + agglomerative",
                 {"detector": "yolo11n", "method": "agglomerative", "embedder": "siglip2", "threshold": 0.12}),
    "hdbscan":  ("YOLO11n + CLIP + HDBSCAN",
                 {"detector": "yolo11n", "method": "hdbscan", "embedder": "clip", "min_cluster_size": 2}),
    "ocr":      ("YOLO11n + CLIP + OCR weighted",
                 {"detector": "yolo11n", "method": "ocr_weighted", "embedder": "clip", "threshold": 0.20, "text_weight": 0.3}),
}
STAGES = ("detect", "embed", "ocr", "cluster", "total")


def stage_ms(res: dict) -> dict:
    """Time per stage for one response, with one-off model loading taken out."""
    d = res["service_timings_ms"].get("detector", {})
    g = res["service_timings_ms"].get("grouping", {})
    load = d.get("model_load", 0) + g.get("embedder_load", 0) + g.get("ocr_load", 0)
    return {"detect": d.get("inference", 0) + d.get("postprocess", 0), "embed": g.get("embed", 0),
            "ocr": g.get("ocr", 0), "cluster": g.get("cluster", 0) + g.get("crop", 0),
            "total": res["timings_ms"]["total"] - load}


def pct(values, q):
    v = sorted(values)
    return v[min(len(v) - 1, int(round(q / 100 * (len(v) - 1))))] if v else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--images", default=str(ROOT / "sample_images"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "methods"))
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--list", action="store_true", help="print the method names and exit")
    args = ap.parse_args()

    if args.list:
        for name, (desc, opts) in METHODS.items():
            print(f"  {name:9s} {desc}   {json.dumps(opts)}")
        return
    try:
        requests.get(f"{args.url}/api/v1/health", timeout=5)
    except requests.ConnectionError:
        sys.exit(f"The app isn't running at {args.url}. Start it in another terminal: python scripts/run_local.py")

    photos = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not photos:
        sys.exit(f"No photos in {args.images}")
    out = Path(args.out)
    rows = []
    for name in args.methods:
        desc, opts = METHODS[name]
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        print(f"\n{name}: {desc}")
        timings, groups = [], []
        for photo in photos:
            with open(photo, "rb") as f:
                r = requests.post(f"{args.url}/api/v1/analyze", files={"image": (photo.name, f, "image/jpeg")},
                                  data={"options": json.dumps({**opts, "no_cache": True})}, timeout=900)
            body = r.json()
            if r.status_code != 200:
                print(f"  FAILED {photo.name}: {body.get('error', {}).get('message')}")
                continue
            vis = requests.get(f"{args.url}{body['visualization']['url']}", timeout=60)
            (folder / f"{photo.stem}_visualization.jpg").write_bytes(vis.content)
            (folder / f"{photo.stem}_result.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
            t = stage_ms(body)
            timings.append(t)
            groups.append(body["summary"]["num_groups"])
            print(f"  {photo.name[:48]:48s} {body['summary']['num_products']:4d} products, "
                  f"{body['summary']['num_groups']:3d} groups   {t['total']:7.0f} ms")
        if not timings:
            continue
        runs = timings[1:] if len(timings) > 1 else timings  # first photo warms the model up
        row = {"method": name, "description": desc, "photos": len(timings), "groups_mean": round(statistics.mean(groups), 1)}
        for s in STAGES:
            row[f"{s}_p50_ms"] = round(pct([t[s] for t in runs], 50), 1)
            row[f"{s}_p95_ms"] = round(pct([t[s] for t in runs], 95), 1)
        rows.append(row)

    if not rows:
        sys.exit("No method produced results.")
    with open(out / "latency.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["# Method comparison", "",
             f"{len(photos)} photos. Median (95th percentile) per photo in ms; the first photo of each method is a "
             "warm-up and one-off model loading is excluded.", "",
             "| method | detect | embed | OCR | cluster | total | groups per photo |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['description']} | {r['detect_p50_ms']} ({r['detect_p95_ms']}) | {r['embed_p50_ms']} | "
                     f"{r['ocr_p50_ms'] or '-'} | {r['cluster_p50_ms']} | **{r['total_p50_ms']}** ({r['total_p95_ms']}) | "
                     f"{r['groups_mean']} |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved per-method visualizations to {out}/<method>/ and the latency table to {out / 'report.md'}")


if __name__ == "__main__":
    main()
