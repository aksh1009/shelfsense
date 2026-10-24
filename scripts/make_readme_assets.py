#!/usr/bin/env python3
"""Build the README's result images and results table from outputs/samples/.

    python scripts/run_samples.py          # with the app running: writes outputs/samples/
    python scripts/make_readme_assets.py   # then: refresh assets/ and the README results section

Writes assets/gallery.jpg (featured results in a grid) and replaces the block between <!-- results:start --> and <!-- results:end --> in README.md.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FEATURED = ["shelf_14", "shelf_06", "shelf_03", "shelf_07"]  # chosen to cover different shelf types


NAMES = {"yolo11n": "YOLO11n", "yolo11s": "YOLO11s", "yolov8l": "YOLOv8l", "clip": "CLIP ViT-B/16",
         "siglip2": "SigLIP2", "dinov2": "DINOv2"}


def font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def stage_ms(res: dict) -> tuple[str, float, float]:
    det = res["service_timings_ms"].get("detector", {})
    cached = res.get("cached", {}).get("detection", False)
    detect = "cached" if cached else f"{det.get('inference', 0) + det.get('postprocess', 0):.0f}"
    return detect, res["timings_ms"]["group"], res["timings_ms"]["total"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default=str(ROOT / "outputs" / "samples"))
    ap.add_argument("--max-side", type=int, default=1400)
    args = ap.parse_args()

    samples = Path(args.samples)
    results = sorted(samples.glob("*_result.json"))
    if not results:
        raise SystemExit(f"No *_result.json in {samples}. Run scripts/run_samples.py first.")

    rows, thumbs = [], {}
    for rj in results:
        stem = rj.name[: -len("_result.json")]
        res = json.loads(rj.read_text(encoding="utf-8"))
        vis = samples / f"{stem}_visualization.jpg"
        if vis.exists():
            im = Image.open(vis).convert("RGB")
            im.thumbnail((args.max_side, args.max_side))
            thumbs[stem] = im
        s = res["summary"]
        detect, group, total = stage_ms(res)
        rows.append(dict(stem=stem, w=res["image"]["width"], h=res["image"]["height"], products=s["num_products"],
                         groups=s["num_groups"], singles=s["num_singletons"], detect=detect, group=group, total=total,
                         res=res))

    # gallery: featured results in a 2-column grid of equal tiles (photos are a mix of portrait and landscape)
    picks = [s for s in FEATURED if s in thumbs] or list(thumbs)[:4]
    if picks:
        tw, th, gap, cols = 900, 760, 14, 2
        rows_n = (len(picks) + cols - 1) // cols
        gallery = Image.new("RGB", (cols * tw + (cols + 1) * gap, rows_n * th + (rows_n + 1) * gap), (38, 50, 58))
        for i, s in enumerate(picks):
            t = thumbs[s].copy()
            t.thumbnail((tw, th))
            x = gap + (i % cols) * (tw + gap) + (tw - t.width) // 2
            y = gap + (i // cols) * (th + gap) + (th - t.height) // 2
            gallery.paste(t, (x, y))
        gallery.save(ROOT / "assets" / "gallery.jpg", quality=84, optimize=True)

    # results section
    r0 = rows[0]["res"]
    det, grp = r0["options"]["detection"], r0["options"]["grouping"]
    dname = r0["models"].get("detector", {}).get("name", det["detector"])
    any_cached = any(r["detect"] == "cached" for r in rows)
    all_cached = all(r["detect"] == "cached" for r in rows)
    lines = [
        "<!-- results:start -->",
        f"Pipeline: **{NAMES.get(dname, dname)}** (confidence {det['conf']}, NMS IoU {det['iou']}, "
        f"containment {det['containment']}) + **{NAMES.get(grp['embedder'], grp['embedder'])}** embeddings, agglomerative clustering at "
        f"cosine distance **{grp['threshold']}**. Times are per photo on the machine that produced them, model loading excluded.",
        "",
        f"**Median per photo: {statistics.median(r['products'] for r in rows):.0f} products, "
        f"{statistics.median(r['groups'] for r in rows):.0f} groups, {statistics.median(r['total'] for r in rows) / 1000:.2f} s "
        f"end to end** ({len(rows)} photos, {sum(r['products'] for r in rows)} products in total).",
        "",
        "<details>",
        f"<summary>Per-photo results ({len(rows)} photos)</summary>",
        "",
        "| Photo | Size | Products | Groups | Single products |" + ("" if all_cached else " Detect (ms) |") + " Group (ms) | Total (ms) |",
        "| --- | --- | ---: | ---: | ---: |" + ("" if all_cached else " ---: |") + " ---: | ---: |",
    ]
    for r in rows:
        name = r["stem"] if len(r["stem"]) <= 34 else r["stem"][:31] + "..."
        link = f"[{name}](outputs/samples/{r['stem'].replace(' ', '%20')}_visualization.jpg)" if r["stem"] in thumbs else name
        det_cell = "" if all_cached else f" {r['detect']} |"
        lines.append(f"| {link} | {r['w']}x{r['h']} | {r['products']} | {r['groups']} | {r['singles']} |{det_cell} "
                     f"{r['group']:.0f} | {r['total']:.0f} |")
    med = lambda k: statistics.median(r[k] for r in rows)
    lines += [
        f"| **Median** | | **{med('products'):.0f}** | **{med('groups'):.0f}** | **{med('singles'):.0f}** |{'' if all_cached else ' |'} "
        f"**{med('group'):.0f}** | **{med('total'):.0f}** |",
        "",
        "Click a photo name for its visualization. Every visualization and its JSON are in [`outputs/samples/`](outputs/samples).",
        "",
        "</details>",
    ]
    if all_cached:
        lines += ["", "Detection times were not recorded in this run; measured detection latency is in "
                      "[Model selection](#model-selection) and the [technical report](docs/REPORT.md#latency)."]
    elif any_cached:
        lines += ["", "> Some detections were reused from cache, so their detection time is not shown. "
                      "`scripts/run_samples.py` disables the cache; re-run it to measure every stage."]
    lines.append("<!-- results:end -->")

    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    a, b = text.find("<!-- results:start -->"), text.find("<!-- results:end -->")
    if a == -1 or b == -1:
        raise SystemExit("README.md has no <!-- results:start --> / <!-- results:end --> markers.")
    readme.write_text(text[:a] + "\n".join(lines) + text[b + len("<!-- results:end -->"):], encoding="utf-8")
    print(f"Wrote assets/gallery.jpg and the README results table ({len(rows)} photos).")


if __name__ == "__main__":
    main()
