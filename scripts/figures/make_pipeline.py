#!/usr/bin/env python3
"""Draw assets/pipeline-light.png and assets/pipeline-dark.png: one real shelf photo at every stage of the pipeline.

    python scripts/figures/make_pipeline.py [--photo shelf_06]

Reads sample_images/<photo>.jpg and outputs/samples/<photo>_result.json (run scripts/run_samples.py first).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shelfai.imaging import crop_boxes, decode_image, letterbox  # noqa: E402
from shelfai.visualization.palette import group_color  # noqa: E402

THEMES = {
    "light": dict(bg=(255, 255, 255), card=(246, 248, 250), line=(209, 217, 224), ink=(31, 35, 40),
                  muted=(89, 99, 110), arrow=(129, 139, 152), tile=(234, 238, 242)),
    "dark": dict(bg=(13, 17, 23), card=(21, 27, 35), line=(61, 68, 77), ink=(230, 237, 243),
                 muted=(145, 152, 161), arrow=(125, 133, 144), tile=(33, 40, 48)),
}
S = 2  # render at 2x for sharpness


def font(size, bold=False):
    for name in (("DejaVuSans-Bold.ttf", "arialbd.ttf") if bold else ("DejaVuSans.ttf", "arial.ttf")):
        try:
            return ImageFont.truetype(name, int(size * S))
        except OSError:
            continue
    return ImageFont.load_default()


def fit(im, w, h):
    im = im.copy()
    im.thumbnail((w, h))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", default="shelf_06")
    args = ap.parse_args()
    res = json.loads((ROOT / "outputs" / "samples" / f"{args.photo}_result.json").read_text(encoding="utf-8"))
    img = decode_image((ROOT / "sample_images" / res["image"]["filename"]).read_bytes())
    boxes = np.array([[d["bbox"][k] for k in ("x1", "y1", "x2", "y2")] for d in res["detections"]], np.float32)
    gid = {g["group_id"]: i for i, g in enumerate(res["groups"])}
    labels = np.array([gid[d["group_id"]] for d in res["detections"]])
    crops = crop_boxes(img, boxes)
    vis = Image.open(ROOT / "outputs" / "samples" / f"{args.photo}_visualization.jpg").convert("RGB")
    n_groups = res["summary"]["num_groups"]

    for name, t in THEMES.items():
        PW, PH, GAP, TOP, BOT = 250, 420, 110, 104, 92
        W = 24 * 2 + 5 * PW + 4 * GAP
        H = TOP + PH + BOT
        canvas = Image.new("RGB", (W * S, H * S), t["bg"])
        d = ImageDraw.Draw(canvas)
        d.text((24 * S, 22 * S), "From shelf photo to product groups", font=font(28, True), fill=t["ink"])
        d.text((24 * S, 62 * S), f"One test photo through every stage: {res['summary']['num_products']} products, "
               f"{n_groups} groups", font=font(18), fill=t["muted"])
        xs = [24 + i * (PW + GAP) for i in range(5)]
        titles = [("1  Shelf photo", f"{img.width} x {img.height} px"),
                  ("2  Detect", f"{len(boxes)} products found"),
                  ("3  Crop", "padded to squares"),
                  ("4  Embed + cluster", "one row per group"),
                  ("5  Result", "colour = group")]
        steps = ["YOLO11n", "crop", "CLIP", "draw"]

        def panel(i):
            x = xs[i]
            d.rounded_rectangle([x * S, TOP * S, (x + PW) * S, (TOP + PH) * S], radius=10 * S, fill=t["card"],
                                outline=t["line"], width=S)
            d.text((x * S, (TOP + PH + 14) * S), titles[i][0], font=font(21, True), fill=t["ink"])
            d.text((x * S, (TOP + PH + 44) * S), titles[i][1], font=font(16), fill=t["muted"])
            return x

        def paste_center(im, x, y, w, h):
            im = fit(im, w * S, h * S)
            canvas.paste(im, (int(x * S + (w * S - im.width) / 2), int(y * S + (h * S - im.height) / 2)))
            return (int(x * S + (w * S - im.width) / 2), int(y * S + (h * S - im.height) / 2), im.width / img.width)

        inner = (PW - 20, PH - 20)
        # 1 photo
        x = panel(0); paste_center(img, x + 10, TOP + 10, *inner)
        # 2 detections: every box in one colour
        x = panel(1)
        ox, oy, sc = paste_center(img, x + 10, TOP + 10, *inner)
        for b in boxes:
            d.rectangle([ox + b[0] * sc, oy + b[1] * sc, ox + b[2] * sc, oy + b[3] * sc],
                        outline=(255, 206, 58), width=max(2, S))
        # 3 crops grid
        x = panel(2)
        cols, cell = 4, (PW - 20 - 3 * 6) // 4
        for k, c in enumerate(crops[: cols * 6]):
            cx, cy = x + 10 + (k % cols) * (cell + 6), TOP + 12 + (k // cols) * (cell + 6)
            if cy + cell > TOP + PH - 8:
                break
            canvas.paste(letterbox(c, cell * S, fill=t["tile"]), (cx * S, cy * S))
        # 4 groups: one row per group, colour bar + members
        x = panel(3)
        multi = [g for g in range(len(res["groups"])) if res["groups"][g]["num_members"] > 1][:7]
        row_h, cell4 = 54, 46
        for r, g in enumerate(multi):
            ry = TOP + 12 + r * (row_h + 2)
            d.rounded_rectangle([(x + 10) * S, ry * S, (x + 16) * S, (ry + cell4) * S], radius=3 * S,
                                fill=group_color(g))
            d.text(((x + 22) * S, (ry + 16) * S), f"G{g + 1}", font=font(14, True), fill=t["ink"])
            members = list(np.flatnonzero(labels == g))[:4]
            for m, idx in enumerate(members):
                canvas.paste(letterbox(crops[idx], cell4 * S, fill=t["tile"]), ((x + 52 + m * (cell4 + 4)) * S, ry * S))
        # 5 result
        x = panel(4); paste_center(vis, x + 10, TOP + 10, *inner)
        # arrows with the model used between stages
        for i, label in enumerate(steps):
            ax0, ax1, ay = xs[i] + PW + 6, xs[i + 1] - 6, TOP + PH / 2
            d.line([ax0 * S, ay * S, ax1 * S, ay * S], fill=t["arrow"], width=2 * S)
            d.polygon([(ax1 * S, ay * S), ((ax1 - 9) * S, (ay - 5) * S), ((ax1 - 9) * S, (ay + 5) * S)], fill=t["arrow"])
            tw = d.textlength(label, font=font(16))
            d.text(((ax0 + ax1) / 2 * S - tw / 2, (ay - 30) * S), label, font=font(16), fill=t["muted"])
        canvas = canvas.resize((W * S // 1, H * S // 1))
        canvas.save(ROOT / "assets" / f"pipeline-{name}.png", optimize=True)
    print("wrote assets/pipeline-light.png and pipeline-dark.png")


if __name__ == "__main__":
    main()
