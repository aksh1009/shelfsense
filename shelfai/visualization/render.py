"""Colour-coded group boxes, group tags and a legend drawn onto the shelf photo."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .palette import group_color, to_hex


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def group_colors(groups: list[dict]) -> dict[str, str]:
    """group_id -> hex colour. Multi-member groups coloured in order; single products grey."""
    return {g["group_id"]: to_hex(group_color(i, g["num_members"] == 1)) for i, g in enumerate(groups)}


def _dashed_rect(d: ImageDraw.ImageDraw, box, color, width: int, dash: int):
    x1, y1, x2, y2 = box
    for (ax, ay, bx, by) in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        length = max(abs(bx - ax), abs(by - ay))
        steps = max(1, int(length // (2 * dash)))
        for s in range(steps + 1):
            t0, t1 = (2 * s * dash) / max(length, 1), min(1.0, (2 * s + 1) * dash / max(length, 1))
            if t0 >= 1:
                break
            d.line([ax + (bx - ax) * t0, ay + (by - ay) * t0, ax + (bx - ax) * t1, ay + (by - ay) * t1], fill=color, width=width)


def render_groups(img: Image.Image, boxes: np.ndarray, labels: np.ndarray, groups: list[dict],
                  removed_boxes: np.ndarray | None = None, max_side: int = 2000) -> Image.Image:
    s = min(1.0, max_side / max(img.size))
    canvas = img.resize((round(img.width * s), round(img.height * s)), Image.BILINEAR) if s < 1 else img.copy()
    d = ImageDraw.Draw(canvas)
    th = max(2, round(max(canvas.size) / 400))
    tag_font = _font(max(11, canvas.width // 70))
    sizes = np.bincount(labels, minlength=len(groups)) if len(labels) else np.zeros(0, int)

    if removed_boxes is not None:
        for b in np.asarray(removed_boxes).reshape(-1, 4):
            _dashed_rect(d, (np.asarray(b, np.float64) * s).tolist(), (255, 255, 255), max(1, th // 2), th * 3)

    for i in sorted(range(len(labels)), key=lambda i: (sizes[labels[i]] > 1, sizes[labels[i]])):
        g = int(labels[i])
        single = sizes[g] == 1
        col = group_color(g, single)
        x1, y1, x2, y2 = (np.asarray(boxes[i], np.float64) * s).tolist()
        d.rectangle([x1, y1, x2, y2], outline=col, width=max(1, th // 2) if single else th)
        if not single and (x2 - x1) > canvas.width / 40:
            tag = f"G{g + 1}"
            tb = d.textbbox((0, 0), tag, font=tag_font)
            d.rectangle([x1, y1, x1 + tb[2] - tb[0] + 2 * th + 2, y1 + tb[3] - tb[1] + 2 * th + 2], fill=col)
            lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
            d.text((x1 + th + 1, y1 + th - tb[1] + 1), tag, fill=(0, 0, 0) if lum > 140 else (255, 255, 255), font=tag_font)
    return _with_legend(canvas, groups, sizes, 0 if removed_boxes is None else len(removed_boxes))


def _with_legend(canvas, groups, sizes, n_removed):
    multi = [i for i, g in enumerate(groups) if g["num_members"] > 1]
    n_single = sum(1 for g in groups if g["num_members"] == 1)
    lh = max(22, canvas.height // 40)
    font, bold = _font(max(12, lh - 8)), _font(max(13, lh - 6))
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    longest = max([f"{n_single} single product{'' if n_single == 1 else 's'}", f"{n_removed} removed (containment)" if n_removed else "",
                   f"+ {len(multi)} more groups"], key=lambda t: probe.textlength(t, font=font))
    width = max(240, canvas.width // 4, int(probe.textlength(longest, font=font)) + lh + 34)  # never clip the legend
    leg = Image.new("RGB", (width, canvas.height), (38, 50, 58))
    d = ImageDraw.Draw(leg)
    y = 14
    d.text((14, y), f"{len(multi)} group{'' if len(multi) == 1 else 's'}", fill=(255, 255, 255), font=bold); y += lh
    d.text((14, y), f"{int(sizes.sum()) if len(sizes) else 0} products", fill=(190, 200, 205), font=font); y += int(lh * 1.4)
    rows = max(1, (canvas.height - y) // lh - 3)
    for i in multi[:rows]:
        d.rectangle([14, y + 3, 14 + lh - 8, y + lh - 5], fill=group_color(i))
        label = groups[i].get("label")
        d.text((lh + 14, y), f"G{i + 1}", fill=(255, 255, 255), font=bold)
        d.text((lh + 14 + int(lh * 2.1), y), f"{groups[i]['num_members']}" + (f"  {label[:16]}" if label else ""),
               fill=(190, 200, 205), font=font)
        y += lh
    if len(multi) > rows:
        d.text((14, y), f"+ {len(multi) - rows} more groups", fill=(190, 200, 205), font=font); y += lh
    if n_single:
        d.rectangle([14, y + 3, 14 + lh - 8, y + lh - 5], outline=group_color(0, True), width=2)
        d.text((lh + 14, y), f"{n_single} single product{'' if n_single == 1 else 's'}", fill=(190, 200, 205), font=font); y += lh
    if n_removed:
        _dashed_rect(d, (14, y + 3, 14 + lh - 8, y + lh - 5), (255, 255, 255), 2, 3)
        d.text((lh + 14, y), f"{n_removed} removed (containment)", fill=(190, 200, 205), font=font)
    out = Image.new("RGB", (canvas.width + width, canvas.height), (38, 50, 58))
    out.paste(canvas, (0, 0)); out.paste(leg, (canvas.width, 0))
    return out
