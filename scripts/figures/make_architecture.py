#!/usr/bin/env python3
"""Draw assets/architecture-light.svg and assets/architecture-dark.svg (the README picks one by theme)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THEMES = {
    "light": dict(bg="#ffffff", card="#f6f8fa", line="#d1d9e0", ink="#1f2328", muted="#59636e",
                  gateway="#2a78d6", detector="#eb6834", grouping="#1baf7a", arrow="#818b98"),
    "dark": dict(bg="#0d1117", card="#151b23", line="#3d444d", ink="#e6edf3", muted="#9198a1",
                 gateway="#3987e5", detector="#d95926", grouping="#199e70", arrow="#7d8590"),
}
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


def card(x, y, w, h, color, title, sub, lines, t):
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{t["card"]}" stroke="{t["line"]}"/>',
           f'<path d="M{x} {y + 10} a10 10 0 0 1 10 -10 h{w - 20} a10 10 0 0 1 10 10 v26 h-{w} z" fill="{color}"/>',
           f'<text x="{x + 14}" y="{y + 23}" font-size="14" font-weight="600" fill="#ffffff">{title}</text>',
           f'<text x="{x + w - 14}" y="{y + 23}" font-size="12" text-anchor="end" fill="#ffffff" opacity="0.9">{sub}</text>']
    for i, ln in enumerate(lines):
        cy = y + 62 + i * 24
        out.append(f'<circle cx="{x + 18}" cy="{cy - 4}" r="2.5" fill="{color}"/>')
        out.append(f'<text x="{x + 28}" y="{cy}" font-size="13" fill="{t["ink"]}">{ln}</text>')
    return out


def arrow(x1, y1, x2, y2, t, label=None, lx=None, ly=None, anchor="middle", step=None, sx=None, sy=None):
    out = [f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{t["arrow"]}" stroke-width="1.6" '
           f'marker-end="url(#head)"/>']
    if label:
        out.append(f'<text x="{lx}" y="{ly}" font-size="12" fill="{t["muted"]}" text-anchor="{anchor}">{label}</text>')
    if step:
        out.append(f'<circle cx="{sx}" cy="{sy}" r="10" fill="{t["bg"]}" stroke="{t["arrow"]}" stroke-width="1.4"/>')
        out.append(f'<text x="{sx}" y="{sy + 4}" font-size="11" font-weight="600" text-anchor="middle" '
                   f'fill="{t["ink"]}">{step}</text>')
    return out


def svg(t):
    W, H = 980, 470
    el = [f'<rect width="{W}" height="{H}" fill="{t["bg"]}"/>',
          f'<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
          f'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{t["arrow"]}"/></marker></defs>',
          f'<text x="24" y="36" font-size="18" font-weight="600" fill="{t["ink"]}">ShelfSense architecture</text>',
          f'<text x="24" y="58" font-size="13" fill="{t["muted"]}">Three processes. Images travel as bytes over HTTP; '
          f'each model loads once at startup.</text>']
    # client
    el += [f'<rect x="24" y="196" width="150" height="96" rx="10" fill="{t["card"]}" stroke="{t["line"]}"/>',
           f'<text x="99" y="234" font-size="14" font-weight="600" text-anchor="middle" fill="{t["ink"]}">Client</text>',
           f'<text x="99" y="256" font-size="12" text-anchor="middle" fill="{t["muted"]}">web page or curl</text>']
    el += card(270, 156, 270, 168, t["gateway"], "Gateway", "Flask :8000",
               ["Web interface and REST API", "Decode image, fix EXIF rotation", "Draw colour-coded visualization",
                "Save result, return JSON"], t)
    el += card(660, 92, 296, 140, t["detector"], "Detector service", ":8001",
               ["YOLO11n trained on SKU-110K", "640 px, conf 0.25, NMS IoU 0.7", "Containment suppression 0.8"], t)
    el += card(660, 266, 296, 140, t["grouping"], "Grouping service", ":8002",
               ["Crop each product, pad to square", "CLIP ViT-B/16, 512-d embeddings", "Agglomerative, cosine 0.20"], t)
    # flows: 1 photo in, 2 detect, 3 group, 4 JSON out
    el += arrow(176, 228, 266, 228, t, "photo", 221, 220, step="1", sx=221, sy=200)
    el += arrow(266, 260, 176, 260, t, "JSON", 221, 278, step="4", sx=221, sy=300)
    el += arrow(542, 190, 656, 152, t, "image bytes", 622, 150, step="2", sx=562, sy=183)
    el += arrow(656, 176, 542, 214, t, "boxes", 612, 214)
    el += arrow(542, 276, 656, 314, t, "image + boxes", 608, 270, step="3", sx=562, sy=283)
    el += arrow(656, 338, 542, 300, t, "group ids", 590, 340)
    el.append(f'<text x="24" y="{H - 22}" font-size="12" fill="{t["muted"]}">Numbers show the order of one request. '
              f'The detector and grouping services are stateless and can be scaled independently.</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
            f'font-family="{FONT}">' + "".join(el) + "</svg>")


for name, t in THEMES.items():
    (ROOT / "assets" / f"architecture-{name}.svg").write_text(svg(t), encoding="utf-8")
print("wrote assets/architecture-light.svg and architecture-dark.svg")
