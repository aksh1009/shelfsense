#!/usr/bin/env python3
"""Draw assets/latency-light.png and assets/latency-dark.png from the measured medians (docs/REPORT.md#latency).

Laptop, RTX 2000 Ada, FP16, 20 test photos, median per photo, model loading excluded.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DETECT = [("YOLO11n", 64), ("YOLO11s", 74), ("YOLOv8l", 103)]  # detection only
GROUP = [("CLIP + agglomerative", 628), ("DINOv2 + agglomerative", 597), ("SigLIP2 + agglomerative", 686),
         ("CLIP + HDBSCAN", 1031), ("CLIP + OCR (weighted)", 6255)]  # grouping only, detection excluded
CHOSEN = {"YOLO11n", "CLIP + agglomerative"}
THEMES = {
    "light": dict(bg="#ffffff", ink="#1f2328", muted="#59636e", grid="#e6e9ec", chosen="#2a78d6", other="#b9bfc6"),
    "dark": dict(bg="#0d1117", ink="#e6edf3", muted="#9198a1", grid="#262c34", chosen="#3987e5", other="#4a525c"),
}


def panel(ax, data, title, t, xmax):
    names = [n for n, _ in data][::-1]
    vals = [v for _, v in data][::-1]
    colors = [t["chosen"] if n in CHOSEN else t["other"] for n in names]
    bars = ax.barh(names, vals, height=0.56, color=colors, edgecolor=t["bg"], linewidth=2)
    for b, n, v in zip(bars, names, vals):
        ax.text(v + xmax * 0.012, b.get_y() + b.get_height() / 2, f"{v:,} ms" + ("  chosen" if n in CHOSEN else ""),
                va="center", fontsize=10, color=t["ink"] if n in CHOSEN else t["muted"],
                fontweight="bold" if n in CHOSEN else "normal")
    ax.set_xlim(0, xmax)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=t["ink"], pad=10)
    ax.tick_params(axis="y", length=0, labelsize=10, labelcolor=t["ink"])
    ax.tick_params(axis="x", length=0, labelsize=9, labelcolor=t["muted"])
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("milliseconds", fontsize=9, color=t["muted"])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor(t["bg"])


for name, t in THEMES.items():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.3), gridspec_kw=dict(width_ratios=[1, 1.45], wspace=0.55))
    fig.patch.set_facecolor(t["bg"])
    panel(a, DETECT, "Detection time per photo", t, 140)
    panel(b, GROUP, "Grouping time per photo (detection excluded)", t, 8200)
    fig.text(0.01, -0.04, "Median over 20 test photos on a laptop GPU (RTX 2000 Ada, FP16), model loading excluded.",
             fontsize=9, color=t["muted"])
    fig.savefig(ROOT / "assets" / f"latency-{name}.png", dpi=160, bbox_inches="tight", facecolor=t["bg"])
    plt.close(fig)
print("wrote assets/latency-light.png and latency-dark.png")
