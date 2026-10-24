"""Distinct colours for groups. The largest groups get the most distinguishable colours."""
from __future__ import annotations

import colorsys

PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 205, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
    (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180),
    (0, 0, 128), (255, 255, 255),
]
SINGLETON = (160, 160, 160)


def group_color(index: int, is_singleton: bool = False) -> tuple[int, int, int]:
    if is_singleton:
        return SINGLETON
    if index < len(PALETTE):
        return PALETTE[index]
    h = (index * 0.618033988749895) % 1.0  # golden-ratio hue steps beyond the fixed palette
    r, g, b = colorsys.hsv_to_rgb(h, 0.8, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)
