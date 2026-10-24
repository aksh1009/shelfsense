"""Shared fixtures. All tests use the fake detector + colour-histogram grouping: no downloads, no GPU."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shelfai.config import load_config  # noqa: E402


@pytest.fixture()
def cfg(tmp_path):
    c = load_config(ROOT / "configs" / "test.yaml")
    c["output_dir"] = str(tmp_path / "outputs")
    return c


def make_jpeg(w=800, h=600, color=(200, 60, 60)) -> bytes:
    img = Image.new("RGB", (w, h), (235, 235, 235))
    for i in range(6):  # a few coloured "products"
        img.paste(Image.new("RGB", (90, 200), color if i % 2 else (60, 60, 200)), (30 + i * 120, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def jpeg_bytes():
    return make_jpeg()


@pytest.fixture()
def sample_path():
    paths = sorted((ROOT / "sample_images").glob("*.jpg"))
    return paths[0] if paths else None
