"""OCR of product crops (EasyOCR), with an in-memory cache so repeated runs on the same photo don't re-read."""
from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np
from PIL import Image


class EasyOCRReader:
    def __init__(self, languages: list[str], gpu: bool):
        import easyocr  # imported here so the app runs without it until an OCR method is used
        self.reader = easyocr.Reader(languages, gpu=gpu, verbose=False)
        self.lock = threading.Lock()
        self.info = {"name": "easyocr", "languages": languages, "gpu": gpu}

    def read(self, img: Image.Image) -> list[dict]:
        with self.lock:
            res = self.reader.readtext(np.asarray(img), detail=1, paragraph=False)
        return [dict(text=t, conf=float(c), height=float(max(p[1] for p in pts) - min(p[1] for p in pts)))
                for pts, t, c in res]


class FakeOCRReader:
    """Reads a placeholder word from the crop's dominant colour. For tests."""
    info = {"name": "fake ocr"}

    def read(self, img: Image.Image) -> list[dict]:
        r, g, b = np.asarray(img.resize((16, 16)).convert("RGB"), np.float32).reshape(-1, 3).mean(0)
        if max(r, g, b) - min(r, g, b) < 25:
            return []
        return [dict(text=["REDCO", "GREENCO", "BLUECO"][int(np.argmax([r, g, b]))], conf=0.9, height=40.0)]


class OCRCache:
    def __init__(self, max_items: int = 20000):
        self.data: OrderedDict[str, list[dict]] = OrderedDict()
        self.max_items = max_items
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.data:
                self.data.move_to_end(key)
                return self.data[key]
        return None

    def put(self, key, value):
        with self.lock:
            self.data[key] = value
            if len(self.data) > self.max_items:
                self.data.popitem(last=False)


def upscale(c: Image.Image, target: int) -> Image.Image:
    s = target / max(c.size)
    return c.resize((round(c.width * s), round(c.height * s)), Image.BICUBIC) if s > 1 else c
