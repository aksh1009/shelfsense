"""Product detectors (SKU-110K checkpoints). Each model is loaded on first use and kept in memory."""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import resolve_device
from ..imaging import to_bgr_array
from .postprocess import postprocess


class _QuietHalfWarning(logging.Filter):
    def filter(self, record):  # newer ultralytics prints a deprecation note for 'half'; the option still works
        return "'half' is deprecated" not in record.getMessage()


class YoloModel:
    def __init__(self, name: str, spec: dict, cfg: dict, models_dir: str, device: str):
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        logging.getLogger("ultralytics").addFilter(_QuietHalfWarning())
        self.cfg, self.device = cfg, resolve_device(device)
        weights = hf_hub_download(spec["repo"], spec["file"], cache_dir=str(Path(models_dir) / "hf"))
        self.model = YOLO(weights)
        self.half = bool(cfg.get("half")) and self.device.startswith("cuda")
        self.lock = threading.Lock()
        self.info = {"name": name, "repo": spec["repo"], "file": spec["file"], "imgsz": cfg["imgsz"],
                     "device": self.device, "half": self.half}

    def predict(self, img: Image.Image, conf: float, iou: float):
        with self.lock:
            res = self.model.predict(to_bgr_array(img), imgsz=self.cfg["imgsz"], conf=conf, iou=iou,
                                     max_det=self.cfg["max_det"], device=self.device, half=self.half, verbose=False)[0]
        return (res.boxes.xyxy.cpu().numpy().astype(np.float32), res.boxes.conf.cpu().numpy().astype(np.float32),
                {f"model_{k}": v for k, v in res.speed.items()})


class FakeModel:
    """Placeholder boxes on a grid, deterministic per image. For tests and trying the interface without downloads."""

    def __init__(self, name: str, *_):
        self.info = {"name": name, "backend": "fake (placeholder boxes)"}

    def predict(self, img: Image.Image, conf: float, iou: float):
        W, H = img.size
        rng = np.random.default_rng(W * 31 + H)
        rows, cols = int(rng.integers(3, 6)), int(rng.integers(5, 10))
        cw, ch = W / cols, H / rows
        boxes, scores = [], []
        for r in range(rows):
            for c in range(cols):
                if rng.random() < 0.85:
                    x1, y1 = c * cw + cw * 0.08, r * ch + ch * 0.1
                    boxes.append([x1, y1, x1 + cw * 0.84, y1 + ch * 0.8]); scores.append(float(rng.uniform(0.2, 0.95)))
                    if rng.random() < 0.15:  # a part-inside-whole duplicate, so containment has something to do
                        boxes.append([x1, y1, x1 + cw * 0.84, y1 + ch * 0.4]); scores.append(float(rng.uniform(0.2, 0.5)))
        return np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(scores, np.float32), {}


class DetectorPool:
    """name -> loaded model. Loading happens once per model, on first use; the load time is reported once."""

    def __init__(self, cfg: dict, models_dir: str, device: str):
        self.cfg, self.models_dir, self.device = cfg, models_dir, device
        self.models: dict[str, object] = {}
        self._lock = threading.Lock()

    def get(self, name: str):
        """-> (model, load_ms). load_ms is 0 when the model was already loaded."""
        if name in self.models:
            return self.models[name], 0.0
        with self._lock:
            if name in self.models:
                return self.models[name], 0.0
            t0 = time.perf_counter()
            if self.cfg.get("backend") == "fake":
                m = FakeModel(name)
            else:
                m = YoloModel(name, self.cfg["models"][name], self.cfg, self.models_dir, self.device)
                m.predict(Image.new("RGB", (640, 480), (128, 128, 128)), 0.25, 0.7)  # warm-up (CUDA kernels)
            self.models[name] = m
            return m, (time.perf_counter() - t0) * 1000

    def detect(self, img: Image.Image, opts: dict):
        model, load_ms = self.get(opts["detector"])
        t0 = time.perf_counter()
        boxes, scores, speed = model.predict(img, opts["conf"], opts["iou"])
        t1 = time.perf_counter()
        kb, ks, rb, rs = postprocess(boxes, scores, opts["conf"], opts["containment"])
        t2 = time.perf_counter()
        timings = {"model_load": load_ms, "inference": (t1 - t0) * 1000, "postprocess": (t2 - t1) * 1000, **speed}
        return kb, ks, rb, rs, model.info, timings

    def loaded(self) -> list[str]:
        return list(self.models)
