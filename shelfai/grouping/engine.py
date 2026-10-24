"""Grouping engine: image + product boxes + options -> group index per product (+ brand words for OCR methods).
Embedders and the OCR reader load on first use and stay in memory."""
from __future__ import annotations

import hashlib
import threading
import time

import numpy as np
from PIL import Image

from ..config import resolve_device
from ..imaging import crop_boxes
from . import methods
from .embedders import HFEmbedder, HistEmbedder
from .ocr import EasyOCRReader, FakeOCRReader, OCRCache, upscale
from .text import brand_tokens, group_labels, propagate_tokens


class GroupingEngine:
    def __init__(self, cfg: dict, models_dir: str, device: str):
        self.cfg, self.models_dir, self.device = cfg, models_dir, device
        self.fake = cfg.get("backend") == "fake"
        self.embedders: dict[str, object] = {}
        self.ocr_reader = None
        self.ocr_cache = OCRCache()
        self._lock = threading.Lock()

    def warmup(self):
        """Import and exercise the clustering libraries once, so the first request isn't paying for it."""
        eye = np.eye(4)
        methods.agglomerative(eye, np.zeros((4, 4), np.float32), 0.2)
        methods.hdbscan(eye, np.zeros((4, 4), np.float32), 2)

    # ------------------------------------------------------------ lazy loading
    def embedder(self, name: str):
        if name in self.embedders:
            return self.embedders[name], 0.0
        with self._lock:
            if name in self.embedders:
                return self.embedders[name], 0.0
            t0 = time.perf_counter()
            e = HistEmbedder(name) if self.fake else HFEmbedder(name, self.cfg["embedders"][name], self.cfg,
                                                               self.models_dir, self.device)
            e.embed([Image.new("RGB", (64, 128), (128, 128, 128))])  # warm-up
            self.embedders[name] = e
            return e, (time.perf_counter() - t0) * 1000

    def ocr(self):
        if self.ocr_reader is not None:
            return self.ocr_reader, 0.0
        with self._lock:
            if self.ocr_reader is not None:
                return self.ocr_reader, 0.0
            t0 = time.perf_counter()
            if self.fake:
                self.ocr_reader = FakeOCRReader()
            else:
                try:
                    self.ocr_reader = EasyOCRReader(self.cfg["ocr"]["languages"],
                                                    gpu=resolve_device(self.device).startswith("cuda"))
                except ImportError as e:
                    raise RuntimeError("The OCR methods need EasyOCR: pip install easyocr") from e
            return self.ocr_reader, (time.perf_counter() - t0) * 1000

    def loaded(self) -> dict:
        return {"embedders": list(self.embedders), "ocr": self.ocr_reader is not None}

    # ------------------------------------------------------------ grouping
    def _tokens(self, image_key: str, crops, boxes, use_cache: bool = True) -> tuple[list[set[str]], float, float]:
        oc = self.cfg["ocr"]
        reader, load_ms = self.ocr()
        t0 = time.perf_counter()
        tokens = []
        for c, b in zip(crops, boxes):
            key = f"{image_key}:{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}"
            reads = self.ocr_cache.get(key) if use_cache else None
            if reads is None:
                reads = reader.read(upscale(c, oc["upscale_to"]))
                self.ocr_cache.put(key, reads)
            tokens.append(brand_tokens(reads, oc["min_len"], oc["min_conf"], oc["prominence"]))
        return tokens, load_ms, (time.perf_counter() - t0) * 1000

    def group(self, img: Image.Image, image_bytes: bytes, boxes: np.ndarray, opts: dict):
        """-> (labels, names {group index: brand word or None}, info, timings_ms)."""
        t = {}
        t0 = time.perf_counter()
        crops = crop_boxes(img, boxes)
        t["crop"] = (time.perf_counter() - t0) * 1000

        emb_model, t["embedder_load"] = self.embedder(opts["embedder"])
        t1 = time.perf_counter()
        emb = emb_model.embed(crops)
        t["embed"] = (time.perf_counter() - t1) * 1000
        V = np.clip(emb.astype(np.float64) @ emb.astype(np.float64).T, -1, 1) if len(emb) else np.zeros((0, 0))

        method, oc = opts["method"], self.cfg["ocr"]
        tokens = None
        if method.startswith("ocr_"):
            tokens, t["ocr_load"], t["ocr"] = self._tokens(hashlib.sha1(image_bytes).hexdigest(), crops, boxes,
                                                           use_cache=not opts.get("no_cache"))
            n_read = int(sum(1 for x in tokens if x))
            tokens = propagate_tokens(tokens, V, oc["propagate"])  # unreadable crops borrow from identical-looking ones

        t2 = time.perf_counter()
        if method == "agglomerative":
            labels = methods.agglomerative(V, boxes, opts["threshold"])
        elif method == "hdbscan":
            labels = methods.hdbscan(V, boxes, opts["min_cluster_size"])
        else:  # ocr_weighted
            labels = methods.ocr_weighted(V, tokens, boxes, opts["threshold"], opts["text_weight"], oc["fuzzy"])
        t["cluster"] = (time.perf_counter() - t2) * 1000

        names = group_labels(labels, tokens) if tokens is not None else {}
        info = {"method": method, "embedder": emb_model.info, **{k: opts[k] for k in ("threshold", "text_weight", "min_cluster_size")}}
        if tokens is not None:
            info["ocr"] = self.ocr_reader.info
            info["crops_read"] = n_read
            info["crops_with_brand_words"] = int(sum(1 for x in tokens if x))
        return labels, names, info, t
