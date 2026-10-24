"""Crop embedders. All return L2-normalised float32 vectors, so a dot product is a cosine similarity."""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import resolve_device
from ..imaging import letterbox


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)


class HistEmbedder:
    """HSV colour histogram (8 x 4 x 4 bins, square-rooted). Stand-in for the neural models in tests."""

    def __init__(self, name: str = "hist"):
        self.info = {"name": name, "backend": "fake (colour histogram)"}

    def embed(self, crops: list[Image.Image]) -> np.ndarray:
        feats = []
        for c in crops:
            hsv = np.asarray(c.resize((64, 64)).convert("HSV")).reshape(-1, 3)
            h, _ = np.histogramdd(hsv, bins=(8, 4, 4), range=((0, 256), (0, 256), (0, 256)))
            feats.append(np.sqrt(h.ravel() / max(h.sum(), 1)))
        return l2_normalize(np.asarray(feats)) if feats else np.zeros((0, 128), np.float32)


class HFEmbedder:
    """clip: CLIP ViT-B/16 projected image embedding | siglip2: SigLIP2 ViT-B/16 pooled output | dinov2: DINOv2-S CLS."""

    def __init__(self, name: str, repo: str, cfg: dict, models_dir: str, device: str):
        import torch
        from transformers import AutoImageProcessor

        self.name, self.cfg = name, cfg
        cache = str(Path(models_dir) / "hf")
        self.device = torch.device(resolve_device(device))
        self.dtype = torch.float16 if (cfg.get("half") and self.device.type == "cuda") else torch.float32
        proc = AutoImageProcessor.from_pretrained(repo, cache_dir=cache)
        self.mean = np.asarray(proc.image_mean, np.float32).reshape(1, 1, 3)
        self.std = np.asarray(proc.image_std, np.float32).reshape(1, 1, 3)
        if name == "clip":
            from transformers import CLIPVisionModelWithProjection
            self.model = CLIPVisionModelWithProjection.from_pretrained(repo, cache_dir=cache)
        elif name == "siglip2":
            from transformers import SiglipVisionModel
            self.model = SiglipVisionModel.from_pretrained(repo, cache_dir=cache)
        else:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(repo, cache_dir=cache)
        self.model = self.model.to(self.device, dtype=self.dtype).eval()
        self.lock = threading.Lock()
        self.info = {"name": name, "repo": repo, "input_size": cfg["input_size"], "device": str(self.device),
                     "half": self.dtype == torch.float16}

    def _tensor(self, crops):
        import torch
        size = self.cfg["input_size"]
        arr = np.stack([np.asarray(letterbox(c, size), np.float32) / 255.0 for c in crops])
        arr = (arr - self.mean) / self.std
        return torch.from_numpy(arr.transpose(0, 3, 1, 2).copy()).to(self.device, dtype=self.dtype)

    def embed(self, crops: list[Image.Image]) -> np.ndarray:
        import torch
        if not crops:
            return np.zeros((0, 1), np.float32)
        out, bs = [], self.cfg["batch_size"]
        with self.lock, torch.inference_mode():
            for i in range(0, len(crops), bs):
                res = self.model(pixel_values=self._tensor(crops[i:i + bs]))
                feat = res.image_embeds if self.name == "clip" else res.pooler_output
                out.append(feat.float().cpu().numpy())
        return l2_normalize(np.concatenate(out))
