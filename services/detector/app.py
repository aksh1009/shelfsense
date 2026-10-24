"""Detector service.

POST /detect?detector=yolov8l&conf=0.25&iou=0.7&containment=0.8
     body: image bytes (Content-Type image/jpeg or image/png), or multipart field 'image'
     -> {"detections": [{"bbox": [x1, y1, x2, y2], "confidence"}], "removed_by_containment": [...], "model", "timings_ms"}
GET  /health
"""
from __future__ import annotations

import time

from flask import Flask, jsonify, request

from services.common import register_error_handlers, serve, setup_logging
from shelfai import schemas
from shelfai.config import get_config
from shelfai.errors import InvalidRequest
from shelfai.imaging import decode_image
from shelfai.options import parse_detection


def create_app(cfg=None, pool=None) -> Flask:
    cfg = cfg or get_config()
    if pool is None:
        from shelfai.detection import DetectorPool
        pool = DetectorPool(cfg["detection"], cfg["models_dir"], cfg["device"])
        pool.get(cfg["detection"]["defaults"]["detector"])  # load + warm up the pipeline's detector before serving
    app = Flask("detector")
    app.config["MAX_CONTENT_LENGTH"] = cfg["services"]["max_upload_mb"] * 1024 * 1024
    register_error_handlers(app)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "service": "detector", "available": list(cfg["detection"]["models"]),
                        "loaded": pool.loaded(), "gpu_memory_mb": _gpu_mb()})

    @app.post("/detect")
    def detect():
        opts = parse_detection(request.args.to_dict(), cfg)
        data = request.files["image"].read() if "image" in request.files else request.get_data()
        if not data:
            raise InvalidRequest("Send the image as the request body (Content-Type: image/jpeg) or multipart field 'image'.")
        t0 = time.perf_counter()
        img = decode_image(data)
        t1 = time.perf_counter()
        kb, ks, rb, rs, info, t = pool.detect(img, opts)
        t = {"decode": (t1 - t0) * 1000, **t, "total": (time.perf_counter() - t0) * 1000}
        return jsonify(schemas.detector_response(kb, ks, rb, rs, img.size, opts, info, t))

    return app


def _gpu_mb():
    try:
        import torch
        return round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else None
    except ImportError:
        return None


def main():
    setup_logging()
    cfg = get_config()
    serve(create_app(cfg), cfg["services"]["detector_port"], threads=4)


if __name__ == "__main__":
    main()
