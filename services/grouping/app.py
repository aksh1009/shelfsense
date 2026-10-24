"""Grouping service.

POST /group   multipart: 'image' (file) + 'detections' (JSON list of {"bbox", "confidence"} or the detector response)
              + optional 'options' (JSON: method, embedder, threshold, text_weight, min_cluster_size)
              -> {"assignments": [{"detection_index", "group_id"}], "groups": [{"group_id", "label", ...}], "model", "timings_ms"}
GET  /health
"""
from __future__ import annotations

import json
import time

from flask import Flask, jsonify, request

from services.common import register_error_handlers, serve, setup_logging
from services.detector.app import _gpu_mb
from shelfai import schemas
from shelfai.config import get_config
from shelfai.errors import InvalidRequest
from shelfai.imaging import decode_image
from shelfai.options import parse_grouping


def _json_field(name, required):
    raw = request.form.get(name)
    if raw is None:
        if required:
            raise InvalidRequest(f"Send '{name}' as a multipart field (JSON).")
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise InvalidRequest(f"'{name}' is not valid JSON.") from e


def create_app(cfg=None, engine=None) -> Flask:
    cfg = cfg or get_config()
    if engine is None:
        from shelfai.grouping import GroupingEngine
        engine = GroupingEngine(cfg["grouping"], cfg["models_dir"], cfg["device"])
        engine.warmup()
        engine.embedder(cfg["grouping"]["defaults"]["embedder"])  # load + warm up the pipeline's image model
    app = Flask("grouping")
    app.config["MAX_CONTENT_LENGTH"] = cfg["services"]["max_upload_mb"] * 1024 * 1024
    register_error_handlers(app)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "service": "grouping", "available": list(cfg["grouping"]["embedders"]),
                        "loaded": engine.loaded(), "gpu_memory_mb": _gpu_mb()})

    @app.post("/group")
    def group():
        if "image" not in request.files:
            raise InvalidRequest("Send the image as multipart field 'image'.")
        payload = _json_field("detections", True)
        opts = parse_grouping(_json_field("options", False), cfg)
        t0 = time.perf_counter()
        data = request.files["image"].read()
        img = decode_image(data)
        boxes, _ = schemas.parse_detections(payload)
        t1 = time.perf_counter()
        labels, names, info, t = engine.group(img, data, boxes, opts)
        t = {"decode": (t1 - t0) * 1000, **t, "total": (time.perf_counter() - t0) * 1000}
        return jsonify(schemas.grouping_response(labels, names, opts, info, t))

    return app


def main():
    setup_logging()
    cfg = get_config()
    serve(create_app(cfg), cfg["services"]["grouping_port"], threads=4)


if __name__ == "__main__":
    main()
