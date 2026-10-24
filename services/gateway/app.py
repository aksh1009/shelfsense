"""Gateway (the Flask web server the client talks to).

Optional 'options' (JSON) override the pipeline in configs/default.yaml - see docs/API.md.

GET  /                                         web interface (upload a shelf photo, see groups + JSON)
POST /api/v1/analyze                           image in -> detections + group ids + visualization URL (JSON)
     multipart field 'image'                   (file upload), or
     JSON {"image_url": "https://..."}         (the gateway downloads it), or
     JSON {"image_base64": "..."}
GET  /api/v1/results/<id>                      stored result JSON
GET  /api/v1/results/<id>/visualization.jpg    stored visualization
GET  /api/v1/health                            gateway + detector + grouping status
GET  /api/v1/samples, /samples/<name>          the bundled sample photos (demo convenience for the web page)
"""
from __future__ import annotations

import base64
import binascii
import json
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template, request, send_file

from services.common import register_error_handlers, serve, setup_logging
from shelfai.config import PROJECT_ROOT, get_config
from shelfai.errors import ImageFetchFailed, ImageTooLarge, InvalidRequest, NotFound
from shelfai.pipeline import build_pipeline

SAMPLES_DIR = PROJECT_ROOT / "sample_images"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _read_options() -> dict:
    raw = request.form.get("options")
    if raw is not None:
        try:
            return json.loads(raw) or {}
        except json.JSONDecodeError as e:
            raise InvalidRequest("'options' is not valid JSON.") from e
    body = request.get_json(silent=True)
    return (body.get("options") or {}) if isinstance(body, dict) else {}


def _read_request_image(max_bytes: int) -> tuple[bytes, str | None]:
    if "image" in request.files:
        f = request.files["image"]
        return f.read(), f.filename
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise InvalidRequest("Send an image as multipart field 'image', or JSON with 'image_url' or 'image_base64'.")
    if body.get("image_base64"):
        data = body["image_base64"]
        data = data.split(",", 1)[1] if data.startswith("data:") else data
        try:
            return base64.b64decode(data, validate=True), body.get("filename")
        except (binascii.Error, ValueError) as e:
            raise InvalidRequest("'image_base64' is not valid base64.") from e
    if body.get("image_url"):
        url = str(body["image_url"])
        if not url.lower().startswith(("http://", "https://")):
            raise InvalidRequest("'image_url' must be an http(s) URL.")
        try:
            r = requests.get(url, timeout=20, stream=True)
            r.raise_for_status()
            data = r.raw.read(max_bytes + 1, decode_content=True)
        except requests.RequestException as e:
            raise ImageFetchFailed(f"Could not download the image: {e}") from e
        if len(data) > max_bytes:
            raise ImageTooLarge(f"The image at the URL is larger than {max_bytes // (1024 * 1024)} MB.")
        return data, url.rsplit("/", 1)[-1][:120] or None
    raise InvalidRequest("Send an image as multipart field 'image', or JSON with 'image_url' or 'image_base64'.")


def create_app(cfg: dict | None = None, pipeline=None) -> Flask:
    cfg = cfg or get_config()
    pipeline = pipeline or build_pipeline(cfg)
    max_bytes = cfg["services"]["max_upload_mb"] * 1024 * 1024
    here = Path(__file__).parent
    app = Flask("gateway", template_folder=str(here / "templates"), static_folder=str(here / "static"))
    app.config["MAX_CONTENT_LENGTH"] = max_bytes + 1024 * 1024
    register_error_handlers(app)

    @app.get("/")
    def index():
        d, g = cfg["detection"]["defaults"], cfg["grouping"]["defaults"]
        return render_template("index.html", mode=cfg["services"]["mode"], detector=d["detector"].replace("yolo", "YOLO"),
                               embedder=g["embedder"], threshold=g["threshold"])

    @app.post("/api/v1/analyze")
    def analyze():
        data, filename = _read_request_image(max_bytes)
        return jsonify(pipeline.analyze(data, filename, _read_options()))

    @app.get("/api/v1/results/<request_id>")
    def result(request_id):
        return send_file(pipeline.storage.path(request_id, "result.json"), mimetype="application/json")

    @app.get("/api/v1/results/<request_id>/visualization.jpg")
    def visualization(request_id):
        return send_file(pipeline.storage.path(request_id, "visualization.jpg"), mimetype="image/jpeg")

    @app.get("/api/v1/health")
    def health():
        out = {"status": "ok", "mode": cfg["services"]["mode"], "services": {}}
        for name, client in (("detector", pipeline.detector), ("grouping", pipeline.grouping)):
            try:
                out["services"][name] = client.health()
            except Exception as e:  # noqa: BLE001 - report, don't fail the health page
                out["services"][name] = {"status": "unreachable", "error": str(e)[:200]}
                out["status"] = "degraded"
        return jsonify(out), (200 if out["status"] == "ok" else 503)

    @app.get("/api/v1/samples")
    def samples():
        names = sorted(p.name for p in SAMPLES_DIR.glob("*") if p.suffix.lower() in IMAGE_EXTS) if SAMPLES_DIR.exists() else []
        return jsonify({"samples": [{"name": n, "url": f"/samples/{n}"} for n in names]})

    @app.get("/samples/<path:name>")
    def sample_file(name):
        p = (SAMPLES_DIR / name).resolve()
        if SAMPLES_DIR.resolve() not in p.parents or not p.is_file():
            raise NotFound("No such sample image.")
        return send_file(p)

    return app


def main():
    setup_logging()
    cfg = get_config()
    serve(create_app(cfg), cfg["services"]["gateway_port"], threads=8)


if __name__ == "__main__":
    main()
