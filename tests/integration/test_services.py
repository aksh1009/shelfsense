"""End-to-end tests: the three services run as real HTTP servers on free ports (the microservice setup)."""
from __future__ import annotations

import json
import socket
import threading

import pytest
import requests
from werkzeug.serving import make_server

from services.detector.app import create_app as detector_app
from services.gateway.app import create_app as gateway_app
from services.grouping.app import create_app as grouping_app
from shelfai.pipeline import build_pipeline


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def stack(cfg):
    servers = []

    def run(app):
        port = _free_port()
        srv = make_server("127.0.0.1", port, app, threaded=True)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{port}"

    cfg["services"]["mode"] = "http"
    cfg["services"]["detector_url"] = run(detector_app(cfg))
    cfg["services"]["grouping_url"] = run(grouping_app(cfg))
    gw = run(gateway_app(cfg, build_pipeline(cfg)))
    yield gw, cfg
    for s in servers:
        s.shutdown()


def test_health(stack):
    gw, _ = stack
    body = requests.get(f"{gw}/api/v1/health").json()
    assert body["status"] == "ok" and body["services"]["detector"]["status"] == "ok"


def test_upload_end_to_end(stack, jpeg_bytes):
    gw, cfg = stack
    r = requests.post(f"{gw}/api/v1/analyze", files={"image": ("shelf.jpg", jpeg_bytes, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success" and body["summary"]["num_products"] == len(body["detections"]) > 0
    ids = {g["group_id"] for g in body["groups"]}
    assert all(d["group_id"] in ids for d in body["detections"])
    assert all(0 <= d["bbox_normalized"]["x1"] <= 1 for d in body["detections"])
    vis = requests.get(f"{gw}{body['visualization']['url']}")
    assert vis.status_code == 200 and vis.headers["Content-Type"] == "image/jpeg"
    stored = requests.get(f"{gw}{body['result_url']}").json()
    assert stored["request_id"] == body["request_id"]


def test_base64_input(stack, jpeg_bytes):
    import base64
    gw, _ = stack
    r = requests.post(f"{gw}/api/v1/analyze", json={"image_base64": base64.b64encode(jpeg_bytes).decode()})
    assert r.status_code == 200 and r.json()["status"] == "success"


@pytest.mark.parametrize("kwargs,code", [
    ({"files": {"image": ("x.jpg", b"not an image", "image/jpeg")}}, "INVALID_IMAGE"),
    ({"json": {}}, "INVALID_REQUEST"),
    ({"json": {"image_url": "ftp://example.com/x.jpg"}}, "INVALID_REQUEST"),
    ({"json": {"image_base64": "%%%"}}, "INVALID_REQUEST"),
])
def test_bad_input_returns_json_error(stack, kwargs, code):
    gw, _ = stack
    r = requests.post(f"{gw}/api/v1/analyze", **kwargs)
    assert r.status_code == 400 and r.json()["error"]["code"] == code


def test_unknown_result_is_404(stack):
    gw, _ = stack
    assert requests.get(f"{gw}/api/v1/results/20200101-000000-abcdef").status_code == 404
    assert requests.get(f"{gw}/api/v1/results/not-a-valid-id").status_code == 404


def test_detector_down_gives_503(cfg, jpeg_bytes):
    cfg["services"]["mode"] = "http"
    cfg["services"]["detector_url"] = f"http://127.0.0.1:{_free_port()}"  # nothing listening
    client = gateway_app(cfg, build_pipeline(cfg)).test_client()
    r = client.post("/api/v1/analyze", data={"image": (__import__("io").BytesIO(jpeg_bytes), "x.jpg")})
    assert r.status_code == 503 and r.get_json()["error"]["code"] == "SERVICE_UNAVAILABLE"


@pytest.mark.parametrize("method", ["agglomerative", "hdbscan", "ocr_weighted"])
@pytest.mark.parametrize("detector", ["yolov8l", "yolo11s", "yolo11n"])
def test_every_method_end_to_end(stack, jpeg_bytes, method, detector):
    gw, _ = stack
    opts = {"detector": detector, "method": method, "embedder": "dinov2", "threshold": 0.3, "show_removed": True}
    r = requests.post(f"{gw}/api/v1/analyze", files={"image": ("s.jpg", jpeg_bytes, "image/jpeg")},
                      data={"options": json.dumps(opts)})
    body = r.json()
    assert r.status_code == 200, body
    assert body["options"]["grouping"]["method"] == method and body["options"]["detection"]["detector"] == detector
    assert len(body["detections"]) == body["summary"]["num_products"]


def test_detection_cache_when_only_grouping_changes(stack, jpeg_bytes):
    gw, _ = stack
    post = lambda o: requests.post(f"{gw}/api/v1/analyze", files={"image": ("s.jpg", jpeg_bytes, "image/jpeg")},
                                   data={"options": json.dumps(o)}).json()
    a = post({"detector": "yolo11n", "method": "agglomerative", "embedder": "clip"})
    b = post({"detector": "yolo11n", "method": "hdbscan", "embedder": "clip"})
    c = post({"detector": "yolo11n", "method": "hdbscan", "embedder": "clip", "conf": 0.5})
    d = post({"detector": "yolo11n", "method": "hdbscan", "embedder": "clip", "no_cache": True})
    assert (a["cached"]["detection"], b["cached"]["detection"], c["cached"]["detection"], d["cached"]["detection"]) == (False, True, False, False)


def test_invalid_option_is_400(stack, jpeg_bytes):
    gw, _ = stack
    r = requests.post(f"{gw}/api/v1/analyze", files={"image": ("s.jpg", jpeg_bytes, "image/jpeg")},
                      data={"options": json.dumps({"detector": "yolo99"})})
    assert r.status_code == 400 and "detector" in r.json()["error"]["message"]


def test_block_contracts(stack, jpeg_bytes):
    """Detector and grouping services, called directly, return the documented formats."""
    _, cfg = stack
    det = requests.post(f"{cfg['services']['detector_url']}/detect", data=jpeg_bytes,
                        headers={"Content-Type": "image/jpeg"}).json()
    assert {"image_size", "num_detections", "detections", "removed_by_containment", "model", "timings_ms"} <= det.keys()
    grp = requests.post(f"{cfg['services']['grouping_url']}/group", files={"image": ("x.jpg", jpeg_bytes, "image/jpeg")},
                        data={"detections": json.dumps(det["detections"]), "options": json.dumps({"method": "hdbscan"})}).json()
    assert len(grp["assignments"]) == det["num_detections"]


def test_local_mode_single_process(cfg, jpeg_bytes):
    cfg["services"]["mode"] = "local"
    client = gateway_app(cfg, build_pipeline(cfg)).test_client()
    r = client.post("/api/v1/analyze", data={"image": (__import__("io").BytesIO(jpeg_bytes), "x.jpg")})
    assert r.status_code == 200 and r.get_json()["status"] == "success"


def test_web_page_and_samples(stack):
    gw, _ = stack
    page = requests.get(gw)
    assert page.status_code == 200 and "Analyze photo" in page.text and "yolo11n" in page.text.lower()
    samples = requests.get(f"{gw}/api/v1/samples").json()["samples"]
    if samples:
        assert requests.get(f"{gw}{samples[0]['url']}").status_code == 200
    assert requests.get(f"{gw}/samples/../configs/default.yaml").status_code == 404
