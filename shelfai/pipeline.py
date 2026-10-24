"""Gateway pipeline: decode -> detect -> group -> visualize -> save -> JSON.

The detector and grouping blocks are reached through clients with one interface each:
Http*Client (separate services) or Local*Client (in-process: single-process mode and tests).
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from . import schemas
from .errors import InvalidRequest, NotFound, ServiceUnavailable, ShelfAIError, UpstreamTimeout
from .imaging import decode_image, encode_jpeg
from .options import parse_options
from .visualization.render import group_colors, render_groups

REQUEST_ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")


def new_request_id() -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


# --------------------------------------------------------------------------- clients
def _post(url, timeout, what, **kwargs) -> dict:
    try:
        r = requests.post(url, timeout=timeout, **kwargs)
    except requests.Timeout as e:
        raise UpstreamTimeout(f"The {what} service did not answer within {timeout:.0f} s.") from e
    except requests.ConnectionError as e:
        raise ServiceUnavailable(f"The {what} service is not reachable at {url}. Start everything with: "
                                 f"python scripts/run_local.py") from e
    try:
        body = r.json()
    except ValueError:
        body = {}
    if r.status_code != 200:
        err = body.get("error", {}) if isinstance(body, dict) else {}
        exc = ShelfAIError(err.get("message") or f"The {what} service returned HTTP {r.status_code}.")
        exc.status, exc.code = (r.status_code if 400 <= r.status_code < 600 else 502), err.get("code", "UPSTREAM_ERROR")
        raise exc
    return body


class HttpDetectorClient:
    def __init__(self, url, timeout):
        self.url, self.timeout = url.rstrip("/"), timeout

    def detect(self, img, jpeg, opts, request_id) -> dict:
        return _post(f"{self.url}/detect", self.timeout, "detector", data=jpeg, params=opts,
                     headers={"Content-Type": "image/jpeg", "X-Request-ID": request_id})

    def health(self):
        return requests.get(f"{self.url}/health", timeout=3).json()


class HttpGroupingClient:
    def __init__(self, url, timeout):
        self.url, self.timeout = url.rstrip("/"), timeout

    def group(self, img, jpeg, detections, opts, request_id) -> dict:
        return _post(f"{self.url}/group", self.timeout, "grouping", files={"image": ("image.jpg", jpeg, "image/jpeg")},
                     data={"detections": json.dumps(detections), "options": json.dumps(opts)},
                     headers={"X-Request-ID": request_id})

    def health(self):
        return requests.get(f"{self.url}/health", timeout=3).json()


class LocalDetectorClient:
    def __init__(self, pool):
        self.pool = pool

    def detect(self, img, jpeg, opts, request_id) -> dict:
        kb, ks, rb, rs, info, t = self.pool.detect(img, opts)
        return schemas.detector_response(kb, ks, rb, rs, img.size, opts, info, t)

    def health(self):
        return {"status": "ok", "service": "detector (in-process)", "loaded": self.pool.loaded()}


class LocalGroupingClient:
    def __init__(self, engine):
        self.engine = engine

    def group(self, img, jpeg, detections, opts, request_id) -> dict:
        boxes, _ = schemas.parse_detections(detections)
        labels, names, info, t = self.engine.group(img, jpeg, boxes, opts)
        return schemas.grouping_response(labels, names, opts, info, t)

    def health(self):
        return {"status": "ok", "service": "grouping (in-process)", "loaded": self.engine.loaded()}


# --------------------------------------------------------------------------- storage
class Storage:
    """Results under <output_dir>/requests/<request_id>/."""

    def __init__(self, output_dir: str):
        self.root = Path(output_dir) / "requests"
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, request_id):
        if not REQUEST_ID_RE.match(request_id or ""):
            raise NotFound("Unknown result id.")
        return self.root / request_id

    def save(self, request_id, visualization: Image.Image, result: dict, quality: int):
        d = self._dir(request_id)
        d.mkdir(parents=True, exist_ok=True)
        visualization.save(d / "visualization.jpg", quality=quality)
        (d / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    def path(self, request_id, name) -> Path:
        p = self._dir(request_id) / name
        if not p.exists():
            raise NotFound(f"No {name} for result {request_id}.")
        return p


# --------------------------------------------------------------------------- pipeline
class DetectionCache:
    """(image, detector options) -> detector response. Lets grouping settings be compared without re-detecting."""

    def __init__(self, size: int):
        self.size, self.data, self.lock = size, OrderedDict(), threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.data:
                self.data.move_to_end(key)
                return self.data[key]
        return None

    def put(self, key, value):
        with self.lock:
            self.data[key] = value
            while len(self.data) > self.size:
                self.data.popitem(last=False)


class Pipeline:
    def __init__(self, cfg, detector_client, grouping_client, storage: Storage):
        self.cfg, self.detector, self.grouping, self.storage = cfg, detector_client, grouping_client, storage
        self.cache = DetectionCache(cfg["services"]["detection_cache_size"])

    def analyze(self, image_bytes: bytes, filename=None, raw_options=None, url_prefix="/api/v1/results") -> dict:
        opts = parse_options(raw_options, self.cfg)
        request_id = new_request_id()
        timings = {}
        t0 = time.perf_counter()
        img = decode_image(image_bytes)
        jpeg = encode_jpeg(img, 95)  # EXIF rotation applied; one normalised format for both services
        t1 = time.perf_counter(); timings["decode"] = (t1 - t0) * 1000

        key = hashlib.sha1(jpeg).hexdigest() + json.dumps(opts["detection"], sort_keys=True)
        det = None if opts["no_cache"] else self.cache.get(key)
        cached_detection = det is not None
        if det is None:
            det = self.detector.detect(img, jpeg, opts["detection"], request_id)
            self.cache.put(key, det)
        t2 = time.perf_counter(); timings["detect"] = (t2 - t1) * 1000
        boxes, scores = schemas.parse_detections(det)

        if len(boxes):
            grp = self.grouping.group(img, jpeg, det["detections"], opts["grouping"], request_id)
        else:
            grp = schemas.grouping_response(np.zeros(0, int), {}, opts["grouping"], {}, {})
        t3 = time.perf_counter(); timings["group"] = (t3 - t2) * 1000
        index = {g["group_id"]: i for i, g in enumerate(grp["groups"])}
        group_ids = [a["group_id"] for a in grp["assignments"]]
        if len(group_ids) != len(boxes):
            raise InvalidRequest("The grouping service returned a different number of assignments than detections.")
        labels = np.asarray([index[g] for g in group_ids], int)

        removed = det.get("removed_by_containment", [])
        removed_boxes = np.asarray([r["bbox"] for r in removed], np.float32).reshape(-1, 4) if opts["show_removed"] else None
        vis = render_groups(img, boxes, labels, grp["groups"], removed_boxes, self.cfg["visualization"]["max_side"])
        t4 = time.perf_counter(); timings["visualize"] = (t4 - t3) * 1000

        result = schemas.analyze_response(
            request_id, filename, img.size, boxes, scores, group_ids, grp["groups"], group_colors(grp["groups"]),
            removed, opts, {"detection": cached_detection}, f"{url_prefix}/{request_id}/visualization.jpg",
            f"{url_prefix}/{request_id}", timings, {"detector": det.get("model", {}), "grouping": grp.get("model", {})},
            {"detector": {} if cached_detection else det.get("timings_ms", {}), "grouping": grp.get("timings_ms", {})})
        self.storage.save(request_id, vis, result, self.cfg["visualization"]["jpeg_quality"])
        t5 = time.perf_counter()
        result["timings_ms"]["save"] = round((t5 - t4) * 1000, 1)
        result["timings_ms"]["total"] = round((t5 - t0) * 1000, 1)
        (self.storage.root / request_id / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result


def build_pipeline(cfg) -> Pipeline:
    svc = cfg["services"]
    if svc["mode"] == "local":
        from .detection import DetectorPool
        from .grouping import GroupingEngine
        pool = DetectorPool(cfg["detection"], cfg["models_dir"], cfg["device"])
        pool.get(cfg["detection"]["defaults"]["detector"])
        dclient = LocalDetectorClient(pool)
        engine = GroupingEngine(cfg["grouping"], cfg["models_dir"], cfg["device"])
        engine.warmup()
        engine.embedder(cfg["grouping"]["defaults"]["embedder"])
        gclient = LocalGroupingClient(engine)
    else:
        dclient = HttpDetectorClient(svc["detector_url"], svc["request_timeout_s"])
        gclient = HttpGroupingClient(svc["grouping_url"], svc["request_timeout_s"])
    return Pipeline(cfg, dclient, gclient, Storage(cfg["output_dir"]))
