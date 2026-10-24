"""JSON contracts of every block (documented in docs/API.md)."""
from __future__ import annotations

import numpy as np

from .errors import InvalidRequest

API_VERSION = "2.0"


def _r(v, nd=1):
    return round(float(v), nd)


def _box_list(boxes, scores):
    return [{"bbox": [_r(a), _r(b), _r(c), _r(d)], "confidence": _r(s, 4)}
            for (a, b, c, d), s in zip(np.asarray(boxes).reshape(-1, 4), np.asarray(scores).ravel())]


def detector_response(boxes, scores, removed_boxes, removed_scores, image_size, options, model, timings_ms) -> dict:
    return {
        "api_version": API_VERSION,
        "image_size": {"width": int(image_size[0]), "height": int(image_size[1])},
        "options": options,
        "num_detections": int(len(boxes)),
        "detections": _box_list(boxes, scores),
        "removed_by_containment": _box_list(removed_boxes, removed_scores),
        "model": model,
        "timings_ms": {k: _r(v) for k, v in timings_ms.items()},
    }


def parse_detections(payload):
    dets = payload.get("detections") if isinstance(payload, dict) else payload
    if not isinstance(dets, list):
        raise InvalidRequest("'detections' must be a list of {bbox: [x1, y1, x2, y2], confidence}.")
    boxes, scores = [], []
    for i, d in enumerate(dets):
        try:
            x1, y1, x2, y2 = (float(v) for v in d["bbox"])
            conf = float(d.get("confidence", 1.0))
        except (KeyError, TypeError, ValueError) as e:
            raise InvalidRequest(f"detections[{i}] is malformed; expected bbox [x1, y1, x2, y2].") from e
        if not (x2 > x1 and y2 > y1):
            raise InvalidRequest(f"detections[{i}] has an empty box {d['bbox']}.")
        boxes.append([x1, y1, x2, y2]); scores.append(conf)
    return np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(scores, np.float32)


def grouping_response(labels, names, options, model, timings_ms) -> dict:
    labels = np.asarray(labels, int)
    k = int(labels.max()) + 1 if len(labels) else 0
    groups = []
    for g in range(k):
        members = np.flatnonzero(labels == g).tolist()
        groups.append({"group_id": f"G{g + 1}", "label": names.get(g), "num_members": len(members),
                       "member_indices": members})
    return {
        "api_version": API_VERSION,
        "options": options,
        "num_groups": sum(1 for g in groups if g["num_members"] > 1),
        "num_singletons": sum(1 for g in groups if g["num_members"] == 1),
        "assignments": [{"detection_index": i, "group_id": f"G{int(g) + 1}"} for i, g in enumerate(labels)],
        "groups": groups,
        "model": model,
        "timings_ms": {kk: _r(v) for kk, v in timings_ms.items()},
    }


def analyze_response(request_id, filename, image_size, boxes, scores, group_ids, groups, colors, removed,
                     options, cached, visualization_url, result_url, timings_ms, models, service_timings) -> dict:
    W, H = image_size
    detections = [{
        "id": i,
        "bbox": {"x1": _r(x1), "y1": _r(y1), "x2": _r(x2), "y2": _r(y2)},
        "bbox_normalized": {"x1": _r(x1 / W, 5), "y1": _r(y1 / H, 5), "x2": _r(x2 / W, 5), "y2": _r(y2 / H, 5)},
        "confidence": _r(s, 4),
        "group_id": gid,
    } for i, ((x1, y1, x2, y2), s, gid) in enumerate(zip(np.asarray(boxes).reshape(-1, 4), scores, group_ids))]
    out_groups = [{"group_id": g["group_id"], "label": g.get("label"), "num_members": g["num_members"],
                   "color": colors[g["group_id"]], "is_singleton": g["num_members"] == 1,
                   "member_ids": g["member_indices"]} for g in groups]
    return {
        "status": "success",
        "request_id": request_id,
        "api_version": API_VERSION,
        "image": {"filename": filename, "width": int(W), "height": int(H)},
        "options": options,
        "summary": {"num_products": len(detections),
                    "num_groups": sum(1 for g in out_groups if not g["is_singleton"]),
                    "num_singletons": sum(1 for g in out_groups if g["is_singleton"]),
                    "num_removed_by_containment": len(removed)},
        "detections": detections,
        "groups": out_groups,
        "removed_by_containment": removed,
        "visualization": {"url": visualization_url, "format": "jpeg"},
        "result_url": result_url,
        "cached": cached,
        "timings_ms": {k: _r(v) for k, v in timings_ms.items()},
        "service_timings_ms": service_timings,
        "models": models,
    }
