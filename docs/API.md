# API

```
client --(1) image--> gateway :8000 --(2) JPEG bytes + detection options--> detector :8001
   ^                     |           <-- boxes + confidences ---------------'
   |                     '--(3) JPEG + boxes + grouping options--> grouping :8002
   '--(4) JSON + visualization URL <-- group id per box ----------'
```

Images travel as bytes, never as file paths. Errors always look like
`{"status": "error", "error": {"code": "...", "message": "..."}}`:

| HTTP | code | cause |
|---|---|---|
| 400 | `INVALID_REQUEST` | missing image, bad JSON / base64 / URL scheme, option out of range, malformed boxes |
| 400 | `INVALID_IMAGE` | not a readable image |
| 404 | `NOT_FOUND` | unknown result id |
| 413 | `IMAGE_TOO_LARGE` | over 25 MB |
| 422 | `IMAGE_FETCH_FAILED` | `image_url` could not be downloaded |
| 503 / 504 | `SERVICE_UNAVAILABLE` / `UPSTREAM_TIMEOUT` | detector or grouping service down or too slow |

## Optional overrides

The app runs one pipeline (YOLO11n, confidence 0.25, NMS 0.7, containment 0.8, CLIP + agglomerative at 0.20; see
`configs/default.yaml`). An API request may override it with multipart field `options` (JSON), or `"options": {...}` in a
JSON request. The web page never does; `scripts/compare_methods.py` uses these overrides to run the alternatives.

| option | values | block |
|---|---|---|
| `detector` | `yolov8l`, `yolo11s`, `yolo11n` | detector |
| `conf` | 0.01 - 0.95 | detector |
| `iou` | NMS IoU, 0.1 - 0.95 | detector |
| `containment` | 0 (off) or 0.3 - 0.99: drop a box this much inside a larger, higher-scoring one | detector |
| `method` | `agglomerative`, `hdbscan`, `ocr_weighted` | grouping |
| `embedder` | `clip`, `siglip2`, `dinov2` | grouping |
| `threshold` | cosine distance, 0.01 - 1.0 (agglomerative and OCR methods) | grouping |
| `text_weight` | 0 - 1 (`ocr_weighted`) | grouping |
| `min_cluster_size` | 2 - 10 (`hdbscan`) | grouping |
| `show_removed` | draw containment-removed boxes, dashed | gateway |
| `no_cache` | re-run detection and OCR even if this photo + settings ran before | gateway |

## 1. Gateway - `POST /api/v1/analyze`

Input: multipart `image` (+ optional `options`), or JSON `{"image_url": ...}` / `{"image_base64": ...}`.

```json
{
  "status": "success",
  "request_id": "20260928-101512-3fa2c1",
  "api_version": "2.0",
  "image": { "filename": "shelf_01.jpg", "width": 1620, "height": 2880 },
  "options": {
    "detection": { "detector": "yolo11n", "conf": 0.25, "iou": 0.7, "containment": 0.8 },
    "grouping": { "method": "agglomerative", "embedder": "clip", "threshold": 0.2, "text_weight": 0.3, "min_cluster_size": 2 },
    "show_removed": false, "no_cache": false
  },
  "summary": { "num_products": 95, "num_groups": 11, "num_singletons": 6, "num_removed_by_containment": 1 },
  "detections": [
    { "id": 0, "bbox": { "x1": 12.5, "y1": 88.0, "x2": 301.2, "y2": 402.7 },
      "bbox_normalized": { "x1": 0.00772, "y1": 0.03056, "x2": 0.18593, "y2": 0.13983 },
      "confidence": 0.8812, "group_id": "G1" }
  ],
  "groups": [
    { "group_id": "G1", "label": "BRANDX", "num_members": 14, "color": "#e6194b", "is_singleton": false, "member_ids": [0, 1, 4] }
  ],
  "removed_by_containment": [ { "bbox": [12.5, 88.0, 301.2, 240.1], "confidence": 0.4102 } ],
  "visualization": { "url": "/api/v1/results/20260928-101512-3fa2c1/visualization.jpg", "format": "jpeg" },
  "result_url": "/api/v1/results/20260928-101512-3fa2c1",
  "cached": { "detection": false },
  "timings_ms": { "decode": 61.0, "detect": 58.3, "group": 211.4, "visualize": 38.9, "save": 12.0, "total": 381.6 },
  "service_timings_ms": {
    "detector": { "decode": 20.1, "model_load": 0.0, "inference": 31.2, "postprocess": 1.1, "total": 52.4 },
    "grouping": { "decode": 19.8, "crop": 2.2, "embedder_load": 0.0, "embed": 41.0, "ocr_load": 0.0, "ocr": 138.7, "cluster": 3.1, "total": 205.0 }
  },
  "models": { "detector": { "name": "yolo11n", "repo": "chistopat/sku110k-yolo11-object-detector" }, "grouping": { "method": "ocr_weighted", "crops_read": 61 } }
}
```

(Numbers show the format only.) `bbox` is in original pixels after EXIF rotation. Groups are numbered by size;
a group of one is a singleton (grey). `label` is the most common brand word, for the OCR methods only.
`model_load`, `embedder_load` and `ocr_load` are non-zero only the first time a model is used.

Other endpoints:

| endpoint | returns |
|---|---|
| `GET /` | the web page |
| `GET /api/v1/results/<id>`, `.../visualization.jpg` | a stored result |
| `GET /api/v1/health` | status of all three services, loaded models, GPU memory |

## 2. Detector - `POST /detect?detector=yolov8l&conf=0.25&iou=0.7&containment=0.8`

Input: image bytes as the body (`Content-Type: image/jpeg`), or multipart `image`.

```json
{
  "api_version": "2.0",
  "image_size": { "width": 1620, "height": 2880 },
  "options": { "detector": "yolo11n", "conf": 0.25, "iou": 0.7, "containment": 0.8 },
  "num_detections": 95,
  "detections": [ { "bbox": [12.5, 88.0, 301.2, 402.7], "confidence": 0.8812 } ],
  "removed_by_containment": [ { "bbox": [12.5, 88.0, 301.2, 240.1], "confidence": 0.4102 } ],
  "model": { "name": "yolo11n", "repo": "chistopat/sku110k-yolo11-object-detector", "file": "weights/sku110k-yolo11-n640.pt", "imgsz": 640, "device": "cuda", "half": true },
  "timings_ms": { "decode": 20.1, "model_load": 0.0, "inference": 31.2, "postprocess": 1.1, "total": 52.4 }
}
```

## 3. Grouping - `POST /group`

Input (multipart): `image`, `detections` (JSON list of `{"bbox", "confidence"}`), optional `options` (JSON).

```json
{
  "api_version": "2.0",
  "options": { "method": "agglomerative", "embedder": "clip", "threshold": 0.2, "text_weight": 0.3, "min_cluster_size": 2 },
  "num_groups": 11,
  "num_singletons": 6,
  "assignments": [ { "detection_index": 0, "group_id": "G1" } ],
  "groups": [ { "group_id": "G1", "label": "BRANDX", "num_members": 14, "member_indices": [0, 1, 4] } ],
  "model": { "method": "ocr_weighted", "embedder": { "name": "clip", "repo": "openai/clip-vit-base-patch16" }, "crops_read": 61, "crops_with_brand_words": 78 },
  "timings_ms": { "decode": 19.8, "crop": 2.2, "embedder_load": 0.0, "embed": 41.0, "ocr_load": 0.0, "ocr": 138.7, "cluster": 3.1, "total": 205.0 }
}
```

`assignments[i]` belongs to `detections[i]` of the input.
