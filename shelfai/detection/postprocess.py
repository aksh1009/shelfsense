"""Detector post-processing: confidence filter and containment suppression."""
from __future__ import annotations

import numpy as np


def containment_filter(boxes: np.ndarray, scores: np.ndarray, thr: float) -> np.ndarray:
    """Indices of boxes to keep (sorted).

    A box is removed when >= `thr` of its area lies inside a LARGER box that scores HIGHER and was kept, e.g. a pack
    half or a bottle neck inside the box of the whole product. NMS misses these because their IoU is only ~0.5.
    If the part scores higher than the whole, both are kept.
    """
    n = len(boxes)
    if n == 0 or thr <= 0:
        return np.arange(n)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep: list[int] = []
    for i in np.argsort(-scores, kind="stable"):
        if keep:
            k = np.asarray(keep)
            iw = np.clip(np.minimum(boxes[i, 2], boxes[k, 2]) - np.maximum(boxes[i, 0], boxes[k, 0]), 0, None)
            ih = np.clip(np.minimum(boxes[i, 3], boxes[k, 3]) - np.maximum(boxes[i, 1], boxes[k, 1]), 0, None)
            inside = (iw * ih) / max(float(area[i]), 1e-9)
            if np.any((inside >= thr) & (area[k] >= area[i])):
                continue
        keep.append(int(i))
    return np.asarray(sorted(keep), int)


def postprocess(boxes: np.ndarray, scores: np.ndarray, conf: float, containment: float):
    """-> (kept boxes, kept scores, removed boxes, removed scores). 'removed' = dropped by containment."""
    m = scores >= conf
    boxes, scores = boxes[m], scores[m]
    keep = containment_filter(boxes, scores, containment)
    drop = np.setdiff1d(np.arange(len(boxes)), keep)
    return boxes[keep], scores[keep], boxes[drop], scores[drop]
