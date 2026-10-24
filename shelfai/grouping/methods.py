"""The grouping methods. Each takes crop embeddings (and, for the OCR methods, brand words) and returns a
group index per product: 0 = largest group, numbered deterministically."""
from __future__ import annotations

import numpy as np

from .text import text_similarity


def order_labels(raw: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Renumber: 0 = largest group; ties by top-left position. Negative raw labels (noise) become singletons."""
    groups: dict[int, list[int]] = {}
    noise = -1
    for i, g in enumerate(raw):
        if g < 0:
            g, noise = noise, noise - 1
        groups.setdefault(int(g), []).append(i)
    order = sorted(groups.values(), key=lambda m: (-len(m), min(boxes[i, 1] for i in m), min(boxes[i, 0] for i in m)))
    labels = np.empty(len(raw), int)
    for gid, members in enumerate(order):
        labels[members] = gid
    return labels


def _cosine_distance(V: np.ndarray) -> np.ndarray:
    D = np.clip(1.0 - V, 0.0, 2.0)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    return D


def _average_linkage(D: np.ndarray, threshold: float) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    return fcluster(linkage(squareform(D, checks=False), "average"), t=threshold, criterion="distance")


def agglomerative(V: np.ndarray, boxes: np.ndarray, threshold: float) -> np.ndarray:
    """Average-linkage agglomerative clustering on cosine distance, cut at `threshold`."""
    if len(V) <= 1:
        return np.zeros(len(V), int)
    return order_labels(_average_linkage(_cosine_distance(V), threshold), boxes)


def hdbscan(V: np.ndarray, boxes: np.ndarray, min_cluster_size: int, min_prob: float = 0.1) -> np.ndarray:
    """HDBSCAN (excess-of-mass selection) on cosine distance. Products with membership probability below `min_prob`
    become singletons: HDBSCAN otherwise attaches outliers to the nearest large group."""
    from sklearn.cluster import HDBSCAN

    n = len(V)
    if n < 3:
        return np.arange(n)
    h = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1, metric="precomputed",
                cluster_selection_method="eom", copy=True).fit(_cosine_distance(V))
    raw = h.labels_.copy()
    raw[h.probabilities_ < min_prob] = -1
    return order_labels(raw, boxes)


def ocr_weighted(V: np.ndarray, tokens: list[set[str]], boxes: np.ndarray, threshold: float, text_weight: float,
                 fuzzy: float) -> np.ndarray:
    """similarity = (1 - w) * visual + w * text, where both crops have brand words; visual only otherwise.
    Then agglomerative clustering on 1 - similarity. (w = text_weight.)"""
    if len(V) <= 1:
        return np.zeros(len(V), int)
    T, has = text_similarity(tokens, fuzzy)
    both = np.outer(has, has)
    S = np.where(both, (1 - text_weight) * V + text_weight * T, V)
    return order_labels(_average_linkage(_cosine_distance(S), threshold), boxes)
