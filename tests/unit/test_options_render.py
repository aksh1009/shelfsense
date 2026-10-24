import numpy as np
import pytest
from PIL import Image

from shelfai import schemas
from shelfai.errors import InvalidRequest
from shelfai.options import parse_options
from shelfai.visualization.render import group_colors, render_groups


def test_defaults_and_validation(cfg):
    o = parse_options({}, cfg)
    assert o["detection"]["detector"] == cfg["detection"]["defaults"]["detector"]
    assert parse_options({"containment": "off"}, cfg)["detection"]["containment"] == 0
    for bad in ({"detector": "nope"}, {"method": "nope"}, {"embedder": "nope"}, {"threshold": 5}, {"conf": "x"}):
        with pytest.raises(InvalidRequest):
            parse_options(bad, cfg)


def test_parse_detections_validates():
    boxes, scores = schemas.parse_detections([{"bbox": [1, 2, 3, 4], "confidence": 0.5}])
    assert boxes.shape == (1, 4)
    with pytest.raises(InvalidRequest):
        schemas.parse_detections([{"bbox": [5, 5, 1, 1]}])


def test_render_legend_labels_and_removed():
    img = Image.new("RGB", (1000, 800), (200, 200, 200))
    labels = np.array([0, 0, 1])
    groups = schemas.grouping_response(labels, {0: "BRANDX"}, {}, {}, {})["groups"]
    out = render_groups(img, np.array([[10, 10, 100, 100], [200, 10, 300, 100], [400, 10, 500, 100]]), labels, groups,
                        removed_boxes=np.array([[20, 20, 60, 60]]))
    assert out.height == 800 and out.width > 1000
    colors = group_colors(groups)
    assert colors["G1"] != colors["G2"] and colors["G2"] == "#a0a0a0"
