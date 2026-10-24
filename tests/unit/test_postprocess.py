import numpy as np

from shelfai.detection.postprocess import containment_filter, postprocess

B = lambda *b: np.array(b, np.float32)


def test_half_inside_higher_scoring_whole_is_removed():
    assert containment_filter(B([0, 0, 10, 20], [0, 0, 10, 10]), np.array([.9, .8]), .8).tolist() == [0]


def test_part_scoring_higher_than_whole_keeps_both():
    assert containment_filter(B([0, 0, 10, 20], [0, 0, 10, 10]), np.array([.6, .9]), .8).tolist() == [0, 1]


def test_neighbours_are_kept():
    assert len(containment_filter(B([0, 0, 10, 20], [9, 0, 19, 20]), np.array([.9, .8]), .8)) == 2


def test_disabled_and_empty():
    assert containment_filter(B([0, 0, 10, 20], [0, 0, 10, 10]), np.array([.9, .8]), 0).tolist() == [0, 1]
    assert containment_filter(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), .8).tolist() == []


def test_postprocess_reports_removed_boxes():
    kb, ks, rb, rs = postprocess(B([0, 0, 10, 20], [0, 0, 10, 10], [50, 50, 60, 60]), np.array([.9, .8, .1], np.float32), .25, .8)
    assert len(kb) == 1 and len(rb) == 1 and rb[0].tolist() == [0, 0, 10, 10]
