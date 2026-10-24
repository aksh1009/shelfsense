"""Grouping methods on synthetic embeddings: two brands, one of which comes in three colourways."""
import numpy as np
import pytest

from shelfai.grouping import methods
from shelfai.grouping.embedders import l2_normalize
from shelfai.grouping.text import brand_tokens, propagate_tokens, text_similarity

rng = np.random.default_rng(0)
D = 64
W, B, K = l2_normalize(rng.normal(size=(3, D)))
OFF = l2_normalize(rng.normal(size=(1, D)))[0]
pts = lambda c, n: [c + 0.05 * rng.normal(size=D) for _ in range(n)]
# 0-2 brand A light, 3-5 brand A blue, 6-7 brand A dark, 8 brand A dark UNREADABLE, 9-10 brand B dark (looks like A dark)
EMB = l2_normalize(np.array(pts(W, 3) + pts(B, 3) + pts(K, 3) + pts(l2_normalize((K + 0.45 * OFF)[None])[0], 2)))
TOK = [{"BRANDA"}] * 8 + [set()] + [{"BRANDB"}] * 2
BOX = np.array([[i * 10, 0, i * 10 + 5, 5] for i in range(len(EMB))], np.float32)
V = EMB @ EMB.T


def same(lab, idx):
    return len({lab[i] for i in idx}) == 1


def test_agglomerative_mixes_lookalikes_as_expected():
    lab = methods.agglomerative(V, BOX, 0.4)
    assert same(lab, [6, 7, 9, 10])  # the known visual failure: brand B joins brand A's dark packs
    assert not same(lab, [0, 3])     # and brand A is split by colour


def test_ocr_weighted_fixes_both_errors():
    lab = methods.ocr_weighted(V, propagate_tokens(TOK, V, 0.85), BOX, 0.4, 0.7, 0.85)
    assert same(lab, range(9)) and same(lab, [9, 10]) and lab[9] != lab[0]


def test_hdbscan_runs_and_small_inputs():
    assert len(methods.hdbscan(V, BOX, 2)) == len(V)
    assert methods.hdbscan(V[:2, :2], BOX[:2], 2).tolist() == [0, 1]


@pytest.mark.parametrize("fn", [lambda v, b: methods.agglomerative(v, b, 0.2),
                                lambda v, b: methods.ocr_weighted(v, [set()] * len(v), b, 0.2, 0.3, 0.85)])
def test_zero_and_one_product(fn):
    assert fn(np.zeros((0, 0)), BOX[:0]).tolist() == []
    assert fn(np.ones((1, 1)), BOX[:1]).tolist() == [0]


def test_deterministic_numbering_largest_first():
    lab = methods.agglomerative(V, BOX, 0.4)
    assert (lab == methods.agglomerative(V, BOX, 0.4)).all()
    assert np.bincount(lab)[0] == np.bincount(lab).max()


def test_brand_tokens_keep_large_non_generic_words():
    reads = [dict(text="brandx", conf=0.9, height=50), dict(text="Extra Soft 28cm", conf=0.9, height=12),
             dict(text="SHAMPOO", conf=0.9, height=48), dict(text="zz", conf=0.1, height=60)]
    assert brand_tokens(reads, 3, 0.3, 0.5) == {"BRANDX"}


def test_text_similarity_overlap():
    T, has = text_similarity([{"BRANDX"}, {"BRANDX", "AIR"}, {"OTHER"}, set()], 0.85)
    assert T[0, 1] == 1.0 and T[0, 2] == 0.0 and has.tolist() == [True, True, True, False]
