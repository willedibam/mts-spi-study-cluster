"""Agreement scoring, cohesion and map choice of the 2026-10-24 consensus atlas."""
import numpy as np
import pandas as pd

from research.zenodo.atlas.consensus_atlas import choose, cohesion, map_agreement


def test_map_agreement_needs_one_region_that_is_mostly_the_group():
    rng = np.random.default_rng(0)
    xy = np.vstack([rng.normal(c, .1, size=(20, 2)) for c in ((0, 0), (5, 0), (0, 5))])
    labels = np.repeat([0, 1, 2], 20)
    labels[40:50] = 0  # group 0 now spans two separate blobs
    agree, regions = map_agreement(xy, labels, [0, 1, 2], .8, .5, {"min_cluster_size": 5, "min_samples": 5})
    assert agree[1][3] and not agree[0][3]
    assert regions.max() + 1 == 3


def test_cohesion_is_higher_for_tight_sets():
    rng = np.random.default_rng(1)
    x = np.vstack([rng.normal(0, .1, size=(10, 3)), rng.normal(3, 1., size=(40, 3))])
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    tight, loose = np.arange(10), np.arange(10, 20)
    assert cohesion(d, tight, np.arange(10, 50)) > cohesion(d, loose, np.setdiff1d(np.arange(50), loose))


def test_choose_respects_the_recall_floor():
    grid = pd.DataFrame({"key": ["a", "b", "c"], "agree": [9., 8., 7.], "agree_min": [9, 8, 7],
                         "recall15_analysis": [.40, .60, .62]})
    assert choose(grid, .95) == "b"
