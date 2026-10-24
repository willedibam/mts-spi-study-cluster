"""Neighbour edges, family enrichment and FDR adjustment of the 2026-10-24 neighbourhood analysis."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from research.zenodo.atlas.neighbourhood_atlas import bh, enrichment, neighbour_edges, parent_strata


def corpus(seed=0):
    """Families A and B share one region of the space, C sits apart; three windows per parent."""
    rng = np.random.default_rng(seed)
    rows = []
    for family, centre, parents in [("A", 0., 12), ("B", .5, 12), ("C", 8., 12), ("D", 16., 12)]:
        for p in range(parents):
            base = rng.normal(centre, 1., size=2)
            for w in range(3):
                rows.append({"source": family, "parent": f"{family}{p}", "M": 5 + 5 * w, "T": 100,
                             "origin": "synthetic" if family in "AC" else "real", "x": base + rng.normal(0, .05, size=2)})
    meta = pd.DataFrame(rows)
    return meta, np.stack(meta.pop("x"))


def test_edges_skip_the_own_parent():
    meta, x = corpus()
    src, dst = neighbour_edges(squareform(pdist(x)), meta.parent.to_numpy(), 4)
    assert len(src) == 4 * len(meta)
    assert not np.any(meta.parent.to_numpy()[src] == meta.parent.to_numpy()[dst])


def test_enrichment_finds_the_planted_pair_only():
    meta, x = corpus()
    edges = neighbour_edges(squareform(pdist(x)), meta.parent.to_numpy(), 4)
    table = enrichment(edges, meta, bins=0, permutations=400, seed=1).set_index(["family_a", "family_b"])
    assert table.loc[("A", "B"), "p"] < .01 and table.loc[("A", "B"), "ratio"] > 1.5
    assert table.loc[("A", "B"), "kind"] == "model–real"
    assert table.loc[("A", "C"), "p"] > .5


def test_one_stratum_without_bins():
    meta, _ = corpus()
    assert parent_strata(meta, 0).nunique() == 1 and len(parent_strata(meta, 0)) == meta.parent.nunique()


def test_bh_matches_hand_values():
    np.testing.assert_allclose(bh(np.array([.01, .04, .03, .5])), [.04, .16 / 3, .16 / 3, .5])
