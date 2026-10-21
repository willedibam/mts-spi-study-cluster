"""Size residualisation, matched-null and rank helpers of the 2026-10-21 M/T control."""
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples

from research.zenodo.atlas.mt_control import (embedded_silhouette, matched_null, other_cluster_rank, parent_pairs,
                                              residualise, size_basis)
from research.zenodo.atlas.tsne_atlas import ward


def sized_corpus(seed=0):
    """Two parents per source, each at four sizes; source offsets aliased with M, plus a shared size effect."""
    rng = np.random.default_rng(seed)
    rows = [(s, f"{s}-{p}", m, t) for s, m in [("small", 5), ("large", 20)] for p in range(2) for t in (100, 300, 1000, 2000)]
    meta = pd.DataFrame(rows, columns=["source", "parent", "M", "T"])
    meta["row"] = np.arange(len(meta))
    source_offset = np.where(meta.source == "large", 5., 0.)[:, None] * np.eye(4)[0]
    size_effect = np.log(meta["T"].to_numpy(float))[:, None] * np.eye(4)[1]
    return meta, source_offset + size_effect + .01 * rng.normal(size=(len(meta), 4))


def test_parent_control_removes_size_effect_but_keeps_aliased_source():
    meta, x = sized_corpus()
    parent, info = residualise(x, meta, "parent")
    pooled, _ = residualise(x, meta, "pooled")
    assert np.ptp(parent[:, 1]) < .1 and info["T_support"] == (100, 2000)  # within-parent T effect gone
    gap = lambda y: y[meta.source == "large", 0].mean() - y[meta.source == "small", 0].mean()
    assert abs(gap(parent) - 5) < .05 and abs(gap(pooled)) < .05  # pooled also removes the source offset


def test_placebo_permutes_size_within_groups_only():
    meta, x = sized_corpus()
    _, info = residualise(x, meta, "parent", permute=np.random.default_rng(1))
    assert info["M_support"] == (5, 20) and size_basis([5], [100]).shape == (1, 5)


def test_embedded_silhouette_reproduces_cluster_silhouette():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(c, .3, size=(20, 3)) for c in (0., 3., 6.)])
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    labels = ward(x, 3)
    sil = silhouette_samples(d, labels, metric="precomputed")
    assert np.isclose(embedded_silhouette(d, labels, np.flatnonzero(labels == 1)), sil[labels == 1].mean())
    rank = other_cluster_rank(d, labels)
    assert (rank[np.arange(len(labels)), labels] == 3).all() and set(np.unique(rank)) == {1, 2, 3}


def test_matched_null_excludes_candidate_sources_and_matches_size():
    rng = np.random.default_rng(0)
    meta = pd.DataFrame({"source": ["a"] * 10 + ["b"] * 10 + ["c"] * 10, "M": [24] * 10 + [24] * 10 + [5] * 10,
                         "T": [51] * 10 + [51] * 10 + [1000] * 10})
    x = np.vstack([rng.normal(c, .2, size=(10, 3)) for c in (0., 5., 10.)])
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    labels = np.repeat([0, 1, 2], 10)
    table, _ = matched_null(d, labels, meta, candidates={0: "a"}, replicates=20, k=1)
    # matched sets are exactly source b (same size, other source): as compact as the candidate
    assert table.match_M_ratio[0] == table.match_T_ratio[0] == 1 and table.matched_sources[0] == "b 10"
    assert abs(table.survival[0]) < .1


def test_parent_pairs_flag_market_transforms():
    meta = pd.DataFrame({"row": [0, 1, 2], "source": ["ForEx", "ForEx", "VAR"], "parent": ["fx", "fx", "var"],
                         "M": [9, 9, 5], "T": [885, 886, 100]})
    d = np.array([[0., 1., 2.], [1., 0., 2.], [2., 2., 0.]])
    pairs = parent_pairs(d, meta, np.zeros(3, int))
    assert pairs.kind.tolist() == ["transform"]
