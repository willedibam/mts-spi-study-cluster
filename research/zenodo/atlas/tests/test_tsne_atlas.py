"""Exaggeration hook, parent grouping and cluster-stability helpers of the 2026-10-21 atlas."""
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from research.zenodo.atlas.tsne_atlas import (best_jaccard, grouped_jaccard, nested_parents, parent_key, raw_descriptors,
                                              representatives, source_family, tsne, ward)


def blobs(seed=0):
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(loc, .3, size=(40, 6)) for loc in (0., 3., 6.)])


def test_unit_exaggeration_is_stock_sklearn():
    x = blobs()
    ours = tsne(x, perplexity=15, exaggeration=1, seed=3, max_iter=300, threads=1)
    stock = TSNE(perplexity=15, init="pca", learning_rate="auto", random_state=3, max_iter=300, n_jobs=1).fit_transform(x)
    np.testing.assert_array_equal(ours, stock)


def test_exaggeration_compacts_the_layout():
    x = blobs()
    plain = tsne(x, perplexity=15, exaggeration=1, seed=3, max_iter=300, threads=1)
    tight = tsne(x, perplexity=15, exaggeration=4, seed=3, max_iter=300, threads=1)
    within = lambda y: np.mean([y[i:i + 40].std(axis=0).mean() for i in (0, 40, 80)]) / y.std(axis=0).mean()
    assert within(tight) < .8 * within(plain)  # tighter clusters relative to the layout


def test_parent_keys_group_windows_markets_and_subjects():
    assert parent_key("wave-1D_M-16_T-100") == parent_key("wave-1D_M-25_T-1000") == "wave-1D"
    assert parent_key("oscillator_sync_k-4_conn_all-M20-T500") == "oscillator_sync_k-4_conn_all"
    assert parent_key("DJIA_N-679_P-9_2010-01-01-lr") == parent_key("DJIA_N-678_P-9_2010-01-01") == "DJIA_2010-01-01"
    assert parent_key("UCLA_neuro_ADHD_S-220_R-60-74") == "UCLA_neuro_ADHD_S-220"
    assert parent_key("NATOPS_C-10_I-0") != parent_key("NATOPS_C-10_I-1")
    assert source_family("PEMS-SF_C-10_I-0") == "PEMS-SF" and source_family("sim21") == "NetSim fMRI"


def test_nested_parents_merges_exact_prefixes_only():
    base = np.arange(40.).reshape(4, 10)
    arrays = {"a": base, "b": base[:2, :6].copy(), "c": base + 1}
    keys = nested_parents(["a", "b", "c"], arrays)
    assert keys[0] == keys[1] != keys[2]


def test_grouped_jaccard_is_one_for_separated_groups():
    x = blobs()
    labels = ward(x, 3)
    parents = np.repeat(np.arange(60), 2)
    table = grouped_jaccard(x, labels, lambda s: ward(s, 3), parents, replicates=5, fraction=.8, seed=1)
    assert np.allclose(table.jaccard_mean, 1) and best_jaccard(labels, labels, 0) == 1


def test_representatives_start_at_medoid_and_cover_sources():
    x = blobs()[:40]
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    meta = pd.DataFrame({"source": ["s"] * 30 + ["t"] * 10})
    rows = representatives(np.arange(40), d, meta, n=4)
    assert rows[0] == np.argmin(d.sum(axis=1)) and {"s", "t"} <= set(meta.source[rows])


def test_raw_descriptors_read_sign_trend_and_outliers():
    t = np.arange(500.)
    rng = np.random.default_rng(0)
    common = rng.normal(size=500).cumsum()
    anti = np.vstack([common, -common + .1 * rng.normal(size=500)])
    assert raw_descriptors(anti)["negative_share_strong"] == 1
    assert raw_descriptors(np.vstack([t, 2 * t + 1]))["trend_r2"] > .99
    spiky = rng.normal(size=(3, 500))
    spiky[0, 10] = 999.
    assert raw_descriptors(spiky)["max_robust_z"] > 100
