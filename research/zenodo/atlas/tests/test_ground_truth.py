"""Descriptor rules, variance partitioning, shufflers and the neighbour test of the 2026-10-24 ground-truth analysis."""
import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.spatial.distance import pdist, squareform

from research.zenodo.atlas import ground_truth as gt
from research.zenodo.atlas.tsne_atlas import source_family

EXAMPLES = {  # archive names -> declared coarse class
    "noise_cauchy_M5_T100": "uncoupled-white", "var_d-0_acf-low_M5_T100": "uncoupled-serial",
    "vma_d-0_acf-high_M10_T500": "uncoupled-serial", "ornstein-uhlenbeck_M-16_T-500": "uncoupled-serial",
    "brownian_geometric_M-16_T-100": "uncoupled-trending", "var_d-0.1_symmetric_acf-high_M5_T100": "linear-lagged",
    "vma_d-0.1_asymmetric_acf-low_M5_T100": "linear-lagged", "TestCLIMnonstat_S4": "linear-lagged",
    "TestWEATHmiss_S3": "nonlinear-lagged", "sim21": "linear-instantaneous",
    "defect_turbulence_alpha-1-895_epsilon-0-1_M5_T100": "nonlinear-lagged",
    "kuramoto-sakaguchi_conn-all_M-9_T-100": "oscillator", "oscillator_sync_k--20_conn_all-M20-T500": "oscillator",
    "vanderpol_conn-rand_M-25_T-500": "oscillator", "oscillator_legion_stim-split_conn-bidir_params-1-M20-T500": "oscillator",
    "oscillator_hysteresis_ow--4_nw--1_is-lin_io-neg-M10-T1000": "oscillator", "wilson-cowan_M-25_T-100": "oscillator",
    "wave-1D_M-25_T-1000": "wave-field", "wave-2D_M-16_T-1000": "wave-field"}


def test_declared_rules_assign_the_expected_classes():
    rules = yaml.safe_load(gt.CONFIG.read_text())["generators"]
    meta = pd.DataFrame({"dataset": list(EXAMPLES), "source": [source_family(n) for n in EXAMPLES]})
    table = gt.descriptors(meta, rules)
    assert table.coarse.tolist() == list(EXAMPLES.values())
    assert set(table.fine[table.coarse == "oscillator"]) == {"phase-difference", "position-forcing", "relaxation", "sigmoid-rate"}
    with pytest.raises(ValueError):
        gt.descriptors(pd.DataFrame({"dataset": ["NATOPS_C-1_I-1"], "source": ["NATOPS"]}), rules)


def test_venn_fractions_sum_to_the_full_model():
    rng = np.random.default_rng(0)
    blocks = {k: rng.normal(size=(60, 3)) for k in "abc"}
    Y = blocks["a"] @ rng.normal(size=(3, 4)) + blocks["b"][:, :1] + rng.normal(size=(60, 4))
    table = gt.subsets(Y, blocks)
    parts = gt.venn3(table, "a", "b", "c")
    assert np.isclose(sum(v for k, v in parts.items() if k != "residual"), table[frozenset("abc")]["adj_r2"])
    assert parts["[a]"] > parts["[c]"]


def crossed(seed=0, effect="class"):
    """Two families, each with parents of both classes and three windows per parent; Y follows one factor."""
    rng = np.random.default_rng(seed)
    rows = []
    for family in "AB":
        for p in range(8):
            cls = "x" if p < 4 else "y"
            for w in range(3):
                rows.append({"source": family, "parent": f"{family}{p}", "unit": f"{family}{cls}", "coarse": cls, "M": [5, 10, 20][w],
                             "T": 100 * (1 + p % 2), "toolbox": family})
    frame = pd.DataFrame(rows)
    shift = (frame.coarse == "y") if effect == "class" else (frame.source == "B")
    Y = rng.normal(size=(len(frame), 6)) + 3 * shift.to_numpy()[:, None]
    return Y, frame


def test_partial_effects_follow_the_planted_factor():
    """Class planted: class | family is detected; family | class rejects at about the nominal rate over seeds."""
    rejections = 0
    for seed in range(20):
        Y, frame = crossed(seed, effect="class")
        blocks = gt.design_blocks(frame, "coarse")
        part = gt.Partition(Y, blocks)
        rng = np.random.default_rng(seed + 100)
        in_family = gt.GroupShuffler(frame.parent, frame.coarse, frame.source)
        assert gt.permutation_test(part, blocks["class"], ["family", "size"], lambda r: gt.onehot(in_family(r)), 99, rng)["p"] <= .02
        in_class = gt.GroupShuffler(frame.parent, frame.source, frame.coarse)
        rejections += gt.permutation_test(part, blocks["family"], ["class", "size"], lambda r: gt.onehot(in_class(r)), 99, rng)["p"] <= .05
    assert rejections <= 4


def test_shuffler_keeps_groups_constant_and_strata_closed():
    _, frame = crossed()
    shuffle = gt.GroupShuffler(frame.parent, frame.coarse, frame.source)
    out = shuffle(np.random.default_rng(3))
    assert (pd.Series(out).groupby(frame.parent).nunique() == 1).all()
    for family, rows in frame.groupby("source").groups.items():
        assert sorted(out[rows]) == sorted(frame.coarse[rows])
    with pytest.raises(ValueError):
        gt.GroupShuffler(frame.source, frame.coarse)


def located(seed=0):
    """Eight single-unit families; families of class x share one place, families of class y another."""
    rng = np.random.default_rng(seed)
    rows, points = [], []
    for f in range(8):
        cls = "x" if f < 4 else "y"
        for p in range(3):
            for w in range(2):
                rows.append({"source": f"F{f}", "unit": f"F{f}", "parent": f"F{f}-{p}", "coarse": cls, "M": 10, "T": 100})
                points.append(rng.normal(0 if cls == "x" else 6, 1, size=3))
    return pd.DataFrame(rows), np.array(points)


def test_neighbour_test_finds_cross_family_same_class_links():
    frame, points = located()
    strata = pd.Series(0, index=frame.parent.unique())
    edges = gt.other_parent_neighbours(squareform(pdist(points)), frame.parent.to_numpy(), 1)
    summary, units = gt.nn_test(edges, frame, "coarse", strata, 300, 50, np.random.default_rng(2))
    by_null = summary.set_index("null")
    assert by_null.loc["units", "p"] < .05 and by_null.loc["units", "ratio"] > 1.5
    assert len(units) == 8 and units.observed.between(0, 1).all()


def test_neighbours_skip_own_parent_and_other_strata():
    frame, points = located()
    strata = np.arange(len(frame)) % 2
    src, dst = gt.other_parent_neighbours(squareform(pdist(points)), frame.parent.to_numpy(), 2, strata)
    assert not np.any(frame.parent.to_numpy()[src] == frame.parent.to_numpy()[dst])
    assert np.all(strata[src] == strata[dst])


def test_raw_checks_see_a_lag_and_white_noise():
    rng = np.random.default_rng(4)
    x = rng.normal(size=2003)
    # channel 1 lags channel 0 by two samples; channel 2 is a noisy zero-lag copy of channel 0
    lagged = np.vstack([x[3:], x[1:-2], x[3:] + .1 * rng.normal(size=2000)])
    row = gt.raw_checks({"a": lagged}, ["a"]).iloc[0]
    assert np.isclose(row.lagged_peak_share, 2 / 3) and abs(row.ac1) < .05 and row.trend_r2 < .01


def test_jackknife_errors_are_finite_and_cover_the_planted_effect():
    Y, frame = crossed(effect="class")
    se = gt.jackknife_fractions(Y, frame, "coarse", "rows")
    value = gt.fractions(Y, frame, "coarse", "rows")
    assert np.isfinite(se).all() and (se >= 0).all()
    assert value["R2adj class"] - 1.96 * se["R2adj class"] > .3
