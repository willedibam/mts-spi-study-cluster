"""Timing over time, block design (2026-10-29): ED, xED and DTW on one 5/5/5 channel layout, 2 x 2 over
lag and warping.

M = 15 in three blocks of five that read one smooth mother (OU, lag-one autocorrelation 0.9), T = 1000
(`mtsspi.generators.generate_delay_blocks_mts`). Block 1 is always aligned; block 2 is aligned or lagged by
1, 2, 3, 4 and 5 samples; block 3 is aligned or warped about zero delay, A_k sin(2 pi t / 250) with
A_k = 1.2, 2.4, 3.6, 4.8 and 6 samples. The "all three" condition is the 5 aligned / 5 lagged / 5 warped
recording. Noise 0.1-0.4, the same five levels in every block, assigned in the order (1, 4, 2, 0, 3), which
is rank-uncorrelated with lag and warp. Replicate k uses the same mother and noise draws in every condition.
Lags are whole samples because xED searches whole-sample shifts (fractional lags leave a residual it cannot
absorb; pilot 2026-10-29). Local pyspi: `pdist_euclidean_rmse`, `xpdist_euclidean_tau-10_min_rmse`,
`dtw_rmse`.

    python -m research.cases.ed_xed_dtw.drift_blocks_261029
"""
from __future__ import annotations

import warnings

import numpy as np

from mtsspi.generators import generate_delay_blocks_mts
from research.cases.case_run import ROOT, Case, run, system_figure
from research.cases.composite_style import AMBER, ROSE, SAGE

SPI_CONFIG = ROOT / "research/cases/ed_xed_dtw/spis/dtw_euclidean_norm.yaml"
KEYS = {"ed": "pdist_euclidean_rmse", "xed": "xpdist_euclidean_tau-10_min_rmse", "dtw": "dtw_rmse"}
BASE = dict(M=15, T=1000, lags=(1, 2, 3, 4, 5), warp=6.0, warp_period=250.0, noise_low=0.1, noise_high=0.4,
            noise_order=(1, 4, 2, 0, 3), ar1_a=0.9)
LEVELS = {"no lag": {**BASE, "blocks": ("aligned", "aligned", "aligned")},
          "lag": {**BASE, "blocks": ("aligned", "lagged", "aligned")},
          "warping": {**BASE, "blocks": ("aligned", "aligned", "warped")},
          "all three": {**BASE, "blocks": ("aligned", "lagged", "warped")}}
COLOURS = {"no lag": "#7B8190", "lag": "#0072B2", "warping": "#D55E00", "all three": "#6A4C93"}
PAIR_INK = {"aligned": "#B4B7BD", "lagged": "#0072B2", "warped": "#D55E00"}


def relation(types):
    """Per ordered pair: "warped" if either channel is warped, else "lagged" if either is lagged, else "aligned".
    Lags and warp amplitudes are distinct within a block, so every such pair differs in delay."""
    t = np.asarray(types)
    warp = (t[:, None] == "warped") | (t[None, :] == "warped")
    lag = (t[:, None] == "lagged") | (t[None, :] == "lagged")
    return np.where(warp, "warped", np.where(lag, "lagged", "aligned"))


def make(kw, rep):
    return generate_delay_blocks_mts(**kw, rng=np.random.default_rng([20261029, rep]), return_internals=True)


def stats(x):
    warnings.filterwarnings("ignore")
    from mtsspi.compute import run_pyspi
    mats = run_pyspi(x, config_path=SPI_CONFIG, normalise=False).matrices
    return {k: mats[v] for k, v in KEYS.items()}


def pair_colour(label, it):
    return np.vectorize(PAIR_INK.get, otypes=[object])(relation(it.types)), None


CASE = Case(name="drift_blocks", levels=LEVELS, make=make, stats=stats,
            spis=(("ed", "ED", SAGE), ("xed", "xED", ROSE), ("dtw", "DTW", AMBER)),
            rungs=(("ed", "xed"), ("xed", "dtw")), level_colours=COLOURS, pair_colour=pair_colour, reps=20,
            extra={"spi_config": str(SPI_CONFIG.relative_to(ROOT)), "seed": [20261029, "rep"],
                   "z_entries": "ordered off-diagonal, M(M-1)"})

if __name__ == "__main__":
    out = ROOT / "results/cases/drift-blocks-261029"
    frame, examples = run(CASE, out)
    print(frame.groupby("level", sort=False).agg(["mean", "std"]).drop(columns="rep").round(3).T.to_string())
    print(system_figure(CASE, frame, examples, out))
