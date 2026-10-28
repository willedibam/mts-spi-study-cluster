"""Timing over time, one system and one knob (2026-10-28): Euclidean distance, lag-tolerant
Euclidean distance (min over shifts |h| <= 10) and DTW as delays go from absent, to constant,
to drifting slowly over time.

M = 12, T = 1,000. Noisy copies of one smooth mother (lag-one autocorrelation 0.9), channel i read
at t - tau_i - D c_i sin(2 pi t / 250 + theta_i) (`mtsspi.generators.generate_drifting_delay_mts`).
Levels: no delay; constant delays tau_i = 0-4 samples; the same delays plus drift D = 3 and D = 6.
c_i ~ U(0, 1), theta_i and the noise order are fixed by the structure seed, so pairs differ in how
much their relative delay drifts. Replicate k reuses the same mother and noise at every level.
Local pyspi: `pdist_euclidean_rmse`, `xpdist_euclidean_tau-10_min_rmse`, `dtw_rmse`.

    python -m research.cases.ed_xed_dtw.drift_261028
"""
from __future__ import annotations

import warnings

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from mtsspi.generators import generate_drifting_delay_mts
from research.cases.case_run import ROOT, Case, run, system_figure
from research.cases.composite_style import AMBER, ROSE, SAGE

SPI_CONFIG = ROOT / "research/cases/ed_xed_dtw/spis/dtw_euclidean_norm.yaml"
KEYS = {"ed": "pdist_euclidean_rmse", "xed": "xpdist_euclidean_tau-10_min_rmse", "dtw": "dtw_rmse"}
BASE = dict(M=12, T=1000, drift_period=250.0, ar1_a=0.9, noise_low=0.1, noise_high=0.4)
LEVELS = {"no delay": {**BASE, "max_lag": 0, "drift": 0}, "delay": {**BASE, "max_lag": 4, "drift": 0},
          "drift 3": {**BASE, "max_lag": 4, "drift": 3}, "drift 6": {**BASE, "max_lag": 4, "drift": 6}}
COLOURS = {"no delay": "#7B8190", "delay": "#0072B2", "drift 3": "#E69F00", "drift 6": "#D55E00"}
CMAP = LinearSegmentedColormap.from_list("drift", ["#B4B7BD", "#E69F00", "#8C3A00"])


def make(kw, rep):
    return generate_drifting_delay_mts(**kw, rng=np.random.default_rng([20261028, rep]), return_internals=True)


def stats(x):
    warnings.filterwarnings("ignore")
    from mtsspi.compute import run_pyspi
    mats = run_pyspi(x, config_path=SPI_CONFIG, normalise=False).matrices
    return {k: mats[v] for k, v in KEYS.items()}


def pair_colour(label, it):
    """Amplitude of the pair's relative drift, |A_i e^{i th_i} - A_j e^{i th_j}| (samples)."""
    a = it.drift_amp * np.exp(1j * it.drift_phase)
    return np.abs(a[:, None] - a[None, :]), CMAP


CASE = Case(name="drift", levels=LEVELS, make=make, stats=stats,
            spis=(("ed", "ED", SAGE), ("xed", "xED", ROSE), ("dtw", "DTW", AMBER)),
            rungs=(("ed", "xed"), ("xed", "dtw")), level_colours=COLOURS, pair_colour=pair_colour, reps=20,
            extra={"spi_config": str(SPI_CONFIG.relative_to(ROOT)), "seed": [20261028, "rep"],
                   "z_entries": "ordered off-diagonal, M(M-1)"})

if __name__ == "__main__":
    out = ROOT / "results/cases/drift-261028"
    frame, examples = run(CASE, out)
    print(frame.groupby("level", sort=False).agg(["mean", "std"]).drop(columns="rep").round(3).T.to_string())
    print(system_figure(CASE, frame, examples, out))
