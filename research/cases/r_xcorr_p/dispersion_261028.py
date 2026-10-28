"""Timing over frequency, one system and one knob (2026-10-28): r, max cross-correlation and the
phase-aligned correlation P as the delays go from absent, to one delay per channel, to a delay that
differs between frequency bands.

M = 12, T = 10,000. Two non-overlapping bands (slow |f| < 0.08, fast 0.10-0.40), each driven by its
own common source delayed per channel (`mtsspi.generators.generate_band_delayed_mts`). Levels: no
delay; delays 0-15 samples shared by both bands (kappa = 0); fast-band delays moved half way
(kappa = 0.5) and fully (kappa = 1) to a fixed permutation of the slow band's. Every level has
identical spectral magnitudes; only cross-spectral phase changes. Replicate k reuses the same
process draws at every level. All four statistics from local pyspi (`spis/r_xcorr_p.yaml`); P is the
fork's `PhaseAlignedCorrelation` (Welch, L = 256), equal to `timing_triplet.phase_aligned_corr` to
2e-16. cohmag is reported against max xcorr and P as the catalogue proxy for P.

    python -m research.cases.r_xcorr_p.dispersion_261028
"""
from __future__ import annotations

import warnings

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from mtsspi.generators import generate_band_delayed_mts
from research.cases.case_run import ROOT, Case, run, system_figure, z
from research.cases.composite_style import ROSE, SAGE, SKY
from research.cases.r_xcorr_p import timing_triplet as tt

BASE = dict(M=12, T=10_000, lag_range=15)
LEVELS = {"no delay": {**BASE, "family": "synchronous"},
          "delay": {**BASE, "family": "dispersive", "dispersion": 0.0},
          "dispersion .5": {**BASE, "family": "dispersive", "dispersion": 0.5},
          "dispersion 1": {**BASE, "family": "dispersive", "dispersion": 1.0}}
SPI_CONFIG = ROOT / "research/cases/r_xcorr_p/spis/r_xcorr_p.yaml"
P_KEY = "phasealigned_welch_nperseg-256"
COLOURS = {"no delay": "#7B8190", "delay": "#0072B2", "dispersion .5": "#B07AA1", "dispersion 1": "#CC79A7"}
CMAP = LinearSegmentedColormap.from_list("mismatch", ["#B4B7BD", "#CC79A7", "#6E2C5A"])


def make(kw, rep):
    return generate_band_delayed_mts(**kw, rng=np.random.default_rng([20261028, rep]), return_internals=True)


def stats(x):
    warnings.filterwarnings("ignore")
    from mtsspi.compute import run_pyspi
    mats = run_pyspi(x, config_path=SPI_CONFIG, normalise=False).matrices
    return {"r": mats[tt.R], "xc": mats[tt.XC], "P": mats[P_KEY], "coh": mats[tt.CM]}


def coh_z(mats):
    return {"x_xc_coh": z(mats["xc"], mats["coh"]), "x_coh_P": z(mats["coh"], mats["P"])}


def pair_colour(label, it):
    return tt.band_mismatch(it), CMAP


CASE = Case(name="dispersion", levels=LEVELS, make=make, stats=stats,
            spis=(("r", "$r$", SAGE), ("xc", r"$\max_h r(h)$", ROSE), ("P", "$P$", SKY)),
            rungs=(("r", "xc"), ("xc", "P")), level_colours=COLOURS, pair_colour=pair_colour, reps=20,
            extra_z=coh_z, extra={"spi_config": str(SPI_CONFIG.relative_to(ROOT)), "P_spi": P_KEY,
                                  "seed": [20261028, "rep"], "z_entries": "ordered off-diagonal, M(M-1)"})

if __name__ == "__main__":
    out = ROOT / "results/cases/dispersion-261028"
    frame, examples = run(CASE, out)
    print(frame.groupby("level", sort=False).agg(["mean", "std"]).drop(columns="rep").round(3).T.to_string())
    print(system_figure(CASE, frame, examples, out, t_window=(0, 200)))
