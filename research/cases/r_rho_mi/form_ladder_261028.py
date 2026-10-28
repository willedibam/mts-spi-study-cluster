"""Form case (2026-10-28): r, rho and MI on one 4/4/4 channel layout, 2 x 2 over the maps.

M = 12 in three blocks of four that observe one shared AR(1) latent. Block 1 is always linear; block 2
is linear or monotone (sinh 2w); block 3 is linear or folded (w^2). The "both" condition is the
4 linear / 4 monotone / 4 fold recording. Strength is set by latent noise placed *before* the map
(log-spaced, the same four levels in every block), so a monotone map leaves Spearman rho and MI
exactly unchanged and moves only Pearson r (`mtsspi.generators.generate_form_ladder_mts`). sinh is
odd, so monotone-fold pairs have r = rho = 0 in population. Replicate k uses the same latent and noise
draws in every condition. Local pyspi: r, rho, KSG MI (k = 4).

    python -m research.cases.r_rho_mi.form_ladder_261028
"""
from __future__ import annotations

import warnings

import numpy as np

from mtsspi.generators import generate_form_ladder_mts
from research.cases.case_run import ROOT, Case, run, system_figure, z
from research.cases.composite_style import AMBER, LAVENDER, SAGE

SPI_CONFIG = ROOT / "research/cases/r_rho_mi/spis/r_rho_mi_261024.yaml"
KEYS = {"r": "cov_EmpiricalCovariance", "rho": "spearmanr", "mi": "mi_kraskov_NN-4"}
BASE = dict(M=12, T=2000, curvature=2.0, noise_low=0.2, noise_high=1.0, ar1_a=0.8)
LEVELS = {"linear": {**BASE, "maps": ("linear", "linear", "linear")},
          "monotone": {**BASE, "maps": ("linear", "monotone", "linear")},
          "fold": {**BASE, "maps": ("linear", "linear", "fold")},
          "both": {**BASE, "maps": ("linear", "monotone", "fold")}}
COLOURS = {"linear": "#7B8190", "monotone": "#0072B2", "fold": "#CC79A7", "both": "#6A4C93"}
PAIR_INK = {"linear": "#B4B7BD", "monotone": "#0072B2", "fold": "#CC79A7"}


def relation(types):
    """Per ordered pair: "fold" if either channel is folded, else "monotone" if either is monotone, else "linear"."""
    t = np.asarray(types)
    fold = (t[:, None] == "fold") | (t[None, :] == "fold")
    mono = (t[:, None] == "monotone") | (t[None, :] == "monotone")
    return np.where(fold, "fold", np.where(mono, "monotone", "linear"))


def make(kw, rep):
    return generate_form_ladder_mts(**kw, rng=np.random.default_rng([20261028, rep]), return_internals=True)


def stats(x):
    warnings.filterwarnings("ignore")
    from mtsspi.compute import run_pyspi
    mats = run_pyspi(x, config_path=SPI_CONFIG, normalise=False).matrices
    return {k: mats[v] for k, v in KEYS.items()}


def linfoot_z(mats):
    mi = np.clip(mats["mi"], 0, None)
    return {"x_rho_milin": z(mats["rho"], np.sqrt(1 - np.exp(-2 * mi)))}


def pair_colour(label, it):
    rel = relation(it.types)
    return np.vectorize(PAIR_INK.get, otypes=[object])(rel), None


CASE = Case(name="form_ladder", levels=LEVELS, make=make, stats=stats,
            spis=(("r", "$r$", SAGE), ("rho", r"$\rho$", LAVENDER), ("mi", "MI", AMBER)),
            rungs=(("r", "rho"), ("rho", "mi")), level_colours=COLOURS, pair_colour=pair_colour, reps=20,
            extra_z=linfoot_z, extra={"spi_config": str(SPI_CONFIG.relative_to(ROOT)), "seed": [20261028, "rep"],
                                      "z_entries": "ordered off-diagonal, M(M-1)"})

if __name__ == "__main__":
    out = ROOT / "results/cases/form-ladder-261028"
    frame, examples = run(CASE, out)
    print(frame.groupby("level", sort=False).agg(["mean", "std"]).drop(columns="rep").round(3).T.to_string())
    print(system_figure(CASE, frame, examples, out))
