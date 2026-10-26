"""How strength heterogeneity across channel pairs moves the r–ρ–MI readouts (2026-10-24).

The archived corpus (``configs/260924_beta.yaml``) fixes the per-channel noise
multiplier spread at E[kappa] = 5 with a 0.01 floor, so every non-quadratic pair
has r between about 0.88 and 1. This sweep reruns the same three constructions
(L, L+M, L+M+Q; beta = pi; M = 32, T = 2,000) over E[kappa], with Pearson,
Spearman and ordinary KSG MI from ``spis/r_rho_mi_261024.yaml``. It is a new,
separate computation, not an extension of the archive.

    python -m research.cases.r_rho_mi.strength_spread_261024 --out results/cases/r-rho-mi-261024
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mtsspi.compute import run_pyspi
from mtsspi.generators import generate_filter_roll_mts

HERE = Path(__file__).resolve().parent
SPI_CONFIG = HERE / "spis" / "r_rho_mi_261024.yaml"
R, RHO, MI = "cov_EmpiricalCovariance", "spearmanr", "mi_kraskov_NN-4"
CONSTRUCTIONS = {"L": (32, 0, 0), "L+M": (16, 16, 0), "L+M+Q": (12, 12, 8)}
SCALES = (1, 2, 5, 10, 20, 50)
BASE = dict(M=32, T=2000, beta=np.pi, noise_std=0.01, noise_std_variable=True,
            ar1_a=0.8, ar1_noise_std=1.0, zscore=True)


def upper(m: np.ndarray) -> np.ndarray:
    return m[np.triu_indices(m.shape[0], 1)]


def z(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def linfoot(mi: np.ndarray) -> np.ndarray:
    return np.sqrt(1.0 - np.exp(-2.0 * np.clip(mi, 0.0, None)))


def one(construction: str, scale: float, rep: int) -> dict:
    nl, nm, nq = CONSTRUCTIONS[construction]
    rng = np.random.default_rng([261024, list(CONSTRUCTIONS).index(construction), SCALES.index(scale), rep])
    x, it = generate_filter_roll_mts(n_linear=nl, n_monotonic=nm, n_nonmonotonic=nq, noise_std_scale=scale,
                                     rng=rng, return_internals=True, **BASE)
    m = run_pyspi(x, config_path=SPI_CONFIG, normalise=False).matrices
    r, rho, mi = upper(m[R]), upper(m[RHO]), upper(m[MI])
    types = np.asarray(it.types)
    iu = np.triu_indices(len(types), 1)
    a, b = types[iu[0]], types[iu[1]]
    ll = (a == "linear") & (b == "linear")
    lm = ((a == "linear") & (b == "monotonic")) | ((a == "monotonic") & (b == "linear"))
    return {"construction": construction, "scale": scale, "rep": rep,
            "z_r_rho": z(r, rho), "z_rho_mi": z(rho, mi), "z_r_mi": z(r, mi), "z_rho_mil": z(rho, linfoot(mi)),
            "z_r_rho_LM_only": z(r[lm], rho[lm]) if lm.sum() > 2 else np.nan,
            "r_LL_sd": float(np.std(r[ll])), "r_LL_min": float(np.min(r[ll])), "r_LL_median": float(np.median(r[ll])),
            "noise_sd_max": float(np.max(it.noise_stds))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for construction in CONSTRUCTIONS:
        for scale in SCALES:
            for rep in range(args.reps):
                rows.append(one(construction, scale, rep))
            print(construction, scale, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / "strength_spread.csv", index=False)
    import pyspi
    (args.out / "strength_spread.json").write_text(json.dumps(
        {"pyspi": getattr(pyspi, "COMPUTATION_VERSION", "unknown"), "spi_config": str(SPI_CONFIG.relative_to(HERE.parents[2])),
         "base": {k: float(v) if isinstance(v, float) else v for k, v in BASE.items()},
         "constructions": CONSTRUCTIONS, "scales": SCALES, "reps": args.reps}, indent=1) + "\n")


if __name__ == "__main__":
    main()
