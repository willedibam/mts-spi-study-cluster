"""Sweep Kuramoto v2: fix omega_mean so we observe enough cycles.

v1 finding: at omega_mean=2π*0.1, dt=0.002, T=1000, oscillators complete
~0.2 cycles total, so PCA on the observation window is effectively
PCA on near-DC signals. This sweep boosts omega_mean and focuses on
near-critical positive K with moderate detuning.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dimensionality import n_pcs_for_variance
from src.generators.dynamical import generate_kuramoto

M = 10
T = 1000
TRANSIENTS = 2000
DT = 0.05  # 25x larger than v1, so omega*T*dt actually rotates
ALPHA = 0.95
N_PCS_MIN = 3
SEEDS = [11, 23, 47, 71, 103]

CONNECTIVITY = ["bidirectional-list", "all-to-all", "ring-unidirectional"]
OUTPUT = ["sin", "phase"]
K_VALUES = [0.1, 0.3, 0.5, 1.0, 2.0, 4.0]
OMEGA_MEAN = [2 * np.pi * 0.1, 2 * np.pi * 0.5]
OMEGA_STD_FRAC = [0.1, 0.3, 0.6]  # as fraction of omega_mean
ETA = [0.0, 0.05]


def run_one(conn, output, K, om_mean, om_std, eta, seed):
    rng = np.random.default_rng(seed)
    data = generate_kuramoto(
        M=M, T=T, dt=DT, K=K,
        k_ring=1,
        omega_mean=om_mean,
        omega_std=om_std,
        eta=eta,
        transients=TRANSIENTS,
        output=output,
        connectivity=conn,
        rng=rng,
        zscore=True,
    )
    return n_pcs_for_variance(data, alpha=ALPHA)


def main():
    rows = []
    combos = list(itertools.product(
        CONNECTIVITY, OUTPUT, K_VALUES, OMEGA_MEAN, OMEGA_STD_FRAC, ETA
    ))
    print(f"M={M} T={T} dt={DT} (cycles/window = {OMEGA_MEAN[0]/(2*np.pi)*T*DT:.1f} or {OMEGA_MEAN[1]/(2*np.pi)*T*DT:.1f})")
    print(f"Running {len(combos)} combos x {len(SEEDS)} seeds = {len(combos)*len(SEEDS)} sims...", flush=True)
    for i, (conn, output, K, om_mean, om_std_frac, eta) in enumerate(combos):
        om_std = om_std_frac * om_mean
        npcs = [run_one(conn, output, K, om_mean, om_std, eta, s) for s in SEEDS]
        rows.append({
            "connectivity": conn,
            "output": output,
            "K": K,
            "omega_mean_hz": om_mean / (2 * np.pi),
            "omega_std_frac": om_std_frac,
            "eta": eta,
            "n_pcs_median": int(np.median(npcs)),
            "n_pcs_min": int(np.min(npcs)),
            "n_pcs_max": int(np.max(npcs)),
            "pass_rate": float(np.mean([n >= N_PCS_MIN for n in npcs])),
        })
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(combos)}", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "scripts" / "sweep_kuramoto_dim_v2_results.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}\n")

    print("=" * 90)
    print(f"Pass criterion: all {len(SEEDS)} seeds >= n_pcs_95 >= {N_PCS_MIN}")
    print("=" * 90)

    passing = df[df["pass_rate"] == 1.0].copy()
    print(f"\n{len(passing)}/{len(df)} combos pass on all seeds.\n")

    print("--- Pass-rate (mean) per connectivity x output ---")
    pivot = df.groupby(["connectivity", "output"])["pass_rate"].mean().unstack()
    print(pivot.to_string())

    print("\n--- Median n_pcs per connectivity x output ---")
    pivot2 = df.groupby(["connectivity", "output"])["n_pcs_median"].median().unstack()
    print(pivot2.to_string())

    print("\n--- Pass-rate per connectivity x output x omega_mean_hz ---")
    pivot3 = df.groupby(["connectivity", "output", "omega_mean_hz"])["pass_rate"].mean().unstack()
    print(pivot3.to_string())

    print("\n--- All passing corners (full pass on all seeds) ---")
    if len(passing):
        passing = passing.sort_values(
            ["connectivity", "output", "K", "omega_std_frac"]
        )
        print(passing.to_string(index=False))
    else:
        print("(none)")


if __name__ == "__main__":
    main()
