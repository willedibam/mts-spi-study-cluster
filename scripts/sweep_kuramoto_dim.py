"""Sweep Kuramoto params and report n_pcs_95.

Identifies which (connectivity, output, K, omega_std, eta) corner reliably
gives n_pcs_95 >= 3 for the proof.yaml setup.
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
DT = 0.002
ALPHA = 0.95
N_PCS_MIN = 3
SEEDS = [11, 23, 47]

CONNECTIVITY = ["bidirectional-list", "all-to-all", "ring-unidirectional"]
OUTPUT = ["sin", "phase"]
K_VALUES = [-8.0, -4.0, -1.0, 1.0, 4.0, 8.0]
OMEGA_STD = [0.01, 0.1, 0.5]
ETA = [0.0, 0.1]


def run_one(conn, output, K, omega_std, eta, seed):
    rng = np.random.default_rng(seed)
    data = generate_kuramoto(
        M=M, T=T, dt=DT, K=K,
        k_ring=1,
        omega_mean=2 * np.pi * 0.1,
        omega_std=omega_std,
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
    combos = list(itertools.product(CONNECTIVITY, OUTPUT, K_VALUES, OMEGA_STD, ETA))
    print(f"Running {len(combos)} combos x {len(SEEDS)} seeds = {len(combos)*len(SEEDS)} sims...", flush=True)
    for i, (conn, output, K, om, eta) in enumerate(combos):
        npcs = [run_one(conn, output, K, om, eta, s) for s in SEEDS]
        rows.append({
            "connectivity": conn,
            "output": output,
            "K": K,
            "omega_std": om,
            "eta": eta,
            "n_pcs_median": int(np.median(npcs)),
            "n_pcs_min": int(np.min(npcs)),
            "n_pcs_max": int(np.max(npcs)),
            "pass_rate": float(np.mean([n >= N_PCS_MIN for n in npcs])),
        })
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(combos)}", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "scripts" / "sweep_kuramoto_dim_results.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}\n")

    print("=" * 80)
    print(f"Pass criterion: all 3 seeds >= n_pcs_95 >= {N_PCS_MIN}")
    print("=" * 80)

    passing = df[df["pass_rate"] == 1.0].copy()
    print(f"\n{len(passing)}/{len(df)} combos pass on all seeds.\n")

    print("\n--- Marginals: pass-rate per connectivity x output ---")
    pivot = df.groupby(["connectivity", "output"])["pass_rate"].mean().unstack()
    print(pivot.to_string())

    print("\n--- Marginals: median n_pcs per connectivity x output ---")
    pivot2 = df.groupby(["connectivity", "output"])["n_pcs_median"].median().unstack()
    print(pivot2.to_string())

    print("\n--- Best 15 corners (highest median n_pcs, ties by pass_rate) ---")
    top = df.sort_values(["n_pcs_median", "pass_rate"], ascending=[False, False]).head(15)
    print(top.to_string(index=False))

    print("\n--- All passing corners (full pass) ---")
    if len(passing):
        print(passing.sort_values("n_pcs_median", ascending=False).to_string(index=False))
    else:
        print("(none)")


if __name__ == "__main__":
    main()
