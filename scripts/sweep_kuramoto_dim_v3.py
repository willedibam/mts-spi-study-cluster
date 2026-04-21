"""Sweep Kuramoto v3: maximise n_pcs_95 around current K=-1 baseline.

Connectivity, output, k_ring, transients all held at proof.yaml values.
We sweep K (both signs), dt, omega_mean, omega_std (as frac of mean), eta.
Baseline (current proof.yaml): K=-1, dt=0.05, omega_mean=pi, omega_std_frac=0.6, eta=0.05.

M=10 (most representative single channel count), T=1000.
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
SEEDS = [11, 23, 47, 71, 103]
ALPHA = 0.95

CONNECTIVITY = "bidirectional-list"
OUTPUT = "sin"

K_VALUES = [-8.0, -4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0, 8.0]
DT_VALUES = [0.02, 0.05, 0.1]
OMEGA_MEAN_HZ = [0.25, 0.5, 1.0]
OMEGA_STD_FRAC = [0.3, 0.6, 0.9]
ETA_VALUES = [0.0, 0.05, 0.1]

BASELINE = dict(K=-1.0, dt=0.05, omega_mean_hz=0.5, omega_std_frac=0.6, eta=0.05)


def run_one(K, dt, om_mean, om_std, eta, seed):
    rng = np.random.default_rng(seed)
    data = generate_kuramoto(
        M=M, T=T, dt=dt, K=K,
        k_ring=1,
        omega_mean=om_mean,
        omega_std=om_std,
        eta=eta,
        transients=TRANSIENTS,
        output=OUTPUT,
        connectivity=CONNECTIVITY,
        rng=rng,
        zscore=True,
    )
    return n_pcs_for_variance(data, alpha=ALPHA)


def main():
    rows = []
    combos = list(itertools.product(
        K_VALUES, DT_VALUES, OMEGA_MEAN_HZ, OMEGA_STD_FRAC, ETA_VALUES
    ))
    print(f"M={M} T={T}, conn={CONNECTIVITY}, output={OUTPUT}")
    print(f"Sweeping {len(combos)} corners x {len(SEEDS)} seeds = {len(combos)*len(SEEDS)} sims", flush=True)

    for i, (K, dt, om_mean_hz, om_std_frac, eta) in enumerate(combos):
        om_mean = 2 * np.pi * om_mean_hz
        om_std = om_std_frac * om_mean
        npcs = [run_one(K, dt, om_mean, om_std, eta, s) for s in SEEDS]
        rows.append({
            "K": K,
            "dt": dt,
            "omega_mean_hz": om_mean_hz,
            "omega_std_frac": om_std_frac,
            "eta": eta,
            "n_pcs_median": float(np.median(npcs)),
            "n_pcs_mean": float(np.mean(npcs)),
            "n_pcs_min": int(np.min(npcs)),
            "n_pcs_max": int(np.max(npcs)),
            "pass_rate": float(np.mean([n >= 3 for n in npcs])),
        })
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(combos)}", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "scripts" / "sweep_kuramoto_dim_v3_results.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}\n")

    print("=" * 90)
    print("BASELINE (current proof.yaml: K=-1, dt=0.05, omega_mean=pi, omega_std_frac=0.6, eta=0.05)")
    print("=" * 90)
    base = df[
        (df["K"] == BASELINE["K"])
        & (df["dt"] == BASELINE["dt"])
        & (df["omega_mean_hz"] == BASELINE["omega_mean_hz"])
        & (df["omega_std_frac"] == BASELINE["omega_std_frac"])
        & (df["eta"] == BASELINE["eta"])
    ]
    print(base.to_string(index=False))
    base_mean = base["n_pcs_mean"].iloc[0] if len(base) else None
    print(f"\nBaseline n_pcs_mean = {base_mean}")

    print("\n" + "=" * 90)
    print("TOP 25 corners by n_pcs_mean (full pass on all 5 seeds, M=10)")
    print("=" * 90)
    full_pass = df[df["pass_rate"] == 1.0].sort_values(
        ["n_pcs_mean", "n_pcs_min"], ascending=False
    )
    print(full_pass.head(25).to_string(index=False))

    print("\n" + "=" * 90)
    print("Median n_pcs_mean per K (margin over baseline K=-1)")
    print("=" * 90)
    by_K = df.groupby("K")["n_pcs_mean"].agg(["median", "mean", "max"])
    print(by_K.to_string())

    print("\n" + "=" * 90)
    print("Best corner per K (by n_pcs_mean, full-pass only)")
    print("=" * 90)
    best_per_K = (
        full_pass.sort_values("n_pcs_mean", ascending=False)
        .groupby("K")
        .head(1)
        .sort_values("K")
    )
    print(best_per_K.to_string(index=False))


if __name__ == "__main__":
    main()
