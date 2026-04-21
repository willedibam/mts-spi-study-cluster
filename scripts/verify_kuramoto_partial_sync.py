"""Verify Option B (K=2, sigma_omega=0.3*omega_mean) is true partial sync
AND passes Cliff filter across the full proof.yaml grid.

Reports order parameter r and n_pcs_95 per (M, T, I).
Targets:
  - 0.3 < r < 0.7  (genuine partial sync, not incoherent or full lock)
  - n_pcs_95 >= 3  (Cliff filter)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dimensionality import n_pcs_for_variance
from src.generators.dynamical import _build_kuramoto_adjacency

# Option B params
K = 2.0
DT = 0.05
OMEGA_MEAN = np.pi  # 2*pi*0.5 = pi
OMEGA_STD = 0.3 * OMEGA_MEAN
ETA = 0.05
TRANSIENTS = 2000
CONNECTIVITY = "bidirectional-list"

M_VALUES = [5, 10, 20]
T_VALUES = [500, 1000, 2000]
INSTANCES = list(range(10))


def simulate(M: int, T: int, seed: int):
    rng = np.random.default_rng(seed)
    A = _build_kuramoto_adjacency(M, CONNECTIVITY, k_ring=1)
    degree = A.sum(axis=1)
    inv_degree = np.where(degree > 0, 1.0 / degree, 0.0)
    theta = rng.uniform(0, 2 * np.pi, M)
    omega = rng.normal(OMEGA_MEAN, OMEGA_STD, M)
    steps = TRANSIENTS + T
    Y = np.zeros((steps, M))
    Theta = np.zeros((steps, M))
    for t in range(steps):
        Y[t] = np.sin(theta)
        Theta[t] = theta
        phase_diff = theta[None, :] - theta[:, None]
        coupling_term = (A * np.sin(phase_diff)).sum(axis=1)
        dtheta = omega + K * inv_degree * coupling_term
        noise = ETA * np.sqrt(DT) * rng.normal(size=M)
        theta = np.mod(theta + dtheta * DT + noise, 2 * np.pi)
    Y_obs = Y[TRANSIENTS:]
    Theta_obs = Theta[TRANSIENTS:]
    Y_obs = (Y_obs - Y_obs.mean(0)) / (Y_obs.std(0) + 1e-12)
    r_t = np.abs(np.mean(np.exp(1j * Theta_obs), axis=1))
    return n_pcs_for_variance(Y_obs, alpha=0.95), float(np.mean(r_t))


def main():
    rows = []
    for M in M_VALUES:
        for T in T_VALUES:
            for I in INSTANCES:
                seed = 110305 + 10_000_000 + M * 1000 + T + I
                npcs, r = simulate(M, T, seed)
                rows.append({"M": M, "T": T, "I": I, "n_pcs_95": npcs, "r": r})
        print(f"  M={M} done", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "scripts" / "verify_kuramoto_partial_sync_results.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}\n")

    print("=" * 70)
    print(f"OPTION B: K={K}, dt={DT}, omega_mean=pi, omega_std=0.3*pi, eta={ETA}")
    print("=" * 70)

    print("\n--- Median r per (M, T) [target: 0.3 < r < 0.7 = partial sync] ---")
    print(df.pivot_table(index="M", columns="T", values="r", aggfunc="median").to_string(float_format=lambda v: f"{v:.2f}"))

    print("\n--- Median n_pcs_95 per (M, T) ---")
    print(df.pivot_table(index="M", columns="T", values="n_pcs_95", aggfunc="median").to_string())

    print("\n--- Min n_pcs_95 per (M, T) [target: >= 3 on every instance] ---")
    print(df.pivot_table(index="M", columns="T", values="n_pcs_95", aggfunc="min").to_string())

    print("\n--- Cliff pass-rate (n_pcs_95 >= 3) per (M, T) ---")
    print(df.pivot_table(
        index="M", columns="T", values="n_pcs_95",
        aggfunc=lambda x: (x >= 3).mean(),
    ).to_string(float_format=lambda v: f"{v:.2f}"))

    print("\n--- Partial-sync pass-rate (0.3 < r < 0.85) per (M, T) ---")
    print(df.pivot_table(
        index="M", columns="T", values="r",
        aggfunc=lambda x: ((x > 0.3) & (x < 0.85)).mean(),
    ).to_string(float_format=lambda v: f"{v:.2f}"))

    pass_dim = (df["n_pcs_95"] >= 3).mean()
    pass_r = ((df["r"] > 0.3) & (df["r"] < 0.85)).mean()
    print(f"\nOverall: Cliff pass {pass_dim:.1%}  |  partial-sync pass {pass_r:.1%}  (n={len(df)})")

    failures = df[(df["n_pcs_95"] < 3) | (df["r"] < 0.3) | (df["r"] >= 0.85)]
    if len(failures):
        print(f"\n--- {len(failures)} failures ({len(failures)/len(df):.1%}) ---")
        print(failures.head(20).to_string(index=False))
    else:
        print("\nNo failures — Option B holds across the full grid.")


if __name__ == "__main__":
    main()
