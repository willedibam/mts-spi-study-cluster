"""Find the Kuramoto setting that is maximally distinguishable from
gaussian-noise, cauchy-noise, var-ring-sym, and sti-ii.

Proxy feature space (physical, not full pyspi):
  - mean |linear correlation| off-diag
  - mean PLV (Hilbert)
  - mean lag-1 autocorr
  - mean kurtosis
  - mean spectral peak height (after Welch)
  - std of dominant frequency across channels

For each candidate Kuramoto setting, compute centroid distance in
z-scored feature space to each reference class. Report the candidate
maximising the MINIMUM distance to any reference class (worst-case
discriminability).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generators.dynamical import _build_kuramoto_adjacency, generate_cml_logistic
from src.generators.linear import (
    generate_cauchy_noise,
    generate_gaussian_noise,
    generate_varma,
)

M = 10
T = 1000
N_SEEDS = 20
TRANSIENTS = 2000


def features(X: np.ndarray) -> np.ndarray:
    T_, M_ = X.shape
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-12)
    C = np.corrcoef(Xn.T)
    iu = np.triu_indices(M_, k=1)
    mean_abs_corr = float(np.mean(np.abs(C[iu])))
    # PLV via Hilbert
    analytic = signal.hilbert(Xn, axis=0)
    phases = np.angle(analytic)
    plv = []
    for i in range(M_):
        for j in range(i+1, M_):
            dphi = phases[:, i] - phases[:, j]
            plv.append(np.abs(np.mean(np.exp(1j * dphi))))
    mean_plv = float(np.mean(plv))
    # lag-1 autocorr per channel
    lag1 = float(np.mean([np.corrcoef(Xn[:-1, k], Xn[1:, k])[0, 1] for k in range(M_)]))
    # kurtosis
    kurt = float(np.mean(stats.kurtosis(X, axis=0, fisher=False)))
    # spectral peak height + dominant freq spread
    f, Pxx = signal.welch(Xn, axis=0, nperseg=min(256, T_))
    Pxx_norm = Pxx / (Pxx.sum(0, keepdims=True) + 1e-12)
    peak_height = float(np.mean(Pxx_norm.max(0)))
    dom_f = f[Pxx_norm.argmax(0)]
    freq_std = float(np.std(dom_f))
    return np.array([mean_abs_corr, mean_plv, lag1, kurt, peak_height, freq_std])


def gen_kuramoto(K, sig_frac, conn, seed, om_mean=np.pi, dt=0.05, eta=0.05):
    rng = np.random.default_rng(seed)
    A = _build_kuramoto_adjacency(M, conn, k_ring=1)
    inv_d = np.where(A.sum(1) > 0, 1.0 / A.sum(1), 0.0)
    theta = rng.uniform(0, 2 * np.pi, M)
    omega = rng.normal(om_mean, sig_frac * om_mean, M)
    Y = np.zeros((TRANSIENTS + T, M))
    for t in range(TRANSIENTS + T):
        Y[t] = np.sin(theta)
        ct = (A * np.sin(theta[None, :] - theta[:, None])).sum(1)
        theta = np.mod(
            theta + (omega + K * inv_d * ct) * dt + eta * np.sqrt(dt) * rng.normal(size=M),
            2 * np.pi,
        )
    return Y[TRANSIENTS:]


def gen_class(name: str, seed: int):
    rng = np.random.default_rng(seed)
    if name == "gaussian":
        return generate_gaussian_noise(M=M, T=T, rng=rng, zscore=True)
    if name == "cauchy":
        return generate_cauchy_noise(M=M, T=T, rng=rng, zscore=False)
    if name == "var-ring":
        return generate_varma(
            M=M, T=T, phi=0.8, coupling=0.4, ma_phi=0.0, ma_coupling=0.0,
            noise_std=0.1, topology="ring-symmetric", transients=100,
            rng=rng, zscore=True,
        )
    if name == "sti-ii":
        return generate_cml_logistic(
            M=M, T=T, alpha=1.75, eps=0.3, transients=1000,
            sample_every=1, rng=rng, zscore=True,
        )
    raise ValueError(name)


def main():
    REF_CLASSES = ["gaussian", "cauchy", "var-ring", "sti-ii"]
    CANDIDATES = [
        # (label, K, sig_frac, conn, om_mean_mult)
        ("current K=-1 sig=0.6 bidir",   -1.0, 0.6,  "bidirectional-list", 1.0),
        ("K=-1 sig=0.3 bidir",           -1.0, 0.3,  "bidirectional-list", 1.0),
        ("K=-2 sig=0.3 bidir (splay)",   -2.0, 0.3,  "bidirectional-list", 1.0),
        ("K=-4 sig=0.3 bidir (splay)",   -4.0, 0.3,  "bidirectional-list", 1.0),
        ("K=2 sig=0.3 bidir (B)",         2.0, 0.3,  "bidirectional-list", 1.0),
        ("K=2 sig=0.5 bidir",             2.0, 0.5,  "bidirectional-list", 1.0),
        ("K=1 sig=0.4 all-to-all",        1.0, 0.4,  "all-to-all",          1.0),
        ("K=1.5 sig=0.4 all-to-all",      1.5, 0.4,  "all-to-all",          1.0),
        ("K=-2 sig=0.3 all-to-all",      -2.0, 0.3,  "all-to-all",          1.0),
        ("K=-4 sig=0.3 all-to-all",      -4.0, 0.3,  "all-to-all",          1.0),
    ]

    print(f"M={M}, T={T}, seeds={N_SEEDS}; computing features...", flush=True)
    rows = []
    for cls in REF_CLASSES:
        for seed in range(N_SEEDS):
            X = gen_class(cls, seed)
            f = features(X)
            rows.append({"class": cls, "seed": seed, **{f"f{i}": v for i, v in enumerate(f)}})
        print(f"  ref {cls} done", flush=True)

    for label, K, sig, conn, ommult in CANDIDATES:
        for seed in range(N_SEEDS):
            X = gen_kuramoto(K, sig, conn, seed, om_mean=ommult * np.pi)
            f = features(X)
            rows.append({"class": label, "seed": seed, **{f"f{i}": v for i, v in enumerate(f)}})
        print(f"  cand {label} done", flush=True)

    df = pd.DataFrame(rows)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    feat_names = ["mean|corr|", "PLV", "lag1", "kurt", "specPeak", "freqStd"]

    # z-score globally
    Z = (df[feat_cols] - df[feat_cols].mean()) / df[feat_cols].std()
    df_z = pd.concat([df[["class"]], Z], axis=1)

    centroids = df_z.groupby("class")[feat_cols].mean()

    print("\n--- Class centroids (z-scored features) ---")
    cent_show = centroids.copy()
    cent_show.columns = feat_names
    print(cent_show.to_string(float_format=lambda v: f"{v:+.2f}"))

    print("\n--- Distance from each Kuramoto candidate to each reference class ---")
    cand_labels = [c[0] for c in CANDIDATES]
    dist_table = pd.DataFrame(index=cand_labels, columns=REF_CLASSES + ["MIN", "MEAN"])
    for cand in cand_labels:
        cv = centroids.loc[cand].values
        for ref in REF_CLASSES:
            rv = centroids.loc[ref].values
            dist_table.loc[cand, ref] = float(np.linalg.norm(cv - rv))
        dists = [dist_table.loc[cand, r] for r in REF_CLASSES]
        dist_table.loc[cand, "MIN"] = float(min(dists))
        dist_table.loc[cand, "MEAN"] = float(np.mean(dists))
    dist_table = dist_table.astype(float).sort_values("MIN", ascending=False)
    print(dist_table.to_string(float_format=lambda v: f"{v:.2f}"))

    print("\n--- Best candidate by worst-case (MIN) discriminability ---")
    best = dist_table.index[0]
    print(f"  {best}")
    print(f"  min-dist={dist_table.loc[best,'MIN']:.2f}  mean-dist={dist_table.loc[best,'MEAN']:.2f}")

    print("\n--- Raw centroid features per class (un-z'd) for sanity ---")
    raw_cent = df.groupby("class")[feat_cols].mean()
    raw_cent.columns = feat_names
    print(raw_cent.loc[REF_CLASSES + cand_labels].to_string(float_format=lambda v: f"{v:.2f}"))

    out = ROOT / "scripts" / "kuramoto_distinguishability_results.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
