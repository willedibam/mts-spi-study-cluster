"""Delay aliasing: identical zero-lag correlation matrices, different timing.

Helpers for ``notebooks/delay-alias_261020.ipynb`` and its Fig. 1 candidate.
SPIs come from the five-SPI pyspi subset in ``spis/delay_alias.yaml``; nothing
here runs the full catalogue.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from mtsspi.compute import run_pyspi
from mtsspi.generators import (
    ar1_lag_attenuation,
    delay_alias_structure,
    generate_delay_aliased_mts,
)

HERE = Path(__file__).resolve().parent
SPI_CONFIG = HERE / "spis" / "delay_alias.yaml"
SEED_BASE = 20261020

ZERO_LAG = "cov_EmpiricalCovariance"
# Each SPI compared with zero-lag Pearson, and the sign that makes it a similarity.
PAIRED = {
    "xcorr_max_sig-True": 1.0,
    "xpdist_euclidean_tau-30_min_rmse": -1.0,
    "cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-5": 1.0,
    "spearmanr": 1.0,
}
LABELS = {
    "xcorr_max_sig-True": "max cross-correlation",
    "xpdist_euclidean_tau-30_min_rmse": "$-$min cross-distance",
    "cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-5": "coherence magnitude",
    "spearmanr": "Spearman (zero-lag control)",
}
COLORS = {"A": "0.55", "B": "#1b6ca8"}


def style():
    """docs/figure-style.md with native Matplotlib text (no external TeX)."""
    plt.rcParams.update({
        "text.usetex": False, "mathtext.fontset": "cm",
        "font.family": "serif", "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.grid": False, "legend.frameon": False,
        "lines.linewidth": 1.7, "lines.markersize": 2.7,
        "figure.dpi": 180, "savefig.dpi": 600, "savefig.bbox": "tight",
    })


def save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=180 if ext == "png" else 600)
    return out_dir / f"{name}.png"


def profile(mat: np.ndarray) -> np.ndarray:
    """Edge profile: the upper-triangle entries of an M x M matrix."""
    return mat[np.triu_indices(mat.shape[0], 1)]


def z(a: np.ndarray, b: np.ndarray) -> float:
    """SPI-SPI coordinate: Pearson correlation of two edge profiles."""
    return float(np.corrcoef(profile(a), profile(b))[0, 1])


def recording_rng(structure_seed: int, lag_range: float, rep: int) -> np.random.Generator:
    """Independent source noise for every (structure, lag range, replicate)."""
    return np.random.default_rng([SEED_BASE, structure_seed, int(round(10 * lag_range)), rep])


def measure(params: dict, lag_ranges, n_rep: int):
    """pyspi subset on ``n_rep`` independent recordings per lag range.

    Returns ``(frame, mats)``. ``frame`` has one row per recording and SPI with
    z against zero-lag Pearson and the profile mean; ``mats[lag]`` is an
    ``(n_rep, 1 + len(PAIRED), M, M)`` array ordered ``[ZERO_LAG, *PAIRED]``
    and oriented as similarities.
    """
    names = [ZERO_LAG, *PAIRED]
    rows, mats = [], {}
    for lag in lag_ranges:
        stack = []
        for rep in range(n_rep):
            rng = recording_rng(params.get("structure_seed", 0), lag, rep)
            data = generate_delay_aliased_mts(lag_range=lag, rng=rng, **params)
            found = run_pyspi(data, config_path=SPI_CONFIG, normalise=False).matrices
            m = [PAIRED.get(name, 1.0) * found[name] for name in names]
            for name, mat in zip(names, m):
                rows.append(dict(lag_range=lag, rep=rep, spi=name,
                                 z=z(m[0], mat), mean=float(profile(mat).mean())))
            stack.append(np.stack(m))
        mats[lag] = np.stack(stack)
    return pd.DataFrame(rows), mats


def population(params: dict, lag_range: float):
    """Exact zero-lag and intrinsic matrices, delays and eigen-margin at one lag range."""
    _, internals = generate_delay_aliased_mts(
        **{**params, "T": 1}, lag_range=lag_range, rng=0, return_internals=True
    )
    return internals


def zero_lag_summaries(C: np.ndarray) -> dict:
    """Functions of a zero-lag correlation matrix: mean, spectrum, weighted clustering."""
    C = C.copy()
    np.fill_diagonal(C, 1.0)
    ev = np.linalg.eigvalsh(C)[::-1]
    W = np.clip(C, 0.0, None)
    np.fill_diagonal(W, 0.0)
    W3 = np.cbrt(W / W.max())
    M = len(C)
    clustering = np.diag(W3 @ W3 @ W3) / ((M - 1) * (M - 2))  # Onnela et al. 2005
    return {"mean r": float(profile(C).mean()), "lambda_1": float(ev[0]),
            "lambda_2": float(ev[1]), "lambda_M": float(ev[-1]),
            "weighted clustering": float(clustering.mean())}


def separation(a, b) -> dict:
    """AUC (P[b > a]) and two-sided Mann-Whitney p for two replicate samples."""
    a, b = np.asarray(a), np.asarray(b)
    test = mannwhitneyu(b, a, alternative="two-sided")
    return {"A mean": a.mean(), "A sd": a.std(ddof=1), "B mean": b.mean(),
            "B sd": b.std(ddof=1), "AUC": test.statistic / (len(a) * len(b)), "p": test.pvalue}


def generic_feasibility(M: int, loading_low: float, loading_high: float,
                        seeds=range(50)) -> pd.DataFrame:
    """Largest delay range, in AR(1) correlation times, keeping S / R PSD for a target not built from delays.

    The target is the one-factor S0 alone (``delay_alias_structure`` with
    ``max_lag_range=0``); delays are continuous, ``s * linspace(0, 1, M)``.
    With ``R_jk = exp(-|tau_j - tau_k| / ell)`` the answer in units of the
    correlation time ``ell = -1 / ln(a)`` does not depend on ``a``.
    """
    positions = np.linspace(0, 1, M)
    rows = []
    for seed in seeds:
        target, _, _ = delay_alias_structure(M, 0, 0.5, loading_low, loading_high, seed)

        def intrinsic(s):
            return target / ar1_lag_attenuation(s * positions, np.exp(-1.0))

        lo, hi = 0.0, 10.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if np.linalg.eigvalsh(intrinsic(mid))[0] >= 0 else (lo, mid)
        rows.append(dict(seed=seed, range_over_ell=lo, z_population=z(target, intrinsic(lo)),
                         target_mean=float(profile(target).mean())))
    return pd.DataFrame(rows)


def plot_fig1(mats: dict, frame: pd.DataFrame, params: dict, lag_max: int, z_pop: dict):
    """One-panel Fig. 1 candidate: zero-lag vs lag-tolerant edge profiles, z inset."""
    xc = 1 + list(PAIRED).index("xcorr_max_sig-True")
    lags = population(params, lag_max).lags
    dtau = profile(np.abs(lags[:, None] - lags[None, :]).astype(float))
    xa, ya = profile(mats[0][0, 0]), profile(mats[0][0, xc])
    xb, yb = profile(mats[lag_max][0, 0]), profile(mats[lag_max][0, xc])
    za = frame.query("lag_range == 0 and rep == 0 and spi == 'xcorr_max_sig-True'").z.item()
    zb = frame.query("lag_range == @lag_max and rep == 0 and spi == 'xcorr_max_sig-True'").z.item()

    cmap = ListedColormap(plt.get_cmap("viridis")(np.linspace(0.0, 0.85, 256)))
    fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
    lim = (0, 0.76)
    ax.plot(lim, lim, color="0.8", lw=0.8, zorder=0)
    ax.scatter(xa, ya, s=7, color=COLORS["A"], lw=0, zorder=2)
    order = np.argsort(dtau)
    sc = ax.scatter(xb[order], yb[order], s=7, c=dtau[order], cmap=cmap, vmin=0, vmax=lag_max,
                    lw=0, zorder=3)
    ax.set(xlim=lim, ylim=lim, aspect="equal", xticks=np.arange(0, 0.8, 0.2), yticks=np.arange(0, 0.8, 0.2),
           xlabel="zero-lag correlation $r_{jk}$",
           ylabel=r"max. cross-correlation $\mathrm{max}_h\, r_{jk}(h)$")
    handles = [Line2D([], [], ls="", marker="o", ms=3.5, mew=0, color=c) for c in (COLORS["A"], cmap(0.55))]
    ax.legend(handles, [f"synchronous, $z = {za:.2f}$", f"delayed, $z = {zb:.2f}$"],
              loc="upper left", handletextpad=0.1, borderaxespad=0.1)
    cb = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=16, pad=0.02, anchor=(0, 0.15))
    cb.set_label(r"$|\tau_j-\tau_k|$ (samples)", fontsize=8)
    cb.set_ticks(np.arange(0, lag_max + 1, 5))
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)

    ins = ax.inset_axes([0.63, 0.12, 0.35, 0.32])
    sub = frame[frame.spi == "xcorr_max_sig-True"]
    levels = np.array(sorted(sub.lag_range.unique()), dtype=float)
    for lag, grp in sub.groupby("lag_range"):
        ins.scatter(np.full(len(grp), lag), grp.z, s=2.5, lw=0, alpha=0.6,
                    color=COLORS["A"] if lag == 0 else COLORS["B"], zorder=3)
    ins.plot(levels, [z_pop[l] for l in levels], color="k", lw=0.7, ls=(0, (2, 2)), zorder=2)
    ins.set_ylim(0, 1.05)
    ins.set_xlim(-1, lag_max + 1)
    ins.set_xticks([0, lag_max])
    ins.set_yticks([0, 0.5, 1])
    ins.tick_params(labelsize=6.5, length=2, pad=1)
    ins.set_xlabel("delay range (samples)", fontsize=6.5, labelpad=0)
    ins.set_ylabel("$z$", fontsize=7.5, labelpad=0)
    return fig
