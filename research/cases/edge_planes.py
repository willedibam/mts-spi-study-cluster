"""Fig. 1-style SPI–SPI edge planes for the r–ρ–MI and DTW cases (2026-10-20 viewers).

Each figure has two panels, one per rung of a nested-invariance ladder. A panel
plots one recording's two edge profiles against each other (restrictive SPI on
x, tolerant SPI on y) with the identity line, and an inset gives z for every
recording of every condition. Inputs are the archived 2026-09-24 pyspi banks
(computation r7), read through ``refresh_case_figures.analyze``; nothing here
runs pyspi. Style follows ``research/cases/delay_alias/delay_alias.py`` on
branch ``claude/quizzical-sutherland-9dca5f``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from research.cases.refresh_case_figures import DTW, ED, MI, R, RHO, XD

GREY, BLUE, ORANGE = "0.55", "#1b6ca8", "#D55E00"
LAG_CMAP = ListedColormap(plt.get_cmap("viridis")(np.linspace(0.0, 0.85, 256)))

# r–ρ–MI constructions: label and colour, in nesting order.
CONSTRUCTIONS = {"iter1_L": ("L", GREY), "iter2_LM": ("+M", BLUE), "iter3_LMNM": ("+Q", ORANGE)}
# DTW conditions (max_lag, p_step); the no-lag control is drawn apart from the lagged sweep.
# Legends sit above each panel, clear of the data and the inset.
LEGEND = dict(loc="lower left", bbox_to_anchor=(-0.02, 1.0), handletextpad=0.1, borderaxespad=0.1)
CONDITIONS = [(0, 0.0), (5, 0.0), (5, 0.1), (5, 0.3), (5, 0.5), (5, 0.7), (5, 0.9)]


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
    return float(np.corrcoef(a, b)[0, 1])


def linfoot(mi: np.ndarray) -> np.ndarray:
    """Linfoot (1957) informational correlation: equals |r| for a bivariate Gaussian."""
    return np.sqrt(1.0 - np.exp(-2.0 * np.clip(mi, 0.0, None)))


def bank(path) -> tuple[dict, dict]:
    path = Path(path)
    with np.load(path / "spi_mpis.npz") as b:
        return {k: profile(b[k]) for k in b.files}, json.loads((path / "meta.json").read_text())


# --------------------------------------------------------------------------- r–ρ–MI

def pair_relation(types) -> np.ndarray:
    """Generating relation of each unordered pair: same family is linear; L–M monotone; any Q-cross non-monotone."""
    iu = np.triu_indices(len(types), 1)
    a, b = np.asarray(types)[iu[0]], np.asarray(types)[iu[1]]
    rel = np.where(a == b, "linear", np.where((a == "non-monotonic") | (b == "non-monotonic"),
                                              "non-monotone", "monotone"))
    return rel


def rmi_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-recording z for the three SPI pairs, with MI raw (archived coordinate) and on the Linfoot scale."""
    rows = []
    for _, rec in frame.iterrows():
        p, meta = bank(rec["path"])
        r, rho, mi, mil = p[R], p[RHO], p[MI], linfoot(p[MI])
        rel = pair_relation(meta["generator"]["types"])
        lm = rel == "monotone"
        rows.append({"class": rec["class"], "beta_tag": rec["beta_tag"], "instance": rec["instance"],
                     "z_r_rho": z(r, rho), "z_rho_mi": z(rho, mi), "z_r_mi": z(r, mi),
                     "z_rho_mil": z(rho, mil), "z_r_mil": z(r, mil),
                     "z_r_rho_LM_only": z(r[lm], rho[lm]) if lm.sum() > 2 else np.nan})
    return pd.DataFrame(rows)


def _plane(ax, lim, ticks, xlabel, ylabel):
    ax.plot(lim, lim, color="0.8", lw=0.8, zorder=0)
    ax.set(xlim=lim, ylim=lim, aspect="equal", xticks=ticks, yticks=ticks, xlabel=xlabel, ylabel=ylabel)


def _inset(ax, rect, groups, xticks, xticklabels, xlabel, ylim=(0, 1.05)):
    """z for every recording (dots) and the replicate mean (dashed); ``groups`` = [(x, values, colour, joined)]."""
    ins = ax.inset_axes(rect)
    for x, values, color, _ in groups:
        ins.scatter(np.full(len(values), x), values, s=2.5, lw=0, alpha=0.6, color=color, zorder=3)
    joined = [(x, np.mean(v)) for x, v, _, j in groups if j]
    ins.plot(*zip(*joined), color="k", lw=0.7, ls=(0, (2, 2)), zorder=2)
    for x, v, _, j in groups:  # a detached control gets its own short mean bar
        if not j:
            ins.plot([x - 0.08, x + 0.08], [np.mean(v)] * 2, color="k", lw=0.7, ls=(0, (2, 2)), zorder=2)
    ins.set_ylim(*ylim)
    ins.set_yticks([0, 0.5, 1])
    ins.set_xticks(xticks)
    ins.set_xticklabels(xticklabels)
    ins.tick_params(labelsize=6.5, length=2, pad=1)
    ins.set_xlabel(xlabel, fontsize=6.5, labelpad=0)
    ins.set_ylabel("$z$", fontsize=7.5, labelpad=0)
    return ins


def plot_rmi(frame: pd.DataFrame, table: pd.DataFrame, beta_tag: str = "beta-pi", instance: int = 0):
    """(a) r vs ρ: L vs L+M. (b) ρ vs Linfoot MI: L+M vs L+M+Q. Insets: z over all three constructions."""
    sel = frame[((frame.beta_tag == beta_tag) | (frame["class"] == "iter1_L"))]
    tab = table[((table.beta_tag == beta_tag) | (table["class"] == "iter1_L"))]
    rec = {c: bank(sel[(sel["class"] == c) & (sel.instance == instance)].iloc[0]["path"]) for c in CONSTRUCTIONS}
    zrow = {c: tab[(tab["class"] == c) & (tab.instance == instance)].iloc[0] for c in CONSTRUCTIONS}

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2), constrained_layout=True)
    handles = lambda colors: [Line2D([], [], ls="", marker="o", ms=3.5, mew=0, color=c) for c in colors]

    ax = axes[0]
    lim = (0.885, 1.004)
    _plane(ax, lim, [0.9, 0.95, 1.0], "Pearson $r_{jk}$", r"Spearman $\rho_{jk}$")
    for c in ("iter1_L", "iter2_LM"):
        p, _ = rec[c]
        ax.scatter(p[R], p[RHO], s=7, lw=0, color=CONSTRUCTIONS[c][1], zorder=2 if c == "iter1_L" else 3)
    ax.legend(handles([GREY, BLUE]), [f"linear, $z = {zrow['iter1_L'].z_r_rho:.2f}$",
                                      f"+ sigmoid, $z = {zrow['iter2_LM'].z_r_rho:.2f}$"],
              **LEGEND)
    p, meta = rec["iter2_LM"]
    types = np.asarray(meta["generator"]["types"])
    iu = np.triu_indices(len(types), 1)
    lm = types[iu[0]] != types[iu[1]]
    mm = (types[iu[0]] == "monotonic") & (types[iu[1]] == "monotonic")
    top = np.flatnonzero(lm)[np.argmax(p[RHO][lm])]
    below = mm & (p[RHO] < p[R] - 0.01)  # M–M pairs visibly under the identity
    ax.annotate("L–M", (p[R][top], p[RHO][top]), xytext=(-8, -2), textcoords="offset points",
                fontsize=7.5, color=BLUE, ha="right", va="top")
    ax.annotate("M–M", (np.max(p[R][below]), np.median(p[RHO][below])), xytext=(6, 0),
                textcoords="offset points", fontsize=7.5, color=BLUE, ha="left", va="center")
    _inset(ax, [0.6, 0.1, 0.38, 0.32],
           [(k, tab[tab["class"] == c].z_r_rho.to_numpy(), col, True)
            for k, (c, (_, col)) in enumerate(CONSTRUCTIONS.items())],
           [0, 1, 2], [lab for lab, _ in CONSTRUCTIONS.values()], "channel families")

    ax = axes[1]
    lim = (-0.04, 1.02)
    _plane(ax, lim, [0, 0.5, 1], r"Spearman $\rho_{jk}$", r"MI as correlation, $\sqrt{1-e^{-2I_{jk}}}$")
    for c in ("iter2_LM", "iter3_LMNM"):
        p, _ = rec[c]
        ax.scatter(p[RHO], linfoot(p[MI]), s=7, lw=0, color=CONSTRUCTIONS[c][1],
                   zorder=3 if c == "iter2_LM" else 2)
    ax.legend(handles([BLUE, ORANGE]), [f"+ sigmoid, $z = {zrow['iter2_LM'].z_rho_mil:.2f}$",
                                        f"+ quadratic, $z = {zrow['iter3_LMNM'].z_rho_mil:.2f}$"],
              **LEGEND)
    p, meta = rec["iter3_LMNM"]
    q = pair_relation(meta["generator"]["types"]) == "non-monotone"
    ax.annotate("L–Q, M–Q pairs", (np.median(p[RHO][q]), np.median(linfoot(p[MI][q]))), xytext=(18, -2),
                textcoords="offset points", fontsize=7.5, color=ORANGE, va="center")
    _inset(ax, [0.6, 0.1, 0.38, 0.32],
           [(k, tab[tab["class"] == c].z_rho_mil.to_numpy(), col, True)
            for k, (c, (_, col)) in enumerate(CONSTRUCTIONS.items())],
           [0, 1, 2], [lab for lab, _ in CONSTRUCTIONS.values()], "channel families")
    for ax, tag in zip(axes, "ab"):
        ax.text(-0.22, 1.2, tag, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
    return fig


# --------------------------------------------------------------------------- DTW

def condition_mask(frame, lag, p_step):
    return (frame.max_lag == lag) & np.isclose(frame.p_step, p_step)


def _dtw_inset(ax, rect, frame, key):
    groups = [(-0.2, frame[condition_mask(frame, 0, 0)][key].to_numpy(), GREY, False)]
    for lag, ps in CONDITIONS[1:]:
        groups.append((ps, frame[condition_mask(frame, lag, ps)][key].to_numpy(),
                       LAG_CMAP(0.55) if ps == 0 else ORANGE, True))
    ins = _inset(ax, rect, groups, [0, 0.5, 0.9], ["0", ".5", ".9"], r"warp-step probability $p$")
    ins.set_xlim(-0.32, 1.0)
    return ins


def plot_dtw(frame: pd.DataFrame, p_show: float = 0.9, instance: int = 0):
    """(a) Euclidean vs shifted: no lag vs lag. (b) shifted vs DTW: lag vs lag + local warp. Insets: z over conditions."""
    pick = lambda lag, ps: frame[condition_mask(frame, lag, ps) & (frame.instance == instance)].iloc[0]
    recs = {"none": pick(0, 0), "lag": pick(5, 0), "warp": pick(5, p_show)}
    data = {k: bank(v["path"]) for k, v in recs.items()}
    lags = np.asarray(data["lag"][1]["generator"]["lags"])
    iu = np.triu_indices(len(lags), 1)
    dtau = np.abs(lags[iu[0]] - lags[iu[1]]).astype(float)
    order = np.argsort(dtau)
    lag_label = f"lag 0–{int(lags.max())}"

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2), constrained_layout=True)
    mk = lambda c: Line2D([], [], ls="", marker="o", ms=3.5, mew=0, color=c)

    ax = axes[0]
    lim = (0, 1.5)
    _plane(ax, lim, [0, 0.5, 1, 1.5], r"Euclidean distance $d_{jk}$",
           r"shifted distance $\min_{|h|\leq 10} d_{jk}(h)$")
    ax.scatter(data["none"][0][ED], data["none"][0][XD], s=7, lw=0, color=GREY, zorder=2)
    sc = ax.scatter(data["lag"][0][ED][order], data["lag"][0][XD][order], s=7, lw=0, c=dtau[order],
                    cmap=LAG_CMAP, vmin=0, vmax=lags.max(), zorder=3)
    ax.legend([mk(GREY), mk(LAG_CMAP(0.55))],
              [f"no lag, $z = {recs['none'].ed_xd:.2f}$", f"{lag_label}, $z = {recs['lag'].ed_xd:.2f}$"],
              **LEGEND)
    _dtw_inset(ax, [0.1, 0.62, 0.38, 0.32], frame, "ed_xd")

    ax = axes[1]
    lim = (0, 1.2)
    _plane(ax, lim, [0, 0.4, 0.8, 1.2], r"shifted distance $\min_{|h|\leq 10} d_{jk}(h)$",
           r"DTW distance $d^{\,\mathrm{DTW}}_{jk}$")
    ax.scatter(data["lag"][0][XD][order], data["lag"][0][DTW][order], s=7, lw=0, c=dtau[order],
               cmap=LAG_CMAP, vmin=0, vmax=lags.max(), zorder=3)
    ax.scatter(data["warp"][0][XD], data["warp"][0][DTW], s=7, lw=0, color=ORANGE, zorder=2)
    ax.legend([mk(LAG_CMAP(0.55)), mk(ORANGE)],
              [f"{lag_label}, $z = {recs['lag'].dtw_xd:.2f}$",
               f"+ local warp ($p = {p_show:g}$), $z = {recs['warp'].dtw_xd:.2f}$"],
              **LEGEND)
    _dtw_inset(ax, [0.1, 0.62, 0.38, 0.32], frame, "dtw_xd")

    cb = fig.colorbar(sc, ax=axes, shrink=0.45, aspect=16, pad=0.02, anchor=(0, 0.15))
    cb.set_label(r"$|\tau_j-\tau_k|$ (samples)", fontsize=8)
    cb.set_ticks(np.arange(0, lags.max() + 1))
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)
    for ax, tag in zip(axes, "ab"):
        ax.text(-0.22, 1.2, tag, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
    return fig


def noise_levels(meta: dict) -> np.ndarray:
    """Per-channel noise SDs of ``generate_lagged_warping_mts`` (uniform noise), by replaying its draw order.

    The generator draws the AR(1) mother (one normal per step after the first),
    then the M noise SDs; linspace lags draw nothing. Not stored in the archive.
    """
    g, p = meta["generator"], meta["generator"]["resolved_params"]
    assert g["name"] == "lagged_warping_mts" and p["noise_dist"] == "uniform" and p["lag_dist"] == "linspace"
    rng = np.random.default_rng(g["seed"])
    rng.normal(0, p["ar1_noise_std"], meta["T"] + max(g["lags"]) - 1)
    return rng.uniform(p["noise_std"] / p["noise_std_scale"], p["noise_std"] * p["noise_std_scale"], meta["M"])


def dtw_checks(frame: pd.DataFrame, max_shift: int = 10) -> pd.DataFrame:
    """Per recording: Euclidean = sqrt(2 - 2r) error, z(shifted, sqrt(2 - 2 max xcorr)), how far each
    distance profile tracks pair noise level s_j^2 + s_k^2 (the strength axis) and, under warping, the
    pair's relative deformation SD_t[(h_j - h_k)(t)] from the replayed generating maps."""
    from research.cases.case_mechanisms import replay  # checks the replay against the archive

    rows = []
    for _, rec in frame.iterrows():
        path = Path(rec["path"])
        p, meta = bank(path)
        x = np.load(path / "timeseries.npy")
        T, M = x.shape
        r = profile(np.corrcoef(x, rowvar=False))
        best = np.full((M, M), -np.inf)
        for h in range(-max_shift, max_shift + 1):
            a, b = (x[h:], x[:T - h]) if h >= 0 else (x[:T + h], x[-h:])
            a = (a - a.mean(0)) / a.std(0)
            b = (b - b.mean(0)) / b.std(0)
            best = np.maximum(best, a.T @ b / len(a))
        xcorr = profile(np.maximum(best, best.T))
        s = noise_levels(meta)
        iu = np.triu_indices(M, 1)
        noise = s[iu[0]] ** 2 + s[iu[1]] ** 2
        deform = np.full(len(noise), np.nan)
        if rec.p_step > 0:
            _, _, internal, _ = replay(rec)
            h = np.asarray(internal.mappings, dtype=float)
            deform = np.array([np.std(h[j] - h[k]) for j, k in zip(*iu)])
        rows.append({"class": rec["class"], "instance": rec.instance, "max_lag": rec.max_lag, "p_step": rec.p_step,
                     "ed_minus_sqrt_2_2r": float(np.max(np.abs(p[ED] - np.sqrt(2 - 2 * r)))),
                     "z_xd_xcorr_distance": z(p[XD], np.sqrt(2 - 2 * xcorr)),
                     "z_r_xcorr": z(r, xcorr), "z_ed_noise": z(p[ED], noise),
                     "z_xd_noise": z(p[XD], noise), "z_dtw_noise": z(p[DTW], noise),
                     "deform_mean": float(np.mean(deform)), "deform_sd_over_pairs": float(np.std(deform)),
                     "z_xd_deform": z(p[XD], deform) if rec.p_step > 0 else np.nan,
                     "z_dtw_deform": z(p[DTW], deform) if rec.p_step > 0 else np.nan})
    return pd.DataFrame(rows)
