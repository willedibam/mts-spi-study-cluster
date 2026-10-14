"""Plots still used by the exploratory DTW and r–rho–MI notebooks.

These retain the old pair colours, lag colouring, marginal plots and sweep
summaries. New September figures have their own study-specific plotters.
"""
from pathlib import Path
import json
import re
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


def apply_plot_style():
    """Legacy figure proportions, with a usable fallback when TeX is absent."""
    plt.rcParams.update({
        "text.usetex": bool(shutil.which("latex") and shutil.which("dvipng")),
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 12,
        "axes.labelsize": 14, "axes.titlesize": 16,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "axes.grid": False, "figure.dpi": 180,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def plot_mts_heatmap(mts, *, title=None, cmap="icefire", vmin=-2, vmax=2, figsize=(9, 6)):
    data = np.asarray(mts)
    if data.ndim != 2:
        raise ValueError("Expected a two-dimensional time series.")
    if data.shape[0] > data.shape[1]:
        data = data.T  # Legacy notebooks supply both T×M and M×T.
    fig, ax = plt.subplots(figsize=figsize)
    ax.pcolormesh(data, shading="flat", vmin=vmin, vmax=vmax,
                  cmap=sns.color_palette(cmap, as_cmap=True))
    ax.set(xticks=[], yticks=[], title=title or "")
    fig.tight_layout()
    plt.show()
    return ax


def _paired_values(directory, spis):
    if len(spis) != 2:
        raise ValueError("Exactly two SPI identifiers are required.")
    directory = Path(directory)
    with np.load(directory / "spi_mpis.npz") as archive:
        a, b = (np.asarray(archive[name], float) for name in spis)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Expected matching square matrices in {directory}")
    i, j = np.triu_indices(len(a), 1)
    x, y = ((v + v.T)[i, j] / 2 for v in (a, b))
    finite = np.isfinite(x) & np.isfinite(y)
    return x[finite], y[finite], i[finite], j[finite]


def _correlation(x, y, metric):
    if metric not in {"pearson", "spearman"}:
        raise ValueError("metric must be 'pearson' or 'spearman'")
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1] if metric == "pearson" else spearmanr(x, y).statistic)


def _annotate(ax, x, y, spis, directory, metric, identity, main_spines=True, trend=True):
    if not len(x):
        raise ValueError(f"No finite paired SPI values in {directory}")
    if trend and np.ptp(x) > 0:
        sns.regplot(x=x, y=y, ax=ax, scatter=False, ci=None,
                    color="#333333", line_kws={"lw": 1})
    if identity:
        limits = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(limits, limits, "--", color="grey", lw=.8)
    symbol = "r" if metric == "pearson" else r"\rho"
    value = _correlation(x, y, metric)
    label = rf"${symbol}={value:.2f}$" if np.isfinite(value) else "Undefined (constant profile)"
    ax.text(.04, .96, label, transform=ax.transAxes, va="top", fontsize=12,
            bbox={"facecolor": "none", "edgecolor": "#333333", "linewidth": .7})
    ax.set(xlabel=spis[0], ylabel=spis[1], title=Path(directory).parent.name)
    ax.set_box_aspect(1)
    ax.spines[["top", "right"]].set_visible(main_spines)


def plot_spi_space_individual(dataset_path, spis, *, ax=None, metric="spearman"):
    x, y, _, _ = _paired_values(dataset_path, spis)
    standalone = ax is None
    if standalone:
        grid = sns.jointplot(x=x, y=y, height=6, s=15, alpha=.6,
                             marginal_kws={"kde": True, "fill": True})
        ax = grid.ax_joint
    else:
        ax.scatter(x, y, s=15, alpha=.6)
    _annotate(ax, x, y, spis, dataset_path, metric, identity=False)
    if standalone:
        plt.show()
    return ax


def plot_spi_space_individual_embed_lags(dataset_path, spis, ax, *,
        lag_feature="diff", cmap="viridis", s=25, alpha=.6,
        main_spines=True, metric="spearman", identity=True):
    x, y, i, j = _paired_values(dataset_path, spis)
    meta = json.loads((Path(dataset_path) / "meta.json").read_text())
    lags = np.asarray(meta["generator"]["lags"], float)
    if lag_feature == "diff":
        values, label = abs(lags[i] - lags[j]), r"$|\tau_i-\tau_j|$"
    elif lag_feature == "max":
        values, label = np.maximum(lags[i], lags[j]), r"$\max(\tau_i,\tau_j)$"
    elif lag_feature == "mean":
        values, label = (lags[i] + lags[j]) / 2, r"$(\tau_i+\tau_j)/2$"
    else:
        raise ValueError("lag_feature must be diff, max or mean")
    points = ax.scatter(x, y, c=values, cmap=cmap, vmin=0,
                        vmax=max(1, values.max()), s=s, alpha=alpha, linewidths=0)
    _annotate(ax, x, y, spis, dataset_path, metric, identity, main_spines)
    cax = ax.inset_axes([1.05, 0, .05, 1])
    ax.figure.colorbar(points, cax=cax, label=label)
    return ax


def plot_spi_space_individual_embed_properties(dataset_path, spis, ax, *,
        thresh_linear=np.pi/8, thresh_monotonic=np.pi/2,
        color_mixed_pairs=True, hollow_same_type_pairs=True,
        s=25, alpha=.4, main_spines=True, metric="spearman", identity=False):
    x, y, i, j = _paired_values(dataset_path, spis)
    meta = json.loads((Path(dataset_path) / "meta.json").read_text())
    a = np.asarray(meta["generator"]["a_values"], float)
    tags = np.where(a < thresh_linear, 0, np.where(a < thresh_monotonic, 1, 2))
    pair_tags = np.sort(np.column_stack([tags[i], tags[j]]), axis=1)
    names = ["L", "M", "NM"]
    markers = {(0, 0): "o", (0, 1): "^", (0, 2): "s", (1, 1): "*", (1, 2): "P", (2, 2): "X"}
    colors = {(0, 1): "#0072B2", (0, 2): "#D55E00", (1, 2): "#009E73"}
    for key, marker in markers.items():
        mask = (pair_tags == key).all(axis=1)
        if not mask.any():
            continue
        same = key[0] == key[1]
        color = colors[key] if not same and color_mixed_pairs else "#8a8a8a"
        ax.scatter(x[mask], y[mask], marker=marker, s=s, alpha=alpha,
                   facecolors="none" if same and hollow_same_type_pairs else color,
                   edgecolors=color, linewidths=.8 if same else .3,
                   label=f"{names[key[0]]}–{names[key[1]]}", zorder=1 if same else 2)
    _annotate(ax, x, y, spis, dataset_path, metric, identity, main_spines)
    return ax


def _sweep_plot(rows, *, ax, hue, title, xlabel, ylabel):
    if not rows:
        raise ValueError("No matching finite SPI pairs: check the run, catalogue and sweep parameter.")
    frame = pd.DataFrame(rows)
    ax = ax if ax is not None else plt.gca()
    sns.lineplot(data=frame, x="x", y="correlation", hue=hue, ax=ax,
                 errorbar=("ci", 95), err_style="band", seed=0)
    for key, value in [("title", title), ("xlabel", xlabel), ("ylabel", ylabel)]:
        if value is not None:
            getattr(ax, f"set_{key}")(value)
    return frame


def plot_spi_corr_sweep(dataset_path, spi_pair, *, folder_template, x_var,
        x_transform=None, metric="spearman", hue=True, title=None,
        xlabel=None, ylabel=None, ax=None):
    root = Path(dataset_path)
    if not root.is_dir():
        raise FileNotFoundError(root)
    pattern = re.compile(re.sub(r"<(\w+)>", lambda m: rf"(?P<{m[1]}>[\d.]+)", folder_template))
    rows = []
    for archive in sorted(root.glob("*/*/spi_mpis.npz")):
        directory = archive.parent
        match = pattern.search(f"{directory.parent.name}/{directory.name}")
        if match is None:
            continue
        groups = {k: float(v) for k, v in match.groupdict().items()}
        x_value = x_transform(groups) if x_transform else groups[x_var]
        x, y, _, _ = _paired_values(directory, spi_pair)
        value = _correlation(x, y, metric)
        if np.isfinite(value):
            rows.append({"x": x_value, "correlation": value, "mts_class": directory.parent.name})
    return _sweep_plot(rows, ax=ax, hue="mts_class" if hue else None,
                       title=title, xlabel=xlabel, ylabel=ylabel)


def plot_spi_corr_meta(dataset_path, spi_pairs, *, x_key, x_transform=None,
        pair_labels=None, metric="spearman", title=None, xlabel=None, ylabel=None, ax=None):
    root = Path(dataset_path)
    if not root.is_dir():
        raise FileNotFoundError(root)
    labels = pair_labels if pair_labels is not None else [f"({a}, {b})" for a, b in spi_pairs]
    if len(labels) != len(spi_pairs):
        raise ValueError("One label is required per SPI pair")
    rows = []
    for path in sorted(root.glob("*/*/meta.json")):
        meta = json.loads(path.read_text())
        raw = meta["generator"]["params"].get(x_key)
        if raw is None:
            continue
        for pair, label in zip(spi_pairs, labels):
            x, y, _, _ = _paired_values(path.parent, pair)
            value = _correlation(x, y, metric)
            if np.isfinite(value):
                rows.append({"x": x_transform(raw) if x_transform else raw,
                             "correlation": value, "spi_pair": label})
    return _sweep_plot(rows, ax=ax, hue="spi_pair", title=title, xlabel=xlabel, ylabel=ylabel)
