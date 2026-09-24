"""Lean notebook figures for the cached SPI baseline exploration."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scripts.spi_baseline_exploration import ROOT, OUT, CML

FIGURES = OUT / "figures"
NAMES = {"pearson": "Two Pearson summaries", "mean": "Per-SPI means", "distribution": "Per-SPI distributions",
         "z": "SPI–SPI", "mean+z": "Means + SPI–SPI", "distribution+z": "Distributions + SPI–SPI"}
METHODS = {"q": "Original SPI–SPI PC1", "mean_PC1": "Mean-SPI PC1", "distribution_PC1": "Distribution-SPI PC1",
    "z_standard_PC1": "Standardized SPI–SPI PC1",
    "mean+z_PC1": "Means + SPI–SPI PC1", "mean_correlation": "Mean Pearson r", "mean_abs_correlation": "Mean |r|",
    "temporal_spectral_entropy": "Spectral entropy", "analytic_phase_coherence": "Hilbert coherence",
    "sample_period2": "Observed period-two proxy", "mean_channel_sd": "Mean channel SD"}
SYSTEMS = {
    "Kuramoto": ("Kuramoto", r"Reduced coupling $\kappa$", 1),
    "CML2D": ("2D logistic CML", r"Map parameter $r$", 3.86212),
    "StuartLandau": ("Stuart–Landau: broad", r"Frequency half-width $\gamma$", None),
    "StuartLandauFine": ("Stuart–Landau: local sweep", r"Frequency half-width $\gamma$", None),
    "MillerHuse": ("Miller–Huse", r"Coupling $g$", .20534),
    "KanekoBand": ("Kaneko: band power", r"Map parameter $\alpha$", None),
    "KanekoEntropy": ("Kaneko: temporal entropy", r"Map parameter $\alpha$", None),
    "Rossler": ("Rössler: entrainment", r"Coupling $C$", None),
}
HEADLINE_SYSTEMS = ("Kuramoto", "StuartLandauFine", "MillerHuse", "KanekoBand", "CML2D", "Rossler", "KanekoEntropy")


def inference_results(systems=None):
    metrics = pd.read_csv(OUT / "inference-metrics.csv")
    info = json.loads((OUT / "inference-provenance.json").read_text())["diagnostics"]
    extra = OUT / "inference-extra-metrics.csv"
    if extra.exists():
        metrics = pd.concat([metrics, pd.read_csv(extra)], ignore_index=True)
        info.update(json.loads((OUT / "inference-extra-provenance.json").read_text())["diagnostics"])
    if systems is not None:
        metrics = metrics[metrics.system.isin(systems)].copy()
        info = {s: info[s] for s in systems}
    return metrics, info


def style():
    mpl.rcParams.update({"text.usetex": bool(shutil.which("latex") and shutil.which("dvipng")),
        "font.family": "serif", "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "xtick.direction": "out", "ytick.direction": "out", "legend.frameon": False,
        "lines.linewidth": 1.7, "lines.markersize": 2.7, "figure.dpi": 150,
        "savefig.dpi": 600, "savefig.bbox": "tight", "figure.constrained_layout.use": True})


def export(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.svg")
    fig.savefig(FIGURES / f"{name}.png", dpi=220)
    return fig


def intuition():
    samples = pd.read_csv(OUT / "intuition-samples.csv")
    with np.load(OUT / "intuition-matrices.npz") as a:
        matrices = [a["C_0.02_-1"], a["L_0.02_1"], a["L_0.02_-1"]]
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.4))
    for ax, matrix, title in zip(axes[0], matrices,
            ["Same contemporaneous profile C", "Lagged profile L: aligned", "Lagged profile L: opposed"]):
        matrix = matrix.copy()
        np.fill_diagonal(matrix, np.nan)
        im = ax.imshow(matrix, norm=mpl.colors.TwoSlopeNorm(vmin=-.04, vcenter=0, vmax=.06), cmap="RdBu_r")
        ax.set(title=title, xlabel="Channel", ylabel="Channel", xticks=range(6), yticks=range(6))
    fig.colorbar(im, ax=list(axes[0]), label="Population correlation (diagonal excluded)", shrink=.75)
    palette = {-1: "#b35806", 1: "#31688e"}
    for (s, orientation), group in samples.groupby(["strength", "orientation"]):
        marker = "o" if s == .02 else "^"
        label = f"{'low' if s == .02 else 'high'} mean; {'aligned' if orientation == 1 else 'opposed'}"
        for ax, y in ((axes[1,0], "mean_L"), (axes[1,1], "z")):
            ax.scatter(group.mean_C, group[y], color=palette[orientation], marker=marker,
                       alpha=.6, s=15, edgecolors="none", label=label)
    axes[1,0].set(xlabel=r"Estimated mean $C_{ij}$", ylabel=r"Estimated mean $L_{ij}$",
                  title="Means retain average dependence")
    axes[1,1].set(xlabel=r"Estimated mean $C_{ij}$", ylabel=r"SPI–SPI $z(C,L)$",
                  title="Adding z resolves correspondence", ylim=(-1.05,1.05))
    axes[1,1].axhline(0, color=".7", lw=.6)
    mask = ~np.eye(6, dtype=bool)
    axes[1,2].plot(np.sort(matrices[1][mask]), color=palette[1], label="Aligned L")
    axes[1,2].plot(np.sort(matrices[2][mask]), "--", color=palette[-1], label="Opposed L")
    axes[1,2].set(xlabel="Sorted ordered-pair index", ylabel="Population lagged correlation",
                  title="Entire marginal distributions match")
    axes[1,2].legend()
    for i in range(0, 3):
        axes[0,i].spines["top"].set_visible(True)
        axes[0,i].spines["right"].set_visible(True)
    handles, labels = axes[1,1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    return export(fig, "intuition")


def proof(kind="umap"):
    with np.load(OUT / "proof-projections.npz") as a:
        data = {k: a[k] for k in a.files if "_" + kind in k or k.endswith("labels")}
    classes = sorted(np.unique(data["inter_labels"]))
    colors = dict(zip(classes, plt.colormaps["tab20"](np.linspace(0, 1, len(classes)))))
    titles = {"mean": r"Per-SPI means $m$", "distribution": "Per-SPI distributions", "z": r"SPI–SPI $z$"}
    fig, axes = plt.subplots(3, 2, figsize=(9.6, 14.2))
    for row, name in enumerate(("mean", "distribution", "z")):
        for col, scope in enumerate(("inter", "cml")):
            ax = axes[row, col]
            labels, xy = data[f"{scope}_labels"], data[f"{name}_{scope}_{kind}"]
            for label in np.unique(labels):
                keep = labels == label
                ax.scatter(*xy[keep].T, color=colors[label], s=9, alpha=.35, linewidths=0,
                           rasterized=True, label=label)
            ax.set(title=f"{titles[name]} — {'13 classes' if scope == 'inter' else '5 CML regimes'}",
                   xlabel=f"{kind.upper()} 1", ylabel=f"{kind.upper()} 2", xticks=[], yticks=[])
            ax.set_box_aspect(1)
            ax.set_aspect("equal", adjustable="datalim")
            for spine in ax.spines.values():
                spine.set_visible(True)
    handles = [Line2D([], [], marker="o", color=colors[c], linestyle="", markersize=4, label=c) for c in classes]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=7)
    return export(fig, f"proof-{kind}")


def proof_table():
    frame = pd.read_csv(OUT / "proof-metrics.csv")
    frame["Representation"] = frame.method.map(NAMES)
    return frame.pivot(index="Representation", columns="scope", values=["balanced_accuracy", "hard_mAP"]).round(3)


def bootstrap_curve(frame, value, seed=260921):
    rows = []
    rng = np.random.default_rng(seed)
    for control, group in frame.groupby("control"):
        values = group.groupby("seed")[value].mean().to_numpy()
        means = values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
        rows.append([control, values.mean(), *np.quantile(means, [.025, .975])])
    return np.array(rows).T


def inference(systems=("Kuramoto", "CML2D"), filename="inference-headlines"):
    metrics, info = inference_results()
    fig, axes = plt.subplots(len(systems), 3, figsize=(12.5, 3.4*len(systems)), squeeze=False)
    methods = ("q", "mean_PC1", "mean_abs_correlation")
    colors = ("#31688e", "#d95f02", "#7570b3")
    for row, system in enumerate(systems):
        title, control_label, boundary = SYSTEMS[system]
        all_rows = pd.read_csv(OUT / f"inference-{system}.csv")
        held = all_rows.query("evaluation and common_eligible").copy()
        dev = all_rows.query("development")
        available = int(all_rows.evaluation.sum())
        coverage = f"; {len(held)}/{available} eligible" if len(held) < available else ""
        for col, (name, color) in enumerate(zip(methods, colors)):
            ax = axes[row, col]
            twin = ax.twinx()
            twin.spines["right"].set_visible(True)
            twin.spines["top"].set_visible(False)
            if name != "mean_abs_correlation":
                sign = info[system]["display_signs_from_development_Q"][name]
                held["display"] = sign * (held[name] - dev[name].mean()) / dev[name].std(ddof=0)
                right_label = "Coordinate (development SD)"
            else:
                held["display"] = held[name]
                right_label = r"Mean $|r|$ (raw units)"
            x, y, lo, hi = bootstrap_curve(held, "Q_reference")
            ax.plot(x,y,"o-",color="#222222",label="Physical Q")
            ax.fill_between(x,lo,hi,color=".4",alpha=.16,lw=0)
            x,y,lo,hi = bootstrap_curve(held,"display")
            twin.plot(x,y,"s-",color=color,label=METHODS[name])
            twin.fill_between(x,lo,hi,color=color,alpha=.16,lw=0)
            rho = metrics.query("system == @system and method == @name").abs_rho.iloc[0]
            label = r"Mean $|r|$" if name == "mean_abs_correlation" else METHODS[name]
            ax.set_title(f"{title}\n{label}: " + rf"$|\rho|={rho:.3f}$" + coverage)
            ax.set(xlabel=control_label, ylabel="Diagnostic Q" if system.startswith("Kaneko") else "Physical Q")
            if boundary is not None:
                ax.axvline(boundary, color=".6", linestyle=":", lw=.8)
            twin.set_ylabel(right_label,color=color)
            twin.tick_params(axis="y",colors=color)
    return export(fig, filename)


def inference_table(compact=False, systems=None):
    frame, _ = inference_results(systems)
    if compact:
        selected = ["q", "mean_PC1", "distribution_PC1", "mean_correlation", "mean_abs_correlation"]
        table = frame.pivot(index="system", columns="method", values="abs_rho")[selected]
        return table.rename(index={k:v[0] for k,v in SYSTEMS.items()}, columns=METHODS).round(3)
    frame["Representation"] = frame.method.map(METHODS)
    frame["Association (95% CI)"] = [f"{r.abs_rho:.3f} [{r.low:.3f}, {r.high:.3f}]" for r in frame.itertuples()]
    frame["Difference from q (95% CI)"] = [f"{r.difference_vs_q:+.3f} [{r.difference_low:+.3f}, {r.difference_high:+.3f}]" for r in frame.itertuples()]
    return frame[["system","Representation","Association (95% CI)","Difference from q (95% CI)","within_control_abs_rho"]].round(3)


def inference_paired_table(systems=None):
    frame, _ = inference_results(systems)
    frame = frame[frame.method.isin(["mean_PC1", "distribution_PC1"])].copy()
    frame["Difference (95% CI)"] = [f"{r.difference_vs_q:+.3f} [{r.difference_low:+.3f}, {r.difference_high:+.3f}]" for r in frame.itertuples()]
    return frame.pivot(index="system", columns="method", values="Difference (95% CI)").rename(
        index={k:v[0] for k,v in SYSTEMS.items()}, columns=METHODS)


def zenodo(color_by="origin"):
    with np.load(OUT / "zenodo-projections.npz") as a:
        data = {k:a[k] for k in a.files}
    meta = pd.read_csv(ROOT / "results/zenodo_7118947/visual-exploration-v2/dataset-metadata.csv").set_index("dataset")
    origin = meta.loc[data["dataset"], "origin"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    for ax, name in zip(axes, ("mean", "distribution", "z")):
        xy=data[f"{name}_umap"]
        if color_by == "origin":
            for category, color in (("real", "#31688e"), ("synthetic", "#999999")):
                keep=origin==category
                ax.scatter(*xy[keep].T, s=9, color=color, alpha=.6, linewidths=0, rasterized=True, label=category)
        else:
            im=ax.scatter(*xy.T, s=9, c=np.log10(data["T"]), cmap="viridis", linewidths=0, rasterized=True)
        ax.set(title=NAMES[name],xlabel="UMAP 1",ylabel="UMAP 2",xticks=[],yticks=[])
    if color_by == "origin":
        handles, labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="outside lower center",ncol=2)
    else:
        fig.colorbar(im,ax=list(axes),label=r"$\log_{10} T$",shrink=.7)
    return export(fig,f"zenodo-{color_by}")
