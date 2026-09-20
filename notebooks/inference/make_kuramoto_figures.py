from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.stats import spearmanr

from src.generators.order_parameter import generate_kuramoto_order_parameter


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
EXAMPLES = ROOT / "data/order_parameter/kuramoto_figure_examples"
CONTRACT = ROOT / "data/order_parameter/kuramoto_final_confirmation_contract"
FEATURES = ROOT / "features/order_parameter/kuramoto_final_confirmation.npz"

VARIANTS = (
    ("kappa0p625", 0.625, "below onset"),
    ("kappa1p0075", 1.0075, "near onset"),
    ("kappa1p65", 1.65, "synchronized"),
)

BLUE = "#31688E"
ORANGE = "#D97732"
PURPLE = "#76528B"
GREEN = "#2A9D8F"
GREY = "#5C6670"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 240,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _example_path(variant: str) -> Path:
    return EXAMPLES / f"M20_T1000_I0_{variant}"


def _reconstruct(variant: str):
    path = _example_path(variant)
    meta = json.loads((path / "meta.json").read_text())
    params = dict(meta["generator"]["resolved_params"])
    params["store_full_phases"] = True
    observed, internals = generate_kuramoto_order_parameter(
        M=int(meta["M"]),
        T=int(meta["T"]),
        rng=np.random.default_rng(int(meta["generator"]["seed"])),
        return_internals=True,
        **params,
    )
    saved = np.load(path / "timeseries.npy")
    if not np.allclose(observed, saved, rtol=0.0, atol=4e-8):
        raise RuntimeError(f"reconstructed observation does not match {path}")
    return saved.astype(float), internals


def _save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def phase_coherence_snapshots(examples) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.55), subplot_kw={"projection": "polar"})
    for ax, (_, kappa, regime), (_, internals) in zip(axes, VARIANTS, examples):
        index = int(np.argmin(np.abs(internals.r_full - np.mean(internals.r_full))))
        theta = internals.full_phases[index]
        vector = np.mean(np.exp(1j * theta))
        ax.scatter(theta, np.ones_like(theta), s=11, alpha=0.5, color=BLUE, edgecolors="none")
        ax.annotate(
            "",
            xy=(np.angle(vector), np.abs(vector)),
            xytext=(0.0, 0.0),
            arrowprops={"arrowstyle": "-|>", "lw": 2.2, "color": ORANGE},
        )
        ax.set_ylim(0, 1.05)
        ax.set_yticklabels([])
        ax.grid(alpha=0.22)
        ax.set_title(
            rf"$\kappa={kappa:g}$  |  {regime}" + "\n" + rf"$R_N(t^*)={abs(vector):.2f}$",
            pad=15,
        )
    fig.suptitle(
        r"Kuramoto order parameter: the mean phase vector of all $N=256$ oscillators",
        y=1.04,
        fontsize=13,
        fontweight="bold",
    )
    fig.text(0.5, -0.01, r"Dots are oscillator phases; the orange vector has length $R_N(t^*)$.", ha="center")
    fig.tight_layout()
    _save(fig, "01-kuramoto-phase-coherence")


def full_system_heatmaps(examples) -> None:
    fig = plt.figure(figsize=(12.8, 6.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(4.0, 1.15))
    heat_axes = []
    for column, ((_, kappa, regime), (_, internals)) in enumerate(zip(VARIANTS, examples)):
        order = np.argsort(internals.frequencies)
        phase = internals.full_phases[:, order].T
        ax = fig.add_subplot(grid[0, column])
        image = ax.imshow(
            phase,
            aspect="auto",
            origin="lower",
            extent=(0, 100, 1, 256),
            cmap="twilight",
            vmin=0,
            vmax=2 * np.pi,
            interpolation="nearest",
        )
        ax.set_title(rf"$\kappa={kappa:g}$  |  {regime}")
        ax.set_ylabel(r"oscillator rank by $\omega_i$" if column == 0 else "")
        ax.set_xticklabels([])
        heat_axes.append(ax)

        trace = fig.add_subplot(grid[1, column])
        trace.plot(np.linspace(0, 100, len(internals.r_full)), internals.r_full, color=ORANGE, lw=1.1)
        trace.axhline(np.mean(internals.r_full), color=GREY, ls="--", lw=0.8)
        trace.set_ylim(0, 1)
        trace.set_xlabel("observed time")
        trace.set_ylabel(r"$R_N(t)$" if column == 0 else "")
    colorbar = fig.colorbar(image, ax=heat_axes, location="right", shrink=0.74, pad=0.015)
    colorbar.set_label(r"phase $\theta_i(t)$")
    colorbar.set_ticks([0, np.pi, 2 * np.pi], labels=["0", r"$\pi$", r"$2\pi$"])
    fig.suptitle(r"Exact paired full systems: $N=256$ phases and their global coherence", fontsize=13, fontweight="bold")
    _save(fig, "02-kuramoto-full-system")


def observed_mts_heatmaps(examples) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.65), constrained_layout=True, sharey=True)
    for column, (ax, ((_, kappa, regime), (observed, _))) in enumerate(zip(axes, zip(VARIANTS, examples))):
        image = ax.imshow(
            observed.T,
            aspect="auto",
            origin="lower",
            extent=(0, 100, 1, 20),
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
        )
        ax.set_title(rf"$\kappa={kappa:g}$  |  {regime}")
        ax.set_xlabel("observed time")
        if column == 0:
            ax.set_ylabel("observed channel")
    colorbar = fig.colorbar(image, ax=axes, location="right", shrink=0.82, pad=0.015)
    colorbar.set_label(r"input $x_i(t)=\cos\theta_i(t)$")
    fig.suptitle(
        r"What SPI–SPI receives: the same $M=20$, $T=1000$ partial observations",
        fontsize=13,
        fontweight="bold",
    )
    _save(fig, "03-kuramoto-observed-mts")


def spi_spi_pipeline() -> None:
    path = _example_path("kappa1p0075")
    observed = np.load(path / "timeseries.npy").astype(float)
    mpi_names = ("cov_EmpiricalCovariance", "corr_pearson_tau-1", "mi_gaussian")
    mpi_titles = ("covariance", "lag-1 correlation", "Gaussian MI")
    with np.load(path / "spi_mpis.npz") as archive:
        mpis = [archive[name].astype(float) for name in mpi_names]
    with np.load(FEATURES, allow_pickle=True) as archive:
        paths = archive["dataset_paths"].astype(str)
        row = int(np.flatnonzero(np.char.endswith(paths, "M20_T1000_I0_kappa1p0075"))[0])
        spi_names = archive["spi_order"].astype(str)
        vector = archive["X_sym"][row].astype(float)
    z_matrix = np.eye(len(spi_names), dtype=float)
    triangle = np.triu_indices(len(spi_names), k=1)
    z_matrix[triangle] = vector
    z_matrix[(triangle[1], triangle[0])] = vector

    fig = plt.figure(figsize=(15.8, 4.0), constrained_layout=True)
    grid = fig.add_gridspec(1, 6, width_ratios=(1.35, 0.82, 0.82, 0.82, 1.25, 1.35))
    ax_input = fig.add_subplot(grid[0, 0])
    ax_input.imshow(observed.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
    ax_input.set(title=r"input $X$", xlabel="time", ylabel="20 channels")
    ax_input.set_xticks([])

    for column, (matrix, title) in enumerate(zip(mpis, mpi_titles), start=1):
        ax = fig.add_subplot(grid[0, column])
        finite = matrix[np.isfinite(matrix)]
        bound = max(abs(np.quantile(finite, 0.02)), abs(np.quantile(finite, 0.98)), 1e-12)
        ax.imshow(matrix, cmap="RdBu_r", vmin=-bound, vmax=bound, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("target channel")
        if column == 1:
            ax.set_ylabel("source channel")
        ax.set_xticks([]); ax.set_yticks([])

    ax_z = fig.add_subplot(grid[0, 4])
    image = ax_z.imshow(z_matrix, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax_z.set(title=r"SPI–SPI matrix $Z$", xlabel="SPI", ylabel="SPI")
    ax_z.set_xticks([]); ax_z.set_yticks([])
    cb = fig.colorbar(image, ax=ax_z, fraction=0.046, pad=0.03)
    cb.set_label("Pearson agreement")

    ax_vector = fig.add_subplot(grid[0, 5])
    ax_vector.imshow(vector[:, None], aspect="auto", origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
    ax_vector.set(xlabel="one feature vector", ylabel="SPI-pair index")
    ax_vector.set_xticks([])
    ax_vector.set_yticks([0, vector.size - 1], labels=["1", f"{vector.size:,}"])
    ax_vector.set_title(r"upper triangle $z(X)$" + f"\n{vector.size:,} features")

    fig.suptitle(
        r"From one MTS to one scalar: 164 interaction matrices $\rightarrow$ SPI–SPI agreement $\rightarrow q=(z-\mu)^\top v_1$",
        fontsize=13,
        fontweight="bold",
    )
    _save(fig, "04-kuramoto-spi-spi-pipeline")


def pc1_tracking() -> None:
    with np.load(CONTRACT / "confirmation_results.npz", allow_pickle=False) as archive:
        frame = pd.DataFrame({name: archive[name] for name in archive.files})
    primary = frame.query("frequency_sampling == 'random' and design == 'paired'").copy()
    sign = np.sign(spearmanr(primary.coordinate_pc1, primary.target_full_future_R).statistic) or 1.0
    primary["q"] = sign * primary.coordinate_pc1

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.0), constrained_layout=True)
    colors = {"R": GREEN, "q": PURPLE}
    for ax, distribution in zip(axes[:2], ("gaussian", "logistic")):
        group = primary[primary.distribution == distribution]
        summaries = group.groupby("kappa").agg(
            R=("target_full_future_R", "mean"),
            R_sem=("target_full_future_R", "sem"),
            q=("q", "mean"),
            q_sem=("q", "sem"),
        ).reset_index()
        for name in ("R", "q"):
            low, high = summaries[name].min(), summaries[name].max()
            scale = high - low
            values = (summaries[name] - low) / scale
            error = 1.96 * summaries[f"{name}_sem"] / scale
            label = r"global $\bar R_N$" if name == "R" else r"frozen PC1 $q$"
            ax.fill_between(summaries.kappa, values - error, values + error, color=colors[name], alpha=0.13)
            ax.plot(summaries.kappa, values, "-o", ms=3.4, lw=1.5, color=colors[name], label=label)
        ax.axvline(1, color=GREY, ls="--", lw=1)
        ax.set(xlabel=r"reduced coupling $\kappa=K/K_c$", ylabel="rescaled cell mean", ylim=(-0.08, 1.08), title=distribution.capitalize())
        ax.legend(frameon=False, loc="upper left")

    axes[2].scatter(
        primary.q,
        primary.target_full_future_R,
        c=primary.kappa,
        cmap="viridis",
        norm=Normalize(primary.kappa.min(), primary.kappa.max()),
        s=10,
        alpha=0.38,
        edgecolors="none",
    )
    axes[2].set(xlabel=r"oriented frozen PC1 $q$", ylabel=r"future global $\bar R_N$", title="Untouched realizations")
    scalar = plt.cm.ScalarMappable(norm=Normalize(primary.kappa.min(), primary.kappa.max()), cmap="viridis")
    colorbar = fig.colorbar(scalar, ax=axes[2], pad=0.02)
    colorbar.set_label(r"$\kappa$")
    fig.suptitle(
        r"Frozen SPI–SPI PC1 tracks the canonical order parameter through synchronization",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(0.34, -0.015, r"Curves are independently rescaled for shape comparison; bands are 95% normal intervals for cell means.", ha="center", fontsize=9)
    _save(fig, "05-kuramoto-pc1-tracking")


def pc1_loadings() -> None:
    with np.load(CONTRACT / "representation_model.npz", allow_pickle=False) as archive:
        spis = archive["core_spis"].astype(str)
        pair_left = archive["pair_left"]
        pair_right = archive["pair_right"]
        selected = archive["pc_feature_indices"]
        weights = archive["pc_component"]
    order = np.argsort(np.abs(weights))[-14:]

    def compact(name: str) -> str:
        replacements = {
            "corr_pearson": "Pearson corr.",
            "corr_spearman": "Spearman corr.",
            "corr_kendall": "Kendall corr.",
            "dtw_constraint-sakoe-chiba_radius-auto": "DTW (Sakoe–Chiba)",
            "dtw_constraint-itakura": "DTW (Itakura)",
            "coint_johansen_trace_stat_order-0_ardiff-1": "Johansen cointegration",
            "cce_kozachenko_n-5": "conditional entropy (Kozachenko)",
            "cce_gaussian_n-5": "conditional entropy (Gaussian)",
            "si_kozachenko_k-1": "stochastic interaction",
        }
        return replacements.get(name, name.replace("_", " "))

    labels = []
    for location in order:
        pair = int(selected[location])
        label = f"{compact(spis[pair_left[pair]])}  ×  {compact(spis[pair_right[pair]])}"
        labels.append(textwrap.fill(label, width=48))
    values = weights[order]
    colors = [ORANGE if value < 0 else BLUE for value in values]
    fig, ax = plt.subplots(figsize=(10.2, 6.7), constrained_layout=True)
    ax.barh(np.arange(len(values)), values, color=colors)
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_yticks(np.arange(len(values)), labels=labels, fontsize=8.5)
    ax.set_xlabel("PC1 loading")
    ax.set_title("Largest absolute loadings in the frozen SPI–SPI PC1", fontsize=13, fontweight="bold")
    fig.text(
        0.99,
        -0.01,
        "Each term is agreement between two SPI interaction patterns; loadings are descriptive, not unique importance scores.",
        ha="right",
        fontsize=9,
        color=GREY,
    )
    _save(fig, "06-kuramoto-pc1-loadings")


def main() -> None:
    _style()
    examples = [(observed, internals) for variant, _, _ in VARIANTS for observed, internals in [_reconstruct(variant)]]
    phase_coherence_snapshots(examples)
    full_system_heatmaps(examples)
    observed_mts_heatmaps(examples)
    spi_spi_pipeline()
    pc1_tracking()
    pc1_loadings()
    print(f"wrote six figures to {OUT}")


if __name__ == "__main__":
    main()
