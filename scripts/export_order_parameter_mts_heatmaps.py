#!/usr/bin/env python3
"""Export stripped representative MTS heatmaps for the benchmark systems."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.corpus_visualization import plot_mts_heatmap
from src.generators.dynamical import generate_cml_logistic
from src.generators.order_parameter import generate_miller_huse, generate_stuart_landau


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "notebooks/inference/figures"


def _parameter_slug(name: str, value: float) -> str:
    return f"{name}-{value:g}".replace(".", "p")


def _save_heatmap(values: np.ndarray, system: str, parameter: str, value: float) -> Path:
    directory = OUTPUT_ROOT / system
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"mts-{_parameter_slug(parameter, value)}.png"
    fig = plt.figure(figsize=(3.6, 1.35))
    ax = fig.add_axes((0, 0, 1, 1))
    plot_mts_heatmap(values, method="robust", ax=ax, colorbar=False)
    ax.set_axis_off()
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(path.relative_to(ROOT), flush=True)
    return path


def export_kuramoto() -> None:
    examples = {
        0.625: "M20_T1000_I0_kappa0p625",
        1.0075: "M20_T1000_I0_kappa1p0075",
        1.65: "M20_T1000_I0_kappa1p65",
    }
    source = ROOT / "data/order_parameter/kuramoto_figure_examples"
    for kappa, directory in examples.items():
        _save_heatmap(np.load(source / directory / "timeseries.npy"), "kuramoto", "kappa", kappa)


def export_stuart_landau() -> None:
    for gamma in (0.55, 0.68, 0.725, 0.77, 0.775, 1.25):
        values = generate_stuart_landau(
            M=32,
            T=1000,
            coupling=0.8,
            frequency_half_width=gamma,
            N_full=32,
            omega_mean=2.0,
            dt=0.02,
            sample_dt=0.1,
            burn_time=200.0,
            future_truth_T=0,
            output="real",
            rng=np.random.default_rng(991377),
            zscore=False,
        )
        _save_heatmap(values, "stuart-landau", "gamma", gamma)


def export_miller_huse() -> None:
    for coupling in (0.185, 0.20517, 0.225):
        values = generate_miller_huse(
            M=32,
            T=1000,
            coupling=coupling,
            mu=3.0,
            lattice_side=128,
            transients=400_000,
            sample_every=1,
            future_truth_T=0,
            truth_start_T=1000,
            observation_mode="distributed",
            initial_state="random",
            rng=np.random.default_rng(725911),
            zscore=False,
        )
        _save_heatmap(values, "miller-huse", "g", coupling)


def export_quadratic_cml() -> None:
    for alpha in (1.60, 1.75, 2.00):
        values = generate_cml_logistic(
            M=32,
            T=1000,
            alpha=alpha,
            eps=0.3,
            transients=2_000_000,
            sample_every=1,
            lattice_size=512,
            observation_mode="distributed",
            rng=np.random.default_rng(355974),
            zscore=False,
        )
        _save_heatmap(values, "quadratic-cml", "alpha", alpha)


def main() -> int:
    export_kuramoto()
    export_stuart_landau()
    export_miller_huse()
    export_quadratic_cml()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
