#!/usr/bin/env python3
"""Build the compact Stuart--Landau SPI--SPI benchmark notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("notebooks/inference/stuart-landau-order-coordinate.ipynb"),
    )
    parser.add_argument("--no-execute", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    cells = [
        nbformat.v4.new_markdown_cell(
            r"""# Stuart–Landau dynamical order-coordinate benchmark

The original Matthews–Strogatz population is

$$
\dot z_j=(1-|z_j|^2+i\omega_j)z_j+K(Z-z_j),\qquad
Z=N^{-1}\sum_jz_j,\qquad R=|Z|.
$$

Frequencies are evenly spaced on $[-\gamma,\gamma]$ in the rotating frame. We fix $K=0.8$ and vary $\gamma$: this is the published Fig. 1 intercept, illustrating locking, large oscillations, irregular order-parameter motion and incoherence at $\gamma=0.6,0.8,1.0,1.2$. The physical scalar is the future-window mean $Q=\langle R\rangle_t$; temporal SD of $R$ and mean oscillator activity are secondary physical coordinates. The complete 289-SPI p90 catalogue enters the SPI–SPI construction."""
        ),
        nbformat.v4.new_code_cell(
            r"""from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path.cwd().resolve()
while ROOT != ROOT.parent and not (ROOT / "src").exists():
    ROOT = ROOT.parent
from src.corpus_visualization import plot_mts_heatmap
from src.generators.order_parameter import generate_stuart_landau

RESULTS = ROOT / "data/order_parameter/stuart_landau_development_analysis"
summary = json.loads((RESULTS / "summary.json").read_text())
scores = pd.read_csv(RESULTS / "scores.csv")
sns.set_theme(style="whitegrid", context="notebook")
print(
    f"rows={summary['rows']}; catalogue={summary['spi_count']} SPIs / "
    f"{summary['meta_feature_count']:,} pairs; retained="
    f"{summary['selected_meta_feature_count']:,}; "
    f"represented SPIs={summary['represented_spi_count']}"
)"""
        ),
        nbformat.v4.new_markdown_cell(
            "## Brief physical orientation\n\n"
            "The heatmaps use the same per-process median/robust-scale `icefire` "
            "display rule as the corpus-geometry notebook. They establish context; "
            "they are not inferential evidence."
        ),
        nbformat.v4.new_code_cell(
            r"""gammas = [0.6, 0.8, 1.0, 1.2]
fig, axes = plt.subplots(2, 4, figsize=(13.5, 5.2), constrained_layout=True)
for column, gamma in enumerate(gammas):
    mts, internals = generate_stuart_landau(
        M=32, T=600, coupling=.8, frequency_half_width=gamma,
        N_full=32, omega_mean=2.0, dt=.02, sample_dt=.1,
        burn_time=200, future_truth_T=0, output="real",
        rng=np.random.default_rng(4100 + column), zscore=False,
        return_internals=True,
    )
    axes[0, column].plot(np.abs(internals.order_parameter), color="C0", lw=.8)
    axes[0, column].set(title=fr"$\gamma={gamma}$", xlabel="Time", ylabel=r"$R(t)$")
    plot_mts_heatmap(mts, method="robust", ax=axes[1, column], colorbar=False)
fig.suptitle(r"Published $K=0.8$ path: mean-field motion and observed MTS", y=1.02)
plt.show()"""
        ),
        nbformat.v4.new_markdown_cell(
            r"""## Leakage-controlled development assay

The feature-validity gate, imputation, centring and PCA use only full-observation rows with $T\ge500$ and instances 0–3. Neither $\gamma$ nor any physical target enters those fits. Instances 4–7 are the internal held-seed check. $T=100$ is applied after fitting as a stress cell. This remains development analysis; a later bank with new seeds and interleaved controls is the independent confirmation."""
        ),
        nbformat.v4.new_code_cell(
            r"""held = scores.query("instance >= 4").copy()
held_full = held.query("arm == 'full'").copy()
physical = held_full.query("T == 1000")

fig, ax = plt.subplots(figsize=(6.5, 4.1), constrained_layout=True)
palette = dict(zip([8, 16, 32], sns.color_palette("viridis", 3)))
for M, group in physical.groupby("M"):
    curve = group.groupby("gamma").Q_R_mean.agg(["mean", "std"]).reset_index()
    ax.fill_between(curve.gamma, curve["mean"]-curve["std"], curve["mean"]+curve["std"], color=palette[M], alpha=.12)
    ax.plot(curve.gamma, curve["mean"], "-o", ms=3, lw=1.8, color=palette[M], label=fr"$M=N={M}$")
ax.set(xlabel=r"frequency half-width $\gamma$ ($K=0.8$)", ylabel=r"future $Q=\langle R\rangle_t$", title="Physical order-parameter path")
ax.legend(frameon=False)
sns.despine(fig)
plt.show()"""
        ),
        nbformat.v4.new_markdown_cell(
            r"""## One frozen $q$ across $M$ and $T$

PC1 sign is arbitrary and is oriented only for readable display after the target-blind fit. Faint lines are held instances; coloured lines are cell means. Every panel uses the same frozen feature mask, centre and loading. The dashed physical curves are standardized only for visual overlay."""
        ),
        nbformat.v4.new_code_cell(
            r"""Q_mean = held_full.Q_R_mean.mean()
Q_sd = held_full.Q_R_mean.std()
held_full["Qz"] = (held_full.Q_R_mean - Q_mean) / Q_sd
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), sharey=True, constrained_layout=True)
for ax, T in zip(axes, [100, 500, 1000]):
    panel = held_full.query("T == @T")
    for M, group in panel.groupby("M"):
        for _, line in group.groupby("instance"):
            line = line.sort_values("gamma")
            ax.plot(line.gamma, line.q, color=palette[M], alpha=.12, lw=.65)
        curve = group.groupby("gamma")[["q", "Qz"]].mean().reset_index()
        ax.plot(curve.gamma, curve.q, color=palette[M], lw=1.8, label=fr"$M={M}$")
        ax.plot(curve.gamma, curve.Qz, color=palette[M], lw=.9, ls="--", alpha=.75)
    pooled = panel.groupby("gamma").q.mean()
    ax.plot(pooled.index, pooled.values, color="black", lw=2.6, alpha=.82, label="pooled q")
    ax.set(title=fr"$T={T}$", xlabel=r"$\gamma$")
axes[0].set_ylabel("standardized coordinate")
axes[-1].legend(frameon=False, fontsize=8)
fig.suptitle(r"Solid: frozen SPI–SPI $q$; dashed: physical $Q$", y=1.03)
sns.despine(fig)
plt.show()

cells = pd.DataFrame(summary["held_full_cell_summary"])
heat = cells.pivot(index="M", columns="T", values="overall_spearman").abs()
fig, ax = plt.subplots(figsize=(4.8, 3.2), constrained_layout=True)
sns.heatmap(heat, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="viridis", ax=ax, cbar_kws={"label": r"held $|\rho(q,Q)|$"})
ax.set_title("M/T robustness under one frozen transform")
plt.show()"""
        ),
        nbformat.v4.new_markdown_cell(
            r"""## Held-seed coordinate recovery and separate numerical inference

The left plot evaluates the unsupervised coordinate. The right plot is explicitly supervised: an isotonic map $q\mapsto\widehat Q$ is fitted on the representation-development rows and applied to held seeds."""
        ),
        nbformat.v4.new_code_cell(
            r"""fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), constrained_layout=True)
points = axes[0].scatter(held_full.q, held_full.Q_R_mean, c=held_full.gamma, cmap="magma", s=17, alpha=.58, linewidth=0)
fig.colorbar(points, ax=axes[0], label=r"$\gamma$")
axes[0].set(xlabel=r"frozen SPI–SPI PC1 $q$", ylabel=r"future physical $Q$", title="Unsupervised coordinate recovery")
axes[1].scatter(held_full.Q_R_mean, held_full.Q_hat, c=held_full.gamma, cmap="magma", s=17, alpha=.58, linewidth=0)
limits = [held_full.Q_R_mean.min(), held_full.Q_R_mean.max()]
axes[1].plot(limits, limits, color=".3", ls="--", lw=1)
axes[1].set(xlabel=r"future physical $Q$", ylabel=r"supervised $\widehat Q(q)$", title="Frozen isotonic readout")
sns.despine(fig)
plt.show()

display(pd.DataFrame({
    "quantity": ["PC1 variance", "held full overall rho", "held full within-gamma rho", "held partial overall rho", "held partial within-gamma rho", "held full isotonic MAE"],
    "value": [summary["pc_explained_variance_ratio"][0], summary["held_full_association"]["overall_spearman"], summary["held_full_association"]["within_gamma_spearman"], summary["held_partial_association"]["overall_spearman"], summary["held_partial_association"]["within_gamma_spearman"], summary["held_full_supervised_isotonic_mae"]],
}).round(3))"""
        ),
        nbformat.v4.new_markdown_cell(
            "## Stability, baselines and exploratory alternatives\n\n"
            "Choosing the PC that maximizes correlation with $Q$ remains supervised "
            "researcher selection even when automated. The oracle table is therefore "
            "diagnostic only; PC1 is the predeclared target-blind scalar."
        ),
        nbformat.v4.new_code_cell(
            r"""display(pd.DataFrame(summary["target_free_source_stability"]).round(3))
display(pd.DataFrame(summary["exploratory_oracle_pc"]).round(3))
display(pd.DataFrame(summary["held_full_input_baselines"]).T.round(3))"""
        ),
    ]

    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3.12"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_execute:
        client = NotebookClient(
            notebook,
            timeout=900,
            kernel_name="python3",
            resources={"metadata": {"path": str(root)}},
        )
        client.execute()
    nbformat.write(notebook, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
