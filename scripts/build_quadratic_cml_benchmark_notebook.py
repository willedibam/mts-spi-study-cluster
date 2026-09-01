#!/usr/bin/env python3
"""Build the quadratic-CML physical-vector SPI--SPI benchmark notebook."""

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
        default=Path("notebooks/inference/quadratic-cml-order-coordinate.ipynb"),
    )
    parser.add_argument("--no-execute", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    cells = [
        nbformat.v4.new_markdown_cell(
            r"""# Quadratic CML regime-coordinate benchmark

The one-dimensional periodic coupled-map lattice is

$$
x_i(t+1)=(1-\epsilon)f_\alpha(x_i(t))+
\frac{\epsilon}{2}\left[f_\alpha(x_{i-1}(t))+f_\alpha(x_{i+1}(t))\right],
\qquad f_\alpha(x)=1-\alpha x^2.
$$

We fix $\epsilon=0.3$ and sweep $\alpha=1.60:0.01:2.00$. This path is literature-grounded but does **not** have one accepted scalar order parameter across all pattern-selection, intermittency/coexistence and fully developed spatiotemporal-chaos regimes. The physical reference is therefore the predeclared vector

$$Q=(H_{\rm temporal},\ H_{\rm dynamic\ pattern},\ P_{k\in[.25,.45]\pi},\ D_2),$$

where $D_2$ is the RMS period-two residual. The primary physical lattice is $N=512$ and SPI–SPI observes nested dispersed $M=8,16,32$ sensors. Separate $M=N=8,16,32$ runs expose finite-size failure and are not pooled with the primary assay."""
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
from src.generators.dynamical import generate_cml_logistic

PHYSICS = ROOT / "data/order_parameter/quadratic_cml_physics_contract"
RESULTS = ROOT / "data/order_parameter/quadratic_cml_development_analysis"
physics = pd.read_csv(PHYSICS / "records.csv")
physics_summary = json.loads((PHYSICS / "summary.json").read_text())
summary = json.loads((RESULTS / "summary.json").read_text())
scores = pd.read_csv(RESULTS / "scores.csv")
sns.set_theme(style="whitegrid", context="notebook")
print(summary["status"], f"rows={summary['rows']}", f"retained pairs={summary['selected_meta_feature_count']:,}")"""
        ),
        nbformat.v4.new_markdown_cell(
            "## Brief physical orientation\n\n"
            "Each field was burned for two million updates. The robust `icefire` "
            "heatmaps scale each observed process independently and serve only as context."
        ),
        nbformat.v4.new_code_cell(
            r"""alphas = [1.70, 1.75, 1.80, 1.90]
fig, axes = plt.subplots(2, 4, figsize=(13.5, 5.2), constrained_layout=True)
for column, alpha in enumerate(alphas):
    observed, full = generate_cml_logistic(
        M=32, T=500, alpha=alpha, eps=.3, transients=2_000_000,
        lattice_size=512, observation_mode="distributed",
        return_full_lattice=True, rng=np.random.default_rng(8100 + column),
        zscore=False,
    )
    axes[0, column].plot(full[-1], color=".18", lw=.55)
    axes[0, column].set(title=fr"$\alpha={alpha:.2f}$", xlabel="Lattice site", ylabel=r"$x_i$")
    plot_mts_heatmap(observed, method="robust", ax=axes[1, column], colorbar=False)
fig.suptitle(r"Full-field snapshot and $M=32$ dispersed observation", y=1.02)
plt.show()"""
        ),
        nbformat.v4.new_markdown_cell(
            "## Physics-only gate\n\n"
            "Faint lines are independent initial conditions; bold lines are cell "
            "means. The shaded interval marks the principal reorganization, not a "
            "claimed exact asymptotic phase boundary."
        ),
        nbformat.v4.new_code_cell(
            r"""metrics = [
    ("selected_band_power", "selected-band power"),
    ("dynamical_spatial_pattern_entropy", "dynamical pattern entropy"),
    ("temporal_spectral_entropy", "temporal spectral entropy"),
    ("period2_activity", "period-2 residual"),
]
palette_N = dict(zip(sorted(physics.N.unique()), sns.color_palette("viridis", physics.N.nunique())))
fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), sharex=True, constrained_layout=True)
for ax, (metric, label) in zip(axes.flat, metrics):
    for N, group in physics.groupby("N"):
        for _, line in group.groupby("instance"):
            line = line.sort_values("alpha")
            ax.plot(line.alpha, line[metric], color=palette_N[N], alpha=.09, lw=.6)
        curve = group.groupby("alpha")[metric].mean()
        ax.plot(curve.index, curve.values, "-o", ms=2.5, color=palette_N[N], lw=1.7, label=f"N={N}")
    ax.axvspan(1.75, 1.80, color=".5", alpha=.08)
    ax.set(ylabel=label)
for ax in axes[-1]: ax.set_xlabel(r"$\alpha$ ($\epsilon=0.3$)")
axes[0, 0].legend(frameon=False)
plt.show()

display(pd.DataFrame(physics_summary["prefix_convergence"]).T.round(3))
print(
    "At alpha=1.75, N=512: ",
    physics_summary["alpha1p75_N512_low_entropy_count_below_0p1"],
    "low-entropy and",
    physics_summary["alpha1p75_N512_high_entropy_count_above_0p3"],
    "high-entropy seeds."
)"""
        ),
        nbformat.v4.new_markdown_cell(
            r"""## Target-blind SPI–SPI coordinates

The full-p90 feature gate, imputation, centring and PCA use only $N=512$, $T=1000$, instances 0–3. Neither $\alpha$ nor $Q$ enters that fit. Instances 4–7 are the internal held-seed check. Because the physical reference branches, the primary unsupervised representation is $(q_1,q_2)$ rather than a forced scalar."""
        ),
        nbformat.v4.new_code_cell(
            r"""held = scores.query("instance >= 4")
large = held.query("arm == 'large'").copy()
palette_M = dict(zip([8, 16, 32], sns.color_palette("viridis", 3)))
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)
for ax, coordinate in zip(axes, ["q1", "q2"]):
    for M, group in large.groupby("M"):
        for _, line in group.groupby("instance"):
            line = line.sort_values("alpha")
            ax.plot(line.alpha, line[coordinate], color=palette_M[M], alpha=.10, lw=.55)
        curve = group.groupby("alpha")[coordinate].mean()
        ax.plot(curve.index, curve.values, color=palette_M[M], lw=1.7, label=f"M={M}")
    ax.set(xlabel=r"$\alpha$", ylabel=coordinate, title=f"Frozen SPI–SPI {coordinate}")
axes[0].legend(frameon=False)
sns.despine(fig)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(9.7, 4.1), constrained_layout=True)
p0 = axes[0].scatter(large.q1, large.q2, c=large.alpha, cmap="magma", s=17, alpha=.58, linewidth=0)
fig.colorbar(p0, ax=axes[0], label=r"$\alpha$")
axes[0].set(xlabel=r"$q_1$", ylabel=r"$q_2$", title="Unsupervised SPI–SPI geometry")
p1 = axes[1].scatter(large.Q_phys1, large.Q_phys2, c=large.alpha, cmap="magma", s=17, alpha=.58, linewidth=0)
fig.colorbar(p1, ax=axes[1], label=r"$\alpha$")
axes[1].set(xlabel=r"physical-vector PC1", ylabel=r"physical-vector PC2", title="Physical reference geometry")
sns.despine(fig)
plt.show()"""
        ),
        nbformat.v4.new_markdown_cell(
            r"""## Evaluation and scope

Pairwise-distance association compares the two-dimensional $q$ geometry with the four-dimensional physical vector without defining a scalar $Q$. The ridge readout is separately supervised. Selecting whichever SPI PC correlates best with a physical component would also be supervised and is not used to define the primary result."""
        ),
        nbformat.v4.new_code_cell(
            r"""display(pd.DataFrame(summary["held_large_by_M"]).round(3))
display(pd.DataFrame(summary["target_free_loading_stability_by_M"]).round(3))
display(pd.DataFrame(summary["held_large_q_component_associations"]).T)
display(pd.DataFrame({
    "quantity": ["q PC1 variance", "q PC2 variance", "overall distance rho", "within-alpha distance rho", "supervised vector MAE"],
    "value": [summary["q_explained_variance_ratio"][0], summary["q_explained_variance_ratio"][1], summary["held_large_two_dimensional_geometry"]["overall_pairwise_distance_spearman"], summary["held_large_two_dimensional_geometry"]["within_alpha_pairwise_distance_spearman"], summary["held_large_supervised_vector_mae"]],
}).round(3))
print("Small-M=N finite-size sensitivity:", summary["small_full_finite_size_geometry"])
print(
    "Interpretation: this system is retained only if its frozen q geometry tracks "
    "the held physical-vector geometry and the alpha=1.75 branches. It is a "
    "regime-coordinate stress test, not canonical scalar-order-parameter recovery."
)"""
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
        NotebookClient(
            notebook,
            timeout=1200,
            kernel_name="python3",
            resources={"metadata": {"path": str(root)}},
        ).execute()
    nbformat.write(notebook, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
