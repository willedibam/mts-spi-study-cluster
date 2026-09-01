#!/usr/bin/env python3
"""Build the Miller--Huse SPI--SPI order-coordinate benchmark notebook."""

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
        default=Path("notebooks/inference/miller-huse-order-coordinate.ipynb"),
    )
    parser.add_argument("--no-execute", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    cells = [
        nbformat.v4.new_markdown_cell(
            r"""# Miller–Huse dynamical order-coordinate benchmark

The synchronous two-dimensional chaotic coupled-map lattice is

$$
x_{ij}(t+1)=(1-4g)f(x_{ij}(t))+g\sum_{(k,l)\in\mathrm{nn}(i,j)}f(x_{kl}(t)),
$$

using the original odd piecewise-linear map at $\mu=3$. The canonical finite-system scalar is

$$Q_{\rm MH}=\left\langle\left|L^{-2}\sum_{ij}\operatorname{sign}x_{ij}(t)\right|\right\rangle_t.$$

We vary $g$ across the refined critical region near $g_c=0.20534$. The physical field is $L=128$ ($N=16{,}384$), while SPI–SPI observes nested dispersed $M=8,16,32$ sites. Full p90 observation of all lattice sites is infeasible because every MPI is dense in $M^2$. Physical truth is averaged over a disjoint two-million-step future; eight block summaries quantify critical slowing."""
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
from src.generators.order_parameter import generate_miller_huse

PHYSICS = ROOT / "data/order_parameter/miller_huse_physics_long_truth"
RESULTS = ROOT / "data/order_parameter/miller_huse_development_analysis"
physics = np.load(PHYSICS / "physics_records.npz", allow_pickle=False)
physics_summary = json.loads((PHYSICS / "physics_summary.json").read_text())
summary = json.loads((RESULTS / "summary.json").read_text())
scores = pd.read_csv(RESULTS / "scores.csv")
sns.set_theme(style="whitegrid", context="notebook")
print(summary["status"], f"rows={summary['rows']}", f"retained pairs={summary['selected_meta_feature_count']:,}")"""
        ),
        nbformat.v4.new_markdown_cell(
            "## Brief physical orientation\n\n"
            "The smaller $L=64$ fields below are qualitative context only. The "
            "claim-bearing truth and all p90 inputs use $L=128$. Heatmaps use the "
            "same per-process robust `icefire` rule as the corpus notebook."
        ),
        nbformat.v4.new_code_cell(
            r"""couplings = [.18, .204, .206, .23]
fig, axes = plt.subplots(3, 4, figsize=(13.4, 8.0), constrained_layout=True)
for column, g in enumerate(couplings):
    mts, internals = generate_miller_huse(
        M=32, T=300, coupling=g, mu=3, lattice_side=64,
        transients=100_000, future_truth_T=0, truth_start_T=300,
        observation_mode="distributed", rng=np.random.default_rng(5100 + column),
        zscore=False, return_internals=True, store_full_field=True,
    )
    axes[0, column].plot(np.abs(internals.spin_magnetization), color="C0", lw=.8)
    axes[0, column].set(title=fr"$g={g}$", xlabel="Time", ylabel=r"$|m_s(t)|$")
    axes[1, column].imshow(np.sign(internals.full_field[-1]), cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    axes[1, column].set(xlabel="x", ylabel="y")
    plot_mts_heatmap(mts, method="robust", ax=axes[2, column], colorbar=False)
fig.suptitle("Domain field, global spin magnetization and dispersed MTS", y=1.01)
plt.show()"""
        ),
        nbformat.v4.new_markdown_cell(
            "## Two-million-step physical-truth gate\n\n"
            "Points are independent initial conditions; curves are cell means. "
            "Vertical bars show the within-trajectory block-mean standard error, "
            "not merely across-seed variability."
        ),
        nbformat.v4.new_code_cell(
            r"""physical = pd.DataFrame({
    "g": physics["control"], "instance": physics["instance"],
    "Q": physics["q_future_abs"],
    "Q_block_se": physics["q_future_abs_blocks"].std(axis=1, ddof=1) / np.sqrt(physics["q_future_abs_blocks"].shape[1]),
})
fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
ax.errorbar(physical.g, physical.Q, yerr=physical.Q_block_se, fmt="o", ms=3, color=".35", alpha=.38, lw=.5)
curve = physical.groupby("g").Q.mean()
ax.plot(curve.index, curve.values, "-o", color="C0", ms=3, lw=1.8)
ax.axvline(.20534, color=".25", ls="--", lw=1, label=r"refined $g_c$")
ax.set(xlabel=r"coupling $g$", ylabel=r"future $Q_{\rm MH}$", title="Canonical dynamical order path")
ax.legend(frameon=False)
sns.despine(fig)
plt.show()
display(pd.DataFrame({"value": physics_summary}).T[["future_block_repeatability_p95", "q_abs_effective_samples_p05", "q_abs_tau_int_p95"]].round(3))"""
        ),
        nbformat.v4.new_markdown_cell(
            r"""## Target-blind SPI–SPI development

Feature hygiene and PCA use instances 0–3 only, without $g$ or physical truth. Instances 4–7 are the internal held-seed check. PC1 sign is arbitrary and oriented after fitting only for readable plots."""
        ),
        nbformat.v4.new_code_cell(
            r"""held = scores.query("instance >= 4").copy()
palette = dict(zip([8, 16, 32], sns.color_palette("viridis", 3)))
fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), constrained_layout=True)
for M, group in held.groupby("M"):
    for _, line in group.groupby("instance"):
        axes[0].plot(line.sort_values("g").g, line.sort_values("g").q, color=palette[M], alpha=.12, lw=.6)
    curve = group.groupby("g").q.mean()
    axes[0].plot(curve.index, curve.values, color=palette[M], lw=1.8, label=f"M={M}")
axes[0].set(xlabel=r"$g$", ylabel=r"frozen SPI–SPI $q$", title="A  Target-blind coordinate")
axes[0].legend(frameon=False)
p = axes[1].scatter(held.q, held.Q_spin_abs, c=held.g, cmap="magma", s=17, alpha=.58, linewidth=0)
fig.colorbar(p, ax=axes[1], label=r"$g$")
axes[1].set(xlabel=r"$q$", ylabel=r"future $Q_{\rm MH}$", title="B  Held-seed recovery")
axes[2].scatter(held.Q_spin_abs, held.Q_hat_q, c=held.g, cmap="magma", s=17, alpha=.58, linewidth=0)
limits = [held.Q_spin_abs.min(), held.Q_spin_abs.max()]
axes[2].plot(limits, limits, color=".3", ls="--", lw=1)
axes[2].set(xlabel=r"future $Q_{\rm MH}$", ylabel=r"supervised $\widehat Q(q)$", title="C  Separate readout")
sns.despine(fig)
plt.show()

display(pd.DataFrame(summary["held_by_M"]).round(3))
display(pd.DataFrame(summary["target_free_source_stability"]).round(3))
display(pd.DataFrame(summary["held_input_baselines"]).T.round(3))
display(pd.DataFrame({
    "quantity": ["PC1 variance", "overall rho", "within-g rho", "pooled-g-mean rho", "q readout MAE", "control-only MAE", "truth block-SE p95"],
    "value": [summary["pc_explained_variance_ratio"][0], summary["held_association"]["overall_spearman"], summary["held_association"]["within_gamma_spearman"], summary["held_pooled_g_mean_spearman"], summary["held_supervised_q_mae"], summary["held_control_only_mae"], summary["truth_uncertainty"]["Q_block_mean_se_p95"]],
}).round(3))"""
        ),
        nbformat.v4.new_markdown_cell(
            "## Interpretation\n\n"
            "The result supports order-coordinate recovery only if the frozen PC1 "
            "is stable across $M$, follows held $Q_{\\rm MH}$ with uncertainty, and "
            "is not merely an artefact of one local sensor set. Numerical prediction "
            "and control-only interpolation are reported separately. No claim of "
            "established two-dimensional Ising universality is made."
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
