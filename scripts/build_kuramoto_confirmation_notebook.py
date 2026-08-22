#!/usr/bin/env python3
"""Build the compact Kuramoto order-parameter confirmation notebook."""

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
        default=Path("notebooks/embeddings/kuramoto-order-parameter-confirmation.ipynb"),
    )
    parser.add_argument("--no-execute", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    cells = [
        nbformat.v4.new_markdown_cell(
            r"""# Unsupervised SPI–SPI recovery of the Kuramoto order parameter

**Prospective terminal test.** The SPI–SPI representation and all gates were frozen before this bank was generated. The primary physical target is the canonical finite-population phase coherence

$$R_N(t)=\left|N^{-1}\sum_j e^{i\theta_j(t)}\right|,$$

averaged over a disjoint future of the hidden $N=256$ system. SPI–SPI sees only $M=20$ channels of $\cos\theta$ for $T=1000$. Numerical $q\mapsto R_N$ calibration is supervised and is reported separately from unsupervised coordinate recovery."""
        ),
        nbformat.v4.new_code_cell(
            r"""from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

ROOT = Path.cwd()
CONTRACT = ROOT / "data/order_parameter/kuramoto_final_confirmation_contract"
summary = json.loads((CONTRACT / "confirmation_summary.json").read_text())
representation = json.loads((CONTRACT / "representation_contract.json").read_text())
eligibility = json.loads((CONTRACT / "confirmation_eligibility.json").read_text())
shift_path = CONTRACT / "shift_control_summary.json"
shift = json.loads(shift_path.read_text()) if shift_path.exists() else None
archive = np.load(CONTRACT / "confirmation_results.npz", allow_pickle=False)
frame = pd.DataFrame({name: archive[name] for name in (
    "class_name", "distribution", "frequency_sampling", "design", "kappa", "instance",
    "coordinate_pc1", "coordinate_diffusion", "target_full_future_R",
    "target_hidden_complement_future_R", "prediction_R", "prediction_kappa_baseline_R",
)})
sns.set_theme(style="whitegrid", context="notebook")
print(f"status={summary['status']}; rows={len(frame)}; target opened={eligibility['outcomes_read']}")"""
        ),
        nbformat.v4.new_markdown_cell(
            r"""## Assay and frozen gates

The first prospective bank is retained as a disclosed eligibility null: three rows made the original 197-SPI core non-finite, and no $R_N$ was read. Exactly one target-blind redesign was allowed. It retained SPIs finite and nonconstant on every old plus eligibility-null input row; this terminal bank permits no further redesign."""
        ),
        nbformat.v4.new_code_cell(
            r"""gate_table = pd.DataFrame({
    "gate": list(summary["gate_results"]),
    "passed": list(summary["gate_results"].values()),
})
display(pd.DataFrame({
    "quantity": ["core SPIs", "SPI–SPI pairs", "PC1 features", "PC1 variance", "bootstrap loading p05", "bootstrap coordinate p05", "worst leave-group loading", "worst leave-group coordinate"],
    "value": [representation["core_spis"], representation["core_meta_features"], representation["retained_pc_features"], representation["pc_explained_variance_ratio"], representation["pc_stability"]["bootstrap_p05_loading_cosine"], representation["pc_stability"]["bootstrap_p05_coordinate_spearman"], representation["pc_stability"]["leave_group_minimum_loading_cosine"], representation["pc_stability"]["leave_group_minimum_coordinate_spearman"]],
}).round(4))
display(gate_table)"""
        ),
        nbformat.v4.new_markdown_cell(
            r"""## Primary result

PC1 has arbitrary sign. Its sign below is oriented after target reveal only for display; every reported gate uses absolute rank association. Random-frequency paired paths are primary. Independent-cell and regular-frequency rows are sensitivities."""
        ),
        nbformat.v4.new_code_cell(
            r"""primary = frame.query("frequency_sampling == 'random' and design == 'paired'").copy()
sign = np.sign(spearmanr(primary.coordinate_pc1, primary.target_full_future_R).statistic) or 1.0
frame["q"] = sign * frame.coordinate_pc1
primary = frame.query("frequency_sampling == 'random' and design == 'paired'").copy()

fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.1), constrained_layout=True)
colors = {"gaussian": "C0", "logistic": "C1"}
for dist, group in primary.groupby("distribution"):
    curve = group.groupby("kappa").target_full_future_R.agg(["mean", "std"]).reset_index()
    axes[0].fill_between(curve.kappa, curve["mean"]-curve["std"], curve["mean"]+curve["std"], color=colors[dist], alpha=.13)
    axes[0].plot(curve.kappa, curve["mean"], "-o", ms=3, color=colors[dist], label=dist)
axes[0].axvline(1, color=".35", ls="--", lw=1)
axes[0].set(xlabel=r"reduced coupling $\kappa=K/K_c$", ylabel=r"future global $\bar R_N$", title="A  Canonical order parameter")
axes[0].legend(frameon=False)

for dist, group in primary.groupby("distribution"):
    axes[1].scatter(group.q, group.target_full_future_R, s=11, alpha=.38, color=colors[dist], label=dist)
axes[1].set(xlabel=r"frozen SPI–SPI PC1 $q$", ylabel=r"future global $\bar R_N$", title="B  Untouched coordinate recovery")
axes[1].legend(frameon=False)

for dist, group in primary.groupby("distribution"):
    axes[2].scatter(group.target_full_future_R, group.prediction_R, s=11, alpha=.38, color=colors[dist], label=dist)
limits = [primary.target_full_future_R.min(), primary.target_full_future_R.max()]
axes[2].plot(limits, limits, color=".25", ls="--", lw=1)
axes[2].set(xlabel=r"observed future $\bar R_N$", ylabel=r"frozen calibrated $\widehat R_N$", title="C  Separate numerical readout")
sns.despine(fig)
plt.show()

pd.DataFrame({
    dist: {
        "overall rho": summary["associations"][dist]["full_future_R"]["overall_spearman"],
        "overall |rho| CI lower": summary["associations"][dist]["full_future_R"]["overall_absolute_ci_lower"],
        "within-kappa rho": summary["associations"][dist]["full_future_R"]["within_kappa_spearman"],
        "within-kappa |rho| CI lower": summary["associations"][dist]["full_future_R"]["within_kappa_absolute_ci_lower"],
        "MAE": summary["calibration"][dist]["mae"],
        "MAE CI upper": summary["calibration"][dist]["mae_ci95"][1],
    } for dist in ("gaussian", "logistic")
}).T.round(3)"""
        ),
        nbformat.v4.new_markdown_cell(
            r"""## Baselines and scope

Simple statistics are mandatory comparators, not advantage gates. A baseline may outperform SPI–SPI without invalidating coordinate recovery; it only prevents a superiority claim."""
        ),
        nbformat.v4.new_code_cell(
            r"""rows = []
for dist in ("gaussian", "logistic"):
    block = summary["associations"][dist]["full_future_R"]
    rows.append({"distribution": dist, "method": "SPI–SPI PC1", "overall": abs(block["overall_spearman"]), "within kappa": abs(block["within_kappa_spearman"])})
    for name, values in summary["baseline_associations"][dist].items():
        rows.append({"distribution": dist, "method": name.replace("_", " "), "overall": abs(values["overall_spearman"]), "within kappa": abs(values["within_kappa_spearman"])})
comparison = pd.DataFrame(rows)
plot = comparison.melt(id_vars=["distribution", "method"], var_name="association", value_name="absolute Spearman")
g = sns.catplot(data=plot, y="method", x="absolute Spearman", hue="association", col="distribution", kind="bar", height=4.2, aspect=1.0, legend_out=False)
g.set(xlim=(0, 1)); g.set_titles("{col_name}"); sns.despine()
plt.show()
display(comparison.round(3))"""
        ),
        nbformat.v4.new_markdown_cell("## Sensitivities"),
        nbformat.v4.new_code_cell(
            r"""sensitivity_rows = []
for dist, blocks in summary["sensitivities"].items():
    for design, values in blocks.items():
        sensitivity_rows.append({
            "distribution": dist, "design": design,
            "overall rho": values["overall_spearman"],
            "within-kappa rho": values["within_kappa_spearman"],
        })
display(pd.DataFrame(sensitivity_rows).round(3))
print(f"Diffusion-map available: {summary['diffusion_map_available']}; rank agreement with PC1: {summary['diffusion_map_pc1_spearman']:.3f}")
if shift is None:
    print("Circular-shift sensitivity not yet present.")
else:
    display(pd.DataFrame(shift["comparisons"]).T.round(3))"""
        ),
        nbformat.v4.new_markdown_cell(
            r"""## Interpretation

The defensible claim depends on the frozen gates above. A pass supports: **SPI–SPI meta-features yield an unsupervised, data-driven coordinate that recovers the changing canonical finite-$N$ Kuramoto order parameter up to a monotone transformation on untouched controls, seeds, and a second frequency law.** The numerical $R_N$ estimate uses a separately supervised calibration. This is a proof of capability, not evidence that SPI–SPI is uniquely optimal or universally recovers order parameters."""
        ),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_execute:
        NotebookClient(notebook, timeout=600, kernel_name="python3").execute(
            cwd=str(root)
        )
    nbformat.write(notebook, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
