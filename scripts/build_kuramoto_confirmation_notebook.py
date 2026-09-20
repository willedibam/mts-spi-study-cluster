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
        default=Path("notebooks/inference/kuramoto-order-parameter-confirmation.ipynb"),
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
from matplotlib.patches import Patch
import seaborn as sns
from scipy.stats import spearmanr

ROOT = Path.cwd().resolve()
while ROOT != ROOT.parent and not (ROOT / "src").exists():
    ROOT = ROOT.parent
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

The first prospective bank is retained as a disclosed eligibility null: three rows made the original 197-SPI core non-finite, and no $R_N$ was read. Exactly one target-blind redesign was allowed. It retained SPIs finite and nonconstant on every old plus eligibility-null input row; this terminal bank permits no further redesign.

The terminal analysis wrote and passed eligibility before loading targets. Its eligibility JSON was subsequently updated to record target access rather than preserving an immutable pre-read copy; the code ordering is auditable, and future runs now create a separate exclusive pre-read artifact."""
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
g = sns.catplot(data=plot, y="method", x="absolute Spearman", hue="association", col="distribution", kind="bar", height=4.2, aspect=1.0, legend=False)
g.set(xlim=(0, 1)); g.set_titles("{col_name}")
handles = [Patch(facecolor=sns.color_palette()[0]), Patch(facecolor=sns.color_palette()[1])]
labels = ["overall", "within kappa"]
g.fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.55, 1.02))
g.fig.subplots_adjust(top=.87); sns.despine()
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
    shift_rows = []
    for dist, values in shift["comparisons"].items():
        shift_rows.append({
            "distribution": dist,
            "raw overall": values["raw_overall_absolute_spearman"],
            "shifted overall": values["shifted_overall_absolute_spearman"],
            "overall difference CI": "[%.3f, %.3f]" % tuple(values["raw_minus_shifted_overall_absolute_spearman_ci95"]),
            "raw within": values["raw_within_kappa_absolute_spearman"],
            "shifted within": values["shifted_within_kappa_absolute_spearman"],
            "within difference CI": "[%.3f, %.3f]" % tuple(values["raw_minus_shifted_within_kappa_absolute_spearman_ci95"]),
        })
    display(pd.DataFrame(shift_rows).round(3))
    print(
        "Shift sensitivity: all frozen features finite = "
        f"{shift['all_frozen_meta_features_finite']}; maximum selected-feature "
        f"missingness = {shift['maximum_selected_feature_missingness']:.3f}."
    )"""
        ),
        nbformat.v4.new_markdown_cell(
            r"""## Interpretation

The defensible claim depends on the frozen gates above. A pass supports: **in this finite-$N$ Kuramoto benchmark, a prospectively frozen non-phase SPI–SPI PC1 learned without coupling or order-parameter labels recovered, up to a monotone transformation, changes in the canonical phase-coherence order parameter from partial observations on untouched controls and random-frequency realizations under Gaussian and logistic frequency laws.** The numerical $R_N$ estimate uses a separately supervised calibration. Both frequency laws appeared during target-free representation development, so this is not unseen-path transfer. Independent channel shifts substantially reduced overall association, consistent with cross-channel temporal alignment contributing to the representation; shift-induced estimator failures and retained association prevent clean causal attribution. This is a proof of capability, not evidence that SPI–SPI is uniquely optimal or universally recovers order parameters."""
        ),
    ]
    historical_cells = cells
    historical_cells[0] = nbformat.v4.new_markdown_cell(
        r"""### Historical assay definition

The prospective assay below used a frozen 164-SPI non-phase core, Gaussian and logistic frequency laws, and partial observation of a hidden larger population. Its result remains valid under that contract, but it is now retained as a restricted-core robustness/provenance analysis rather than the primary presentation."""
    )
    full_catalogue_cells = [
        nbformat.v4.new_markdown_cell(
            r"""# Unsupervised SPI–SPI recovery of the Kuramoto order parameter

**Primary full-catalogue reanalysis.** Every one of the 289 SPIs produced by `benchmarked_p90.yaml` enters the unified ordered SPI–SPI construction, giving 41,616 candidate meta-features before ordinary development-only validity and variance filtering. The physical model is restricted to one Gaussian random-frequency population. PC1 is fitted without coupling or order-parameter values.

This reconstruction uses the already disclosed terminal bank, so it is a **retrospective reanalysis**, not a new prospective confirmation. Its scientific purpose is to test the intended full-catalogue representation. The original prospectively frozen 164-SPI assay is preserved below a divider for provenance."""
        ),
        nbformat.v4.new_code_cell(
            r"""from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

ROOT = Path.cwd().resolve()
while ROOT != ROOT.parent and not (ROOT / "src").exists():
    ROOT = ROOT.parent
FULL = ROOT / "data/order_parameter/kuramoto_full_catalogue_reanalysis"
full_summary = json.loads((FULL / "summary.json").read_text())
full_archive = np.load(FULL / "results.npz", allow_pickle=False)
full_frame = pd.DataFrame({name: full_archive[name] for name in (
    "class_name", "design", "kappa", "instance", "coordinate_pc1",
    "target_full_future_R", "selected_feature_missingness",
    "mean_abs_correlation", "analytic_phase_coherence",
    "temporal_spectral_entropy",
)})
primary_rows = full_archive["primary_row_indices"].astype(int)
full_frame["prediction_R"] = np.nan
full_frame.loc[primary_rows, "prediction_R"] = full_archive["prediction_primary"]
sns.set_theme(style="whitegrid", context="notebook")
print(
    f"{full_summary['catalogue_spis']} SPIs -> "
    f"{full_summary['catalogue_meta_features']:,} candidate pairs -> "
    f"{full_summary['retained_meta_features']:,} retained meta-features; "
    f"represented SPIs={full_summary['represented_spis']}"
)
print(
    f"development rows={full_summary['development_rows']}; "
    f"terminal rows={full_summary['terminal_rows']}; "
    f"PC1 variance={full_summary['pc1_explained_variance_ratio']:.3f}"
)"""
        ),
        nbformat.v4.new_markdown_cell(
            r"""## Full-catalogue order-coordinate recovery

PC1 sign is arbitrary. It is oriented against $R_N$ only for display; the representation fit and all feature hygiene are target-free. The paired random-frequency design is primary and independent-cell rows are a sensitivity."""
        ),
        nbformat.v4.new_code_cell(
            r"""primary_full = full_frame.query("design == 'paired'").copy()
display_sign = np.sign(spearmanr(primary_full.coordinate_pc1, primary_full.target_full_future_R).statistic) or 1.0
full_frame["q"] = display_sign * full_frame.coordinate_pc1
primary_full = full_frame.query("design == 'paired'").copy()

def standardized(values):
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / values.std()

primary_full["q_standardized"] = standardized(primary_full.q)
primary_full["R_standardized"] = standardized(primary_full.target_full_future_R)
curve = primary_full.groupby("kappa").agg(
    Q=("target_full_future_R", "mean"), Q_sd=("target_full_future_R", "std"),
    q=("q_standardized", "mean"), q_sd=("q_standardized", "std"),
    Rz=("R_standardized", "mean"), Rz_sd=("R_standardized", "std"),
).reset_index()
q_variance = primary_full.groupby("kappa").q.var()

fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.3), constrained_layout=True)
axes[0, 0].fill_between(curve.kappa, curve.Rz-curve.Rz_sd, curve.Rz+curve.Rz_sd, color="C0", alpha=.12)
axes[0, 0].fill_between(curve.kappa, curve.q-curve.q_sd, curve.q+curve.q_sd, color="C1", alpha=.12)
axes[0, 0].plot(curve.kappa, curve.Rz, "-o", ms=3, color="C0", label=r"physical $R_N$")
axes[0, 0].plot(curve.kappa, curve.q, "-o", ms=3, color="C1", label=r"full-p90 PC1 $q$")
axes[0, 0].axvline(1, color=".35", ls="--", lw=1)
axes[0, 0].set(xlabel=r"reduced coupling $\kappa=K/K_c$", ylabel="standardized coordinate", title="A  Changing order coordinates")
axes[0, 0].legend(frameon=False)

points = axes[0, 1].scatter(primary_full.q, primary_full.target_full_future_R, c=primary_full.kappa, cmap="viridis", s=13, alpha=.55, linewidth=0)
fig.colorbar(points, ax=axes[0, 1], label=r"$\kappa$")
axes[0, 1].set(xlabel=r"full-p90 SPI–SPI PC1 $q$", ylabel=r"future global $\bar R_N$", title="B  Retrospective held-bank recovery")

axes[1, 0].plot(q_variance.index, q_variance.values, "-o", ms=3, color="C2")
axes[1, 0].axvline(full_summary["target_free_q_variance_peak_kappa"], color=".35", ls="--", lw=1)
axes[1, 0].set(xlabel=r"$\kappa$", ylabel=r"across-master $\mathrm{Var}(q)$", title="C  Target-free transition-localization sensitivity")

axes[1, 1].scatter(primary_full.target_full_future_R, primary_full.prediction_R, s=13, alpha=.48, color="C3")
limits = [primary_full.target_full_future_R.min(), primary_full.target_full_future_R.max()]
axes[1, 1].plot(limits, limits, color=".25", ls="--", lw=1)
axes[1, 1].set(xlabel=r"observed $\bar R_N$", ylabel=r"supervised $\widehat R_N(q)$", title="D  Separate isotonic readout")
sns.despine(fig)
plt.show()

display(pd.DataFrame({
    "quantity": [
        "overall Spearman", "overall 95% CI", "within-kappa Spearman",
        "within-kappa 95% CI", "isotonic MAE", "isotonic MAE 95% CI",
        "maximum selected-feature missingness", "p99 selected-feature missingness",
    ],
    "value": [
        full_summary["primary_association"]["overall_spearman"],
        full_summary["primary_association"]["overall_ci95"],
        full_summary["primary_association"]["within_kappa_spearman"],
        full_summary["primary_association"]["within_kappa_ci95"],
        full_summary["supervised_isotonic_readout"]["mae"],
        full_summary["supervised_isotonic_readout"]["mae_ci95"],
        full_summary["terminal_selected_feature_missingness_max"],
        full_summary["terminal_selected_feature_missingness_p99"],
    ],
}))"""
        ),
        nbformat.v4.new_markdown_cell(
            r"""## Baselines and interpretation

The comparison below is descriptive. The full-catalogue result demonstrates whether the intended unsupervised estimator-geometry contains a stable changing order coordinate; it does not require SPI–SPI to outperform purpose-built synchronization summaries."""
        ),
        nbformat.v4.new_code_cell(
            r"""baseline_rows = [{
    "method": "full-p90 SPI–SPI PC1",
    "overall |rho|": abs(full_summary["primary_association"]["overall_spearman"]),
    "within-kappa |rho|": abs(full_summary["primary_association"]["within_kappa_spearman"]),
}]
for name, values in full_summary["baseline_associations"].items():
    baseline_rows.append({
        "method": name.replace("_", " "),
        "overall |rho|": abs(values["overall_spearman"]),
        "within-kappa |rho|": abs(values["within_kappa_spearman"]),
    })
display(pd.DataFrame(baseline_rows).set_index("method").round(3))
display(pd.DataFrame(full_summary["source_stability"]).T.round(3))
print(
    "Interpretation: all 289 SPIs entered the feature construction; only "
    "development-fitted meta-feature validity and variance gates were applied. "
    "Because terminal outcomes had already been disclosed, this is a retrospective "
    "full-catalogue recovery result rather than a new prospective confirmation."
)"""
        ),
        nbformat.v4.new_markdown_cell(
            r"""---

## Historical prospective restricted-core assay

Everything below this divider is retained for provenance. It used 164 deliberately non-phase SPIs, two frequency laws, and partial observation of a hidden $N=256$ population. It is no longer the primary workflow."""
        ),
    ]
    cells = full_catalogue_cells + historical_cells
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
