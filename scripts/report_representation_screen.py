"""Render Stage A learning curves and a compact descriptive report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def report(path: Path) -> None:
    result = json.loads(path.read_text())
    root = path.parent
    config = result["protocol"]
    budgets = config["labelled_realizations_per_class"]
    views = config["representations"]
    colors = dict(zip(views, ["#777777", "#31688e", "#35b779", "#d99626", "#7b3294", "#b2182b"]))
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "mathtext.fontset": "cm", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": False,
                         "legend.frameon": False, "figure.dpi": 180})
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1), constrained_layout=True)
    for ax, name, title in zip(axes, ["same_cell", "both_M_and_T_changed"],
                               ["Same observation dimensions", "Both M and T changed"]):
        for view in views:
            scores = result["summary"][name]["scores"][view]
            y = [scores[str(n)]["balanced_accuracy"] for n in budgets]
            ci = np.asarray([scores[str(n)]["conditional_ci95"] for n in budgets])
            ax.plot(budgets, y, "o-", lw=1.7, ms=3, color=colors[view], label=view)
            ax.fill_between(budgets, ci[:, 0], ci[:, 1], color=colors[view], alpha=.12)
        ax.set(title=title, xlabel="Labelled source realizations per class", ylabel="Balanced accuracy")
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets, labels=[str(n) for n in budgets])
        ax.set_ylim(0, 1.025)
        ax.axhline(1 / 14, color="#999999", lw=.8, ls=":")
    axes[1].legend(fontsize=8, ncol=2, loc="lower left", bbox_to_anchor=(.02, .14))
    fig.savefig(root / "learning-curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(root / "learning-curves.svg", bbox_inches="tight")
    plt.close(fig)
    primary = result["summary"]["both_M_and_T_changed"]
    lines = ["# Stage A: exploratory p90 representation screen", "",
             f"Run status: `{config['status']}`. Training-derived clipping: `{config['preprocessing'].get('clip_standard_deviations', 'none')}` standard deviations. The clipped run is a post-initial-screen sensitivity; the original results remain separate.", "",
             "Training uses M=16,T=1000, instances 0–9; evaluation uses historical instances 10–29 across nine observation cells. No new independent confirmation or neural comparator is included.", "",
             "The primary endpoint changes both M and T: four cells, 1,120 rows, grouped into 280 class/instance units for resampling. Means average five nested-training-subset seeds. Intervals resample evaluation groups and are conditional on the fitted models; seed variation is reported separately in results.json.", "",
             "## Primary learning curve", "", "| Representation | " + " | ".join(f"n={n}/class" for n in budgets) + " |", "|---|" + "---|" * len(budgets)]
    for view in views:
        cells = []
        for n in budgets:
            s = primary["scores"][view][str(n)]
            lo, hi = s["conditional_ci95"]
            cells.append(f"{s['balanced_accuracy']:.3f} [{lo:.3f}, {hi:.3f}]")
        lines.append("| " + view + " | " + " | ".join(cells) + " |")
    lines += ["", "## Paired differences", "", "| Comparison | " + " | ".join(f"n={n}/class" for n in budgets) + " |", "|---|" + "---|" * len(budgets)]
    for comparison, values in primary["paired_differences"].items():
        cells = []
        for n in budgets:
            d = values[str(n)]
            lo, hi = d["conditional_ci95"]
            cells.append(f"{d['difference']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        lines.append("| " + comparison + " | " + " | ".join(cells) + " |")
    lines += ["", "## Normalized area under the learning curve", "",
              "Trapezoidal area versus log(label budget), normalized to the budget interval. This summarizes all three reported budgets rather than selecting one.", "",
              "| Representation | Area [conditional 95% interval] |", "|---|---|"]
    for view, value in primary["learning_curve_area"].items():
        lo, hi = value["conditional_ci95"]
        lines.append(f"| {view} | {value['normalized_area_vs_log_labels']:.3f} [{lo:.3f}, {hi:.3f}] |")
    lines += ["", "## Class-level failure analysis at the largest label budget", "",
              "Joint-shift mean accuracy, averaged across the four cells and five fitted subsets.", "",
              "| Class | " + " | ".join(views) + " |", "|---|" + "---|" * len(views)]
    classes = sorted(set(r["class"] for r in result["per_class_cell"]))
    for label in classes:
        values = []
        for view in views:
            selected = [r["accuracy"] for r in result["per_class_cell"] if r["class"] == label and r["view"] == view and r["n"] == max(budgets) and r["M"] != 16 and r["T"] != 1000]
            values.append(f"{np.mean(selected):.3f}")
        lines.append("| " + label + " | " + " | ".join(values) + " |")
    lines += ["", "## Interpretation limits", "",
              "- m: full-p90 marginal summaries; z: ordered-edge Pearson; g: weighted-graph summaries; validity: per-SPI finite/nonconstant indicators.",
              "- u (when present): the repository's existing 82-feature raw baseline, combining pooled single-channel temporal/marginal summaries and pairwise correlation/lag-one/spectral summaries. It is not a neural encoder.",
              "- Same PCA dimension cap and linear-head tuning grid; m/g standardized, z centred, each block balanced to unit training variance. This is one explicit learner/preprocessing contract, not each representation's best possible performance.",
              "- All imputation, feature selection, PCA and C tuning use only the sampled labelled training recordings. Evaluation records do not fit transforms or choose hyperparameters.",
              "- Varying M in these historical generators can change the physical system. This is not a sensor-subsampling experiment.",
              "- Bootstrap intervals are descriptive, unadjusted and conditional on the fitted training subsets. They do not establish a confirmatory superiority claim or generalization to new generator families.",
              "- Reported runtime is for fitting/evaluation on cached features. It excludes the historical p90 computation, feature extraction and data transfer; it is not an end-to-end compute comparison with raw summaries.",
              f"- Total convergence warnings: {result['convergence_warning_count']}. Runtime: {result['elapsed_seconds'] / 60:.1f} min.", "",
              "Artifacts: results.json, predictions.npz, splits.json, learning-curves.svg/png."]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    print(root / "report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    report(parser.parse_args().results)
