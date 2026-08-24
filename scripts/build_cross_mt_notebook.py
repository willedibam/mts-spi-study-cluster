#!/usr/bin/env python3
"""Build the dated, reproducible successor to proof_p90_260712.ipynb."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _cell(cell_type: str, source: str) -> dict:
    cell = {
        "cell_type": cell_type,
        "id": hashlib.sha256(f"{cell_type}\0{source}".encode()).hexdigest()[:12],
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def build_notebook(output: str | Path) -> None:
    cells = [
        _cell(
            "markdown",
            """# Frozen SPI–SPI transfer across heterogeneous $M$ and $T$

This is the dated successor to `proof_p90_260712.ipynb`. It evaluates an analysis frozen on instances 0–9 and applied without refitting to instances 10–29. Quantitative held-out tests are primary; PCA and UMAP are illustrative. The study tests transfer across observation dimensions, not mathematical invariance.
""",
        ),
        _cell(
            "code",
            """from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

ROOT = Path.cwd().resolve()
while not (ROOT / "src").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
RESULT_PATH = ROOT / "results/cross_mt_transfer_260824/confirmation-results.json"
COORDINATE_PATH = ROOT / "results/cross_mt_transfer_260824/confirmation-coordinates.npz"
MANIFEST_PATH = ROOT / "results/cross_mt_transfer_260824/development-manifest.json"

result = json.loads(RESULT_PATH.read_text())
manifest = json.loads(MANIFEST_PATH.read_text())
coordinates = np.load(COORDINATE_PATH, allow_pickle=False)
assert result["status"] == "confirmation_evaluated"
assert manifest["status"] == "development_frozen_confirmation_unseen"
print(f"development={manifest['rows']:,}; confirmation={result['rows']:,}; schema={manifest['schema']['schema_sha256'][:12]}…")
""",
        ),
        _cell(
            "markdown",
            r"""## Frozen design

- 14 classes; $M\in\{8,16,32\}$; $T\in\{500,1000,2000\}$.
- Each fold excludes one complete $(M,T)$ cell from development, then evaluates independent confirmation instances in that cell.
- Primary representation: $z_{\mathrm{sym}}$. Prespecified sensitivities: $z_{\mathrm{dir}}$ and block-balanced $[z_{\mathrm{sym}},z_{\mathrm{dir}}]$.
- Baselines: fixed-dimensional pooled univariate summaries (50), dependence summaries (32), and their concatenation (82).
- All validity filtering, median imputation, centring/scaling, PCA and classifiers were fitted on development/training rows only.
""",
        ),
        _cell(
            "code",
            r"""display_names = {
    "sym": r"$z_{\mathrm{sym}}$ (primary)",
    "dir": r"$z_{\mathrm{dir}}$",
    "augmented_balanced": r"balanced $z_{\mathrm{aug}}$",
    "pooled_univariate": "pooled univariate",
    "pooled_dependence": "pooled dependence",
    "pooled_combined": "pooled combined",
}
names = [
    "sym", "dir", "augmented_balanced",
    "pooled_univariate", "pooled_dependence", "pooled_combined",
]
rows = []
for name in names:
    values = result["results"][name]
    rows.append({
        "representation": display_names[name],
        "balanced_accuracy": values["classification"]["balanced_accuracy"]["estimate"],
        "BA_95%_CI": values["classification"]["balanced_accuracy"]["confidence_interval"],
        "hard_retrieval_mAP": values["retrieval"]["both_M_and_T_changed"]["mean_average_precision"]["estimate"],
        "mAP_95%_CI": values["retrieval"]["both_M_and_T_changed"]["mean_average_precision"]["confidence_interval"],
        "joint_cell_leakage": values["size_leakage"]["cell"]["estimate"],
        "class_eta2": values["geometry"]["class_eta_squared"],
        "cell_eta2": values["geometry"]["cell_eta_squared"],
    })
summary = pd.DataFrame(rows).set_index("representation")
summary
""",
        ),
        _cell(
            "code",
            r'''primary = result["results"]["sym"]
ba = primary["classification"]["balanced_accuracy"]
hard_map = primary["retrieval"]["both_M_and_T_changed"]["mean_average_precision"]
hard_null = primary["retrieval"]["both_M_and_T_changed"]["random_ranking_null"]["mean_average_precision"]
cell_leak = primary["size_leakage"]["cell"]
minimum_ba = min(row["balanced_accuracy"] for row in primary["folds"].values())
minimum_map = min(row["hard_mean_average_precision"] for row in primary["folds"].values())
best_baseline_ba = max(
    result["results"][name]["classification"]["balanced_accuracy"]["estimate"]
    for name in names[3:]
)
best_baseline_map = max(
    result["results"][name]["retrieval"]["both_M_and_T_changed"]["mean_average_precision"]["estimate"]
    for name in names[3:]
)

display(Markdown(
    f"**Held-out result.** $z_{{\\mathrm{{sym}}}}$ achieved balanced accuracy "
    f"{ba['estimate']:.3f} (95% CI {ba['confidence_interval'][0]:.3f}–{ba['confidence_interval'][1]:.3f}; "
    f"permutation $p={ba['permutation_p_value']:.3g}$) and hard cross-size mAP {hard_map['estimate']:.3f} "
    f"(95% CI {hard_map['confidence_interval'][0]:.3f}–{hard_map['confidence_interval'][1]:.3f}; "
    f"random-ranking expectation {hard_null:.3f}). The worst held cell retained accuracy {minimum_ba:.3f} "
    f"and mAP {minimum_map:.3f}. These estimates are descriptively above the best pooled baselines "
    f"({best_baseline_ba:.3f} accuracy; {best_baseline_map:.3f} mAP), but the frozen protocol did not "
    f"include paired baseline-difference inference, so this is not a formal superiority claim. "
    f"Joint $(M,T)$ remained decodable within class at {cell_leak['estimate']:.3f} "
    f"(chance $1/9$; $p={cell_leak['permutation_p_value']:.3g}$): the evidence supports quantitative "
    f"cross-size transfer, not representation invariance."
))
''',
        ),
        _cell(
            "code",
            r"""x = np.arange(len(names))
fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), dpi=150)

def estimate_and_error(path):
    estimates, errors = [], []
    for name in names:
        node = result["results"][name]
        for key in path:
            node = node[key]
        estimates.append(node["estimate"])
        lo, hi = node["confidence_interval"]
        errors.append((node["estimate"] - lo, hi - node["estimate"]))
    return np.asarray(estimates), np.asarray(errors).T

ba, ba_err = estimate_and_error(["classification", "balanced_accuracy"])
hard_map, map_err = estimate_and_error(["retrieval", "both_M_and_T_changed", "mean_average_precision"])
leakage, leak_err = estimate_and_error(["size_leakage", "cell"])
class_eta = [result["results"][name]["geometry"]["class_eta_squared"] for name in names]
cell_eta = [result["results"][name]["geometry"]["cell_eta_squared"] for name in names]

axes[0].errorbar(x, ba, yerr=ba_err, fmt="o", capsize=3)
axes[0].axhline(1/14, color="0.5", ls="--", lw=1)
axes[0].set(title="Unseen-cell classification", ylabel="balanced accuracy")
axes[1].errorbar(x, hard_map, yerr=map_err, fmt="o", capsize=3)
axes[1].axhline(result["results"]["sym"]["retrieval"]["both_M_and_T_changed"]["random_ranking_null"]["mean_average_precision"], color="0.5", ls="--", lw=1)
axes[1].set(title="Both $M$ and $T$ changed", ylabel="retrieval mAP")
axes[2].errorbar(x, leakage, yerr=leak_err, fmt="o", capsize=3)
axes[2].axhline(1/9, color="0.5", ls="--", lw=1)
axes[2].set(title="Class-conditioned size leakage", ylabel="joint-cell accuracy")
width = 0.36
axes[3].bar(x-width/2, class_eta, width, label="class")
axes[3].bar(x+width/2, cell_eta, width, label="$(M,T)$ cell")
axes[3].set(title="Confirmation geometry", ylabel=r"$\eta^2$")
axes[3].legend(frameon=False)
for ax in axes:
    ax.set_xticks(x, [display_names[name] for name in names], rotation=50, ha="right")
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.show()
""",
        ),
        _cell(
            "markdown",
            r"""Dashed lines are chance/random-ranking references. High class accuracy and retrieval support transfer; high cell leakage or cell $\eta^2$ contradicts invariance. These quantities should be read together rather than selecting the most favourable panel.
""",
        ),
        _cell(
            "code",
            r"""folds = result["results"]["sym"]["folds"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=150)
for M in (8, 16, 32):
    cells = [f"M{M}_T{T}" for T in (500, 1000, 2000)]
    axes[0].plot((500, 1000, 2000), [folds[cell]["balanced_accuracy"] for cell in cells], "-o", label=f"M={M}")
    axes[1].plot((500, 1000, 2000), [folds[cell]["hard_mean_average_precision"] for cell in cells], "-o", label=f"M={M}")
axes[0].set(title=r"$z_{\mathrm{sym}}$ classification by held cell", ylabel="balanced accuracy")
axes[1].set(title=r"$z_{\mathrm{sym}}$ hard retrieval by held cell", ylabel="mAP")
for ax in axes:
    ax.set(xlabel="T", xscale="log")
    ax.set_xticks((500, 1000, 2000), ("500", "1000", "2000"))
    ax.minorticks_off()
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False)
fig.tight_layout()
plt.show()
""",
        ),
        _cell(
            "markdown",
            r"""## Shared-coordinate illustrations

PCA and UMAP below were fitted to development $z_{\mathrm{sym}}$ only; confirmation rows were transformed afterward. Their visual appearance is not used as evidence of invariance.
""",
        ),
        _cell(
            "code",
            """labels = coordinates["confirmation_y"].astype(str)
M = coordinates["confirmation_M"].astype(int)
T = coordinates["confirmation_T"].astype(int)
pca = coordinates["confirmation_pca_sym"]
umap_xy = coordinates["confirmation_umap_sym"]
classes = np.unique(labels)
cells = np.asarray([f"M{m}, T{t}" for m, t in zip(M, T)])
cell_names = np.unique(cells)
class_palette = dict(zip(classes, plt.cm.tab20(np.linspace(0, 1, len(classes)))))
cell_palette = dict(zip(cell_names, plt.cm.viridis(np.linspace(0, 1, len(cell_names)))))

fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=150)
for ax, xy, title in ((axes[0,0], pca, "PCA: class"), (axes[1,0], umap_xy, "UMAP: class")):
    for label in classes:
        member = labels == label
        ax.scatter(xy[member,0], xy[member,1], s=5, alpha=.22, color=class_palette[label], label=label)
    ax.set_title(title)
for ax, xy, title in ((axes[0,1], pca, "PCA: observation cell"), (axes[1,1], umap_xy, "UMAP: observation cell")):
    for cell in cell_names:
        member = cells == cell
        ax.scatter(xy[member,0], xy[member,1], s=5, alpha=.22, color=cell_palette[cell], label=cell)
    ax.set_title(title)
for ax in axes.ravel():
    ax.spines[["top", "right"]].set_visible(False)
axes[0,0].legend(bbox_to_anchor=(-.05, 1), loc="upper right", frameon=False, fontsize=7)
axes[0,1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=7)
fig.tight_layout()
plt.show()
""",
        ),
        _cell(
            "markdown",
            """## Interpretation boundary

The defensible conclusion is determined by confirmation performance relative to the prespecified pooled baselines, random-ranking/classification nulls, and measured size leakage. Successful classification alone is insufficient: it can coexist with a strong encoding of $M$ or $T$. Conversely, nonzero size leakage does not erase useful transfer, but rules out a strict invariance claim.
""",
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="notebooks/embeddings/proof_cross_mt_transfer_260824.ipynb",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_notebook(parse_args().output)
