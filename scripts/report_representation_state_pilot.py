"""Paired, master-grouped learning curves for completed Stage B fits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
import yaml

from src.representation_screen import evaluation_cells, bootstrap_group_means
from src.representation_state_data import file_hash, load_state_data


def report(config_path, data_root, result_roots, output):
    protocol = yaml.safe_load(config_path.read_text())
    manifest, _ = load_state_data(data_root, config_path)
    evaluation = [r for r in manifest["rows"] if r["role"] == "evaluation"]
    row_ids = [r["row_id"] for r in evaluation]
    groups = np.asarray([r["master_id"] for r in evaluation])
    target = np.asarray([r["target"] for r in evaluation])
    strata = np.asarray([r["coupling_index"] for r in evaluation])
    cells = evaluation_cells(np.asarray([r["M"] for r in evaluation]), np.asarray([r["T"] for r in evaluation]),
                             protocol["observations"]["source"])
    seeds = protocol["methods"]["subset_seeds"]
    budgets = protocol["sampling"]["labelled_training_masters_per_coupling"]
    predictions, metadata = {}, {}
    for root in result_roots:
        for path in sorted(root.glob("*.json")):
            meta = json.loads(path.read_text())
            if "identity" not in meta:
                continue
            if meta["identity"]["protocol_sha256"] != file_hash(config_path):
                raise ValueError(f"mixed protocols: {path}")
            method = meta["identity"]["method"]
            key = (method, meta["identity"]["n_per_coupling"], meta["identity"]["seed"])
            if key in predictions:
                raise ValueError(f"duplicate fit: {key}")
            archive = path.with_suffix(".npz")
            if file_hash(archive) != meta["predictions_sha256"]:
                raise ValueError(f"prediction hash mismatch: {path}")
            with np.load(archive, allow_pickle=False) as bank:
                assert bank["row_id"].tolist() == row_ids
                np.testing.assert_array_equal(bank["target"], target)
                predictions[key] = bank["prediction"]
            metadata[key] = meta
    methods = sorted({key[0] for key in predictions})
    for method in methods:
        if any((method, n, seed) not in predictions for n in budgets for seed in seeds):
            raise ValueError(f"incomplete learning curve for {method}; do not average unequal seed sets")
    if not methods:
        raise ValueError("no completed fits")
    errors = np.asarray([[[np.abs(predictions[method, n, seed] - target) for seed in seeds]
                          for n in budgets] for method in methods])
    summary, group_errors = {}, {}
    for cell, mask in cells.items():
        units = np.unique(groups[mask])
        values = np.stack([errors[..., mask & (groups == group)].mean(axis=(-1, -2)) for group in units], axis=-1)
        group_errors[cell] = values
        unit_strata = np.asarray([strata[np.flatnonzero(groups == group)[0]] for group in units])
        boot = bootstrap_group_means(values, unit_strata, 2000, 260907)
        summary[cell] = {method: {"MAE": values[i].mean(axis=-1).tolist(),
                                  "conditional_95_CI": np.quantile(boot[i], [.025, .975], axis=-1).T.tolist()}
                         for i, method in enumerate(methods)}
    primary = "both_M_and_T_changed"
    comparisons = {}
    pairs = [("z", "m"), ("z", "observables"), ("z", "neural"), ("neural", "observables"),
             ("m+z", "m"), ("m+g+z", "m+g"), ("random_encoder", "observables"),
             ("neural", "random_encoder"), ("z", "random_encoder")]
    units = np.unique(groups)
    unit_strata = np.asarray([strata[np.flatnonzero(groups == group)[0]] for group in units])
    for left, right in pairs:
        if left not in methods or right not in methods:
            continue
        difference = group_errors[primary][methods.index(left)] - group_errors[primary][methods.index(right)]
        boot = bootstrap_group_means(difference, unit_strata, 2000, 260907)
        comparisons[f"{left}_minus_{right}"] = {
            "MAE_difference": difference.mean(axis=-1).tolist(),
            "conditional_95_CI": np.quantile(boot, [.025, .975], axis=-1).T.tolist(),
            "negative_favours": left}
    per_coupling = {}
    for i, method in enumerate(methods):
        per_coupling[method] = {str(k): errors[i][..., cells[primary] & (strata == k)].mean(axis=(1, 2)).tolist()
                                for k in np.unique(strata)}
    results = {"protocol_sha256": file_hash(config_path), "data_manifest_sha256": file_hash(data_root / "manifest.json"),
               "methods": methods, "budgets_per_coupling": budgets, "labels_total": protocol["sampling"]["total_label_budgets"],
               "subset_seeds": seeds, "evaluation_masters": len(units), "summary": summary,
               "paired_primary_comparisons": comparisons, "per_coupling_primary_MAE": per_coupling,
               "fitting_seconds": {method: sum(m["seconds"] for key, m in metadata.items() if key[0] == method) for method in methods},
               "neural_optimization": [{"n": key[1], "seed": key[2], "training_MAE": meta["training_MAE"],
                                         "refit_epochs": meta["details"]["refit_epochs"],
                                         "selected_lr": meta["details"]["chosen_learning_rate"],
                                         "selected_weight_decay": meta["details"]["chosen_weight_decay"]}
                                        for key, meta in metadata.items() if key[0] == "neural"],
               "uncertainty": "Exploratory 95% paired bootstrap within coupling, resampling independent evaluation masters; averages views and five training subsets. Conditional on fitted models, not fresh training populations. No multiplicity adjustment.",
               "claim_boundary": protocol["claim_boundary"]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    # Keep the figure legible; the table and numeric artifact retain every model.
    display = [m for m in ("m", "z", "m+g+z", "observables", "phase_direct", "neural", "random_encoder", "validity") if m in methods]
    names = {"m": "SPI marginals", "z": "SPI–SPI z", "m+g+z": "Marginals + graphs + z",
             "observables": "Coherence + correlation", "phase_direct": "Coherence (no labels)",
             "neural": "Raw CNN + attention", "random_encoder": "Frozen random encoder + ridge", "validity": "SPI validity"}
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    x = results["labels_total"]
    for ax, cell, title in zip(axes, ("same_cell", "M_only_changed", primary),
                               ("Source observation shape", "Changed sensor count", "Changed count and duration"), strict=True):
        for method in display:
            value = summary[cell][method]
            line, = ax.plot(x, value["MAE"], marker="o", ms=3, label=names[method])
            lo, hi = np.asarray(value["conditional_95_CI"]).T
            ax.fill_between(x, lo, hi, color=line.get_color(), alpha=.08)
        ax.set(xscale="log", title=title, xlabel="Independent labelled realizations", xticks=x)
        ax.set_xticklabels(x)
        ax.xaxis.set_minor_locator(NullLocator())
    axes[0].set_ylabel("Mean absolute error (lower is better)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(output / f"learning-curves.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    lines = ["# Stage B: future coherence under observation changes", "",
             "Exploratory fixed-population Kuramoto pilot; 96 independent evaluation masters. Training uses only M16/T1000. "
             "The primary shift tests M8 and M32 at T500, ending at the same prediction time. Physical N remains 32.", "",
             "All tuned methods use identical 16/32/64-label budgets, including inner validation. "
             "Direct observables use no labels. Neural training starts from scratch; no pretraining or observation augmentation.", "",
             "| Method | 16 labels | 32 labels | 64 labels |", "|---|---:|---:|---:|"]
    for method in methods:
        vals = summary[primary][method]["MAE"]
        lines.append(f"| {method} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
    lines += ["", "![Learning curves](learning-curves.png)", "", results["uncertainty"], "",
              "The neural comparator is a small temporal CNN with aligned cross-channel attention and invariant pooling. "
              "Passing this comparison does not establish superiority over tuned pretrained time-series models or transformers generally.", "",
              "Physics supplies a strong simple coherence observable here. A result on this pilot alone does not motivate real label scarcity, "
              "cross-generator transfer, spatial coverage invariance, or a high-tier publication claim.", "",
              "Fitting timings exclude p90 extraction and data transfer; the feature bank records summed extraction CPU time separately. "
              "GPU/CPU timings and pretraining exposure must remain separate in cost comparisons.", ""]
    if "random_encoder" in methods:
        lines += ["The frozen random encoder is an exploratory addition made after seeing the initial neural results. "
                  "It uses the same initial encoder weights for each matched seed, extracts pooled features without training, "
                  "and fits the same training-only PCA/ridge procedure as the statistical features. No pretrained weights "
                  "or extra labels are used; extraction time is stored once per initialization separately from fitting time.", ""]
    (output / "report.md").write_text("\n".join(lines))
    print(json.dumps({"methods": methods, "primary": summary[primary]}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--results", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report(args.config, args.data, args.results, args.output)
