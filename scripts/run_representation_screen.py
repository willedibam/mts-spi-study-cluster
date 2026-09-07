"""Run the prespecified exploratory catalogue-matched Stage A screen."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score, log_loss
from threadpoolctl import threadpool_limits

from src.representation_screen import (
    bootstrap_group_means, classifier, evaluation_cells, fit_view, select_c, training_subsets,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(config_path: Path, bank_path: Path, output: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    if (output / "results.json").exists():
        raise FileExistsError("completed result exists; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with np.load(bank_path, allow_pickle=False) as archive:
        bank = {name: archive[f"X_{name}"] for name in ("m", "g", "z", "validity")}
        metadata = {name: archive[name] for name in ("label", "M", "T", "instance", "role", "group", "row_id")}
        if archive["feature_contract"].item() != config["feature_contract"]:
            raise ValueError("wrong feature contract")
        manifest_digest = archive["manifest_sha256"].item()
    # The extraction manifest freezes the exact protocol before evaluation.
    report = json.loads(bank_path.with_suffix(".json").read_text())
    if report["artifact_sha256"] != sha(bank_path) or report["manifest_sha256"] != manifest_digest:
        raise ValueError("bank provenance mismatch")
    if report["protocol_sha256"] != sha(config_path):
        # Explicit analysis sensitivities may reuse the frozen bank and splits.
        base_path = Path(config.get("base_protocol", ""))
        if not base_path.is_file() or sha(base_path) != report["protocol_sha256"]:
            raise ValueError("protocol changed since bank preparation")
        base = yaml.safe_load(base_path.read_text())
        for key in base:
            if key not in ("study_id", "status", "preprocessing", "representations", "evaluation") and config.get(key) != base[key]:
                raise ValueError(f"preprocessing variant changed {key}")
        for key in base["evaluation"]:
            if key != "paired_comparisons" and config["evaluation"].get(key) != base["evaluation"][key]:
                raise ValueError(f"analysis variant changed evaluation {key}")
    extra_input = None
    if config.get("raw_controls"):
        path = Path(config["raw_controls"])
        raw_report = json.loads(path.with_suffix(".json").read_text())
        if raw_report["artifact_sha256"] != sha(path) or raw_report["manifest_sha256"] != manifest_digest:
            raise ValueError("raw control bank provenance mismatch")
        with np.load(path, allow_pickle=False) as raw:
            np.testing.assert_array_equal(raw["row_id"], metadata["row_id"])
            bank["u"] = raw["X_u"]
        extra_input = {"path": str(path), "sha256": raw_report["artifact_sha256"]}
    labels = metadata["label"]
    pool = np.flatnonzero(metadata["role"] == "training_pool")
    evaluation = np.flatnonzero(metadata["role"] == "evaluation")
    if set(metadata["group"][pool]) & set(metadata["group"][evaluation]):
        raise ValueError("training/evaluation group leakage")
    source = config["source_cell"]
    if not np.all((metadata["M"][pool] == source["M"]) & (metadata["T"][pool] == source["T"])):
        raise ValueError("training pool contains other observation cells")
    if len(set(metadata["group"][pool])) != len(pool):
        raise ValueError("training pool repeats independent groups")
    classes = np.unique(labels[pool])
    if len(set(metadata["row_id"])) != len(labels):
        raise ValueError("duplicate row identifiers")
    if set(classes) != set(labels[evaluation]):
        raise ValueError("evaluation class mismatch")
    if set(metadata["instance"][pool]) != set(config["training_instances"]) or set(metadata["instance"][evaluation]) != set(config["evaluation_instances"]):
        raise ValueError("instance split differs from protocol")
    for label in classes:
        expected = {(m, t, i) for m in config["M_values"] for t in config["T_values"] for i in config["evaluation_instances"]}
        rows = evaluation[labels[evaluation] == label]
        actual = set(zip(metadata["M"][rows].tolist(), metadata["T"][rows].tolist(), metadata["instance"][rows].tolist()))
        if actual != expected or len(rows) != len(expected):
            raise ValueError(f"incomplete evaluation grid: {label}")
        rows = pool[labels[pool] == label]
        if len(rows) != len(config["training_instances"]) or set(metadata["instance"][rows]) != set(config["training_instances"]):
            raise ValueError(f"incomplete source training pool: {label}")
    views, budgets, seeds = config["representations"], config["labelled_realizations_per_class"], config["subset_seeds"]
    cells = evaluation_cells(metadata["M"][evaluation], metadata["T"][evaluation], source)
    predicted = np.empty((len(views), len(budgets), len(seeds), len(evaluation)), dtype=np.int16)
    probabilities = np.empty(predicted.shape + (len(classes),), dtype=np.float32)
    class_index = {label: i for i, label in enumerate(classes)}
    truth = np.asarray([class_index[label] for label in labels[evaluation]])
    fits, split_records = [], []
    for si, seed in enumerate(seeds):
        subsets = training_subsets(labels, pool, budgets, seed)
        for ni, n in enumerate(budgets):
            train = subsets[n]
            split_records.append({"seed": seed, "labelled_per_class": n,
                                  "row_ids": metadata["row_id"][train].tolist()})
            for vi, view in enumerate(views):
                tick = time.monotonic()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    c, cv_scores = select_c(bank, view, train, labels, config["preprocessing"], config["classifier"], seed)
                    transformer, train_scores = fit_view(bank, view, train, config["preprocessing"])
                    model = classifier(train_scores, labels[train], c, config["classifier"])
                    test_scores = transformer.transform(bank, evaluation)
                    prediction = model.predict(test_scores)
                    probability = model.predict_proba(test_scores)
                np.testing.assert_array_equal(model.classes_, classes)
                predicted[vi, ni, si] = [class_index[label] for label in prediction]
                probabilities[vi, ni, si] = probability
                metrics = {name: {"balanced_accuracy": float(balanced_accuracy_score(labels[evaluation][mask], prediction[mask])),
                                  "log_loss": float(log_loss(labels[evaluation][mask], probability[mask], labels=classes))}
                           for name, mask in cells.items()}
                fit = {"view": view, "labelled_per_class": n, "seed": seed, "C": c,
                       "inner_cv": cv_scores, "retained_features": {b.name: len(b.keep) for b in transformer.blocks},
                       "pca_dimensions": train_scores.shape[1], "seconds": time.monotonic() - tick,
                       "convergence_warnings": [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)],
                       "metrics": metrics}
                fits.append(fit)
                print(f"n={n} seed={seed} {view}: joint-shift BA={metrics['both_M_and_T_changed']['balanced_accuracy']:.4f} ({fit['seconds']:.1f}s)", flush=True)
    correct = predicted == truth[None, None, None, :]
    summary = {}
    for cell_name, mask in cells.items():
        groups = np.unique(metadata["group"][evaluation][mask])
        grouped = np.stack([correct[..., mask & (metadata["group"][evaluation] == group)].mean(axis=-1).mean(axis=-1) for group in groups], axis=-1)
        # shape: views, budgets, independent evaluation groups. Average training
        # subsets before bootstrapping; overlapping subsets are not new datasets.
        group_labels = np.asarray([labels[evaluation][np.flatnonzero(metadata["group"][evaluation] == group)[0]] for group in groups])
        draws = bootstrap_group_means(grouped, group_labels, config["evaluation"]["bootstrap_repetitions"], config["evaluation"]["bootstrap_seed"])
        scores = {}
        for vi, view in enumerate(views):
            scores[view] = {str(n): {"balanced_accuracy": float(grouped[vi, ni].mean()),
                                    "conditional_ci95": np.quantile(draws[vi, ni], [.025, .975]).tolist(),
                                    "subset_seed_range": [float(x) for x in (correct[vi, ni][:, mask].mean(axis=-1).min(), correct[vi, ni][:, mask].mean(axis=-1).max())]}
                           for ni, n in enumerate(budgets)}
        differences = {}
        for left, right in config["evaluation"]["paired_comparisons"]:
            li, ri = views.index(left), views.index(right)
            differences[f"{left} minus {right}"] = {str(n): {"difference": float((grouped[li, ni] - grouped[ri, ni]).mean()),
                                                             "conditional_ci95": np.quantile(draws[li, ni] - draws[ri, ni], [.025, .975]).tolist()}
                                                      for ni, n in enumerate(budgets)}
        widths = np.diff(np.log(budgets))
        area_draws = np.sum((draws[:, 1:] + draws[:, :-1]) * widths[None, :, None] / 2, axis=1) / widths.sum()
        mean_scores = grouped.mean(axis=-1)
        areas = np.sum((mean_scores[:, 1:] + mean_scores[:, :-1]) * widths[None, :] / 2, axis=1) / widths.sum()
        curve_summary = {view: {"normalized_area_vs_log_labels": float(areas[vi]),
                                "conditional_ci95": np.quantile(area_draws[vi], [.025, .975]).tolist()}
                         for vi, view in enumerate(views)}
        curve_differences = {f"{left} minus {right}": {
            "difference": float(areas[views.index(left)] - areas[views.index(right)]),
            "conditional_ci95": np.quantile(area_draws[views.index(left)] - area_draws[views.index(right)], [.025, .975]).tolist()}
            for left, right in config["evaluation"]["paired_comparisons"]}
        summary[cell_name] = {"rows": int(mask.sum()), "groups": len(groups), "scores": scores,
                              "paired_differences": differences, "learning_curve_area": curve_summary,
                              "paired_learning_curve_area_differences": curve_differences}
    per_class_cell = []
    for vi, view in enumerate(views):
        for ni, n in enumerate(budgets):
            for label in classes:
                for m in config["M_values"]:
                    for t in config["T_values"]:
                        mask = (labels[evaluation] == label) & (metadata["M"][evaluation] == m) & (metadata["T"][evaluation] == t)
                        per_class_cell.append({"view": view, "n": n, "class": str(label), "M": m, "T": t,
                                               "accuracy": float(correct[vi, ni][:, mask].mean())})
    np.savez_compressed(output / "predictions.npz", predicted=predicted, probabilities=probabilities,
                        truth=truth, classes=classes, views=views, budgets=budgets, seeds=seeds,
                        **{name: values[evaluation] for name, values in metadata.items()})
    (output / "splits.json").write_text(json.dumps({"training": split_records, "evaluation_row_ids": metadata["row_id"][evaluation].tolist()}, indent=2) + "\n")
    result = {"status": "exploratory_screen_complete", "protocol": config,
              "raw_control_input": extra_input,
              "config_sha256": sha(config_path), "bank_sha256": report["artifact_sha256"],
              "runner_sha256": sha(Path(__file__)), "module_sha256": sha(Path("src/representation_screen.py")),
              "predictions_sha256": sha(output / "predictions.npz"), "splits_sha256": sha(output / "splits.json"),
              "elapsed_seconds": time.monotonic() - started,
              "convergence_warning_count": sum(len(f["convergence_warnings"]) for f in fits),
              "summary": summary, "fits": fits, "per_class_cell": per_class_cell}
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {output / 'results.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    with threadpool_limits(limits=args.threads):
        run(args.config, args.bank, args.output)


if __name__ == "__main__":
    main()
