"""Focused three-VAR-class diagnostic with fixed matched learning curves."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import sklearn
import yaml

from src.representation_mechanism import permute_dyads, linear_dynamics_features
from src.representation_attribution import fit_head, select_head
from src.representation_screen import fit_view, training_subsets, evaluation_cells, bootstrap_group_means
from src.representation_state_data import file_hash
from src.spi_spi_contract import build_unified_feature_values
from src.run_external_corpus import _atomic_json, _atomic_savez


def run(config_path, output):
    if output.exists():
        raise FileExistsError(output)
    config = yaml.safe_load(config_path.read_text())
    base_path = Path(config["base_protocol"])
    if file_hash(base_path) != config["base_protocol_sha256"]:
        raise ValueError("base protocol changed")
    base = yaml.safe_load(base_path.read_text())
    manifest_path = Path(config["manifest"])
    manifest = json.loads(manifest_path.read_text())
    full_rows = manifest["records"]
    chosen = np.asarray([i for i, r in enumerate(full_rows) if r["label"] in config["classes"]])
    rows = [full_rows[i] for i in chosen]
    ids = [r["row_id"] for r in rows]
    order = manifest["spi_order"]
    bank_path, rich_path = Path(config["bank"]), Path(config["rich_bank"])
    bm = json.loads(bank_path.with_suffix(".json").read_text())
    rm = json.loads(rich_path.with_suffix(".json").read_text())
    assert file_hash(bank_path) == bm["artifact_sha256"] and file_hash(rich_path) == rm["artifact_sha256"]
    assert file_hash(manifest_path) == bm["manifest_sha256"] == rm["identity"]["manifest_sha256"]
    with np.load(bank_path, allow_pickle=False) as a:
        np.testing.assert_array_equal(a["row_id"][chosen], ids)
        z = a["X_z"][chosen]
        validity = a["X_validity"][chosen]
    with np.load(rich_path, allow_pickle=False) as a:
        np.testing.assert_array_equal(a["row_id"][chosen], ids)
        rich = a["X_m"][chosen]
    raw_manifest_path = Path(config["raw_manifest"])
    raw_manifest = json.loads(raw_manifest_path.read_text())
    assert raw_manifest["manifest_sha256"] == file_hash(manifest_path)
    raw_sources = {r["row_id"]: r for r in raw_manifest["sources"]}
    nulls = [[] for _ in config["null_seeds"]]
    direct = []
    started = time.perf_counter()
    shared_max_difference = 0.0
    for j, row in enumerate(rows):
        path = Path(config["mpi_mirror"]) / row["dataset_path"].lstrip("/") / "spi_mpis.npz"
        assert file_hash(path) == row["mpi_sha256"]
        with np.load(path, allow_pickle=False) as a:
            mpis = {name: a[name] for name in order}
        observed_z = build_unified_feature_values(mpis, order)[0]
        np.testing.assert_allclose(observed_z, z[j], atol=1e-7, rtol=1e-6, equal_nan=True)
        shared = permute_dyads(mpis, order, np.random.default_rng([config["shared_seed"], int(chosen[j])]), shared=True)
        shared_z = build_unified_feature_values(shared, order)[0]
        np.testing.assert_allclose(shared_z, observed_z, atol=1e-7, rtol=1e-6, equal_nan=True)
        shared_max_difference = max(shared_max_difference, float(np.nanmax(abs(shared_z - observed_z))))
        for k, seed in enumerate(config["null_seeds"]):
            moved = permute_dyads(mpis, order, np.random.default_rng([seed, int(chosen[j])]))
            null = build_unified_feature_values(moved, order)[0]
            np.testing.assert_array_equal(np.isfinite(null), np.isfinite(z[j]))
            nulls[k].append(null)
        source = raw_sources[row["row_id"]]
        raw_path = Path(source["path"])
        assert file_hash(raw_path) == source["sha256"]
        raw = np.load(raw_path, allow_pickle=False)
        assert raw.shape == (row["T"], row["M"])
        direct.append(linear_dynamics_features(raw, config["ridge_fraction"]))
        if (j + 1) % 100 == 0:
            print(f"Extracted {j + 1}/{len(rows)}", flush=True)
    extraction_seconds = time.perf_counter() - started
    matrices = {"z": ("z", z), "m_rich": ("m", rich), "VAR": ("u", np.asarray(direct)),
                "validity": ("validity", validity)}
    matrices.update({f"z_null_{seed}": ("z", np.asarray(values)) for seed, values in zip(config["null_seeds"], nulls)})
    methods = list(matrices)
    output.mkdir(parents=True)
    _atomic_savez(output / "features.npz", {**{name: value for name, (_, value) in matrices.items()}, "row_id": np.asarray(ids)})
    full_pool = np.asarray([i for i, r in enumerate(full_rows) if r["role"] == "training_pool"])
    full_labels = np.asarray([r["label"] for r in full_rows])
    lookup = {int(global_index): local_index for local_index, global_index in enumerate(chosen)}
    classes, labels = np.unique([r["label"] for r in rows], return_inverse=True)
    evaluation = np.asarray([i for i, r in enumerate(rows) if r["role"] == "evaluation"])
    budgets, seeds = base["labelled_realizations_per_class"], base["subset_seeds"]
    predictions = np.empty((len(methods), len(budgets), len(seeds), len(evaluation)), dtype=np.int16)
    fits = []
    for si, seed in enumerate(seeds):
        # Filter the original full-screen subsets; do not draw easier new ones.
        for ni, (n, full_train) in enumerate(training_subsets(full_labels, full_pool, budgets, seed).items()):
            train = np.asarray([lookup[int(i)] for i in full_train if int(i) in lookup])
            assert len(train) == n * len(classes) and not set(train) & set(evaluation)
            for vi, name in enumerate(methods):
                view, matrix = matrices[name]
                bank = {view: matrix}
                c, inner = select_head(bank, view, train, labels, base["preprocessing"], base["classifier"], seed, "logistic")
                transform, x = fit_view(bank, view, train, base["preprocessing"])
                model = fit_head(x, labels[train], c, "logistic", base["classifier"])
                predictions[vi, ni, si] = model.predict(transform.transform(bank, evaluation))
                fits.append({"method": name, "n": n, "seed": seed, "C": c, "inner": inner,
                             "train_row_ids": [ids[i] for i in train]})
            print(f"Fitted n={n} seed={seed}", flush=True)
    groups = np.asarray([rows[i]["group"] for i in evaluation])
    strata = labels[evaluation]
    correct = predictions == strata
    null_positions = [i for i, name in enumerate(methods) if name.startswith("z_null_")]
    # Mean performance of independent null runs, not a prediction ensemble.
    correct = np.concatenate([correct, correct[null_positions].mean(axis=0, keepdims=True)], axis=0)
    names = [*methods, "z_null_mean"]
    cells = evaluation_cells(np.asarray([rows[i]["M"] for i in evaluation]),
                             np.asarray([rows[i]["T"] for i in evaluation]), base["source_cell"])
    summary, group_values = {}, {}
    units = np.unique(groups)
    unit_labels = np.asarray([strata[np.flatnonzero(groups == unit)[0]] for unit in units])
    for cell, mask in cells.items():
        values = np.stack([correct[..., mask & (groups == unit)].mean(axis=(-1, -2)) for unit in units], axis=-1)
        boot = bootstrap_group_means(values, unit_labels, 2000, 1729)
        summary[cell] = {name: {"balanced_accuracy": values[i].mean(axis=-1).tolist(),
                                "conditional_95_CI": np.quantile(boot[i], [.025, .975], axis=-1).T.tolist()}
                         for i, name in enumerate(names)}
        group_values[cell] = values
    primary = "both_M_and_T_changed"
    comparisons = {}
    for left, right in [("z", "z_null_mean"), ("z", "m_rich"), ("VAR", "z"), ("z_null_mean", "validity")]:
        diff = group_values[primary][names.index(left)] - group_values[primary][names.index(right)]
        boot = bootstrap_group_means(diff, unit_labels, 2000, 1729)
        comparisons[f"{left}_minus_{right}"] = {"difference": diff.mean(axis=-1).tolist(),
                                                "conditional_95_CI": np.quantile(boot, [.025, .975], axis=-1).T.tolist()}
    _atomic_savez(output / "predictions.npz", {"prediction": predictions, "methods": np.asarray(methods),
                  "target": strata, "row_id": np.asarray(ids)[evaluation], "classes": classes})
    result = {"status": "exploratory_three_class_mechanism_check", "summary": summary,
              "paired_primary": comparisons, "fits": fits, "labelled_per_class": budgets, "subset_seeds": seeds,
              "null_seeds": config["null_seeds"], "independent_evaluation_groups": len(units),
              "shared_permutation_max_abs_difference": shared_max_difference,
              "extraction_seconds": extraction_seconds, "total_seconds": time.perf_counter() - started,
              "numpy": np.__version__, "sklearn": sklearn.__version__,
              "hashes": {str(p): file_hash(Path(p)) for p in [config_path, manifest_path, raw_manifest_path, bank_path, rich_path,
                  __file__, "src/representation_mechanism.py", "src/representation_attribution.py", "src/representation_screen.py"]},
              "features_sha256": file_hash(output / "features.npz"), "predictions_sha256": file_hash(output / "predictions.npz")}
    _atomic_json(output / "results.json", result)
    lines = ["# Three-class VAR mechanism diagnostic", "",
             "All three existing VAR classes, including the previously difficult low-self-memory/moderate-coupling class. "
             "Every method is refit on the same filtered historical training subsets: 6/12/24 total labels. "
             "This is an exploratory three-way task, not directly comparable with the earlier 14-class accuracy.", "",
             "| Representation | 2/class | 4/class | 8/class |", "|---|---:|---:|---:|"]
    for name in ["m_rich", "z", "z_null_mean", "VAR", "validity"]:
        lines.append(f"| {name} | " + " | ".join(f"{x:.4f}" for x in summary[primary][name]["balanced_accuracy"]) + " |")
    lines += ["", "Primary shift changes physical M and recording T. The null is mean accuracy across three "
              "independently shuffled-and-refitted runs, not an ensemble; individual runs and paired conditional "
              "95% intervals are in results.json. Evaluation units are 60 class-instance groups, not edges or null draws.", "",
              "The null preserves marginals, reciprocity and feature validity. A loss implicates cross-SPI alignment "
              "but cannot distinguish useful statistical aliases from different physical mechanisms. VAR summaries "
              "use the correct model family; this is a diagnostic model-based reference, not a universal encoder.", ""]
    (output / "report.md").write_text("\n".join(lines))
    print(json.dumps(summary[primary], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)
