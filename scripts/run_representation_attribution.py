"""Exploratory Stage A factorial check: marginal richness and nonlinear readout."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import sklearn
import yaml

from src.representation_attribution import RICH_NAMES, rich_marginals, fit_head, select_head
from src.representation_screen import fit_view, training_subsets, evaluation_cells, bootstrap_group_means
from src.representation_state_data import file_hash
from src.run_external_corpus import _atomic_json, _atomic_savez


def extract(manifest_path, mirror, output):
    manifest = json.loads(manifest_path.read_text())
    identity = {"manifest_sha256": file_hash(manifest_path),
                "module_sha256": file_hash(Path("src/representation_attribution.py"))}
    if output.exists():
        previous = json.loads(output.with_suffix(".json").read_text())
        if previous["identity"] != identity or previous["artifact_sha256"] != file_hash(output):
            raise ValueError("rich feature cache identity mismatch")
        return
    values = []
    order = manifest["spi_order"]
    start = time.perf_counter()
    for row in manifest["records"]:
        path = mirror / row["dataset_path"].lstrip("/") / "spi_mpis.npz"
        if file_hash(path) != row["mpi_sha256"]:
            raise ValueError(f"MPI hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(order):
                raise ValueError("catalogue mismatch")
            mpis = {name: archive[name] for name in order}
        values.append(rich_marginals(mpis, order))
        if len(values) % 200 == 0:
            print(f"Rich marginals: {len(values)}/{len(manifest['records'])}", flush=True)
    _atomic_savez(output, {"X_m": np.asarray(values),
                          "row_id": np.asarray([r["row_id"] for r in manifest["records"]]),
                          "feature_names": np.asarray([f"{s}::{n}" for s in order for n in RICH_NAMES])})
    _atomic_json(output.with_suffix(".json"), {"identity": identity, "artifact_sha256": file_hash(output),
                                              "seconds": time.perf_counter() - start})


def run(config_path, output):
    if output.exists():
        raise FileExistsError(output)
    config = yaml.safe_load(config_path.read_text())
    base_path = Path(config["base_protocol"])
    if file_hash(base_path) != config["base_protocol_sha256"]:
        raise ValueError("base protocol changed")
    base = yaml.safe_load(base_path.read_text())
    bank_path = Path(config["bank"])
    bank_meta = json.loads(bank_path.with_suffix(".json").read_text())
    if file_hash(bank_path) != bank_meta["artifact_sha256"]:
        raise ValueError("base bank hash mismatch")
    manifest_path = Path(config["manifest"])
    if file_hash(manifest_path) != bank_meta["manifest_sha256"]:
        raise ValueError("source manifest mismatch")
    rich_path = Path(config["rich_bank"])
    extract(manifest_path, Path(config["mpi_mirror"]), rich_path)
    with np.load(bank_path, allow_pickle=False) as archive:
        bank = {key: archive[f"X_{key}"] for key in ("m", "z")}
        meta = {key: archive[key] for key in ("label", "role", "row_id", "M", "T", "group")}
    with np.load(rich_path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["row_id"], meta["row_id"])
        rich = archive["X_m"]
    pool = np.flatnonzero(meta["role"] == "training_pool")
    evaluation = np.flatnonzero(meta["role"] == "evaluation")
    if set(meta["group"][pool]) & set(meta["group"][evaluation]):
        raise ValueError("group leakage")
    classes, labels = np.unique(meta["label"], return_inverse=True)
    budgets, seeds = base["labelled_realizations_per_class"], base["subset_seeds"]
    methods = [(view, head) for view in config["views"] for head in config["heads"]]
    names = [f"{view}/{head}" for view, head in methods]
    predictions = np.empty((len(methods), len(budgets), len(seeds), len(evaluation)), dtype=np.int16)
    fits = []
    start = time.perf_counter()
    for si, seed in enumerate(seeds):
        for ni, (n, train) in enumerate(training_subsets(labels, pool, budgets, seed).items()):
            for vi, (view, head) in enumerate(methods):
                tick = time.perf_counter()
                active = {"m": rich if "rich" in view else bank["m"], "z": bank["z"]}
                block_view = view.replace("m_rich", "m")
                c, inner = select_head(active, block_view, train, labels, base["preprocessing"], base["classifier"], seed, head)
                transform, x = fit_view(active, block_view, train, base["preprocessing"])
                model = fit_head(x, labels[train], c, head, base["classifier"])
                predictions[vi, ni, si] = model.predict(transform.transform(active, evaluation))
                fits.append({"method": names[vi], "n": n, "seed": seed, "C": c, "inner": inner,
                             "train_indices": train.tolist(), "seconds": time.perf_counter() - tick})
                print(f"[DONE] {names[vi]} n={n} seed={seed}", flush=True)
    cells = evaluation_cells(meta["M"][evaluation], meta["T"][evaluation], base["source_cell"])
    correct = predictions == labels[evaluation]
    groups = meta["group"][evaluation]
    strata = labels[evaluation]
    summary, group_values = {}, {}
    for cell, mask in cells.items():
        units = np.unique(groups[mask])
        values = np.stack([correct[..., mask & (groups == unit)].mean(axis=(-1, -2)) for unit in units], axis=-1)
        unit_labels = np.asarray([strata[np.flatnonzero(groups == unit)[0]] for unit in units])
        boot = bootstrap_group_means(values, unit_labels, 2000, 1729)
        summary[cell] = {name: {"balanced_accuracy": values[i].mean(axis=-1).tolist(),
                                "conditional_95_CI": np.quantile(boot[i], [.025, .975], axis=-1).T.tolist()}
                         for i, name in enumerate(names)}
        group_values[cell] = values
    pairs = [(f"{a}/{h}", f"{b}/{h}") for h in config["heads"]
             for a, b in [("m_rich", "m"), ("z", "m_rich"), ("m_rich+z", "m_rich")]]
    pairs += [(f"{v}/rbf", f"{v}/logistic") for v in config["views"]]
    primary = "both_M_and_T_changed"
    units = np.unique(groups)
    unit_labels = np.asarray([strata[np.flatnonzero(groups == unit)[0]] for unit in units])
    comparisons = {}
    for left, right in pairs:
        values = group_values[primary][names.index(left)] - group_values[primary][names.index(right)]
        boot = bootstrap_group_means(values, unit_labels, 2000, 1729)
        comparisons[f"{left}_minus_{right}"] = {"difference": values.mean(axis=-1).tolist(),
                                                "conditional_95_CI": np.quantile(boot, [.025, .975], axis=-1).T.tolist()}
    per_class = {name: {str(cls): correct[i][..., cells[primary] & (strata == j)].mean(axis=(1, 2)).tolist()
                       for j, cls in enumerate(classes)} for i, name in enumerate(names)}
    output.mkdir(parents=True)
    _atomic_savez(output / "predictions.npz", {"prediction": predictions, "target": labels[evaluation],
                    "row_id": meta["row_id"][evaluation], "classes": classes, "methods": np.asarray(names)})
    result = {"status": "exploratory_post_stage_b_attribution", "protocol_sha256": file_hash(config_path),
              "bank_sha256": file_hash(bank_path), "rich_bank_sha256": file_hash(rich_path),
              "code_sha256": {p: file_hash(Path(p)) for p in [__file__, "src/representation_attribution.py", "src/representation_screen.py"]},
              "numpy": np.__version__, "sklearn": sklearn.__version__, "seconds": time.perf_counter() - start,
              "predictions_sha256": file_hash(output / "predictions.npz"), "summary": summary,
              "paired_primary": comparisons, "per_class_primary": per_class, "fits": fits,
              "labelled_per_class": budgets, "subset_seeds": seeds}
    _atomic_json(output / "results.json", result)
    lines = ["# Stage A: marginal richness and nonlinear readout", "",
             "Exploratory reuse of inspected data. All methods use the same training-only clipping/PCA32, "
             "two-fold selection and five C values. RBF uses fixed gamma='scale'; this is one bounded "
             "nonlinear comparator, not an exhaustive marginal learner. No extra labels or pretraining.", "",
             "Rich marginals contain mean, standard deviation, skewness, Pearson kurtosis and 19 quantiles per SPI (23 total). "
             "They preserve no cross-SPI edge alignment. Primary shift includes physical-size changes, not only sensor subsampling.", "",
             "| Method | 2/class | 4/class | 8/class |", "|---|---:|---:|---:|"]
    for name in names:
        lines.append(f"| {name} | " + " | ".join(f"{v:.4f}" for v in summary[primary][name]["balanced_accuracy"]) + " |")
    lines += ["", "Intervals in results.json are paired class-stratified bootstrap intervals over independent "
              "class-instance groups, averaging observation cells and fitted subsets. They condition on fitted "
              "models and have no multiplicity adjustment. A residual z gain cannot establish its mechanism: "
              "these finite marginal descriptors and the PCA/readout still discard information.", ""]
    (output / "report.md").write_text("\n".join(lines))
    print(json.dumps(summary[primary], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)
