"""Training-only PCA-cap selection sensitivity for the completed state pilot."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
import yaml

from src.representation_screen import fit_view, training_subsets
from src.representation_state_data import file_hash, load_state_data
from src.run_external_corpus import _atomic_json, _atomic_savez


def select_cap(bank, view, train, target, strata, protocol, caps, seed):
    """All candidate fitting/selection is confined to the labelled training set."""
    alphas = sorted(protocol["methods"]["ridge_alpha_grid"], reverse=True)
    scores = {(cap, alpha): [] for cap in sorted(caps) for alpha in alphas}
    folds = []
    for fit, val in StratifiedKFold(2, shuffle=True, random_state=seed).split(train, strata[train]):
        for cap in sorted(caps):
            pre = {**protocol["methods"]["preprocessing"], "pca_dimensions": cap}
            transform, x = fit_view(bank, view, train[fit], pre)
            v = transform.transform(bank, train[val])
            for alpha in alphas:
                model = Ridge(alpha=alpha).fit(x, target[train[fit]])
                scores[cap, alpha].append(float(np.abs(np.clip(model.predict(v), 0, 1) - target[train[val]]).mean()))
        folds.append({"fit": train[fit].tolist(), "validation": train[val].tolist()})
    best = min(np.mean(s) for s in scores.values())
    # Prefer a smaller cap, then stronger ridge, for numerical ties. Caps can
    # yield identical effective dimensions when inner folds have few records.
    chosen = next(key for key, values in scores.items() if np.mean(values) <= best + 1e-12)
    return chosen, {"candidates": [{"cap": k[0], "alpha": k[1], "fold_MAE": v} for k, v in scores.items()], "folds": folds}


def run(config, output):
    if output.exists():
        raise FileExistsError(output)
    settings = yaml.safe_load(config.read_text())
    protocol_path = Path(settings["protocol"])
    if file_hash(protocol_path) != settings["protocol_sha256"]:
        raise ValueError("base protocol changed")
    protocol = yaml.safe_load(protocol_path.read_text())
    data = Path(settings["data"])
    manifest, _ = load_state_data(data, protocol_path)
    rows = manifest["rows"]
    pool = np.asarray([i for i, r in enumerate(rows) if r["role"] == "training_pool"])
    evaluation = np.asarray([i for i, r in enumerate(rows) if r["role"] == "evaluation"])
    target = np.asarray([r["target"] for r in rows])
    strata = np.asarray([r["coupling_index"] for r in rows])
    feature_path = Path(settings["bank"])
    provenance = json.loads(feature_path.with_suffix(".json").read_text())
    assert file_hash(feature_path) == provenance["artifact_sha256"]
    with np.load(feature_path, allow_pickle=False) as a:
        np.testing.assert_array_equal(a["row_id"], [r["row_id"] for r in rows])
        assert a["manifest_sha256"].item() == file_hash(data / "manifest.json")
        bank = {key: a[f"X_{key}"] for key in ["m", "z"]}
    simple = np.load(data / "observables.npy", allow_pickle=False)
    output.mkdir(parents=True)
    for seed in protocol["methods"]["subset_seeds"]:
        for n, train in training_subsets(strata, pool, protocol["sampling"]["labelled_training_masters_per_coupling"], seed).items():
            for view in settings["views"]:
                start = time.perf_counter()
                active, key = ({"u": simple}, "u") if view == "observables" else (bank, view)
                (cap, alpha), details = select_cap(active, key, train, target, strata, protocol, settings["caps"], seed)
                pre = {**protocol["methods"]["preprocessing"], "pca_dimensions": cap}
                transform, x = fit_view(active, key, train, pre)
                model = Ridge(alpha=alpha).fit(x, target[train])
                prediction = np.clip(model.predict(transform.transform(active, evaluation)), 0, 1)
                method = view + "_pca_selected"
                stem = output / f"{method.replace('+', '_')}-n{n}-s{seed}"
                _atomic_savez(stem.with_suffix(".npz"), {"prediction": prediction, "target": target[evaluation],
                              "row_id": np.asarray([rows[i]["row_id"] for i in evaluation]),
                              "train_indices": train, "evaluation_indices": evaluation})
                details.update(chosen_cap=cap, retained_dimensions=x.shape[1], chosen_alpha=alpha)
                _atomic_json(stem.with_suffix(".json"), {"identity": {
                    "method": method, "n_per_coupling": n, "seed": seed,
                    "protocol_sha256": file_hash(protocol_path), "sensitivity_sha256": file_hash(config),
                    "runner_sha256": file_hash(Path(__file__)), "preprocessing_sha256": file_hash(Path("src/representation_screen.py")),
                    "manifest_sha256": file_hash(data / "manifest.json"), "bank_sha256": file_hash(feature_path)},
                    "details": details, "labels_total": len(train), "seconds": time.perf_counter() - start,
                    "predictions_sha256": file_hash(stem.with_suffix(".npz")),
                    "status": "exploratory_dimension_selection_after_primary_results"})
                print(f"{method} n={n} seed={seed} cap={cap} alpha={alpha}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)
