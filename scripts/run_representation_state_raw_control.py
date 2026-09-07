"""Apply the existing pooled raw features with the matched state-pilot head."""
import argparse
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import Ridge
import yaml

from scripts.run_representation_state_pilot import select_ridge
from src.cross_mt_transfer import pooled_baseline_features
from src.representation_screen import fit_view, training_subsets
from src.representation_state_data import file_hash, load_state_data, observed_view
from src.run_external_corpus import _atomic_json, _atomic_savez


def run(config_path, data, output):
    if output.exists():
        raise FileExistsError(output)
    protocol = yaml.safe_load(config_path.read_text())
    manifest, masters = load_state_data(data, config_path)
    rows = manifest["rows"]
    pool = np.asarray([i for i, r in enumerate(rows) if r["role"] == "training_pool"])
    evaluation = np.asarray([i for i, r in enumerate(rows) if r["role"] == "evaluation"])
    target = np.asarray([r["target"] for r in rows])
    strata = np.asarray([r["coupling_index"] for r in rows])
    start = time.perf_counter()
    values, names = [], None
    for row in rows:
        raw = observed_view(masters[row["master_index"]], row["M"], row["T"])
        features, current_names = pooled_baseline_features(raw)["pooled_combined"]
        if names is not None and names != current_names:
            raise ValueError("pooled raw feature schema changed")
        names = current_names
        values.append(features)
    values = np.asarray(values)
    phase = np.load(data / "observables.npy", allow_pickle=False)[:, 1:2]
    feature_sets = {"pooled_raw": values, "pooled_raw_phase": np.concatenate([values, phase], axis=1)}
    extraction_seconds = time.perf_counter() - start
    output.mkdir(parents=True)
    feature_path = output / "features.npz"
    _atomic_savez(feature_path, {**feature_sets, "feature_names": np.asarray(names),
                               "row_id": np.asarray([r["row_id"] for r in rows])})
    identity = {"protocol_sha256": file_hash(config_path), "manifest_sha256": file_hash(data / "manifest.json"),
                "runner_sha256": file_hash(Path(__file__)), "frontend_sha256": file_hash(Path("src/cross_mt_transfer.py")),
                "preprocessing_sha256": file_hash(Path("src/representation_screen.py")),
                "status": "post_neural_result_exploratory_addition", "pretraining": "none"}
    for seed in protocol["methods"]["subset_seeds"]:
        subsets = training_subsets(strata, pool, protocol["sampling"]["labelled_training_masters_per_coupling"], seed)
        for n, train in subsets.items():
            for method, matrix in feature_sets.items():
                start = time.perf_counter()
                bank = {"u": matrix}
                alpha, details = select_ridge(bank, "u", train, target, strata, protocol, seed)
                transform, scores = fit_view(bank, "u", train, protocol["methods"]["preprocessing"])
                fitted = Ridge(alpha=alpha).fit(scores, target[train])
                prediction = np.clip(fitted.predict(transform.transform(bank, evaluation)), 0, 1)
                stem = output / f"{method}-n{n}-s{seed}"
                _atomic_savez(stem.with_suffix(".npz"), {"prediction": prediction, "target": target[evaluation],
                                                        "train_indices": train, "evaluation_indices": evaluation,
                                                        "row_id": np.asarray([rows[i]["row_id"] for i in evaluation])})
                details.update(chosen_alpha=alpha, feature_count=matrix.shape[1],
                               extraction_seconds_once=extraction_seconds, features_sha256=file_hash(feature_path))
                _atomic_json(stem.with_suffix(".json"), {"identity": {**identity, "method": method, "seed": seed, "n_per_coupling": n},
                             "details": details, "predictions_sha256": file_hash(stem.with_suffix(".npz")),
                             "labels_total": len(train), "train_indices": train.tolist(), "evaluation_indices": evaluation.tolist(),
                             "training_MAE": float(abs(np.clip(fitted.predict(scores), 0, 1) - target[train]).mean()),
                             "evaluation_MAE": float(abs(prediction - target[evaluation]).mean()), "seconds": time.perf_counter() - start})
    print(f"Completed pooled raw controls: {len(rows)} records, {values.shape[1]} features, extraction {extraction_seconds:.1f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(args.config, args.data, args.output)
