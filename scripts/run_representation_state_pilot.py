"""Matched label-budget ridge/raw-neural regression for the Stage B pilot.

Outputs are independently resumable by method, budget and subset seed. Resume
requires identical inputs, protocol and model/training code. Test labels never
enter preprocessing, hyperparameter selection or epoch selection.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import platform
import time

import numpy as np
import sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
import yaml

from src.representation_screen import fit_view, training_subsets
from src.representation_state_data import file_hash, load_state_data, observed_view, source_pool_for_seed
from src.run_external_corpus import _atomic_json, _atomic_savez


def select_ridge(bank, view, train, targets, strata, protocol, seed):
    methods = protocol["methods"]
    candidates = sorted(methods["ridge_alpha_grid"], reverse=True)
    cv = StratifiedKFold(2, shuffle=True, random_state=seed)
    scores = {alpha: [] for alpha in candidates}
    folds = []
    for fit, val in cv.split(train, strata[train]):
        fitted, tx = fit_view(bank, view, train[fit], methods["preprocessing"])
        vx = fitted.transform(bank, train[val])
        for alpha in candidates:
            model = Ridge(alpha=alpha).fit(tx, targets[train[fit]])
            prediction = np.clip(model.predict(vx), 0, 1)
            scores[alpha].append(float(np.abs(prediction - targets[train[val]]).mean()))
        folds.append({"fit": train[fit].tolist(), "validation": train[val].tolist()})
    chosen = min(candidates, key=lambda alpha: np.mean(scores[alpha]))
    return chosen, {"candidates": {str(k): v for k, v in scores.items()}, "folds": folds}


def run(config_path, data_root, output, methods, device, seeds=None, budgets=None, feature_bank=None, source_family=None):
    protocol = yaml.safe_load(config_path.read_text())
    manifest, masters = load_state_data(data_root, config_path)
    rows = manifest["rows"]
    pool = np.asarray([i for i, r in enumerate(rows) if r["role"] == "training_pool"
                       and (source_family is None or r.get("family") == source_family)])
    if not len(pool):
        raise ValueError("empty source training pool")
    evaluation = np.asarray([i for i, r in enumerate(rows) if r["role"] == "evaluation"])
    targets = np.asarray([r["target"] for r in rows])
    strata = np.asarray([r["coupling_index"] for r in rows])
    if len({rows[i]["master_id"] for i in pool}) != len(pool):
        raise ValueError("inner CV requires one source view per training master")
    simple = np.load(data_root / "observables.npy", allow_pickle=False)
    bank = None
    if feature_bank is not None:
        with np.load(feature_bank, allow_pickle=False) as archive:
            if archive["row_id"].tolist() != [r["row_id"] for r in rows]:
                raise ValueError("feature bank rows do not match")
            if str(archive["manifest_sha256"].item()) != file_hash(data_root / "manifest.json"):
                raise ValueError("feature bank source manifest mismatch")
            bank = {name: archive[f"X_{name}"] for name in ("m", "g", "z", "validity")}
    selected_seeds = seeds or protocol["methods"]["subset_seeds"]
    selected_budgets = budgets or protocol["sampling"]["labelled_training_masters_per_coupling"]
    if not set(selected_seeds) <= set(protocol["methods"]["subset_seeds"]):
        raise ValueError("seed outside protocol")
    if not set(selected_budgets) <= set(protocol["sampling"]["labelled_training_masters_per_coupling"]):
        raise ValueError("budget outside protocol")
    output.mkdir(parents=True, exist_ok=True)
    common_identity = {"protocol_sha256": file_hash(config_path),
                       "manifest_sha256": file_hash(data_root / "manifest.json"),
                       "runner_sha256": file_hash(Path(__file__)),
                       "preprocessing_sha256": file_hash(Path("src/representation_screen.py")),
                       "data_module_sha256": file_hash(Path("src/representation_state_data.py")),
                       "feature_bank_sha256": None if feature_bank is None else file_hash(feature_bank),
                       "numpy": np.__version__, "sklearn": sklearn.__version__, "python": platform.python_version()}
    if source_family is not None:
        common_identity["source_family"] = source_family
    x_source = None
    if "neural" in methods:
        import torch
        from src.representation_state_neural import fit_encoder, predict
        torch.set_num_threads(2)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA explicitly requested but unavailable")
        x_source = torch.as_tensor(np.stack([observed_view(masters[rows[i]["master_index"]],
                                                          rows[i]["M"], rows[i]["T"]) for i in pool]),
                                   dtype=torch.float32, device=device)
        # No input tensors are made from the held-out masters until fitting ends.
        pool_lookup = {int(row): i for i, row in enumerate(pool)}
        y_source = torch.as_tensor(targets[pool], dtype=torch.float32, device=device)
    for seed in selected_seeds:
        cohort = source_pool_for_seed(rows, pool, protocol, seed)
        subsets = training_subsets(strata, cohort, selected_budgets, seed)
        for n, train in subsets.items():
            for method in methods:
                stem = output / f"{method.replace('+', '_')}-n{n}-s{seed}"
                identity = {**common_identity, "method": method, "n_per_coupling": n, "seed": seed}
                if method == "neural":
                    identity.update(neural_sha256=file_hash(Path("src/representation_state_neural.py")),
                                    torch=torch.__version__, device=device,
                                    hardware=torch.cuda.get_device_name() if device == "cuda" else platform.machine())
                if stem.with_suffix(".json").exists():
                    old = json.loads(stem.with_suffix(".json").read_text())
                    if old["identity"] != identity or old["predictions_sha256"] != file_hash(stem.with_suffix(".npz")):
                        raise ValueError(f"resume identity mismatch: {stem}")
                    print(f"[SKIP] {stem.name}", flush=True)
                    continue
                start = time.perf_counter()
                details = {}
                if method == "mean":
                    prediction = np.full(len(evaluation), targets[train].mean())
                    training_prediction = np.full(len(train), targets[train].mean())
                elif method in ("correlation_direct", "phase_direct"):
                    col = 0 if method == "correlation_direct" else 1
                    prediction, training_prediction = simple[evaluation, col], simple[train, col]
                    details["labels_used"] = 0
                elif method == "neural":
                    spec = protocol["methods"]["raw_encoder_spec"]
                    cv = StratifiedKFold(2, shuffle=True, random_state=seed)
                    folds = [(train[a], train[b]) for a, b in cv.split(train, strata[train])]
                    candidates = []
                    for lr, wd in itertools.product(spec["learning_rates"], spec["weight_decays"]):
                        histories = []
                        for fold, (fit, val) in enumerate(folds):
                            a = [pool_lookup[int(i)] for i in fit]
                            b = [pool_lookup[int(i)] for i in val]
                            fitted, log = fit_encoder(x_source[a], y_source[a], spec, lr, wd, seed,
                                                      validation=(x_source[b], y_source[b]))
                            histories.append(log)
                            del fitted
                            print(f"[CV] n={n} seed={seed} lr={lr} wd={wd} fold={fold} "
                                  f"MAE={log['validation_MAE']:.4f} epoch={log['best_epoch']}", flush=True)
                        candidates.append({"learning_rate": lr, "weight_decay": wd, "folds": histories,
                                           "mean_MAE": float(np.mean([h["validation_MAE"] for h in histories]))})
                    chosen = min(candidates, key=lambda item: item["mean_MAE"])
                    epochs = max(1, int(np.median([h["best_epoch"] for h in chosen["folds"]]) + .5))
                    a = [pool_lookup[int(i)] for i in train]
                    fitted, log = fit_encoder(x_source[a], y_source[a], spec,
                                              chosen["learning_rate"], chosen["weight_decay"], seed, epochs=epochs)
                    prediction = np.empty(len(evaluation))
                    cells = sorted({(rows[i]["M"], rows[i]["T"]) for i in evaluation})
                    for m, t in cells:
                        positions = np.asarray([j for j, i in enumerate(evaluation) if (rows[i]["M"], rows[i]["T"]) == (m, t)])
                        for chunk in np.array_split(positions, max(1, int(np.ceil(len(positions) / spec["batch_size"])))):
                            x = torch.as_tensor(np.stack([observed_view(masters[rows[evaluation[j]]["master_index"]], m, t)
                                                          for j in chunk]), dtype=torch.float32, device=device)
                            prediction[chunk] = predict(fitted, x, spec["batch_size"])
                    training_prediction = predict(fitted, x_source[a], spec["batch_size"])
                    torch.save({"state_dict": {k: v.detach().cpu() for k, v in fitted.state_dict().items()},
                                "spec": spec, "identity": identity}, stem.with_suffix(".pt"))
                    details = {"candidates": candidates, "chosen_learning_rate": chosen["learning_rate"],
                               "chosen_weight_decay": chosen["weight_decay"], "refit_epochs": epochs,
                               "refit": log, "checkpoint_sha256": file_hash(stem.with_suffix(".pt")),
                               "folds": [{"fit": a.tolist(), "validation": b.tolist()} for a, b in folds]}
                else:
                    if method in ("correlation", "phase", "observables"):
                        columns = {"correlation": [0], "phase": [1], "observables": [0, 1]}[method]
                        active_bank, view = {"u": simple[:, columns]}, "u"
                    else:
                        if bank is None or method not in protocol["methods"]["views"]:
                            raise ValueError(f"unknown or unavailable method {method}")
                        active_bank, view = bank, method
                    alpha, details = select_ridge(active_bank, view, train, targets, strata, protocol, seed)
                    transformer, scores = fit_view(active_bank, view, train, protocol["methods"]["preprocessing"])
                    fitted = Ridge(alpha=alpha).fit(scores, targets[train])
                    prediction = fitted.predict(transformer.transform(active_bank, evaluation))
                    training_prediction = fitted.predict(scores)
                    details["chosen_alpha"] = alpha
                    details["retained_dimensions"] = scores.shape[1]
                prediction = np.clip(prediction, 0, 1)
                _atomic_savez(stem.with_suffix(".npz"), {"prediction": prediction, "evaluation_indices": evaluation,
                                                        "train_indices": train, "target": targets[evaluation],
                                                        "row_id": np.asarray([rows[i]["row_id"] for i in evaluation])})
                report = {"identity": identity, "details": details,
                          "training_MAE": float(np.abs(np.clip(training_prediction, 0, 1) - targets[train]).mean()),
                          "evaluation_MAE": float(np.abs(prediction - targets[evaluation]).mean()),
                          "seconds": time.perf_counter() - start,
                          "predictions_sha256": file_hash(stem.with_suffix(".npz")),
                          "train_indices": train.tolist(), "evaluation_indices": evaluation.tolist(),
                          "labels_total": len(train), "status": "exploratory_pilot"}
                _atomic_json(stem.with_suffix(".json"), report)
                print(f"[DONE] {stem.name} MAE={report['evaluation_MAE']:.4f} seconds={report['seconds']:.1f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["mean", "correlation_direct", "phase_direct", "correlation", "phase", "observables"])
    parser.add_argument("--feature-bank", type=Path)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--budgets", nargs="+", type=int)
    parser.add_argument("--source-family")
    args = parser.parse_args()
    run(args.config, args.data, args.output, args.methods, args.device, args.seeds, args.budgets, args.feature_bank, args.source_family)
