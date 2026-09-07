"""Exploratory frozen-initialization control with the matched PCA/ridge head."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import Ridge
import torch
import yaml

from scripts.run_representation_state_pilot import select_ridge
from src.representation_screen import fit_view, training_subsets
from src.representation_state_data import file_hash, load_state_data, observed_view
from src.representation_state_neural import AlignedChannelEncoder, seed_torch
from src.run_external_corpus import _atomic_json, _atomic_savez


def run(config_path, data, output, device):
    protocol = yaml.safe_load(config_path.read_text())
    manifest, masters = load_state_data(data, config_path)
    rows = manifest["rows"]
    pool = np.asarray([i for i, r in enumerate(rows) if r["role"] == "training_pool"])
    evaluation = np.asarray([i for i, r in enumerate(rows) if r["role"] == "evaluation"])
    targets = np.asarray([r["target"] for r in rows])
    strata = np.asarray([r["coupling_index"] for r in rows])
    budgets = protocol["sampling"]["labelled_training_masters_per_coupling"]
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    for seed in protocol["methods"]["subset_seeds"]:
        identity = {"protocol_sha256": file_hash(config_path), "manifest_sha256": file_hash(data / "manifest.json"),
                    "runner_sha256": file_hash(Path(__file__)), "neural_sha256": file_hash(Path("src/representation_state_neural.py")),
                    "preprocessing_sha256": file_hash(Path("src/representation_screen.py")),
                    "method": "random_encoder", "seed": seed, "device": device, "torch": str(torch.__version__),
                    "status": "post_neural_result_exploratory_addition", "pretraining": "none"}
        seed_torch(seed)
        model = AlignedChannelEncoder(protocol["methods"]["raw_encoder_spec"]).to(device).eval()
        # Capture the invariant pooled vector immediately before the readout;
        # no weights, normalizers or hyperparameters are fitted in extraction.
        captured = []
        hook = model.head[0].register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().cpu().numpy()))
        values = np.empty((len(rows), 128))
        start = time.perf_counter()
        with torch.no_grad():
            for m, t in sorted({(r["M"], r["T"]) for r in rows}):
                indices = np.asarray([i for i, r in enumerate(rows) if (r["M"], r["T"]) == (m, t)])
                for chunk in np.array_split(indices, int(np.ceil(len(indices) / 16))):
                    x = torch.as_tensor(np.stack([observed_view(masters[rows[i]["master_index"]], m, t) for i in chunk]),
                                        dtype=torch.float32, device=device)
                    model(x)
                    values[chunk] = captured.pop()
        hook.remove()
        extraction_seconds = time.perf_counter() - start
        bank = {"u": values}
        feature_path = output / f"initial-features-s{seed}.npz"
        _atomic_savez(feature_path, {"X": values, "row_id": np.asarray([r["row_id"] for r in rows])})
        for n, train in training_subsets(strata, pool, budgets, seed).items():
            start = time.perf_counter()
            alpha, details = select_ridge(bank, "u", train, targets, strata, protocol, seed)
            transform, scores = fit_view(bank, "u", train, protocol["methods"]["preprocessing"])
            fitted = Ridge(alpha=alpha).fit(scores, targets[train])
            prediction = np.clip(fitted.predict(transform.transform(bank, evaluation)), 0, 1)
            stem = output / f"random_encoder-n{n}-s{seed}"
            _atomic_savez(stem.with_suffix(".npz"), {"prediction": prediction, "target": targets[evaluation],
                                                    "train_indices": train, "evaluation_indices": evaluation,
                                                    "row_id": np.asarray([rows[i]["row_id"] for i in evaluation])})
            details.update(chosen_alpha=alpha, extraction_seconds_once_per_seed=extraction_seconds,
                           features_sha256=file_hash(feature_path))
            _atomic_json(stem.with_suffix(".json"), {"identity": {**identity, "n_per_coupling": n}, "details": details,
                         "predictions_sha256": file_hash(stem.with_suffix(".npz")), "labels_total": len(train),
                         "train_indices": train.tolist(), "evaluation_indices": evaluation.tolist(),
                         "training_MAE": float(abs(np.clip(fitted.predict(scores), 0, 1) - targets[train]).mean()),
                         "evaluation_MAE": float(abs(prediction - targets[evaluation]).mean()),
                         "seconds": time.perf_counter() - start})
        print(f"Completed frozen encoder seed {seed}; extraction {extraction_seconds:.1f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    args = p.parse_args()
    run(args.config, args.data, args.output, args.device)
