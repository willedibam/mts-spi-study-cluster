"""Build deterministic masters, future labels and a pyspi named-array corpus."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import yaml

from src.generators.order_parameter import generate_kuramoto_order_parameter, kuramoto_critical_coupling
from src.representation_state_data import file_hash, observed_view, simple_observables


def build(config_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    config = yaml.safe_load(config_path.read_text())
    params = config["generator"].copy()
    params.pop("name")
    kappas = params.pop("reduced_couplings")
    params.pop("coupling_scale")
    critical = kuramoto_critical_coupling(params["frequency_distribution"], params["omega_std"])
    rows, master_records, arrays, targets, corpus, observables = [], [], [], [], {}, []
    start = time.perf_counter()
    for split_id, role, key in ((0, "training_pool", "training_masters_per_coupling"),
                                (1, "evaluation", "evaluation_masters_per_coupling")):
        for coupling, kappa in enumerate(kappas):
            for replicate in range(config["sampling"][key]):
                seed_parts = [config["sampling"]["master_seed"], split_id, coupling, replicate]
                raw, truth = generate_kuramoto_order_parameter(
                    **params, K=kappa * critical, rng=np.random.default_rng(np.random.SeedSequence(seed_parts)),
                    return_internals=True, store_full_phases=False)
                assert raw.shape == (params["T"], params["N_full"]) and np.isfinite(raw).all()
                assert len(truth.r_full_future) == params["future_truth_T"]
                target = float(truth.r_full_future.mean())
                assert np.isfinite(target) and 0 <= target <= 1
                index = len(arrays)
                master_id = f"s{split_id}-k{coupling}-r{replicate:02d}"
                arrays.append(raw)
                targets.append(target)
                master_records.append({"master_id": master_id, "role": role, "seed_parts": seed_parts,
                                       "coupling_index": coupling, "replicate": replicate,
                                       "sensor_order": truth.observation_indices.tolist(),
                                       "past_full_coherence": float(truth.r_full.mean())})
                source = config["observations"]["source"]
                cells = [(source["M"], source["T"])] if role == "training_pool" else [
                    (m, t) for m in config["observations"]["M_values"] for t in config["observations"]["T_values"]]
                for m, t in cells:
                    name = f"{master_id}-M{m}-T{t}"
                    view = observed_view(raw, m, t)
                    corpus[name] = view
                    observables.append(simple_observables(view))
                    rows.append({"row_id": name, "master_id": master_id, "master_index": index,
                                 "role": role, "coupling_index": coupling, "M": m, "T": t,
                                 "target": target, "corpus_index": len(rows) + 1})
            print(f"Generated {role} coupling {coupling}: {len(arrays)} masters", flush=True)
    output.mkdir(parents=True)
    np.save(output / "masters.npy", np.stack(arrays), allow_pickle=False)
    np.save(output / "targets.npy", np.asarray(targets), allow_pickle=False)
    np.save(output / "observables.npy", np.stack(observables), allow_pickle=False)
    corpus.update(__dataset_names__=np.asarray([r["row_id"] for r in rows]),
                  __labels_json__=np.asarray(["[]"] * len(rows)),
                  __shapes__=np.asarray([[r["T"], r["M"]] for r in rows]),
                  __axis_order__=np.asarray(["observation", "process"]))
    np.savez_compressed(output / "views.npz", **corpus)
    artifacts = {name: file_hash(output / name) for name in ("masters.npy", "targets.npy", "observables.npy", "views.npz")}
    manifest = {"config_sha256": file_hash(config_path), "artifacts": artifacts,
                "generator_sha256": file_hash(Path("src/generators/order_parameter.py")),
                "builder_sha256": file_hash(Path(__file__)), "rows": rows, "masters": master_records,
                "seconds": time.perf_counter() - start, "target_range": [min(targets), max(targets)],
                "status": "exploratory_pilot_not_confirmation"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"masters": len(arrays), "rows": len(rows), "seconds": manifest["seconds"],
                      "target_range": manifest["target_range"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.config, args.output)
