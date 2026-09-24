"""Seed the twelve stochastic p90 summaries without rerunning p90.

Run only after the original dataset writer has finished. A durable sidecar
preserves the original matrices and allows recovery between atomic writes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from src.compute import run_pyspi
from src.mapping import DatasetMapping, ExperimentConfig
from src.run_experiments import _file_sha256, _repository_provenance, _pyspi_version
from src.utils import project_root, timestamp

NAMES = ("bary_sgddtw_mean", "bary_sgddtw_max", "bary-sq_sgddtw_mean", "bary-sq_sgddtw_max")
NAMES += tuple(f"{prefix}_{estimator}" for estimator in ("EllipticEnvelope", "MinCovDet")
               for prefix in ("cov", "cov-sq", "prec", "prec-sq"))
POLICY = "isolated-stochastic-numpy-python-seed-v1"
CONFIG = project_root() / "configs/pyspi/cases/proof_stochastic_replay.yaml"


def _write_npz(path, arrays):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temp.replace(path)


def replay(directory: Path, *, compute=run_pyspi):
    mpi_path, meta_path = directory / "spi_mpis.npz", directory / "meta.json"
    meta = json.loads(meta_path.read_text())
    seed = int(meta["generator"]["seed"])
    receipt_path = directory / "stochastic-seeded-replay.npz"
    with np.load(mpi_path) as archive:
        matrices = {name: archive[name] for name in archive.files}
    if not all(name in matrices for name in NAMES):
        raise ValueError(f"Missing stochastic p90 variants: {directory}")
    if receipt_path.exists():
        with np.load(receipt_path) as archive:
            receipt = json.loads(str(archive["receipt"]))
            old = {name: archive[f"original_{name}"] for name in NAMES}
            new = {name: archive[f"seeded_{name}"] for name in NAMES}
        if receipt["policy"] != POLICY or receipt["seed"] != seed:
            raise ValueError("Replay receipt seed/policy mismatch")
        if receipt["input_sha256"] != _file_sha256(directory / "timeseries.npy"):
            raise ValueError("Replay input has changed")
        for name in NAMES:
            if not any(np.array_equal(matrices[name], candidate[name], equal_nan=True) for candidate in (old, new)):
                raise ValueError(f"Unexpected change to {name}")
    else:
        started = time.perf_counter()
        result = compute(np.load(directory / "timeseries.npy"), config_path=CONFIG,
                         normalise=meta["normalise"], n_jobs=1, random_seed=seed)
        if set(result.matrices) != set(NAMES):
            raise ValueError("Unexpected stochastic replay catalogue")
        old, new = {name: matrices[name] for name in NAMES}, result.matrices
        receipt = {
            "policy": POLICY, "seed": seed, "timestamp": timestamp(),
            "input_sha256": _file_sha256(directory / "timeseries.npy"),
            "original_archive_sha256": _file_sha256(mpi_path),
            "original_errors": {name: meta["pyspi"].get("errors", {}).get(name) for name in NAMES},
            "original_seconds": {name: meta.get("paths", {}).get("per_spi", {}).get(name) for name in NAMES},
            "config_sha256": _file_sha256(CONFIG), "pyspi_version": _pyspi_version(),
            "code": _repository_provenance(), "compute_seconds": time.perf_counter() - started,
            "errors": result.errors or {}, "per_spi": result.timings or {},
            "names": list(NAMES),
        }
        _write_npz(receipt_path, {
            "receipt": np.array(json.dumps(receipt)),
            **{f"original_{name}": old[name] for name in NAMES},
            **{f"seeded_{name}": new[name] for name in NAMES},
        })
    if not all(np.array_equal(matrices[name], new[name], equal_nan=True) for name in NAMES):
        matrices.update(new)
        _write_npz(mpi_path, matrices)
    # This records a partial recomputation; the original p90 provenance remains.
    meta["pyspi"]["stochastic_replay"] = {
        **receipt, "sidecar": receipt_path.name,
        "repaired_archive_sha256": _file_sha256(mpi_path),
    }
    errors = meta["pyspi"].setdefault("errors", {})
    for name in NAMES:
        errors.pop(name, None)
    errors.update(receipt["errors"])
    temp = meta_path.with_name(meta_path.name + ".tmp")
    temp.write_text(json.dumps(meta, indent=2) + "\n")
    temp.replace(meta_path)
    print(f"[SEEDED] {directory.name} seed={seed}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--job-index", required=True, type=int)
    args = parser.parse_args()
    mapping = DatasetMapping(ExperimentConfig.from_file(args.experiment_config))
    replay(Path(mapping.spec_for_index(args.job_index).dataset_dir))


if __name__ == "__main__":
    main()
