#!/usr/bin/env python3
"""Physics-only Miller--Huse or kinetic-Ising scout; no SPI computation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generators.order_parameter import (  # noqa: E402
    generate_kinetic_ising,
    generate_miller_huse,
)
from src.utils import load_yaml  # noqa: E402


def _jobs(config: dict) -> list[dict]:
    jobs: list[dict] = []
    for path in config["paths"]:
        for control in path["controls"]:
            for side in config["lattice_side"]:
                for initial_state in config.get("initial_states", ["random"]):
                    for instance in range(int(config["instances"])):
                        jobs.append(
                            {
                                "path": str(path["name"]),
                                "path_params": dict(path.get("params") or {}),
                                "control": float(control),
                                "lattice_side": int(side),
                                "initial_state": str(initial_state),
                                "instance": instance,
                            }
                        )
    return jobs


def _seed(config: dict, job: dict) -> int:
    parts = [
        str(config.get("base_seed", 305177)),
        str(config["model"]),
        str(job["lattice_side"]),
        str(job["initial_state"]),
        str(job["instance"]),
    ]
    if config.get("seed_pairing", "paired") == "independent":
        parts.extend([str(job["path"]), f"{job['control']:.12g}"])
    digest = hashlib.blake2s("|".join(parts).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**32 - 1) or 1


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2)))


def _patch_summaries(observed: np.ndarray, patch_shape: tuple[int, int]) -> dict:
    spins = np.where(observed >= 0.0, 1.0, -1.0)
    height, width = patch_shape
    field = spins.reshape(spins.shape[0], height, width)
    walls = []
    if height > 1:
        walls.append((field[:, 1:, :] != field[:, :-1, :]).reshape(field.shape[0], -1))
    if width > 1:
        walls.append((field[:, :, 1:] != field[:, :, :-1]).reshape(field.shape[0], -1))
    domain_wall_density = float(np.concatenate(walls, axis=1).mean()) if walls else 0.0
    correlations = np.corrcoef(observed, rowvar=False)
    upper = correlations[np.triu_indices(observed.shape[1], 1)]
    finite = np.isfinite(upper)
    return {
        "magnetization_rms": _rms(spins.mean(axis=1)),
        "magnetization_abs": float(np.mean(np.abs(spins.mean(axis=1)))),
        "domain_wall_density": domain_wall_density,
        "flip_rate": float(np.mean(spins[1:] != spins[:-1])) if len(spins) > 1 else 0.0,
        "mean_absolute_correlation": (
            float(np.mean(np.abs(upper[finite]))) if np.any(finite) else None
        ),
        "finite_correlation_fraction": float(finite.mean()),
        "constant_channel_fraction": float(np.mean(np.std(observed, axis=0) == 0.0)),
    }


def run_one(config: dict, job: dict) -> dict:
    model = str(config["model"])
    seed = _seed(config, job)
    record_T = int(config["record_T"])
    truth_block_T = int(config["truth_block_T"])
    truth_blocks = int(config.get("truth_blocks", 2))
    future_T = truth_block_T * truth_blocks
    patch_shape = tuple(int(value) for value in config.get("patch_shape", [4, 5]))
    M = int(np.prod(patch_shape))
    common = {
        "M": M,
        "T": record_T,
        "lattice_side": job["lattice_side"],
        "sample_every": int(config.get("sample_every", 1)),
        "future_truth_T": future_T,
        "patch_shape": patch_shape,
        "initial_state": job["initial_state"],
        "rng": np.random.default_rng(seed),
        "return_internals": True,
    }
    start = time.perf_counter()
    if model == "miller_huse":
        observed, internals = generate_miller_huse(
            coupling=job["control"],
            transients=int(config.get("transients", 200_000)),
            store_full_field=False,
            **job["path_params"],
            **common,
        )
        current = internals.spin_magnetization
        future = internals.spin_magnetization_future
        future_hidden = internals.spin_magnetization_unobserved_future
        resolved = {
            "coupling": job["control"],
            "mu": float(job["path_params"].get("mu", 3.0)),
            "transients": int(config.get("transients", 200_000)),
        }
    elif model == "kinetic_ising":
        observed, internals = generate_kinetic_ising(
            reduced_coupling=job["control"],
            equilibration_sweeps=int(config.get("equilibration_sweeps", 200)),
            kinetic_burn_sweeps=int(config.get("kinetic_burn_sweeps", 0)),
            store_full_spins=False,
            **job["path_params"],
            **common,
        )
        current = internals.magnetization
        future = internals.magnetization_future
        future_hidden = internals.magnetization_unobserved_future
        resolved = {
            "reduced_coupling": internals.reduced_coupling,
            "beta": internals.beta,
            "J_x": float(job["path_params"].get("J_x", 1.0)),
            "J_y": float(job["path_params"].get("J_y", 1.0)),
            "exact_spontaneous_magnetization": internals.exact_spontaneous_magnetization,
            "equilibration_sweeps": int(config.get("equilibration_sweeps", 200)),
            "kinetic_burn_sweeps": int(config.get("kinetic_burn_sweeps", 0)),
        }
    else:
        raise ValueError(f"unsupported model {model!r}")

    future_blocks = np.split(future, truth_blocks)
    hidden_blocks = np.split(future_hidden, truth_blocks)
    second = float(np.mean(future**2))
    fourth = float(np.mean(future**4))
    binder = float(1.0 - fourth / (3.0 * second**2)) if second > 0.0 else None
    q_future_abs = float(np.mean(np.abs(future)))
    record = {
        "model": model,
        **job,
        "seed": seed,
        "seed_pairing": str(config.get("seed_pairing", "paired")),
        "M": M,
        "patch_shape": list(patch_shape),
        "record_T": record_T,
        "truth_block_T": truth_block_T,
        "truth_blocks": truth_blocks,
        "resolved": resolved,
        "q_current_abs": float(np.mean(np.abs(current))),
        "q_current_rms": _rms(current),
        "q_future_abs": q_future_abs,
        "q_future_rms": _rms(future),
        "q_future_hidden_abs": float(np.mean(np.abs(future_hidden))),
        "q_future_hidden_rms": _rms(future_hidden),
        "q_future_abs_blocks": [float(np.mean(np.abs(block))) for block in future_blocks],
        "q_future_rms_blocks": [_rms(block) for block in future_blocks],
        "q_future_hidden_abs_blocks": [
            float(np.mean(np.abs(block))) for block in hidden_blocks
        ],
        "q_future_hidden_rms_blocks": [_rms(block) for block in hidden_blocks],
        "magnetization_m2": second,
        "magnetization_m4": fourth,
        "susceptibility": float(job["lattice_side"] ** 2 * (second - q_future_abs**2)),
        "binder_cumulant": binder,
        "first_last_current_block_difference": float(
            abs(
                np.mean(np.abs(current[: record_T // 2]))
                - np.mean(np.abs(current[record_T // 2 :]))
            )
        ),
        "patch": _patch_summaries(observed, patch_shape),
        "elapsed_seconds": time.perf_counter() - start,
    }
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    jobs = _jobs(config)
    if args.count_only:
        print(len(jobs))
        return 0
    indices = [args.task_index] if args.task_index is not None else range(1, len(jobs) + 1)
    output_dir = ROOT / Path(config["output_dir"])
    for index in indices:
        if index is None or not 1 <= index <= len(jobs):
            raise IndexError(f"task index must be in 1..{len(jobs)}, got {index}")
        output = output_dir / f"part-{index:06d}.json"
        if output.exists() and not args.overwrite:
            print(f"[skip] {output}")
            continue
        record = run_one(config, jobs[index - 1])
        record.update(task_index=index, task_count=len(jobs))
        _atomic_json(output, record)
        print(
            f"[{index}/{len(jobs)}] {record['model']} {record['path']} "
            f"L={record['lattice_side']} control={record['control']:.5g} "
            f"I={record['instance']} Q={record['q_future_abs']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
