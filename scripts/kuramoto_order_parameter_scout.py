#!/usr/bin/env python3
"""Physics-only Kuramoto scout; no SPI computation."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generators.order_parameter import (  # noqa: E402
    generate_kuramoto_order_parameter,
    kuramoto_critical_coupling,
)
from src.utils import load_yaml  # noqa: E402


def _seed(config: dict, distribution: str, n_full: int, kappa: float, instance: int) -> int:
    parts = [str(config.get("base_seed", 110305)), distribution, str(n_full), str(instance)]
    if config.get("seed_pairing", "paired") == "independent":
        parts.append(f"{kappa:.12g}")
    digest = hashlib.blake2s("|".join(parts).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**32 - 1) or 1


def _jobs(config: dict) -> list[tuple[str, int, float, float, int]]:
    return list(
        product(
            [str(value) for value in config["frequency_distributions"]],
            [int(value) for value in config["N_full"]],
            [float(value) for value in config["kappa"]],
            [float(value) for value in config.get("dt", [0.02])],
            range(int(config["instances"])),
        )
    )


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_one(config: dict, job: tuple[str, int, float, float, int]) -> dict:
    distribution, n_full, kappa, dt, instance = job
    m_values = sorted({int(value) for value in config["M"]})
    t_values = sorted({int(value) for value in config["T"]})
    if m_values[-1] > n_full:
        raise ValueError(f"maximum M={m_values[-1]} exceeds N_full={n_full}")
    record_samples = t_values[-1]
    omega_std = float(config.get("omega_std", 1.0))
    critical = kuramoto_critical_coupling(distribution, omega_std)
    coupling = kappa * critical
    seed = _seed(config, distribution, n_full, kappa, instance)
    start = time.perf_counter()
    _, internals = generate_kuramoto_order_parameter(
        M=m_values[-1],
        T=record_samples,
        K=coupling,
        N_full=n_full,
        dt=dt,
        sample_dt=float(config.get("sample_dt", 0.1)),
        burn_time=float(config.get("burn_time", 100.0)),
        omega_mean=float(config.get("omega_mean", 1.0)),
        omega_std=omega_std,
        frequency_distribution=distribution,
        frequency_sampling=str(config.get("frequency_sampling", "random")),
        output="cos",
        rng=np.random.default_rng(seed),
        return_internals=True,
        store_full_phases=True,
    )
    if internals.full_phases is None:
        raise RuntimeError("full phases are required for nested-M diagnostics")

    r_full = internals.r_full
    blocks = int(config.get("stationarity_blocks", 8))
    block_records = []
    for index, indices in enumerate(np.array_split(np.arange(record_samples), blocks)):
        values = r_full[indices]
        block_records.append(
            {
                "block": index,
                "start": int(indices[0]),
                "stop": int(indices[-1] + 1),
                "mean": float(values.mean()),
                "std": float(values.std()),
            }
        )

    subset_records: dict[str, dict] = {}
    for m in m_values:
        indices = internals.observation_indices[:m]
        r_m = np.abs(np.mean(np.exp(1j * internals.full_phases[:, indices]), axis=1))
        subset_records[str(m)] = {
            "mean": float(r_m.mean()),
            "std": float(r_m.std()),
            "temporal_correlation_with_r_full": float(np.corrcoef(r_m, r_full)[0, 1]),
            "prefix_mean": {str(t): float(r_m[:t].mean()) for t in t_values},
        }

    return {
        "frequency_distribution": distribution,
        "frequency_sampling": str(config.get("frequency_sampling", "random")),
        "N_full": n_full,
        "kappa": kappa,
        "K": coupling,
        "critical_coupling_continuum": critical,
        "dt": dt,
        "sample_dt": float(config.get("sample_dt", 0.1)),
        "burn_time": float(config.get("burn_time", 100.0)),
        "record_samples": record_samples,
        "instance": instance,
        "seed": seed,
        "seed_pairing": str(config.get("seed_pairing", "paired")),
        "r_full_mean": float(r_full.mean()),
        "r_full_std": float(r_full.std()),
        "susceptibility": float(n_full * r_full.var()),
        "r_full_prefix_mean": {str(t): float(r_full[:t].mean()) for t in t_values},
        "stationarity_blocks": block_records,
        "subsets": subset_records,
        "frequency_mean": float(internals.frequencies.mean()),
        "frequency_std": float(internals.frequencies.std()),
        "elapsed_seconds": time.perf_counter() - start,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-index", type=int, help="1-based task index")
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    jobs = _jobs(config)
    if args.count_only:
        print(len(jobs))
        return 0
    indices = [args.task_index] if args.task_index is not None else list(range(1, len(jobs) + 1))
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
            f"[{index}/{len(jobs)}] {record['frequency_distribution']} "
            f"N={record['N_full']} kappa={record['kappa']:.3f} "
            f"dt={record['dt']:.3g} I={record['instance']} "
            f"R={record['r_full_mean']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
