#!/usr/bin/env python3
"""Physics-only Stuart--Landau phase-path replication; no SPI computation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generators.order_parameter import generate_stuart_landau  # noqa: E402
from src.utils import load_yaml  # noqa: E402


def _jobs(config: dict) -> list[dict]:
    couplings = config.get("couplings", [config.get("coupling", 0.8)])
    return [
        {
            "N": int(size),
            "coupling": float(coupling),
            "gamma": float(gamma),
            "instance": instance,
        }
        for size in config["population_sizes"]
        for coupling in couplings
        for gamma in config["frequency_half_widths"]
        for instance in range(int(config["instances"]))
    ]


def _seed(config: dict, job: dict) -> int:
    payload = f"{config['base_seed']}|{job['N']}|{job['instance']}".encode()
    return int.from_bytes(hashlib.blake2s(payload, digest_size=8).digest(), "big") % (2**32 - 1) or 1


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_one(config: dict, job: dict) -> dict:
    seed = _seed(config, job)
    start = time.perf_counter()
    _, internals = generate_stuart_landau(
        M=min(job["N"], 32),
        T=int(config["record_T"]),
        N_full=job["N"],
        coupling=job["coupling"],
        frequency_half_width=job["gamma"],
        omega_mean=float(config.get("omega_mean", 2.0)),
        dt=float(config["dt"]),
        sample_dt=float(config["sample_dt"]),
        burn_time=float(config["burn_time"]),
        rng=np.random.default_rng(seed),
        return_internals=True,
    )
    amplitude = np.abs(internals.order_parameter)
    activity = internals.mean_activity
    blocks = int(config.get("blocks", 4))
    amplitude_blocks = np.array_split(amplitude, blocks)
    activity_blocks = np.array_split(activity, blocks)
    return {
        **job,
        "seed": seed,
        "record_T": int(config["record_T"]),
        "burn_time": float(config["burn_time"]),
        "R_mean": float(np.mean(amplitude)),
        "R_std": float(np.std(amplitude)),
        "R_min": float(np.min(amplitude)),
        "R_max": float(np.max(amplitude)),
        "activity_mean": float(np.mean(activity)),
        "activity_std": float(np.std(activity)),
        "R_block_means": [float(np.mean(block)) for block in amplitude_blocks],
        "R_block_stds": [float(np.std(block)) for block in amplitude_blocks],
        "activity_block_means": [float(np.mean(block)) for block in activity_blocks],
        "elapsed_seconds": time.perf_counter() - start,
    }


def _aggregate(config: dict) -> int:
    output_dir = ROOT / Path(config["output_dir"])
    paths = sorted(output_dir.glob("part-*.json"))
    expected = len(_jobs(config))
    if len(paths) != expected:
        raise RuntimeError(f"found {len(paths)}/{expected} scout parts in {output_dir}")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    names = ("N", "coupling", "gamma", "instance", "seed", "R_mean", "R_std", "R_min", "R_max", "activity_mean", "activity_std", "elapsed_seconds")
    arrays = {name: np.asarray([record[name] for record in records]) for name in names}
    block_means = np.asarray([record["R_block_means"] for record in records])
    arrays["R_block_means"] = block_means

    large = arrays["N"] == max(config["population_sizes"])
    gamma = arrays["gamma"]
    couplings = np.unique(arrays["coupling"])
    if couplings.size != 1:
        return _aggregate_plane(config, arrays, block_means, expected)
    def _cell_mean(key: str, value: float) -> float:
        return float(np.mean(arrays[key][large & np.isclose(gamma, value)]))
    anchors = {str(value): {key: _cell_mean(key, value) for key in ("R_mean", "R_std", "activity_mean")} for value in (0.6, 0.8, 1.0, 1.2)}
    block_range = np.ptp(block_means, axis=1)
    summary = {
        "status": "pass" if (
            anchors["0.6"]["R_mean"] > 0.6
            and anchors["0.6"]["R_std"] < 0.02
            and anchors["0.8"]["R_std"] > 0.08
            and anchors["1.2"]["R_mean"] < 0.12
            and anchors["1.2"]["activity_mean"] > 0.1
        ) else "fail",
        "expected_parts": expected,
        "large_population": int(max(config["population_sizes"])),
        "published_path_anchors": anchors,
        "R_block_range_p95": float(np.quantile(block_range, 0.95)),
        "elapsed_seconds_median": float(np.median(arrays["elapsed_seconds"])),
        "elapsed_seconds_p95": float(np.quantile(arrays["elapsed_seconds"], 0.95)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "physics_records.npz", **arrays)
    (output_dir / "physics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5), dpi=160, constrained_layout=True)
    for size in config["population_sizes"]:
        mask = arrays["N"] == size
        controls = np.unique(gamma[mask])
        for axis, key, label in zip(
            axes,
            ("R_mean", "R_std", "activity_mean"),
            (r"$\langle R\rangle$", r"$\mathrm{sd}(R)$", r"$\langle N^{-1}\sum|z_j|^2\rangle$"),
            strict=True,
        ):
            means = [np.mean(arrays[key][mask & np.isclose(gamma, value)]) for value in controls]
            axis.plot(controls, means, "-o", ms=3, label=f"N={size}")
            axis.set(xlabel=r"frequency half-width $\gamma$", ylabel=label)
            axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=7)
    fig.savefig(output_dir / "physics_scout.png", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _aggregate_plane(
    config: dict,
    arrays: dict[str, np.ndarray],
    block_means: np.ndarray,
    expected: int,
) -> int:
    output_dir = ROOT / Path(config["output_dir"])
    couplings = np.unique(arrays["coupling"])
    gammas = np.unique(arrays["gamma"])
    sizes = np.unique(arrays["N"])
    if sizes.size != 1:
        raise ValueError("phase-plane aggregation requires one population size")
    grids: dict[str, np.ndarray] = {}
    for key in ("R_mean", "R_std", "activity_mean"):
        grids[key] = np.asarray(
            [
                np.mean(
                    arrays[key][
                        np.isclose(arrays["coupling"], coupling)
                        & np.isclose(arrays["gamma"], gamma)
                    ]
                )
                for coupling in couplings
                for gamma in gammas
            ]
        ).reshape(len(couplings), len(gammas))
    activity = grids["activity_mean"]
    mean_r = grids["R_mean"]
    sd_r = grids["R_std"]
    finite_floor = 3.0 * np.sqrt(np.maximum(activity, 0.0) / float(sizes[0]))
    phase = np.full(mean_r.shape, 2, dtype=np.int8)  # unsteady
    phase[(sd_r < 0.02) & (mean_r > finite_floor)] = 1  # locking
    phase[(sd_r < 0.04) & (mean_r <= finite_floor)] = 0  # incoherence
    phase[activity < 0.01] = 3  # amplitude death
    summary = {
        "status": "complete_coarse_phase_plane",
        "expected_parts": expected,
        "population_size": int(sizes[0]),
        "couplings": couplings.tolist(),
        "frequency_half_widths": gammas.tolist(),
        "phase_codes": {"0": "incoherence", "1": "locking", "2": "unsteady", "3": "death"},
        "R_block_range_p95": float(np.quantile(np.ptp(block_means, axis=1), 0.95)),
    }
    np.savez_compressed(
        output_dir / "physics_records.npz",
        **arrays,
        R_mean_grid=mean_r,
        R_std_grid=sd_r,
        activity_mean_grid=activity,
        phase_code_grid=phase,
        grid_couplings=couplings,
        grid_frequency_half_widths=gammas,
    )
    (output_dir / "physics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.5), dpi=160, constrained_layout=True)
    extent = [gammas.min(), gammas.max(), couplings.min(), couplings.max()]
    for axis, values, title, cmap in zip(
        axes,
        (mean_r, sd_r, activity, phase),
        (r"$\langle R\rangle$", r"$\mathrm{sd}(R)$", "mean activity", "coarse state code"),
        ("viridis", "magma", "cividis", "tab10"),
        strict=True,
    ):
        image = axis.imshow(values, origin="lower", aspect="auto", extent=extent, cmap=cmap)
        fig.colorbar(image, ax=axis, shrink=0.8)
        axis.set(xlabel=r"frequency half-width $\gamma$", ylabel=r"coupling $K$", title=title)
    fig.savefig(output_dir / "physics_scout.png", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    jobs = _jobs(config)
    if args.count_only:
        print(len(jobs))
        return 0
    if args.aggregate:
        return _aggregate(config)
    indices = [args.task_index] if args.task_index is not None else range(1, len(jobs) + 1)
    output_dir = ROOT / Path(config["output_dir"])
    for index in indices:
        if index is None or not 1 <= index <= len(jobs):
            raise IndexError(f"task index must be in 1..{len(jobs)}, got {index}")
        output = output_dir / f"part-{index:06d}.json"
        if output.exists() and not args.overwrite:
            continue
        record = _run_one(config, jobs[index - 1])
        record.update(task_index=index, task_count=len(jobs))
        _atomic_json(output, record)
        print(f"[{index}/{len(jobs)}] N={record['N']} K={record['coupling']:.3f} gamma={record['gamma']:.3f} I={record['instance']} R={record['R_mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
