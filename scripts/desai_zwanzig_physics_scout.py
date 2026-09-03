#!/usr/bin/env python3
"""Physics-only finite-size scout for the Desai--Zwanzig transition."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
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

from src.generators.order_parameter import generate_desai_zwanzig  # noqa: E402
from src.utils import load_yaml  # noqa: E402


def _jobs(config: dict) -> list[dict]:
    return [
        {
            "N": int(size),
            "sigma": float(sigma),
            "initial_mean": float(initial_mean),
            "instance": int(instance),
        }
        for size in config["population_sizes"]
        for sigma in config["sigmas"]
        for initial_mean in config["initial_means"]
        for instance in range(int(config["instances"]))
    ]


def _seed(config: dict, job: dict) -> int:
    payload = f"{config['base_seed']}|{job['N']}|{job['instance']}".encode()
    value = int.from_bytes(hashlib.blake2s(payload, digest_size=8).digest(), "big")
    return value % (2**32 - 1) or 1


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _run_one(config: dict, job: dict) -> dict:
    seed = _seed(config, job)
    start = time.perf_counter()
    _, internals = generate_desai_zwanzig(
        M=min(job["N"], 32),
        T=int(config["T"]),
        sigma=job["sigma"],
        N_full=job["N"],
        alpha=float(config["alpha"]),
        theta=float(config["theta"]),
        sigma_m=float(config["sigma_m"]),
        nu=float(config["nu"]),
        dt=float(config["dt"]),
        sample_dt=float(config["sample_dt"]),
        burn_time=float(config["burn_time"]),
        truth_start_T=int(config["truth_start_T"]),
        future_truth_T=int(config["future_truth_T"]),
        initial_mean=job["initial_mean"],
        initial_std=float(config["initial_std"]),
        rng=np.random.default_rng(seed),
        return_internals=True,
    )
    mean_field = internals.mean_field_future
    absolute = np.abs(mean_field)
    blocks = np.array_split(absolute, int(config["blocks"]))
    signs = np.sign(mean_field)
    nonzero = signs != 0
    sign_changes = int(np.sum(signs[1:][nonzero[1:] & nonzero[:-1]] != signs[:-1][nonzero[1:] & nonzero[:-1]]))
    return {
        **job,
        "seed": seed,
        "Q_mean_abs": float(np.mean(absolute)),
        "Q_mean_signed": float(np.mean(mean_field)),
        "Q_rms": float(np.sqrt(np.mean(mean_field**2))),
        "Q_sd": float(np.std(mean_field)),
        "Q_mean_abs_blocks": [float(np.mean(block)) for block in blocks],
        "sign_changes": sign_changes,
        "elapsed_seconds": time.perf_counter() - start,
    }


def _run_and_write(config: dict, index: int, overwrite: bool) -> str:
    jobs = _jobs(config)
    job = jobs[index - 1]
    output_dir = ROOT / Path(config["output_dir"])
    output = output_dir / f"part-{index:06d}.json"
    if output.exists() and not overwrite:
        return f"[{index}/{len(jobs)}] cached"
    record = _run_one(config, job)
    record.update(task_index=index, task_count=len(jobs))
    _atomic_json(output, record)
    return (
        f"[{index}/{len(jobs)}] N={record['N']} sigma={record['sigma']:.3f} "
        f"start={record['initial_mean']:+.0f} Q={record['Q_mean_abs']:.3f}"
    )


def _aggregate(config: dict) -> int:
    output_dir = ROOT / Path(config["output_dir"])
    paths = sorted(output_dir.glob("part-*.json"))
    expected = len(_jobs(config))
    if len(paths) != expected:
        raise RuntimeError(f"found {len(paths)}/{expected} parts in {output_dir}")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    scalar_names = (
        "N",
        "sigma",
        "initial_mean",
        "instance",
        "seed",
        "Q_mean_abs",
        "Q_mean_signed",
        "Q_rms",
        "Q_sd",
        "sign_changes",
        "elapsed_seconds",
    )
    arrays = {
        name: np.asarray([record[name] for record in records])
        for name in scalar_names
    }
    blocks = np.asarray([record["Q_mean_abs_blocks"] for record in records])
    arrays["Q_mean_abs_blocks"] = blocks

    boundary_by_size = {}
    branch_gap_by_size = {}
    for size in config["population_sizes"]:
        size_mask = arrays["N"] == size
        controls = np.unique(arrays["sigma"][size_mask])
        curve = np.asarray(
            [
                np.mean(
                    arrays["Q_mean_abs"][
                        size_mask & np.isclose(arrays["sigma"], control)
                    ]
                )
                for control in controls
            ]
        )
        slopes = np.abs(np.diff(curve) / np.diff(controls))
        steepest = int(np.argmax(slopes))
        branch_gaps = []
        for control in controls:
            cell = size_mask & np.isclose(arrays["sigma"], control)
            negative = cell & (arrays["initial_mean"] < 0)
            positive = cell & (arrays["initial_mean"] > 0)
            branch_gaps.append(
                abs(
                    np.mean(arrays["Q_mean_abs"][negative])
                    - np.mean(arrays["Q_mean_abs"][positive])
                )
            )
        boundary_by_size[str(size)] = {
            "steepest_Q_interval": [
                float(controls[steepest]), float(controls[steepest + 1])
            ],
            "maximum_absolute_Q_slope": float(slopes[steepest]),
        }
        branch_gap_by_size[str(size)] = float(max(branch_gaps))

    summary = {
        "status": "complete_finite_size_scout",
        "expected_parts": expected,
        "reference_mean_field_sigma_c": float(config["reference_sigma_c"]),
        "boundary_by_population_size": boundary_by_size,
        "maximum_initial_branch_Q_gap_by_size": branch_gap_by_size,
        "Q_block_range_p95_by_size": {
            str(size): float(
                np.quantile(
                    np.ptp(blocks[arrays["N"] == size], axis=1), 0.95
                )
            )
            for size in config["population_sizes"]
        },
        "elapsed_seconds_median": float(np.median(arrays["elapsed_seconds"])),
        "elapsed_seconds_p95": float(np.quantile(arrays["elapsed_seconds"], 0.95)),
    }
    np.savez_compressed(output_dir / "physics_records.npz", **arrays)
    (output_dir / "physics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3), dpi=180, constrained_layout=True)
    for size in config["population_sizes"]:
        mask = arrays["N"] == size
        controls = np.unique(arrays["sigma"][mask])
        for axis, key, label in (
            (axes[0], "Q_mean_abs", r"$Q=\langle|M_1|\rangle_t$"),
            (axes[1], "Q_sd", r"$\mathrm{sd}_t(M_1)$"),
        ):
            mean = np.asarray(
                [
                    np.mean(arrays[key][mask & np.isclose(arrays["sigma"], control)])
                    for control in controls
                ]
            )
            axis.plot(controls, mean, "-o", ms=2.5, label=rf"$N={size}$")
            axis.set(xlabel=r"noise amplitude $\sigma$", ylabel=label)
            axis.grid(alpha=0.15)
    for axis in axes:
        axis.axvline(float(config["reference_sigma_c"]), color="0.35", ls=":", lw=1)
    axes[0].legend(frameon=False, fontsize=7)
    fig.savefig(output_dir / "physics_scout.png", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--workers", type=int, default=1)
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
    if args.task_index is not None:
        print(_run_and_write(config, args.task_index, args.overwrite))
        return 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(_run_and_write, config, index, args.overwrite)
            for index in range(1, len(jobs) + 1)
        ]
        for future in futures:
            print(future.result(), flush=True)
    return _aggregate(config)


if __name__ == "__main__":
    raise SystemExit(main())
