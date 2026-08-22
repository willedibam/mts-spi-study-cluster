#!/usr/bin/env python3
"""Aggregate a spin-order physics scout into compact numeric diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.spin_order_parameter_scout import _jobs  # noqa: E402
from src.utils import load_yaml  # noqa: E402


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
        return float("nan")
    return float(spearmanr(x[mask], y[mask]).statistic)


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _cell_residual(values: np.ndarray, paths: np.ndarray, controls: np.ndarray) -> np.ndarray:
    residual = np.empty_like(values, dtype=np.float64)
    for path in np.unique(paths):
        for control in np.unique(controls[paths == path]):
            mask = (paths == path) & (controls == control)
            residual[mask] = values[mask] - np.mean(values[mask])
    return residual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    output_dir = ROOT / Path(config["output_dir"])
    paths = sorted(output_dir.glob("part-*.json"))
    expected = len(_jobs(config))
    if len(paths) != expected:
        raise RuntimeError(f"found {len(paths)}/{expected} scout parts in {output_dir}")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]

    path_names = np.asarray([record["path"] for record in records])
    controls = np.asarray([record["control"] for record in records], dtype=np.float64)
    sides = np.asarray([record["lattice_side"] for record in records], dtype=np.int32)
    instances = np.asarray([record["instance"] for record in records], dtype=np.int32)
    q = np.asarray([record["q_future_abs"] for record in records], dtype=np.float64)
    q_rms = np.asarray([record["q_future_rms"] for record in records], dtype=np.float64)
    q_hidden = np.asarray(
        [record["q_future_hidden_abs"] for record in records], dtype=np.float64
    )
    q_current = np.asarray([record["q_current_abs"] for record in records], dtype=np.float64)
    binder = np.asarray(
        [np.nan if record["binder_cumulant"] is None else record["binder_cumulant"] for record in records],
        dtype=np.float64,
    )
    susceptibility = np.asarray([record["susceptibility"] for record in records])
    patch_q = np.asarray([record["patch"]["magnetization_rms"] for record in records])
    wall_density = np.asarray([record["patch"]["domain_wall_density"] for record in records])
    flip_rate = np.asarray([record["patch"]["flip_rate"] for record in records])
    constant_fraction = np.asarray(
        [record["patch"]["constant_channel_fraction"] for record in records]
    )
    elapsed = np.asarray([record["elapsed_seconds"] for record in records])
    block_pairs = []
    for record in records:
        blocks = np.asarray(record["q_future_abs_blocks"], dtype=np.float64)
        block_pairs.extend(np.abs(np.diff(blocks)).tolist())
    block_pairs = np.asarray(block_pairs, dtype=np.float64)

    q_range = float(np.ptp(q))
    repeat_p95 = float(np.quantile(block_pairs, 0.95)) if block_pairs.size else float("nan")
    q_residual = _cell_residual(q, path_names, controls)
    patch_residual = _cell_residual(patch_q, path_names, controls)
    summary = {
        "model": str(config["model"]),
        "expected_parts": expected,
        "observed_parts": len(records),
        "q_range": q_range,
        "future_block_repeatability_p95": repeat_p95,
        "future_block_repeatability_p95_fraction_of_q_range": (
            repeat_p95 / q_range if q_range > 0 else None
        ),
        "current_future_difference_p95": float(np.quantile(np.abs(q_current - q), 0.95)),
        "full_hidden_difference_p95": float(np.quantile(np.abs(q_hidden - q), 0.95)),
        "patch_q_overall_spearman": _finite_or_none(_safe_spearman(patch_q, q)),
        "patch_q_within_cell_spearman": _finite_or_none(
            _safe_spearman(patch_residual, q_residual)
        ),
        "domain_wall_overall_spearman": _finite_or_none(
            _safe_spearman(wall_density, q)
        ),
        "flip_rate_overall_spearman": _finite_or_none(_safe_spearman(flip_rate, q)),
        "constant_channel_fraction_max": float(np.max(constant_fraction)),
        "elapsed_seconds_median": float(np.median(elapsed)),
        "elapsed_seconds_p95": float(np.quantile(elapsed, 0.95)),
    }
    if str(config["model"]) == "kinetic_ising" and len(np.unique(path_names)) == 2:
        matched = []
        first, second = np.unique(path_names)
        shared_controls = np.intersect1d(controls[path_names == first], controls[path_names == second])
        for control in shared_controls:
            first_mean = q[(path_names == first) & (controls == control)].mean()
            second_mean = q[(path_names == second) & (controls == control)].mean()
            matched.append(abs(float(first_mean - second_mean)))
        summary["matched_control_path_gap_max"] = max(matched) if matched else None
        summary["matched_control_path_gap_mean"] = float(np.mean(matched)) if matched else None

    np.savez_compressed(
        output_dir / "physics_records.npz",
        path=path_names,
        control=controls,
        lattice_side=sides,
        instance=instances,
        q_future_abs=q,
        q_future_rms=q_rms,
        q_future_hidden_abs=q_hidden,
        q_current_abs=q_current,
        binder_cumulant=binder,
        susceptibility=susceptibility,
        patch_q_rms=patch_q,
        patch_domain_wall_density=wall_density,
        patch_flip_rate=flip_rate,
        constant_channel_fraction=constant_fraction,
        elapsed_seconds=elapsed,
        future_block_pair_differences=block_pairs,
    )
    (output_dir / "physics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0), dpi=160)
    for path_name in np.unique(path_names):
        for side in np.unique(sides[path_names == path_name]):
            mask = (path_names == path_name) & (sides == side)
            unique_controls = np.unique(controls[mask])
            means = np.asarray([q[mask & (controls == value)].mean() for value in unique_controls])
            ses = np.asarray(
                [
                    q[mask & (controls == value)].std(ddof=1)
                    / np.sqrt((mask & (controls == value)).sum())
                    if (mask & (controls == value)).sum() > 1
                    else 0.0
                    for value in unique_controls
                ]
            )
            label = f"{path_name}, L={side}"
            axes[0, 0].errorbar(unique_controls, means, yerr=1.96 * ses, marker="o", ms=3, label=label)
            binder_means = [np.nanmean(binder[mask & (controls == value)]) for value in unique_controls]
            susceptibility_means = [
                np.mean(susceptibility[mask & (controls == value)]) for value in unique_controls
            ]
            axes[0, 1].plot(unique_controls, binder_means, "-o", ms=3, label=label)
            axes[1, 0].plot(unique_controls, susceptibility_means, "-o", ms=3, label=label)
    axes[1, 1].scatter(patch_q, q, c=controls, s=12, alpha=0.7)
    axes[0, 0].set(xlabel="control", ylabel=r"$Q=\langle |m|\rangle$", title="Order curve")
    axes[0, 1].set(xlabel="control", ylabel="Binder cumulant", title="Finite-size diagnostic")
    axes[1, 0].set(xlabel="control", ylabel="susceptibility", title="Fluctuation diagnostic")
    axes[1, 1].set(xlabel="local patch Q", ylabel="future full-system Q", title="Local observability")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "physics_scout.png", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
