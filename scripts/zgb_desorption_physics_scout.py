#!/usr/bin/env python3
"""Physics-only ZGB-k scout around the active/high-CO coexistence line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:  # pragma: no cover - the production environment has numba
    def njit(*_args, **_kwargs):
        return lambda function: function


ROOT = Path(__file__).resolve().parents[1]


@njit(cache=True)
def _simulate_zgb_k(
    side: int,
    co_probability: float,
    desorption_probability: float,
    burn_mcss: int,
    record_mcss: int,
    seed: int,
    start_high_co: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the standard random-sequential ZGB-k rules; one MCSS is L^2 trials."""

    np.random.seed(seed)
    field = np.ones((side, side), dtype=np.int8) if start_high_co else np.zeros((side, side), dtype=np.int8)
    co_coverage = np.empty(record_mcss, dtype=np.float64)
    o_coverage = np.empty(record_mcss, dtype=np.float64)
    reaction_rate = np.empty(record_mcss, dtype=np.float64)
    di = np.asarray((-1, 1, 0, 0), dtype=np.int64)
    dj = np.asarray((0, 0, -1, 1), dtype=np.int64)
    candidates = np.empty(4, dtype=np.int64)
    sites = side * side

    for mcss in range(burn_mcss + record_mcss):
        reactions = 0
        for _ in range(sites):
            row = np.random.randint(side)
            col = np.random.randint(side)
            if np.random.random() < desorption_probability:
                if field[row, col] == 1:
                    field[row, col] = 0
                continue
            if field[row, col] != 0:
                continue

            if np.random.random() < co_probability:
                count = 0
                for direction in range(4):
                    rr = (row + di[direction]) % side
                    cc = (col + dj[direction]) % side
                    if field[rr, cc] == -1:
                        candidates[count] = direction
                        count += 1
                if count:
                    direction = candidates[np.random.randint(count)]
                    field[(row + di[direction]) % side, (col + dj[direction]) % side] = 0
                    reactions += 1
                else:
                    field[row, col] = 1
                continue

            direction = np.random.randint(4)
            other_row = (row + di[direction]) % side
            other_col = (col + dj[direction]) % side
            if field[other_row, other_col] != 0:
                continue

            for target_row, target_col in ((row, col), (other_row, other_col)):
                count = 0
                for neighbor_direction in range(4):
                    rr = (target_row + di[neighbor_direction]) % side
                    cc = (target_col + dj[neighbor_direction]) % side
                    if field[rr, cc] == 1:
                        candidates[count] = neighbor_direction
                        count += 1
                if count:
                    neighbor_direction = candidates[np.random.randint(count)]
                    field[
                        (target_row + di[neighbor_direction]) % side,
                        (target_col + dj[neighbor_direction]) % side,
                    ] = 0
                    reactions += 1
                else:
                    field[target_row, target_col] = -1

        if mcss >= burn_mcss:
            index = mcss - burn_mcss
            co_coverage[index] = np.sum(field == 1) / sites
            o_coverage[index] = np.sum(field == -1) / sites
            reaction_rate[index] = reactions / sites
    return co_coverage, o_coverage, reaction_rate


def _seed(base_seed: int, side: int, instance: int, start: str) -> int:
    return int(base_seed + 1009 * side + 7919 * instance + (1 if start == "high_co" else 0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=64)
    parser.add_argument("--desorption", type=float, default=0.02)
    parser.add_argument("--y-min", type=float, default=0.520)
    parser.add_argument("--y-max", type=float, default=0.545)
    parser.add_argument("--y-step", type=float, default=0.001)
    parser.add_argument("--instances", type=int, default=2)
    parser.add_argument("--burn-mcss", type=int, default=2000)
    parser.add_argument("--record-mcss", type=int, default=2000)
    parser.add_argument("--base-seed", type=int, default=614911)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/order_parameter/zgb_desorption_physics_scout"),
    )
    args = parser.parse_args()
    if args.side < 4 or args.instances < 1 or args.burn_mcss < 0 or args.record_mcss < 2:
        raise ValueError("invalid lattice, instance, burn, or record size")
    if not 0 <= args.desorption <= 1 or args.y_step <= 0:
        raise ValueError("probabilities must be valid and y-step positive")

    controls = np.arange(args.y_min, args.y_max + args.y_step / 2, args.y_step)
    rows = []
    for start in ("empty", "high_co"):
        for instance in range(args.instances):
            for y in controls:
                co, oxygen, reactions = _simulate_zgb_k(
                    args.side,
                    float(y),
                    args.desorption,
                    args.burn_mcss,
                    args.record_mcss,
                    _seed(args.base_seed, args.side, instance, start),
                    start == "high_co",
                )
                rows.append(
                    {
                        "L": args.side,
                        "y": float(y),
                        "k": args.desorption,
                        "start": start,
                        "instance": instance,
                        "theta_CO_mean": float(np.mean(co)),
                        "theta_CO_std": float(np.std(co)),
                        "theta_O_mean": float(np.mean(oxygen)),
                        "reaction_rate_mean": float(np.mean(reactions)),
                        "theta_CO_block_range": float(
                            np.ptp([np.mean(block) for block in np.array_split(co, 4)])
                        ),
                    }
                )
                print(
                    f"L={args.side} y={y:.4f} k={args.desorption:.3f} "
                    f"start={start} I={instance} theta_CO={np.mean(co):.3f}",
                    flush=True,
                )

    frame = pd.DataFrame(rows)
    curves = frame.groupby(["start", "y"], as_index=False).mean(numeric_only=True)
    branch_intervals = {}
    for start, curve in curves.groupby("start"):
        curve = curve.sort_values("y")
        differences = np.abs(np.diff(curve["theta_CO_mean"]))
        index = int(np.argmax(differences))
        branch_intervals[start] = {
            "largest_jump_interval": [
                float(curve.iloc[index]["y"]),
                float(curve.iloc[index + 1]["y"]),
            ],
            "largest_theta_CO_jump": float(differences[index]),
        }
    pivot = curves.pivot(index="y", columns="start", values="theta_CO_mean")
    summary = {
        "status": "complete_physics_pilot",
        "model": "ZGB-k random-sequential kinetic Monte Carlo",
        "L": args.side,
        "desorption_probability": args.desorption,
        "controls": controls.tolist(),
        "instances_per_start": args.instances,
        "burn_mcss": args.burn_mcss,
        "record_mcss": args.record_mcss,
        "branch_intervals": branch_intervals,
        "maximum_start_state_gap": float(np.max(np.abs(pivot["high_co"] - pivot["empty"]))),
        "theta_CO_block_range_p95": float(frame["theta_CO_block_range"].quantile(0.95)),
    }
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "records.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), constrained_layout=True)
    colors = {"empty": "#31688e", "high_co": "#b2182b"}
    for start, curve in curves.groupby("start"):
        axes[0].plot(curve["y"], curve["theta_CO_mean"], "-o", ms=2.5, color=colors[start], label=start)
        axes[1].plot(curve["y"], curve["reaction_rate_mean"], "-o", ms=2.5, color=colors[start], label=start)
    axes[0].set(xlabel=r"CO adsorption probability $y$", ylabel=r"CO coverage $\theta_{\rm CO}$")
    axes[1].set(xlabel=r"CO adsorption probability $y$", ylabel=r"reaction rate per site and MCSS")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    fig.savefig(output / "physics_scout.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
