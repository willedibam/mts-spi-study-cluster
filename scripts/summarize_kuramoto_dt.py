#!/usr/bin/env python3
"""Summarize paired Kuramoto timestep checks without figures or tabular exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.kuramoto_order_parameter_scout import _jobs  # noqa: E402
from src.order_parameter_analysis import safe_spearman  # noqa: E402
from src.utils import load_yaml  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    output_dir = ROOT / Path(config["output_dir"])
    paths = sorted(output_dir.glob("part-*.json"))
    expected = len(_jobs(config))
    if len(paths) != expected:
        raise RuntimeError(f"found {len(paths)}/{expected} parts in {output_dir}")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    grouped: dict[tuple, dict[float, float]] = {}
    for record in records:
        key = (
            record["frequency_distribution"],
            int(record["N_full"]),
            float(record["kappa"]),
            int(record["instance"]),
        )
        grouped.setdefault(key, {})[float(record["dt"])] = float(record["r_full_mean"])
    steps = sorted({float(record["dt"]) for record in records})
    if len(steps) != 2 or any(set(values) != set(steps) for values in grouped.values()):
        raise RuntimeError("expected exactly two paired timestep values for every realization")
    fine, coarse = steps[0], steps[1]
    fine_values = np.array([values[fine] for values in grouped.values()])
    coarse_values = np.array([values[coarse] for values in grouped.values()])
    difference = coarse_values - fine_values
    summary = {
        "pairs": int(len(grouped)),
        "fine_dt": fine,
        "coarse_dt": coarse,
        "signed_mean_difference": float(np.mean(difference)),
        "median_absolute_difference": float(np.median(np.abs(difference))),
        "p95_absolute_difference": float(np.quantile(np.abs(difference), 0.95)),
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
        "spearman": safe_spearman(coarse_values, fine_values),
    }
    output = output_dir / "dt_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
