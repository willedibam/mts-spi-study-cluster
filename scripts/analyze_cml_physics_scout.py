#!/usr/bin/env python3
"""Aggregate quadratic-CML physics-only scout parts and draw the gate figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plot_style import apply_plot_style  # noqa: E402


METRICS = {
    "selected_band_power": "selected-band power",
    "dynamical_spatial_pattern_entropy": "dynamical pattern entropy",
    "temporal_spectral_entropy": "temporal spectral entropy",
    "period2_activity": "period-2 residual",
}


def _value(record: dict, metric: str) -> float:
    if metric == "turbulent_fraction_0p05":
        return float(record["turbulent_fraction"]["0.05"])
    return float(record[metric])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    raw = []
    for directory in args.input_dirs:
        for path in sorted(directory.glob("part-*.json")):
            raw.append(json.loads(path.read_text(encoding="utf-8")))
    if not raw:
        raise RuntimeError("no scout parts found")
    by_key = {
        (record["alpha"], record["eps"], record["lattice_size"], record["instance"]): record
        for record in raw
    }
    records = list(by_key.values())
    metric_names = [
        *METRICS,
        "static_spatial_pattern_entropy",
        "period2_mean_abs",
        "spatial_neighbour_correlation",
        "field_std",
        "turbulent_fraction_0p05",
    ]
    rows = []
    for record in records:
        rows.append(
            {
                "alpha": float(record["alpha"]),
                "eps": float(record["eps"]),
                "N": int(record["lattice_size"]),
                "instance": int(record["instance"]),
                **{metric: _value(record, metric) for metric in metric_names},
            }
        )
    frame = pd.DataFrame(rows).sort_values(["N", "instance", "alpha"])

    block_summary = {}
    prefix_summary = {}
    for metric in metric_names:
        iqr = float(frame[metric].quantile(0.75) - frame[metric].quantile(0.25)) or 1.0
        block_ranges = []
        for record in records:
            values = [_value(block, metric) for block in record["stationarity_blocks"]]
            block_ranges.append((max(values) - min(values)) / iqr)
        block_summary[metric] = {
            "median_range_over_iqr": float(np.median(block_ranges)),
            "p95_range_over_iqr": float(np.quantile(block_ranges, 0.95)),
        }
    prefixes = sorted(
        {int(value) for record in records for value in record["prefix_summaries"]}
    )
    for T in prefixes:
        errors = []
        for metric in metric_names:
            iqr = float(frame[metric].quantile(0.75) - frame[metric].quantile(0.25)) or 1.0
            errors.extend(
                abs(_value(record["prefix_summaries"][str(T)], metric) - _value(record, metric))
                / iqr
                for record in records
                if str(T) in record["prefix_summaries"]
            )
        prefix_summary[str(T)] = {
            "median_absolute_error_over_iqr": float(np.median(errors)),
            "p95_absolute_error_over_iqr": float(np.quantile(errors, 0.95)),
        }

    branch = frame.query("alpha == 1.75 and N == 512").sort_values("instance")
    summary = {
        "records": int(len(frame)),
        "alpha_values": sorted(frame["alpha"].unique().tolist()),
        "lattice_sizes": sorted(frame["N"].unique().tolist()),
        "instances": sorted(frame["instance"].unique().tolist()),
        "stationarity": block_summary,
        "prefix_convergence": prefix_summary,
        "alpha1p75_N512_temporal_entropy_by_instance": branch[
            "temporal_spectral_entropy"
        ].tolist(),
        "alpha1p75_N512_low_entropy_count_below_0p1": int(
            (branch["temporal_spectral_entropy"] < 0.1).sum()
        ),
        "alpha1p75_N512_high_entropy_count_above_0p3": int(
            (branch["temporal_spectral_entropy"] > 0.3).sum()
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "records.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    apply_plot_style()
    figure, axes = plt.subplots(2, 2, figsize=(8.0, 5.8), sharex=True)
    colors = {128: "#3b4cc0", 256: "#1fa187", 512: "#d1495b"}
    for axis, (metric, label) in zip(axes.flat, METRICS.items()):
        for N in sorted(frame["N"].unique()):
            group = frame.loc[frame["N"].eq(N)]
            for _, instance in group.groupby("instance"):
                axis.plot(
                    instance["alpha"],
                    instance[metric],
                    color=colors.get(int(N), "0.45"),
                    alpha=0.10,
                    linewidth=0.65,
                )
            curve = group.groupby("alpha")[metric].agg(["mean", "sem"]).reset_index()
            axis.plot(
                curve["alpha"],
                curve["mean"],
                marker="o",
                markersize=2.6,
                linewidth=1.7,
                color=colors.get(int(N), "0.45"),
                label=f"N={int(N)}",
            )
            axis.fill_between(
                curve["alpha"],
                curve["mean"] - curve["sem"],
                curve["mean"] + curve["sem"],
                color=colors.get(int(N), "0.45"),
                alpha=0.13,
                linewidth=0,
            )
        axis.axvspan(1.75, 1.80, color="0.5", alpha=0.08, linewidth=0)
        axis.set_ylabel(label)
        axis.grid(alpha=0.18)
    for axis in axes[-1]:
        axis.set_xlabel(r"local-map nonlinearity $\alpha$ ($\epsilon=0.3$)")
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8)
    figure.suptitle("Quadratic CML physics gate after a 2,000,000-step burn", y=0.995)
    figure.tight_layout()
    figure.savefig(args.output_dir / "physics-gate.png", dpi=250, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
