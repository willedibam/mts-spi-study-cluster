#!/usr/bin/env python3
"""Aggregate Kuramoto physics-scout parts and render validation diagnostics."""

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

from scripts.kuramoto_order_parameter_scout import _jobs  # noqa: E402
from src.utils import load_yaml  # noqa: E402


def _correlation(frame: pd.DataFrame, x: str, y: str) -> float:
    if len(frame) < 3 or frame[x].nunique() < 2 or frame[y].nunique() < 2:
        return float("nan")
    return float(frame[[x, y]].corr(method="spearman").iloc[0, 1])


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
    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "physics_records.csv", index=False)

    curve_keys = ["frequency_distribution", "N_full", "kappa", "dt"]
    curves = (
        frame.groupby(curve_keys, as_index=False)
        .agg(
            r_mean=("r_full_mean", "mean"),
            r_std=("r_full_mean", "std"),
            susceptibility=("susceptibility", "mean"),
            n=("r_full_mean", "size"),
        )
    )
    curves["r_se"] = curves["r_std"] / np.sqrt(curves["n"])
    curves.to_csv(output_dir / "physics_curves.csv", index=False)

    subset_rows = []
    prefix_rows = []
    for record in records:
        base = {key: record[key] for key in curve_keys}
        base["instance"] = record["instance"]
        for m, values in record["subsets"].items():
            subset_rows.append(
                {
                    **base,
                    "M": int(m),
                    "r_m_mean": values["mean"],
                    "r_full_mean": record["r_full_mean"],
                }
            )
        longest = max(int(value) for value in record["r_full_prefix_mean"])
        reference = record["r_full_prefix_mean"][str(longest)]
        for length, value in record["r_full_prefix_mean"].items():
            prefix_rows.append(
                {
                    **base,
                    "T": int(length),
                    "absolute_error_from_longest": abs(float(value) - float(reference)),
                }
            )
    subsets = pd.DataFrame(subset_rows)
    cell_keys = ["frequency_distribution", "N_full", "kappa", "dt", "M"]
    subsets["r_m_residual"] = subsets["r_m_mean"] - subsets.groupby(cell_keys)[
        "r_m_mean"
    ].transform("mean")
    subsets["r_full_residual"] = subsets["r_full_mean"] - subsets.groupby(cell_keys)[
        "r_full_mean"
    ].transform("mean")
    residual_correlations = (
        subsets.groupby(["frequency_distribution", "N_full", "dt", "M"])
        .apply(lambda group: _correlation(group, "r_m_residual", "r_full_residual"), include_groups=False)
        .rename("within_kappa_spearman")
        .reset_index()
    )
    residual_correlations.to_csv(output_dir / "subset_correlations.csv", index=False)

    prefixes = pd.DataFrame(prefix_rows)
    prefix_summary = (
        prefixes.groupby(["frequency_distribution", "N_full", "dt", "T"], as_index=False)
        .agg(
            mean_absolute_error=("absolute_error_from_longest", "mean"),
            p95_absolute_error=("absolute_error_from_longest", lambda x: np.quantile(x, 0.95)),
        )
    )
    prefix_summary.to_csv(output_dir / "time_convergence.csv", index=False)

    stationarity = []
    for record in records:
        blocks = record["stationarity_blocks"]
        stationarity.append(abs(float(blocks[-1]["mean"]) - float(blocks[0]["mean"])))
    summary = {
        "expected_parts": expected,
        "observed_parts": len(paths),
        "mean_absolute_first_to_last_block_drift": float(np.mean(stationarity)),
        "p95_absolute_first_to_last_block_drift": float(np.quantile(stationarity, 0.95)),
        "median_elapsed_seconds": float(frame["elapsed_seconds"].median()),
    }
    (output_dir / "physics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), dpi=180)
    for (distribution, n_full, dt), group in curves.groupby(
        ["frequency_distribution", "N_full", "dt"]
    ):
        group = group.sort_values("kappa")
        label = f"{distribution}, N={n_full}, dt={dt:g}"
        axes[0, 0].plot(group["kappa"], group["r_mean"], "-o", ms=3, label=label)
        axes[0, 0].fill_between(
            group["kappa"],
            group["r_mean"] - 1.96 * group["r_se"],
            group["r_mean"] + 1.96 * group["r_se"],
            alpha=0.15,
        )
        axes[0, 1].plot(group["kappa"], group["susceptibility"], "-o", ms=3, label=label)
    axes[0, 0].set(xlabel=r"$K/K_c$", ylabel=r"mean $R_N$", title="Finite-system order curve")
    axes[0, 1].set(xlabel=r"$K/K_c$", ylabel=r"$N\,\mathrm{Var}_t(R_N)$", title="Dynamic susceptibility")

    for (distribution, n_full, dt), group in residual_correlations.groupby(
        ["frequency_distribution", "N_full", "dt"]
    ):
        axes[1, 0].plot(
            group["M"], group["within_kappa_spearman"], "-o", ms=3,
            label=f"{distribution}, N={n_full}, dt={dt:g}",
        )
    axes[1, 0].axhline(0, color="0.6", lw=0.7)
    axes[1, 0].set(
        xlabel="observed M", ylabel="within-kappa Spearman",
        title=r"Information in purpose-built $R_M$",
    )

    for (distribution, n_full, dt), group in prefix_summary.groupby(
        ["frequency_distribution", "N_full", "dt"]
    ):
        axes[1, 1].plot(
            group["T"], group["mean_absolute_error"], "-o", ms=3,
            label=f"{distribution}, N={n_full}, dt={dt:g}",
        )
    axes[1, 1].set(
        xscale="log", yscale="log", xlabel="T samples",
        ylabel="mean absolute error vs longest T", title="Time-window convergence",
    )
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=min(3, len(labels)), fontsize=7)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(output_dir / "physics_scout.png", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
