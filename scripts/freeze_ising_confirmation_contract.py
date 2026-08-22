#!/usr/bin/env python3
"""Derive the target-only Ising confirmation contract before SPI evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression


CALIBRATION_CONTROLS = np.asarray(
    [0.75, 0.92, 0.98, 1.005, 1.025, 1.1, 1.4], dtype=np.float64
)
ACCURACY_MARGIN = 0.06
TRACKING_SPEARMAN_LOWER_BOUND = 0.80
PATH_EQUIVALENCE_MARGIN = 0.04
N_BOOTSTRAPS = 5000


def _is_calibration(controls: np.ndarray) -> np.ndarray:
    return np.any(
        np.isclose(controls[:, None], CALIBRATION_CONTROLS[None, :]), axis=1
    )


def _cell_rows(archive: np.lib.npyio.NpzFile, indices: np.ndarray) -> np.ndarray:
    paths = archive["path"][indices]
    controls = archive["control"][indices]
    truth = archive["q_future_abs"][indices]
    local = archive["patch_q_rms"][indices]
    rows = []
    for path in ("isotropic", "anisotropic"):
        for control in np.unique(controls):
            selected = (paths == path) & np.isclose(controls, control)
            rows.append(
                (path, float(control), float(truth[selected].mean()), float(local[selected].mean()))
            )
    return np.asarray(rows, dtype=object)


def _mae_triplet(cells: np.ndarray) -> np.ndarray:
    paths = cells[:, 0].astype(str)
    controls = cells[:, 1].astype(float)
    truth = cells[:, 2].astype(float)
    local = cells[:, 3].astype(float)
    calibration = (paths == "isotropic") & _is_calibration(controls)
    prediction = IsotonicRegression(out_of_bounds="clip").fit(
        local[calibration], truth[calibration]
    ).predict(local)
    held_control = (paths == "isotropic") & ~_is_calibration(controls)
    held_path = paths == "anisotropic"
    held_both = held_path & ~_is_calibration(controls)
    return np.asarray(
        [
            np.mean(np.abs(prediction[held_control] - truth[held_control])),
            np.mean(np.abs(prediction[held_path] - truth[held_path])),
            np.mean(np.abs(prediction[held_both] - truth[held_both])),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("data/order_parameter/kinetic_ising_truth_bank_records.npz"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with np.load(args.records) as archive:
        paths = archive["path"]
        controls = archive["control"]
        instances = archive["instance"]
        point_cells = _cell_rows(archive, np.arange(paths.size))
        cell_ses = []
        path_means = {}
        for path in ("isotropic", "anisotropic"):
            for control in np.unique(controls):
                selected = (paths == path) & np.isclose(controls, control)
                values = archive["q_future_abs"][selected]
                cell_ses.append(float(values.std(ddof=1) / np.sqrt(values.size)))
                path_means[path, float(control)] = float(values.mean())

        rng = np.random.default_rng(20260822)
        draws = np.empty((N_BOOTSTRAPS, 3), dtype=np.float64)
        sampled = np.empty(paths.size, dtype=np.int64)
        for draw in range(N_BOOTSTRAPS):
            cursor = 0
            for path in ("isotropic", "anisotropic"):
                for control in np.unique(controls):
                    cell = np.flatnonzero((paths == path) & np.isclose(controls, control))
                    chosen = rng.choice(cell, size=cell.size, replace=True)
                    sampled[cursor : cursor + cell.size] = chosen
                    cursor += cell.size
            draws[draw] = _mae_triplet(_cell_rows(archive, sampled))

    point_truth = point_cells[:, 2].astype(float)
    point_local = point_cells[:, 3].astype(float)
    gaps = np.asarray(
        [
            abs(path_means["isotropic", float(control)] - path_means["anisotropic", float(control)])
            for control in np.unique(controls)
        ]
    )
    labels = ("held_control", "held_path", "held_path_and_control")
    result = {
        "source": str(args.records),
        "uses_spi_features": False,
        "primary_target": "independent cell E[|m|]",
        "local_oracle": "cell mean patch RMS magnetization with isotonic calibration",
        "calibration_path": "isotropic",
        "calibration_controls": CALIBRATION_CONTROLS.tolist(),
        "cell_level_local_oracle_spearman": float(
            spearmanr(point_local, point_truth).statistic
        ),
        "local_oracle_point_mae": dict(zip(labels, _mae_triplet(point_cells), strict=True)),
        "local_oracle_bootstrap_p95_mae": dict(
            zip(labels, np.quantile(draws, 0.95, axis=0), strict=True)
        ),
        "truth_bank_maximum_cell_se": float(max(cell_ses)),
        "matched_path_gap_mean": float(gaps.mean()),
        "matched_path_gap_maximum": float(gaps.max()),
        "frozen_gates": {
            "numerical_recovery_mae_max": ACCURACY_MARGIN,
            "tracking_absolute_spearman_lower_bound_min": TRACKING_SPEARMAN_LOWER_BOUND,
            "matched_path_absolute_difference_max": PATH_EQUIVALENCE_MARGIN,
        },
        "bootstrap": {
            "draws": N_BOOTSTRAPS,
            "unit": "independent truth-chain mean within each path/control cell",
            "seed": 20260822,
        },
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
