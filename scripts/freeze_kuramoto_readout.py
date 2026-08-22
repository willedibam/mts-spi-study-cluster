#!/usr/bin/env python3
"""Freeze supervised interpretation of the already-frozen Kuramoto PC1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.freeze_kuramoto_confirmation import _records  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-dir",
        type=Path,
        default=ROOT / "data/order_parameter/kuramoto_order_benchmark",
    )
    parser.add_argument(
        "--contract-dir",
        type=Path,
        default=ROOT / "data/order_parameter/kuramoto_confirmation_contract",
    )
    args = parser.parse_args()

    representation_path = args.contract_dir / "representation_model.npz"
    contract = json.loads((args.contract_dir / "representation_contract.json").read_text())
    if contract["status"] != "eligible" or contract["outcomes_read"]:
        raise RuntimeError("representation must pass target-free gates before fitting the readout")
    if _sha256(representation_path) != contract["representation_model_sha256"]:
        raise RuntimeError("representation-model hash mismatch")

    frame = _records(args.development_dir)
    model = np.load(representation_path, allow_pickle=False)
    coordinate = model["development_pc1"]
    if coordinate.shape != (len(frame),):
        raise RuntimeError("development coordinate order/shape mismatch")
    target = np.empty(len(frame), dtype=np.float64)
    sensitivity = np.empty(len(frame), dtype=np.float64)
    for row, path in enumerate(frame["path"]):
        with np.load(path / "ground_truth.npz") as truth:
            target[row] = float(np.mean(truth["r_full_future"]))
            sensitivity[row] = float(np.mean(truth["r_unobserved_future"]))

    gaussian = frame["distribution"].eq("gaussian").to_numpy()
    readout = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(
        coordinate[gaussian], target[gaussian]
    )
    control = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(
        frame.loc[gaussian, "kappa"].to_numpy(), target[gaussian]
    )
    output_path = args.contract_dir / "readout_model.npz"
    np.savez_compressed(
        output_path,
        pc1_x=readout.X_thresholds_,
        pc1_y=readout.y_thresholds_,
        kappa_x=control.X_thresholds_,
        kappa_y=control.y_thresholds_,
        intercept=np.asarray(float(np.mean(target[gaussian]))),
        development_target=target,
        development_sensitivity_target=sensitivity,
    )
    summary = {
        "status": "frozen",
        "representation_model_sha256": _sha256(representation_path),
        "readout_model_sha256": _sha256(output_path),
        "training_rows": int(gaussian.sum()),
        "training_path": "all disclosed Gaussian development rows",
        "primary_target": "mean disjoint-future full-system R_N",
        "representation": "unsupervised and frozen before this readout",
        "readout": "supervised isotonic q-to-R map; increasing direction selected on development only",
        "confirmation_gates": {
            "all_frozen_meta_features_finite": True,
            "minimum_overall_absolute_spearman_ci95_lower_each_random_path": 0.70,
            "minimum_within_kappa_absolute_spearman_ci95_lower_each_random_path": 0.30,
            "minimum_within_kappa_hidden_complement_absolute_spearman_ci95_lower_each_random_path": 0.30,
            "maximum_calibrated_mae_ci95_upper_each_random_path": 0.10,
            "calibrated_pc1_minus_intercept_mae_ci95_upper_each_random_path": 0.0,
        },
        "interpretation": (
            "Passing supports an unsupervised order-parameter coordinate and a separately "
            "supervised numerical readout; it does not require advantage over kappa or simple statistics."
        ),
    }
    (args.contract_dir / "readout_contract.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
