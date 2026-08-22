#!/usr/bin/env python3
"""Target-blind diagnosis of failed Kuramoto confirmation eligibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_kuramoto_confirmation import _records  # noqa: E402
from src.order_parameter_features import (  # noqa: E402
    build_meta_feature_matrix,
    stable_spi_names,
    validate_spi_catalogs,
)
from src.utils import load_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data/order_parameter/kuramoto_confirmation",
    )
    parser.add_argument(
        "--contract-dir",
        type=Path,
        default=ROOT / "data/order_parameter/kuramoto_confirmation_contract",
    )
    args = parser.parse_args()

    frame = _records(args.data_dir)
    model = np.load(args.contract_dir / "representation_model.npz", allow_pickle=False)
    core_spis = model["core_spis"].astype(str).tolist()
    core_spi_set = set(core_spis)
    catalog = validate_spi_catalogs(frame["path"].tolist())
    matrix, _ = build_meta_feature_matrix(frame["path"].tolist(), catalog, core_spis)
    selected_indices = model["pc_feature_indices"].astype(int)
    selected_missing = ~np.isfinite(matrix[:, selected_indices])
    row_fraction = selected_missing.mean(axis=1)
    pair_left = model["pair_left"][selected_indices].astype(int)
    pair_right = model["pair_right"][selected_indices].astype(int)

    class_summary = {}
    for name, group in frame.groupby("class_name"):
        values = row_fraction[group.index.to_numpy()]
        class_summary[name] = {
            "rows": int(values.size),
            "rows_with_missingness": int(np.sum(values > 0.0)),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "maximum": float(np.max(values)),
        }

    details = []
    for row in np.flatnonzero(row_fraction > 0.0):
        missing_features = np.flatnonzero(selected_missing[row])
        incidence = np.bincount(
            np.r_[pair_left[missing_features], pair_right[missing_features]],
            minlength=len(core_spis),
        )
        implicated = np.flatnonzero(incidence)
        order = implicated[np.argsort(incidence[implicated])[::-1]]
        meta = load_json(frame.loc[row, "path"] / "meta.json")
        core_errors = {
            name: error
            for name, error in meta["pyspi"].get("errors", {}).items()
            if name in core_spi_set
        }
        details.append(
            {
                "class_name": frame.loc[row, "class_name"],
                "distribution": frame.loc[row, "distribution"],
                "frequency_sampling": frame.loc[row, "frequency_sampling"],
                "design": frame.loc[row, "design"],
                "kappa": float(frame.loc[row, "kappa"]),
                "instance": int(frame.loc[row, "instance"]),
                "selected_missing_fraction": float(row_fraction[row]),
                "selected_missing_features": int(missing_features.size),
                "implicated_core_spis": int(implicated.size),
                "top_implicated_spis": [
                    {"name": core_spis[index], "missing_pair_incidence": int(incidence[index])}
                    for index in order[:20]
                ],
                "recorded_core_errors": core_errors,
            }
        )

    confirmation_stable, rates = stable_spi_names(
        frame["path"].tolist(), catalog, min_valid_fraction=1.0
    )
    confirmation_stable_set = set(confirmation_stable)
    safe_core = [name for name in core_spis if name in confirmation_stable_set]
    summary = {
        "outcomes_read": False,
        "rows": len(frame),
        "core_spis": len(core_spis),
        "selected_meta_features": int(selected_indices.size),
        "rows_with_missingness": int(np.sum(row_fraction > 0.0)),
        "features_ever_missing": int(np.any(selected_missing, axis=0).sum()),
        "maximum_row_missingness": float(np.max(row_fraction)),
        "confirmation_zero_failure_spis_total": len(confirmation_stable),
        "frozen_core_spis_zero_failure_on_confirmation": len(safe_core),
        "frozen_core_spis_not_zero_failure": [name for name in core_spis if name not in confirmation_stable_set],
        "minimum_frozen_core_validity_rate": float(min(rates[name] for name in core_spis)),
        "class_summary": class_summary,
        "missing_row_details": details,
    }
    output = args.contract_dir / "confirmation_eligibility_diagnosis.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
