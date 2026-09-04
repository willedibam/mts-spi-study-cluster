#!/usr/bin/env python3
"""Apply the frozen Desai--Zwanzig T=1000 coordinate to shorter prefixes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_desai_zwanzig_fine_boundary import (  # noqa: E402
    _association,
    _load_targets,
    _metadata_frame,
    _sha256,
)
from scripts.analyze_stuart_landau_confirmation import _steepest_interval  # noqa: E402
from scripts.analyze_stuart_landau_development import _load_artifact  # noqa: E402
from src.order_parameter_analysis import safe_spearman  # noqa: E402
from src.utils import load_yaml  # noqa: E402


def _reference_path(path: Path, reference_root: Path, T: int) -> Path:
    name = path.name.replace(f"_T{T}_", "_T1000_", 1)
    return reference_root / path.parent.name / name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reference-data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--primary-summary", type=Path, required=True)
    parser.add_argument("--primary-scores", type=Path, required=True)
    parser.add_argument("--analysis-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    contract = load_yaml(args.analysis_contract)
    design = contract["expected_design"]
    eligibility_contract = contract["eligibility"]
    expected_T = {int(value) for value in design["T_values"]}
    expected_sigmas = {float(value) for value in design["sigmas"]}
    confirmation_instances = {
        int(value) for value in design["confirmation_instances"]
    }

    payload = _load_artifact(args.features)
    values = np.asarray(payload["X"], dtype=np.float32)
    frame = _metadata_frame(payload, args.data_root)
    analysis_rows = frame["T"].isin(expected_T).to_numpy()
    frame = frame.loc[analysis_rows].reset_index(drop=True)
    values = values[analysis_rows]
    with np.load(args.model, allow_pickle=False) as archive:
        model = {name: archive[name] for name in archive.files}
    primary_summary = json.loads(args.primary_summary.read_text())

    schema_matches = str(np.asarray(model["schema_sha256"]).item()) == str(
        np.asarray(payload["schema_sha256"]).item()
    )
    spi_order_matches = np.array_equal(
        np.asarray(model["spi_order"], dtype=str),
        np.asarray(payload["spi_order"], dtype=str),
    )
    keep = np.asarray(model["keep_indices"], dtype=np.int64)
    selected = values[:, keep]
    missingness = np.mean(~np.isfinite(selected), axis=1)
    filled = np.where(
        np.isfinite(selected), selected, np.asarray(model["impute_values"])
    )
    transformed = (
        filled - np.asarray(model["center"])
    ) * np.asarray(model["block_scale"])
    raw_q = transformed @ np.asarray(model["components"])[0]
    frame["q"] = (
        raw_q - float(np.asarray(model["q_center"]).item())
    ) / float(np.asarray(model["q_scale"]).item())
    display_sign = float(primary_summary["display_sign_uses_development_Q_only"])
    frame["q_display"] = display_sign * frame["q"]
    frame["selected_missingness"] = missingness

    expected_rows = len(expected_T) * len(expected_sigmas) * int(design["instances"])
    design_valid = (
        len(frame) == expected_rows
        and set(frame["arm"]) == {"mean_field"}
        and set(frame["M"]) == {int(design["M"])}
        and set(frame["T"]) == expected_T
        and set(frame["N_full"]) == {int(design["N_full"])}
        and set(frame["instance"]) == set(range(int(design["instances"])))
        and set(frame["sigma"]) == expected_sigmas
        and set(frame["integration_scheme"]) == {"milstein"}
    )
    provenance_valid = all(
        frame[column].nunique() == 1
        for column in (
            "experiment_commit",
            "experiment_config_sha256",
            "pyspi_config_sha256",
            "pyspi_version",
            "experiment_dirty",
        )
    ) and not bool(frame["experiment_dirty"].iloc[0])

    prefix_matches = []
    for row in frame.itertuples(index=False):
        reference = _reference_path(row.path, args.reference_data_root, int(row.T))
        prefix_matches.append(
            reference.exists()
            and np.array_equal(
                np.load(row.path / "timeseries.npy"),
                np.load(reference / "timeseries.npy")[: int(row.T)],
            )
        )
    prefix_matches = np.asarray(prefix_matches, dtype=bool)

    threshold = float(
        eligibility_contract["maximum_selected_feature_missingness_per_row"]
    )
    retained = missingness <= threshold
    confirmation = frame["instance"].isin(confirmation_instances).to_numpy()
    retained_counts = (
        frame.assign(_retained=retained & confirmation)
        .groupby(["T", "sigma"])["_retained"]
        .sum()
    )
    excluded_fraction = float(np.mean(~retained))
    gates = {
        "complete_expected_design": bool(design_valid),
        "homogeneous_clean_provenance": bool(provenance_valid),
        "frozen_schema_and_full_spi_order": bool(
            schema_matches
            and spi_order_matches
            and np.asarray(payload["spi_order"]).size == 289
            and values.shape[1] == 41616
        ),
        "exact_T1000_prefixes": bool(prefix_matches.all()),
        "selected_missingness_policy_passes": bool(
            excluded_fraction
            <= float(eligibility_contract["maximum_excluded_fraction"])
            and int(retained_counts.min())
            >= int(
                eligibility_contract[
                    "minimum_retained_confirmation_instances_per_control_T"
                ]
            )
        ),
        "all_retained_coordinates_finite": bool(
            np.isfinite(frame.loc[retained, "q_display"]).all()
        ),
    }
    eligibility = {
        "status": "eligible" if all(gates.values()) else "ineligible",
        "outcomes_read": False,
        "rows": int(len(frame)),
        "retained_rows": int(retained.sum()),
        "excluded_rows": int((~retained).sum()),
        "excluded_fraction": excluded_fraction,
        "minimum_retained_confirmation_per_control_T": int(retained_counts.min()),
        "maximum_selected_missingness": float(missingness.max()),
        "gates": gates,
        "feature_artifact_sha256": _sha256(args.features),
        "frozen_model_sha256": _sha256(args.model),
        "analysis_contract_sha256": _sha256(args.analysis_contract),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "eligibility_pre_outcome.json").write_text(
        json.dumps(eligibility, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not all(gates.values()):
        raise RuntimeError(f"Desai--Zwanzig T sensitivity is ineligible: {gates}")

    targets = _load_targets(frame["path"].tolist())
    for column in targets:
        frame[column] = targets[column].to_numpy()
    frame = frame.loc[retained].reset_index(drop=True)
    confirmation = frame["instance"].isin(confirmation_instances).to_numpy()

    primary = pd.read_csv(args.primary_scores)
    primary = primary.query("arm == 'mean_field'").copy()
    primary_keyed = primary.set_index(["instance", "sigma"])
    associations = {}
    paired_T1000 = {}
    localization = {}
    target_max_difference = {}
    for T in sorted(expected_T):
        mask = confirmation & frame["T"].eq(T).to_numpy()
        associations[str(T)] = _association(frame, mask, "q_display")
        selected_frame = frame.loc[mask].copy()
        reference = primary_keyed.loc[
            list(zip(selected_frame["instance"], selected_frame["sigma"], strict=True))
        ]
        paired_T1000[str(T)] = safe_spearman(
            selected_frame["q_display"].to_numpy(),
            reference["q_display"].to_numpy(),
        )
        target_max_difference[str(T)] = float(
            np.max(
                np.abs(
                    selected_frame["Q_mean_abs"].to_numpy()
                    - reference["Q_mean_abs"].to_numpy()
                )
            )
        )
        curve = selected_frame.groupby("sigma")["q_display"].mean()
        localization[str(T)] = _steepest_interval(curve)

    summary = {
        "status": "frozen_T1000_coordinate_applied_to_exact_shorter_prefixes",
        "representation_refit_on_short_records": False,
        "associations_by_T": associations,
        "paired_q_spearman_with_T1000": paired_T1000,
        "q_steepest_interval_by_T": localization,
        "maximum_paired_future_Q_difference_by_T": target_max_difference,
        "eligibility_precedes_outcome_access": True,
    }
    frame.drop(columns="path").to_csv(args.output_dir / "scores.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
