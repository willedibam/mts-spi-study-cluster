#!/usr/bin/env python3
"""Apply the frozen Miller--Huse SPI--SPI coordinate to confirmation data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_miller_huse_development import (  # noqa: E402
    _cell_summary,
    _metadata_frame,
    _targets,
)
from scripts.analyze_stuart_landau_development import (  # noqa: E402
    _association,
    _input_baselines,
    _load_artifact,
)
from src.order_parameter_analysis import (  # noqa: E402
    clustered_bootstrap_mae,
    clustered_bootstrap_spearman,
    safe_spearman,
)
from src.utils import load_json  # noqa: E402


EXPECTED_G = {0.185, 0.195, 0.20125, 0.20325, 0.2047, 0.20517, 0.2057, 0.20875, 0.225}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provenance(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for path in frame["path"]:
        meta = load_json(path / "meta.json")
        records.append(
            {
                "experiment_commit": str(meta["experiment"]["git_commit"]),
                "experiment_config_sha256": str(
                    meta["experiment"]["config_sha256"]
                ),
                "pyspi_config_sha256": str(meta["pyspi"]["config_sha256"]),
                "pyspi_version": json.dumps(meta["pyspi"]["version"], sort_keys=True),
                "experiment_dirty": bool(meta["experiment"]["git_dirty"]),
            }
        )
    return pd.DataFrame(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--development-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-selected-missingness", type=float, default=0.05)
    parser.add_argument("--bootstraps", type=int, default=2000)
    args = parser.parse_args()

    payload = _load_artifact(args.features)
    values = np.asarray(payload["X"], dtype=np.float32)
    frame = _metadata_frame(payload, args.data_root)
    provenance = _provenance(frame)
    frame = pd.concat([frame.reset_index(drop=True), provenance], axis=1)
    with np.load(args.model, allow_pickle=False) as archive:
        model = {name: archive[name] for name in archive.files}

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
    q_raw = (
        transformed @ np.asarray(model["components"])[0]
    ) * float(model["pc1_display_sign"])
    frame["q"] = (q_raw - float(model["q_center"])) / float(model["q_scale"])
    frame["selected_missingness"] = missingness

    design_valid = (
        len(frame) == 216
        and set(frame["M"]) == {8, 16, 32}
        and set(frame["T"]) == {1000}
        and set(frame["instance"]) == set(range(8))
        and set(frame["g"]) == EXPECTED_G
        and set(frame["mu"]) == {3.0}
        and set(frame["lattice_side"]) == {128}
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
    gates = {
        "schema_matches_frozen_model": bool(schema_matches),
        "spi_order_matches_frozen_model": bool(spi_order_matches),
        "complete_confirmation_design": bool(design_valid),
        "homogeneous_clean_provenance": bool(provenance_valid),
        "all_coordinates_finite": bool(np.isfinite(frame["q"]).all()),
        "selected_missingness_within_gate": bool(
            missingness.max() <= args.maximum_selected_missingness
        ),
    }
    eligibility = {
        "status": "eligible" if all(gates.values()) else "ineligible",
        "outcomes_read": False,
        "rows": int(len(frame)),
        "gates": gates,
        "maximum_selected_missingness": float(missingness.max()),
        "p99_selected_missingness": float(np.quantile(missingness, 0.99)),
        "feature_artifact_sha256": _sha256(args.features),
        "frozen_model_sha256": _sha256(args.model),
        "confirmation_controls": sorted(frame["g"].unique().tolist()),
        "experiment_commit": str(frame["experiment_commit"].iloc[0]),
        "experiment_config_sha256": str(
            frame["experiment_config_sha256"].iloc[0]
        ),
        "pyspi_config_sha256": str(frame["pyspi_config_sha256"].iloc[0]),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    eligibility_path = args.output_dir / "eligibility_pre_outcome.json"
    with eligibility_path.open("x", encoding="utf-8") as handle:
        json.dump(eligibility, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if not all(gates.values()):
        raise RuntimeError(f"confirmation is ineligible: {gates}")

    targets = _targets(frame["path"].tolist())
    for column in targets:
        frame[column] = targets[column].to_numpy()
    development = pd.read_csv(args.development_scores)
    development_fit = development["instance"].lt(4)
    isotonic_q = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(
        development.loc[development_fit, "q"],
        development.loc[development_fit, "Q_spin_abs"],
    )
    isotonic_control = IsotonicRegression(
        increasing="auto", out_of_bounds="clip"
    ).fit(
        development.loc[development_fit, "g"],
        development.loc[development_fit, "Q_spin_abs"],
    )
    frame["Q_hat_q"] = isotonic_q.predict(frame["q"])
    frame["Q_hat_control"] = isotonic_control.predict(frame["g"])

    baseline = _input_baselines(frame["path"].tolist())
    baseline_summary = {}
    for column in baseline:
        frame[column] = baseline[column].to_numpy()
        decoder = IsotonicRegression(
            increasing="auto", out_of_bounds="clip"
        ).fit(
            development.loc[development_fit, column],
            development.loc[development_fit, "Q_spin_abs"],
        )
        frame[f"Q_hat_{column}"] = decoder.predict(frame[column])
        baseline_summary[column] = {
            **_association(
                frame[column].to_numpy(),
                frame["Q_spin_abs"].to_numpy(),
                frame["g_group"].to_numpy(),
            ),
            "supervised_isotonic_mae": float(
                np.mean(np.abs(frame[f"Q_hat_{column}"] - frame["Q_spin_abs"]))
            ),
        }

    overall_boot, within_boot = clustered_bootstrap_spearman(
        frame["q"],
        frame["Q_spin_abs"],
        frame["g_group"],
        frame["instance"],
        n_resamples=args.bootstraps,
        seed=9311,
    )
    q_mae_boot = clustered_bootstrap_mae(
        frame["Q_spin_abs"],
        frame["Q_hat_q"],
        frame["instance"],
        n_resamples=args.bootstraps,
        seed=9312,
    )
    pooled = frame.groupby("g")[["q", "Q_spin_abs"]].mean()
    summary = {
        "status": "independent_confirmation",
        "eligibility_precedes_outcome_access": True,
        "rows": int(len(frame)),
        "association": {
            **_association(
                frame["q"].to_numpy(),
                frame["Q_spin_abs"].to_numpy(),
                frame["g_group"].to_numpy(),
            ),
            "overall_ci95": np.quantile(overall_boot, [0.025, 0.975]).tolist(),
            "within_g_ci95": np.quantile(within_boot, [0.025, 0.975]).tolist(),
        },
        "pooled_g_mean_spearman": safe_spearman(
            pooled["q"], pooled["Q_spin_abs"]
        ),
        "cell_summary": _cell_summary(frame, np.ones(len(frame), dtype=bool)),
        "supervised_q_readout": {
            "mae": float(np.mean(np.abs(frame["Q_hat_q"] - frame["Q_spin_abs"]))),
            "mae_ci95": np.quantile(q_mae_boot, [0.025, 0.975]).tolist(),
        },
        "control_only_readout": {
            "mae": float(
                np.mean(np.abs(frame["Q_hat_control"] - frame["Q_spin_abs"]))
            )
        },
        "input_baselines": baseline_summary,
        "truth_uncertainty": {
            "Q_block_mean_se_median": float(frame["Q_block_mean_se"].median()),
            "Q_block_mean_se_p95": float(frame["Q_block_mean_se"].quantile(0.95)),
        },
        "maximum_selected_missingness": float(missingness.max()),
        "p99_selected_missingness": float(np.quantile(missingness, 0.99)),
        "frozen_model_sha256": eligibility["frozen_model_sha256"],
        "feature_artifact_sha256": eligibility["feature_artifact_sha256"],
    }
    frame.drop(columns="path").to_csv(args.output_dir / "scores.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
