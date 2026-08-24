#!/usr/bin/env python3
"""Fit and freeze every development-only model before confirmation generation."""
from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score

from src.cross_mt_transfer import (
    FrozenCellModel,
    FrozenSharedModel,
    build_pooled_baseline_matrices,
    combine_feature_artifacts,
    feature_view,
    file_sha256,
    fit_logistic_classifier,
    fit_projection,
    load_feature_artifact,
)
from src.utils import load_yaml


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _row_key(label: str, M: int, T: int, instance: int) -> str:
    return f"{label}|M{M}|T{T}|I{instance}"


def _validate_pyspi(artifact: dict[str, np.ndarray], protocol: dict[str, Any]) -> None:
    provenance = str(artifact["pyspi_provenance_json"].item())
    if protocol["expected_pyspi_computation"] not in provenance:
        raise ValueError("artifact has the wrong pyspi computation version")
    if protocol["expected_pyspi_config_sha256"] not in provenance:
        raise ValueError("artifact has the wrong pyspi configuration")
    if '"status":"complete"' not in provenance:
        raise ValueError("artifact pyspi provenance is incomplete")


def _ordered_development(
    combined: dict[str, np.ndarray], protocol: dict[str, Any]
) -> tuple[dict[str, np.ndarray], list[str]]:
    classes = [str(value) for value in protocol["classes"]]
    M_values = [int(value) for value in protocol["M_values"]]
    T_values = [int(value) for value in protocol["T_values"]]
    instances = [int(value) for value in protocol["development_instances"]]
    expected = [
        _row_key(label, M, T, instance)
        for label, M, T, instance in product(classes, M_values, T_values, instances)
    ]
    actual = [
        _row_key(str(label), int(M), int(T), int(instance))
        for label, M, T, instance in zip(
            combined["y"], combined["M"], combined["T"], combined["instance"], strict=True
        )
    ]
    if len(actual) != len(set(actual)):
        raise ValueError("development rows are not unique")
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise ValueError(f"development grid mismatch: missing={missing[:5]}, extra={extra[:5]}")
    by_key = {key: index for index, key in enumerate(actual)}
    order = np.asarray([by_key[key] for key in expected], dtype=int)
    for key in ("y", "dataset_paths", "M", "T", "instance", "X_sym", "X_dir"):
        combined[key] = combined[key][order]
    return combined, expected


def _view_specifications(
    development: dict[str, np.ndarray], baseline_matrices: dict[str, np.ndarray]
) -> dict[str, tuple[np.ndarray, np.ndarray, bool, bool]]:
    views: dict[str, tuple[np.ndarray, np.ndarray, bool, bool]] = {}
    for name in ("sym", "dir", "augmented_balanced"):
        values, blocks, balanced = feature_view(development, name)
        views[name] = (values, blocks, balanced, False)
    for name, values in baseline_matrices.items():
        views[name] = (values.astype(np.float64), np.repeat(name, values.shape[1]), False, True)
    return views


def _projection_parameters(protocol: dict[str, Any], standardize: bool, balanced: bool) -> dict[str, Any]:
    return {
        "minimum_valid_fraction": float(protocol["preprocessing"]["minimum_valid_fraction"]),
        "variance_threshold": float(protocol["preprocessing"]["variance_threshold"]),
        "block_balanced": balanced,
        "standardize": standardize,
        "dimensions": int(protocol["projection"]["dimensions"]),
        "random_state": int(protocol["projection"]["random_state"]),
    }


def _classifier_parameters(protocol: dict[str, Any]) -> dict[str, Any]:
    config = protocol["classification"]
    return {
        "C": float(config["C"]),
        "solver": str(config["solver"]),
        "max_iter": int(config["max_iter"]),
        "tolerance": float(config["tolerance"]),
        "random_state": int(protocol["projection"]["random_state"]),
    }


def freeze(arguments: argparse.Namespace) -> dict[str, Any]:
    protocol = load_yaml(arguments.protocol)
    proof = load_feature_artifact(arguments.proof_features)
    cml = load_feature_artifact(arguments.cml_development_features)
    for artifact in (proof, cml):
        _validate_pyspi(artifact, protocol)
    development, row_keys = _ordered_development(
        combine_feature_artifacts((proof, cml)), protocol
    )

    baseline_matrices, baseline_names, timeseries_hashes = build_pooled_baseline_matrices(
        development["dataset_paths"].astype(str), workers=arguments.workers
    )
    baseline_cache = Path(arguments.baseline_cache)
    baseline_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        baseline_cache,
        row_keys=np.asarray(row_keys),
        dataset_paths=development["dataset_paths"],
        timeseries_sha256=np.asarray(timeseries_hashes),
        pooled_univariate=baseline_matrices["pooled_univariate"],
        pooled_dependence=baseline_matrices["pooled_dependence"],
        pooled_combined=baseline_matrices["pooled_combined"],
        pooled_univariate_names=np.asarray(baseline_names["pooled_univariate"]),
        pooled_dependence_names=np.asarray(baseline_names["pooled_dependence"]),
        pooled_combined_names=np.asarray(baseline_names["pooled_combined"]),
    )

    labels = development["y"].astype(str)
    M_values = development["M"].astype(int)
    T_values = development["T"].astype(int)
    cells = np.asarray([f"M{M}_T{T}" for M, T in zip(M_values, T_values, strict=True)])
    class_order = np.asarray(protocol["classes"], dtype=str)
    classifier_parameters = _classifier_parameters(protocol)
    bundle: dict[str, Any] = {
        "study_id": protocol["study_id"],
        "protocol_sha256": file_sha256(arguments.protocol),
        "row_keys": np.asarray(row_keys),
        "class_order": class_order,
        "cell_order": np.asarray(
            [f"M{M}_T{T}" for M, T in product(protocol["M_values"], protocol["T_values"])],
            dtype=str,
        ),
        "models": {},
    }
    diagnostics: dict[str, Any] = {}
    view_specifications = _view_specifications(development, baseline_matrices)
    for view_name, (values, blocks, balanced, standardize) in view_specifications.items():
        cell_models: dict[str, FrozenCellModel] = {}
        fold_diagnostics: dict[str, Any] = {}
        for held_cell in bundle["cell_order"]:
            training = cells != held_cell
            projection, coordinates = fit_projection(
                values[training],
                blocks,
                **_projection_parameters(protocol, standardize, balanced),
            )
            classifier = fit_logistic_classifier(
                coordinates, labels[training], **classifier_parameters
            )
            cell_models[str(held_cell)] = FrozenCellModel(
                held_cell=str(held_cell),
                projection=projection,
                classifier=classifier,
                gallery_coordinates=coordinates.astype(np.float32),
                gallery_labels=labels[training],
                gallery_M=M_values[training],
                gallery_T=T_values[training],
            )
            fold_diagnostics[str(held_cell)] = {
                "training_rows": int(np.sum(training)),
                "retained_features": int(projection.feature_transform.keep_indices.size),
                "pca_dimensions": int(projection.pca.n_components_),
                "pca_variance": float(np.sum(projection.pca.explained_variance_ratio_)),
            }

        shared_projection, shared_coordinates = fit_projection(
            values,
            blocks,
            **_projection_parameters(protocol, standardize, balanced),
        )
        size_classifiers: dict[str, dict[str, Any]] = {}
        targets = {"M": M_values.astype(str), "T": T_values.astype(str), "cell": cells}
        for label in class_order:
            member = labels == label
            size_classifiers[str(label)] = {
                target_name: fit_logistic_classifier(
                    shared_coordinates[member], target[member], **classifier_parameters
                )
                for target_name, target in targets.items()
            }
        bundle["models"][view_name] = {
            "cell": cell_models,
            "shared": FrozenSharedModel(shared_projection, size_classifiers),
            "development_coordinates": shared_coordinates.astype(np.float32),
        }
        diagnostics[view_name] = {
            "input_features": int(values.shape[1]),
            "folds": fold_diagnostics,
            "shared_retained_features": int(shared_projection.feature_transform.keep_indices.size),
            "shared_pca_dimensions": int(shared_projection.pca.n_components_),
            "shared_pca_variance": float(np.sum(shared_projection.pca.explained_variance_ratio_)),
        }

    cml_config = protocol["development_evidence"]["cml_panel"]
    cml_member = np.isin(labels, np.asarray(cml_config["classes"], dtype=str))
    instances = development["instance"].astype(int)
    cml_training = cml_member & np.isin(
        instances, np.asarray(cml_config["training_instances"], dtype=int)
    )
    cml_evaluation = cml_member & np.isin(
        instances, np.asarray(cml_config["evaluation_instances"], dtype=int)
    )
    cml_evidence: dict[str, Any] = {}
    for view_name in ("sym", "dir", "augmented_balanced"):
        values, blocks, balanced, standardize = view_specifications[view_name]
        projection, coordinates = fit_projection(
            values[cml_training],
            blocks,
            **_projection_parameters(protocol, standardize, balanced),
        )
        classifier = fit_logistic_classifier(
            coordinates, labels[cml_training], **classifier_parameters
        )
        prediction = classifier.predict(projection.transform(values[cml_evaluation]))
        cml_evidence[view_name] = {
            "training_rows": int(np.sum(cml_training)),
            "evaluation_rows": int(np.sum(cml_evaluation)),
            "retained_features": int(projection.feature_transform.keep_indices.size),
            "pca_variance": float(np.sum(projection.pca.explained_variance_ratio_)),
            "held_instance_balanced_accuracy": float(
                balanced_accuracy_score(labels[cml_evaluation], prediction)
            ),
        }

    model_bundle = Path(arguments.model_bundle)
    model_bundle.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_bundle, compress=3)
    manifest = {
        "study_id": protocol["study_id"],
        "status": "development_frozen_confirmation_unseen",
        "protocol_path": str(arguments.protocol),
        "protocol_sha256": file_sha256(arguments.protocol),
        "development_artifacts": {
            "proof": {"path": str(arguments.proof_features), "sha256": file_sha256(arguments.proof_features)},
            "cml_additions": {
                "path": str(arguments.cml_development_features),
                "sha256": file_sha256(arguments.cml_development_features),
            },
        },
        "rows": len(row_keys),
        "row_keys_sha256": hashlib.sha256(_canonical_json(row_keys).encode()).hexdigest(),
        "schema": {
            "feature_contract": str(development["feature_contract"].item()),
            "metric": str(development["metric"].item()),
            "schema_sha256": str(development["schema_sha256"].item()),
            "sym_schema_sha256": str(development["sym_schema_sha256"].item()),
            "dir_schema_sha256": str(development["dir_schema_sha256"].item()),
            "spi_order": development["spi_order"].astype(str).tolist(),
            "directed_spis": development["spi_order"][development["directed_flags"].astype(bool)].astype(str).tolist(),
            "symmetric_features": int(development["X_sym"].shape[1]),
            "directional_features": int(development["X_dir"].shape[1]),
        },
        "baseline_cache": {"path": str(baseline_cache), "sha256": file_sha256(baseline_cache)},
        "model_bundle": {"path": str(model_bundle), "sha256": file_sha256(model_bundle)},
        "diagnostics": diagnostics,
        "development_evidence": {
            "cml_panel_current_pyspi_v3": cml_evidence,
        },
    }
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--proof-features", required=True)
    parser.add_argument("--cml-development-features", required=True)
    parser.add_argument("--baseline-cache", required=True)
    parser.add_argument("--model-bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    result = freeze(parse_args())
    print(json.dumps({key: result[key] for key in ("study_id", "status", "rows", "schema")}, indent=2))
