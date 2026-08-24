#!/usr/bin/env python3
"""Evaluate the frozen cross-M,T models on untouched confirmation rows."""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.cross_mt_transfer import (
    build_pooled_baseline_matrices,
    feature_view,
    file_sha256,
    geometry_metrics,
    held_label_permutation_test,
    load_feature_artifact,
    retrieval_scores,
    stratified_bootstrap_interval,
)
from src.utils import load_yaml


def _row_key(label: str, M: int, T: int, instance: int) -> str:
    return f"{label}|M{M}|T{T}|I{instance}"


def _validate_and_order_confirmation(
    artifact: dict[str, np.ndarray], protocol: dict[str, Any], manifest: dict[str, Any]
) -> tuple[dict[str, np.ndarray], list[str]]:
    schema = manifest["schema"]
    for key in ("feature_contract", "metric", "schema_sha256", "sym_schema_sha256", "dir_schema_sha256"):
        if str(artifact[key].item()) != str(schema[key]):
            raise ValueError(f"confirmation schema mismatch for {key}")
    provenance = str(artifact["pyspi_provenance_json"].item())
    if protocol["expected_pyspi_computation"] not in provenance:
        raise ValueError("confirmation has the wrong pyspi computation version")
    if protocol["expected_pyspi_config_sha256"] not in provenance:
        raise ValueError("confirmation has the wrong pyspi configuration")
    if '"status":"complete"' not in provenance:
        raise ValueError("confirmation pyspi provenance is incomplete")

    expected = [
        _row_key(label, M, T, instance)
        for label, M, T, instance in product(
            protocol["classes"],
            protocol["M_values"],
            protocol["T_values"],
            protocol["confirmation_instances"],
        )
    ]
    actual = [
        _row_key(str(label), int(M), int(T), int(instance))
        for label, M, T, instance in zip(
            artifact["y"], artifact["M"], artifact["T"], artifact["instance"], strict=True
        )
    ]
    if len(actual) != len(set(actual)):
        raise ValueError("confirmation rows are not unique")
    if set(expected) != set(actual):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(f"confirmation grid mismatch: missing={missing[:5]}, extra={extra[:5]}")
    by_key = {key: index for index, key in enumerate(actual)}
    order = np.asarray([by_key[key] for key in expected], dtype=int)
    for key in ("y", "dataset_paths", "M", "T", "instance", "X_sym", "X_dir"):
        artifact[key] = artifact[key][order]
    return artifact, expected


def _view_specifications(
    confirmation: dict[str, np.ndarray], baseline_matrices: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    views = {
        name: feature_view(confirmation, name)[0]
        for name in ("sym", "dir", "augmented_balanced")
    }
    views.update({name: values.astype(np.float64) for name, values in baseline_matrices.items()})
    return views


def _metric_with_interval(
    value: float,
    metric: Callable[[np.ndarray], float],
    strata: np.ndarray,
    protocol: dict[str, Any],
    seed_offset: int,
) -> dict[str, float | list[float]]:
    uncertainty = protocol["uncertainty"]
    interval = stratified_bootstrap_interval(
        metric,
        strata,
        repetitions=int(uncertainty["bootstrap_repetitions"]),
        confidence_level=float(uncertainty["confidence_level"]),
        random_state=int(uncertainty["random_state"]) + seed_offset,
    )
    return {"estimate": float(value), "confidence_interval": list(interval)}


def _class_eta(coordinates: np.ndarray, labels: np.ndarray) -> float:
    grand = np.mean(coordinates, axis=0)
    total = float(np.sum((coordinates - grand) ** 2))
    explained = 0.0
    for label in np.unique(labels):
        member = labels == label
        explained += float(np.sum(member)) * float(
            np.sum((np.mean(coordinates[member], axis=0) - grand) ** 2)
        )
    return explained / total if total > 0 else np.nan


def _cell_eta(coordinates: np.ndarray, cells: np.ndarray) -> float:
    return _class_eta(coordinates, cells)


def _retrieval_null(gallery_size: int, relevant: int) -> dict[str, float]:
    harmonic = float(np.sum(1.0 / np.arange(1, gallery_size + 1)))
    expected_ap = harmonic / gallery_size + (
        (relevant - 1) * (gallery_size - harmonic) / (gallery_size * (gallery_size - 1))
    )
    no_relevant_top5 = 1.0
    for offset in range(5):
        no_relevant_top5 *= (gallery_size - relevant - offset) / (gallery_size - offset)
    return {
        "mean_average_precision": expected_ap,
        "recall_at_1": relevant / gallery_size,
        "recall_at_5": 1.0 - no_relevant_top5,
    }


def analyze(arguments: argparse.Namespace) -> dict[str, Any]:
    protocol = load_yaml(arguments.protocol)
    manifest = json.loads(Path(arguments.manifest).read_text(encoding="utf-8"))
    if file_sha256(arguments.protocol) != manifest["protocol_sha256"]:
        raise ValueError("protocol differs from the development-frozen manifest")
    if file_sha256(arguments.model_bundle) != manifest["model_bundle"]["sha256"]:
        raise ValueError("model bundle differs from the development-frozen manifest")
    bundle = joblib.load(arguments.model_bundle)
    if bundle["protocol_sha256"] != manifest["protocol_sha256"]:
        raise ValueError("model bundle has the wrong protocol")

    confirmation, row_keys = _validate_and_order_confirmation(
        load_feature_artifact(arguments.confirmation_features), protocol, manifest
    )
    baseline_matrices, baseline_names, timeseries_hashes = build_pooled_baseline_matrices(
        confirmation["dataset_paths"].astype(str), workers=arguments.workers
    )
    baseline_cache = Path(arguments.baseline_cache)
    baseline_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        baseline_cache,
        row_keys=np.asarray(row_keys),
        dataset_paths=confirmation["dataset_paths"],
        timeseries_sha256=np.asarray(timeseries_hashes),
        pooled_univariate=baseline_matrices["pooled_univariate"],
        pooled_dependence=baseline_matrices["pooled_dependence"],
        pooled_combined=baseline_matrices["pooled_combined"],
        pooled_univariate_names=np.asarray(baseline_names["pooled_univariate"]),
        pooled_dependence_names=np.asarray(baseline_names["pooled_dependence"]),
        pooled_combined_names=np.asarray(baseline_names["pooled_combined"]),
    )

    labels = confirmation["y"].astype(str)
    M_values = confirmation["M"].astype(int)
    T_values = confirmation["T"].astype(int)
    cells = np.asarray([f"M{M}_T{T}" for M, T in zip(M_values, T_values, strict=True)])
    strata = np.asarray([f"{label}|{cell}" for label, cell in zip(labels, cells, strict=True)])
    repetitions = int(protocol["uncertainty"]["permutation_repetitions"])
    random_state = int(protocol["uncertainty"]["random_state"])
    results: dict[str, Any] = {}
    coordinate_payload: dict[str, np.ndarray] = {
        "confirmation_y": labels,
        "confirmation_M": M_values,
        "confirmation_T": T_values,
        "confirmation_instance": confirmation["instance"].astype(int),
        "development_row_keys": bundle["row_keys"],
    }

    for view_index, (view_name, values) in enumerate(
        _view_specifications(confirmation, baseline_matrices).items()
    ):
        frozen = bundle["models"][view_name]
        predicted = np.empty(len(labels), dtype=object)
        top_k = min(3, len(bundle["class_order"]))
        top3 = np.empty((len(labels), top_k), dtype=object)
        retrieval = {
            candidate_set: {
                "average_precision": np.empty(len(labels)),
                "recall_at_1": np.empty(len(labels)),
                "recall_at_5": np.empty(len(labels)),
            }
            for candidate_set in ("other_cell", "both_M_and_T_changed")
        }
        fold_results: dict[str, Any] = {}
        for held_cell in bundle["cell_order"]:
            member = cells == held_cell
            model = frozen["cell"][str(held_cell)]
            coordinates = model.projection.transform(values[member])
            predicted[member] = model.classifier.predict(coordinates)
            probability = model.classifier.predict_proba(coordinates)
            top3[member] = model.classifier.classes_[
                np.argsort(probability, axis=1)[:, -top_k:]
            ]
            cell_retrieval = retrieval_scores(
                coordinates,
                labels[member],
                model.gallery_coordinates,
                model.gallery_labels,
            )
            for metric_name, scores in cell_retrieval.items():
                retrieval["other_cell"][metric_name][member] = scores
            held_M = int(M_values[member][0])
            held_T = int(T_values[member][0])
            hard_gallery = (model.gallery_M != held_M) & (model.gallery_T != held_T)
            hard_retrieval = retrieval_scores(
                coordinates,
                labels[member],
                model.gallery_coordinates[hard_gallery],
                model.gallery_labels[hard_gallery],
            )
            for metric_name, scores in hard_retrieval.items():
                retrieval["both_M_and_T_changed"][metric_name][member] = scores
            fold_results[str(held_cell)] = {
                "rows": int(np.sum(member)),
                "balanced_accuracy": float(balanced_accuracy_score(labels[member], predicted[member])),
                "macro_f1": float(f1_score(labels[member], predicted[member], average="macro")),
                "top3_accuracy": float(np.mean(np.any(top3[member] == labels[member, None], axis=1))),
                "mean_average_precision": float(np.mean(cell_retrieval["average_precision"])),
                "recall_at_1": float(np.mean(cell_retrieval["recall_at_1"])),
                "recall_at_5": float(np.mean(cell_retrieval["recall_at_5"])),
                "hard_mean_average_precision": float(np.mean(hard_retrieval["average_precision"])),
                "hard_recall_at_1": float(np.mean(hard_retrieval["recall_at_1"])),
                "hard_recall_at_5": float(np.mean(hard_retrieval["recall_at_5"])),
            }

        correct_top3 = np.any(top3 == labels[:, None], axis=1)
        classification = {
            "balanced_accuracy": _metric_with_interval(
                balanced_accuracy_score(labels, predicted),
                lambda index: float(balanced_accuracy_score(labels[index], predicted[index])),
                strata,
                protocol,
                100 * view_index + 1,
            ),
            "macro_f1": _metric_with_interval(
                f1_score(labels, predicted, average="macro"),
                lambda index: float(f1_score(labels[index], predicted[index], average="macro")),
                strata,
                protocol,
                100 * view_index + 2,
            ),
            "top3_accuracy": _metric_with_interval(
                float(np.mean(correct_top3)),
                lambda index: float(np.mean(correct_top3[index])),
                strata,
                protocol,
                100 * view_index + 3,
            ),
        }
        null_mean, p_value = held_label_permutation_test(
            lambda permuted: float(balanced_accuracy_score(permuted, predicted)),
            labels,
            cells,
            classification["balanced_accuracy"]["estimate"],
            repetitions=repetitions,
            random_state=random_state + 100 * view_index + 4,
        )
        classification["balanced_accuracy"].update(
            {"permutation_null_mean": null_mean, "permutation_p_value": p_value}
        )

        retrieval_results: dict[str, Any] = {}
        first_model = next(iter(frozen["cell"].values()))
        first_cell = str(next(iter(frozen["cell"])))
        first_M = int(first_cell.split("_")[0][1:])
        first_T = int(first_cell.split("_")[1][1:])
        first_hard = (first_model.gallery_M != first_M) & (first_model.gallery_T != first_T)
        gallery_masks = {
            "other_cell": np.ones(len(first_model.gallery_labels), dtype=bool),
            "both_M_and_T_changed": first_hard,
        }
        for candidate_offset, (candidate_set, candidate_scores) in enumerate(retrieval.items()):
            retrieval_results[candidate_set] = {}
            for metric_offset, (metric_name, scores) in enumerate(candidate_scores.items()):
                display_name = (
                    "mean_average_precision" if metric_name == "average_precision" else metric_name
                )
                retrieval_results[candidate_set][display_name] = _metric_with_interval(
                    float(np.mean(scores)),
                    lambda index, scores=scores: float(np.mean(scores[index])),
                    strata,
                    protocol,
                    100 * view_index + 10 + 5 * candidate_offset + metric_offset,
                )
            gallery_labels = first_model.gallery_labels[gallery_masks[candidate_set]]
            relevant = int(np.sum(gallery_labels == gallery_labels[0]))
            retrieval_results[candidate_set]["random_ranking_null"] = _retrieval_null(
                len(gallery_labels), relevant
            )

        shared = frozen["shared"]
        shared_coordinates = shared.projection.transform(values)
        coordinate_payload[f"confirmation_pca_{view_name}"] = shared_coordinates.astype(np.float32)
        coordinate_payload[f"development_pca_{view_name}"] = frozen[
            "development_coordinates"
        ].astype(np.float32)
        leakage: dict[str, Any] = {}
        target_values = {"M": M_values.astype(str), "T": T_values.astype(str), "cell": cells}
        for target_offset, (target_name, target) in enumerate(target_values.items()):
            size_prediction = np.empty(len(labels), dtype=object)
            for label in protocol["classes"]:
                member = labels == label
                size_prediction[member] = shared.size_classifiers[str(label)][target_name].predict(
                    shared_coordinates[member]
                )
            target_strata = np.asarray(
                [f"{label}|{value}" for label, value in zip(labels, target, strict=True)]
            )
            estimate = float(balanced_accuracy_score(target, size_prediction))
            leakage[target_name] = _metric_with_interval(
                estimate,
                lambda index, target=target, prediction=size_prediction: float(
                    balanced_accuracy_score(target[index], prediction[index])
                ),
                target_strata,
                protocol,
                100 * view_index + 20 + target_offset,
            )
            null_mean, p_value = held_label_permutation_test(
                lambda permuted, prediction=size_prediction: float(
                    balanced_accuracy_score(permuted, prediction)
                ),
                target,
                labels,
                estimate,
                repetitions=repetitions,
                random_state=random_state + 100 * view_index + 30 + target_offset,
            )
            leakage[target_name].update(
                {"permutation_null_mean": null_mean, "permutation_p_value": p_value}
            )

        geometry = geometry_metrics(shared_coordinates, labels, cells)
        geometry["class_eta_confidence_interval"] = list(
            stratified_bootstrap_interval(
                lambda index: _class_eta(shared_coordinates[index], labels[index]),
                strata,
                repetitions=int(protocol["uncertainty"]["bootstrap_repetitions"]),
                confidence_level=float(protocol["uncertainty"]["confidence_level"]),
                random_state=random_state + 100 * view_index + 40,
            )
        )
        geometry["cell_eta_confidence_interval"] = list(
            stratified_bootstrap_interval(
                lambda index: _cell_eta(shared_coordinates[index], cells[index]),
                strata,
                repetitions=int(protocol["uncertainty"]["bootstrap_repetitions"]),
                confidence_level=float(protocol["uncertainty"]["confidence_level"]),
                random_state=random_state + 100 * view_index + 41,
            )
        )
        null_mean, p_value = held_label_permutation_test(
            lambda permuted: _class_eta(shared_coordinates, permuted),
            labels,
            cells,
            geometry["class_eta_squared"],
            repetitions=repetitions,
            random_state=random_state + 100 * view_index + 42,
        )
        geometry.update(
            {"class_eta_permutation_null_mean": null_mean, "class_eta_permutation_p_value": p_value}
        )

        results[view_name] = {
            "role": "primary" if view_name == protocol["representations"]["primary"] else "sensitivity_or_baseline",
            "classification": classification,
            "retrieval": retrieval_results,
            "size_leakage": leakage,
            "geometry": geometry,
            "folds": fold_results,
        }

    try:
        import umap

        umap_config = protocol["illustration"]["umap"]
        primary = protocol["representations"]["primary"]
        development_pca = coordinate_payload[f"development_pca_{primary}"]
        confirmation_pca = coordinate_payload[f"confirmation_pca_{primary}"]
        embedding = umap.UMAP(
            n_neighbors=int(umap_config["n_neighbors"]),
            min_dist=float(umap_config["min_dist"]),
            metric=str(umap_config["metric"]),
            random_state=int(umap_config["random_state"]),
            n_jobs=1,
        ).fit(development_pca)
        coordinate_payload["development_umap_sym"] = embedding.embedding_.astype(np.float32)
        coordinate_payload["confirmation_umap_sym"] = embedding.transform(
            confirmation_pca
        ).astype(np.float32)
        umap_status = "complete_development_fit_confirmation_transform"
    except ImportError:
        umap_status = "unavailable"

    coordinates_output = Path(arguments.coordinates_output)
    coordinates_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(coordinates_output, **coordinate_payload)
    result = {
        "study_id": protocol["study_id"],
        "status": "confirmation_evaluated",
        "rows": len(labels),
        "protocol_sha256": manifest["protocol_sha256"],
        "development_manifest_sha256": file_sha256(arguments.manifest),
        "model_bundle_sha256": manifest["model_bundle"]["sha256"],
        "confirmation_feature_sha256": file_sha256(arguments.confirmation_features),
        "confirmation_baseline_cache_sha256": file_sha256(baseline_cache),
        "coordinates_sha256": file_sha256(coordinates_output),
        "umap_status": umap_status,
        "results": results,
    }
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-bundle", required=True)
    parser.add_argument("--confirmation-features", required=True)
    parser.add_argument("--baseline-cache", required=True)
    parser.add_argument("--coordinates-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    result = analyze(parse_args())
    print(json.dumps({"study_id": result["study_id"], "status": result["status"], "rows": result["rows"]}, indent=2))
