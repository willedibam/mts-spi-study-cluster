"""Exploratory comparisons of symmetric and directional SPI--SPI blocks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import Isomap
from sklearn.metrics import balanced_accuracy_score

from src.cml_order_parameter import spatial_power_distribution
from src.spi_spi_analysis import fit_feature_transform, fit_frozen_pc1


def _load(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        payload = {name: archive[name] for name in archive.files}
    if payload.get("feature_contract", np.array("")).item() != "direction_preserving_v2":
        raise ValueError(f"not a direction_preserving_v2 artifact: {path}")
    return payload


def _assert_same_schema(artifacts: list[dict[str, np.ndarray]]) -> None:
    hashes = {str(artifact["schema_sha256"].item()) for artifact in artifacts}
    if len(hashes) != 1:
        raise ValueError("feature artifacts have different schemas")


def _view(
    artifact: dict[str, np.ndarray], name: str
) -> tuple[np.ndarray, np.ndarray, bool]:
    if name == "sym":
        return artifact["X_sym"], np.repeat("sym", artifact["X_sym"].shape[1]), False
    if name == "dir":
        return artifact["X_dir"], np.repeat("dir", artifact["X_dir"].shape[1]), False
    if name == "augmented_balanced":
        return (
            np.concatenate((artifact["X_sym"], artifact["X_dir"]), axis=1),
            np.concatenate(
                (
                    np.repeat("sym", artifact["X_sym"].shape[1]),
                    np.repeat("dir", artifact["X_dir"].shape[1]),
                )
            ),
            True,
        )
    raise ValueError(name)


def _residualize(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    for group in np.unique(groups):
        member = groups == group
        result[member] -= result[member].mean()
    return result


def _abs_spearman(first: np.ndarray, second: np.ndarray) -> float:
    return float(abs(spearmanr(first, second).statistic))


def _load_cml_observables(paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected_power: list[float] = []
    temporal_entropy: list[float] = []
    for dataset_path in paths:
        field = np.load(Path(str(dataset_path)) / "timeseries.npy")
        probabilities = spatial_power_distribution(field)
        # For M=20 these are exactly k/pi=0.3 and 0.4, frozen by the
        # original discovery split before its confirmation analysis.
        selected_power.append(float(probabilities[[2, 3]].sum()))
        centred = field - field.mean(axis=0, keepdims=True)
        power = np.mean(np.abs(np.fft.rfft(centred, axis=0)) ** 2, axis=1)[1:]
        probability = power / power.sum()
        positive = probability > 0
        temporal_entropy.append(
            float(
                -np.sum(probability[positive] * np.log(probability[positive]))
                / np.log(len(probability))
            )
        )
    return np.asarray(selected_power), np.asarray(temporal_entropy)


def analyze_cml(arguments: argparse.Namespace) -> dict[str, object]:
    artifact = _load(arguments.features)
    alpha = np.asarray(
        [float(str(value).replace("a", "").replace("p", ".")) for value in artifact["variant"]]
    )
    instance = artifact["instance"].astype(int)
    alpha_index = np.rint((alpha - 1.6) * 100).astype(int)
    development = (instance <= 9) & (alpha_index % 2 == 0)
    confirmation = (instance >= 10) & (alpha_index % 2 == 1)
    q_selected, temporal_entropy = _load_cml_observables(artifact["dataset_paths"])

    results: dict[str, object] = {}
    for view_name in ("sym", "dir", "augmented_balanced"):
        values, blocks, balanced = _view(artifact, view_name)
        transform = fit_feature_transform(
            values[development],
            blocks,
            minimum_valid_fraction=1.0,
            variance_threshold=1e-8,
            block_balanced=balanced,
        )
        fitted = transform.transform(values[development])
        projected = transform.transform(values[confirmation])
        dimensions = min(10, fitted.shape[0] - 1, fitted.shape[1])
        pca = PCA(n_components=dimensions, svd_solver="randomized", random_state=0)
        fitted_pca = pca.fit_transform(fitted)
        projected_pca = pca.transform(projected)
        neighbours = min(15, len(fitted_pca) - 1)
        isomap = Isomap(n_neighbors=neighbours, n_components=1, eigen_solver="arpack")
        isomap.fit(fitted_pca)
        coordinate = isomap.transform(projected_pca)[:, 0]
        pc1 = pca.transform(projected)[:, 0]
        alpha_test = alpha[confirmation]
        target_test = q_selected[confirmation]
        results[view_name] = {
            "input_features": int(values.shape[1]),
            "retained_features": int(transform.keep_indices.size),
            "pca_dimensions": int(dimensions),
            "pca_variance": float(pca.explained_variance_ratio_.sum()),
            "isomap_vs_Q_selected": _abs_spearman(coordinate, target_test),
            "isomap_vs_alpha": _abs_spearman(coordinate, alpha_test),
            "isomap_vs_Q_selected_within_alpha": _abs_spearman(
                _residualize(coordinate, alpha_test),
                _residualize(target_test, alpha_test),
            ),
            "pc1_vs_Q_selected": _abs_spearman(pc1, target_test),
        }
    results["temporal_entropy_vs_Q_selected"] = _abs_spearman(
        temporal_entropy[confirmation], q_selected[confirmation]
    )
    return {
        "analysis": "exploratory_directional_cml_sensitivity",
        "development_rows": int(development.sum()),
        "confirmation_rows": int(confirmation.sum()),
        "results": results,
    }


def _kuramoto_metadata(paths: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target: list[float] = []
    kappa: list[float] = []
    distribution: list[str] = []
    for dataset_path in paths:
        path = Path(str(dataset_path))
        metadata = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        with np.load(path / "ground_truth.npz") as truth:
            target.append(float(np.mean(truth["r_full_future"])))
        kappa.append(float(metadata["generator"]["control"]["reduced_value"]))
        distribution.append(
            str(metadata["generator"]["resolved_params"]["frequency_distribution"])
        )
    return np.asarray(target), np.asarray(kappa), np.asarray(distribution)


def analyze_kuramoto(arguments: argparse.Namespace) -> dict[str, object]:
    development_artifacts = [_load(path) for path in arguments.development_features]
    confirmation = _load(arguments.confirmation_features)
    _assert_same_schema(development_artifacts + [confirmation])

    fitted_coordinates: dict[str, tuple[np.ndarray, dict[str, object]]] = {}
    for view_name in ("sym", "dir", "augmented_balanced"):
        development_views = [_view(artifact, view_name) for artifact in development_artifacts]
        values = np.concatenate([view[0] for view in development_views], axis=0)
        blocks = development_views[0][1]
        balanced = development_views[0][2]
        confirmation_values = _view(confirmation, view_name)[0]
        model = fit_frozen_pc1(
            values,
            blocks,
            minimum_valid_fraction=1.0,
            variance_threshold=0.05,
            block_balanced=balanced,
        )
        coordinate = model.transform(confirmation_values)
        fitted_coordinates[view_name] = (coordinate, {
            "input_features": int(values.shape[1]),
            "retained_features": int(model.feature_transform.keep_indices.size),
            "pc1_variance": model.explained_variance_ratio,
        })

    # The target is opened only after every representation has been fitted.
    target, kappa, distribution = _kuramoto_metadata(confirmation["dataset_paths"])
    primary = np.char.find(confirmation["y"].astype(str), "regular") < 0
    paired = np.char.find(confirmation["y"].astype(str), "paired") >= 0
    evaluation = primary & paired
    results: dict[str, object] = {}
    for view_name, (coordinate, view_results) in fitted_coordinates.items():
        for law in ("gaussian", "logistic"):
            selected = evaluation & (distribution == law)
            view_results[law] = {
                "pc1_vs_future_R": _abs_spearman(coordinate[selected], target[selected]),
                "pc1_vs_future_R_within_kappa": _abs_spearman(
                    _residualize(coordinate[selected], kappa[selected]),
                    _residualize(target[selected], kappa[selected]),
                ),
            }
        results[view_name] = view_results
    return {
        "analysis": "exploratory_directional_kuramoto_sensitivity",
        "development_rows": int(sum(len(a["y"]) for a in development_artifacts)),
        "confirmation_rows": int(evaluation.sum()),
        "results": results,
    }


def analyze_proof(arguments: argparse.Namespace) -> dict[str, object]:
    artifact = _load(arguments.features)
    instance = artifact["instance"].astype(int)
    development = instance <= 5
    confirmation = instance >= 6
    labels = artifact["y"].astype(str)
    results: dict[str, object] = {}
    for view_name in ("sym", "dir", "augmented_balanced"):
        values, blocks, balanced = _view(artifact, view_name)
        transform = fit_feature_transform(
            values[development],
            blocks,
            minimum_valid_fraction=0.95,
            variance_threshold=1e-8,
            block_balanced=balanced,
        )
        fitted = transform.transform(values[development])
        projected = transform.transform(values[confirmation])
        dimensions = min(50, fitted.shape[0] - 1, fitted.shape[1])
        pca = PCA(n_components=dimensions, svd_solver="randomized", random_state=0)
        fitted = pca.fit_transform(fitted)
        projected = pca.transform(projected)
        classifier = LogisticRegression(max_iter=3000, C=1.0).fit(
            fitted, labels[development]
        )
        predicted = classifier.predict(projected)
        results[view_name] = {
            "input_features": int(values.shape[1]),
            "retained_features": int(transform.keep_indices.size),
            "pca_dimensions": int(dimensions),
            "pca_variance": float(pca.explained_variance_ratio_.sum()),
            "held_instance_balanced_accuracy": float(
                balanced_accuracy_score(labels[confirmation], predicted)
            ),
        }
    return {
        "analysis": "exploratory_directional_proof_sensitivity",
        "development_rows": int(development.sum()),
        "confirmation_rows": int(confirmation.sum()),
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="analysis", required=True)
    for name in ("cml", "proof"):
        child = subparsers.add_parser(name)
        child.add_argument("--features", required=True)
        child.add_argument("--output", required=True)
    kuramoto = subparsers.add_parser("kuramoto")
    kuramoto.add_argument("--development-features", nargs="+", required=True)
    kuramoto.add_argument("--confirmation-features", required=True)
    kuramoto.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    result = {
        "cml": analyze_cml,
        "kuramoto": analyze_kuramoto,
        "proof": analyze_proof,
    }[arguments.analysis](arguments)
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
