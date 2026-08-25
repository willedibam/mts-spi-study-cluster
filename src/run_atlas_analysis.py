"""Fit the unsupervised Zenodo SPI--SPI atlas from a unified feature artifact."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits
from umap import UMAP

from .atlas_analysis import (
    cluster_medoids,
    density_subsample_stability,
    embedding_quality,
    fit_atlas_pca,
    fit_atlas_transform,
    gmm_grid,
    hdbscan_grid,
    kmeans_grid,
    load_unified_artifact,
    predictive_subsample_stability,
    procrustes_stability,
)
from .utils import load_yaml


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(_json_value(payload), handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _selected_gmm_by_dimension(
    table: pd.DataFrame,
    models: dict[tuple[int, int, str], GaussianMixture],
    scores: np.ndarray,
    seeds: Sequence[int],
    fraction: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    minimum_size = max(5, int(np.ceil(0.01 * len(scores))))
    for dimension, group in table.groupby("dimensions"):
        eligible = group[group["minimum_cluster_size"] >= minimum_size]
        if eligible.empty:
            eligible = group
        winner = eligible.loc[eligible["bic"].idxmin()]
        n_components = int(winner["clusters"])
        covariance_type = str(winner["covariance_type"])
        one = group[group["clusters"] == 1]
        delta_one = float(one["bic"].min() - winner["bic"]) if not one.empty else np.nan
        if n_components > 1:
            stability, _ = predictive_subsample_stability(
                scores[:, : int(dimension)],
                lambda seed, k=n_components, cov=covariance_type: GaussianMixture(
                    n_components=k,
                    covariance_type=cov,
                    reg_covar=1e-6,
                    n_init=2,
                    max_iter=500,
                    random_state=seed,
                ),
                seeds=seeds,
                fraction=fraction,
            )
        else:
            stability = np.nan
        row = winner.to_dict()
        row.update({"delta_bic_vs_one": delta_one, "subsample_stability": stability})
        rows.append(row)
    return pd.DataFrame(rows)


def _select_primary_gmm(
    winners: pd.DataFrame, *, minimum_stability: float
) -> pd.Series | None:
    supported = winners[
        (winners["clusters"] > 1)
        & (winners["delta_bic_vs_one"] > 10)
        & np.isfinite(winners["subsample_stability"])
        & (winners["subsample_stability"] >= minimum_stability)
    ]
    if supported.empty:
        return None
    # Dimension is not chosen by cross-dimensional BIC. Prefer the most stable
    # supported partition, breaking near-ties toward the smaller representation.
    best_stability = float(supported["subsample_stability"].max())
    near = supported[supported["subsample_stability"] >= best_stability - 0.02]
    return near.sort_values(["dimensions", "bic"]).iloc[0]


def _embedding_grid(
    reference: np.ndarray,
    config: dict[str, Any],
    seeds: Sequence[int],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    candidates: dict[tuple[Any, ...], list[np.ndarray]] = {}

    umap_config = config["umap"]
    for neighbours in umap_config["n_neighbors"]:
        if neighbours >= len(reference):
            continue
        for min_dist in umap_config["min_dist"]:
            for metric in umap_config["metric"]:
                key = ("umap", int(neighbours), float(min_dist), str(metric))
                candidates[key] = []
                for seed in seeds:
                    embedding = UMAP(
                        n_components=2,
                        n_neighbors=int(neighbours),
                        min_dist=float(min_dist),
                        metric=str(metric),
                        random_state=int(seed),
                        n_jobs=1,
                    ).fit_transform(reference)
                    candidates[key].append(embedding)
                    rows.append(
                        {
                            "method": "umap",
                            "n_neighbors": neighbours,
                            "min_dist": min_dist,
                            "metric": metric,
                            "perplexity": np.nan,
                            "seed": seed,
                            **embedding_quality(reference, embedding),
                        }
                    )

    tsne_config = config["tsne"]
    for perplexity in tsne_config["perplexity"]:
        if perplexity >= len(reference):
            continue
        key = ("tsne", float(perplexity))
        candidates[key] = []
        for seed in seeds:
            embedding = TSNE(
                n_components=2,
                perplexity=float(perplexity),
                init="pca",
                learning_rate="auto",
                max_iter=int(tsne_config.get("max_iter", 1000)),
                random_state=int(seed),
            ).fit_transform(reference)
            candidates[key].append(embedding)
            rows.append(
                {
                    "method": "tsne",
                    "n_neighbors": np.nan,
                    "min_dist": np.nan,
                    "metric": "euclidean",
                    "perplexity": perplexity,
                    "seed": seed,
                    **embedding_quality(reference, embedding),
                }
            )

    trials = pd.DataFrame(rows)
    selected_embeddings: dict[str, np.ndarray] = {}
    selected_configs: dict[str, dict[str, Any]] = {}
    for method in ("umap", "tsne"):
        subset = trials[trials["method"] == method].copy()
        group_columns = (
            ["method", "n_neighbors", "min_dist", "metric"]
            if method == "umap"
            else ["method", "perplexity"]
        )
        metric_columns = [
            column
            for column in (
                "trustworthiness_15",
                "trustworthiness_30",
                "neighbour_recall_15",
                "neighbour_recall_30",
            )
            if column in subset
        ]
        summary = (
            subset.groupby(group_columns, dropna=False)[metric_columns]
            .mean()
            .reset_index()
        )
        stability_values: list[float] = []
        for _, row in summary.iterrows():
            key = (
                ("umap", int(row["n_neighbors"]), float(row["min_dist"]), str(row["metric"]))
                if method == "umap"
                else ("tsne", float(row["perplexity"]))
            )
            stability_values.append(procrustes_stability(candidates[key]))
        summary["seed_stability"] = stability_values
        score_columns = [
            column
            for column in ("trustworthiness_15", "neighbour_recall_15", "seed_stability")
            if column in summary
        ]
        summary["selection_score"] = np.mean(
            [summary[column].rank(pct=True) for column in score_columns], axis=0
        )
        winner = summary.loc[summary["selection_score"].idxmax()]
        key = (
            ("umap", int(winner["n_neighbors"]), float(winner["min_dist"]), str(winner["metric"]))
            if method == "umap"
            else ("tsne", float(winner["perplexity"]))
        )
        selected_embeddings[method] = candidates[key][0]
        selected_configs[method] = winner.to_dict()
    return trials, selected_embeddings, selected_configs


def run(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = load_yaml(path)
    feature_path = Path(config["feature_artifact"]).resolve()
    output_dir = Path(config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = load_unified_artifact(feature_path)
    values = np.asarray(artifact["X"], dtype=np.float64)
    random_state = int(config.get("random_state", 1729))
    threads = int(config.get("threads", 1))
    pca_config = config["pca"]
    thresholds = [float(value) for value in config["validity_thresholds"]]
    primary_threshold = float(config["primary_validity_threshold"])
    if primary_threshold not in thresholds:
        raise ValueError("primary_validity_threshold must be listed in validity_thresholds")

    sensitivity: list[dict[str, Any]] = []
    fitted: dict[float, tuple[Any, Any, np.ndarray]] = {}
    with threadpool_limits(limits=threads):
        for threshold in thresholds:
            transform = fit_atlas_transform(
                values,
                minimum_valid_fraction=threshold,
                variance_threshold=float(config.get("variance_threshold", 1e-8)),
            )
            transformed = transform.transform(values)
            pca, scores = fit_atlas_pca(
                transformed,
                n_components=int(pca_config["max_components"]),
                random_state=random_state,
            )
            fitted[threshold] = (transform, pca, scores)
            cumulative = np.cumsum(pca.explained_variance_ratio_)
            sensitivity.append(
                {
                    "validity_threshold": threshold,
                    "retained_features": len(transform.keep_indices),
                    **{
                        f"pca_cumulative_{dimension}": float(cumulative[min(dimension, len(cumulative)) - 1])
                        for dimension in (2, 5, 10, 20, 40, 80)
                    },
                }
            )

        transform, pca, scores = fitted[primary_threshold]
        cluster_config = config["clustering"]
        dimensions = [
            int(value)
            for value in cluster_config["pca_dimensions"]
            if int(value) <= scores.shape[1]
        ]
        cluster_range = list(
            range(int(cluster_config["minimum_clusters"]), int(cluster_config["maximum_clusters"]) + 1)
        )
        gmm_table, gmm_models = gmm_grid(
            scores,
            dimensions=dimensions,
            components=cluster_range,
            covariance_types=cluster_config["gmm_covariance_types"],
            random_state=random_state,
        )
        stability_seeds = [int(seed) for seed in cluster_config["stability_seeds"]]
        fraction = float(cluster_config["subsample_fraction"])
        gmm_winners = _selected_gmm_by_dimension(
            gmm_table, gmm_models, scores, stability_seeds, fraction
        )
        minimum_gmm_stability = float(cluster_config["minimum_subsample_stability"])
        primary_gmm_row = _select_primary_gmm(
            gmm_winners,
            minimum_stability=minimum_gmm_stability,
        )
        nontrivial_gmm = gmm_winners[
            (gmm_winners["clusters"] > 1)
            & (gmm_winners["delta_bic_vs_one"] > 10)
            & np.isfinite(gmm_winners["subsample_stability"])
        ]
        diagnostic_gmm_row = (
            nontrivial_gmm.sort_values(
                ["subsample_stability", "dimensions"], ascending=[False, True]
            ).iloc[0]
            if not nontrivial_gmm.empty
            else None
        )

        kmeans_table, kmeans_models = kmeans_grid(
            scores,
            dimensions=dimensions,
            clusters=[value for value in cluster_range if value >= 2],
            random_state=random_state,
        )
        kmeans_winners: list[dict[str, Any]] = []
        for dimension, group in kmeans_table.groupby("dimensions"):
            winner = group.loc[group["silhouette"].idxmax()]
            n_clusters = int(winner["clusters"])
            stability, _ = predictive_subsample_stability(
                scores[:, : int(dimension)],
                lambda seed, k=n_clusters: KMeans(n_clusters=k, n_init=20, random_state=seed),
                seeds=stability_seeds,
                fraction=fraction,
            )
            row = winner.to_dict()
            row["subsample_stability"] = stability
            kmeans_winners.append(row)
        kmeans_winners_table = pd.DataFrame(kmeans_winners)
        best_kmeans_stability = float(kmeans_winners_table["subsample_stability"].max())
        primary_kmeans_row = (
            kmeans_winners_table[
                kmeans_winners_table["subsample_stability"]
                >= best_kmeans_stability - 0.02
            ]
            .sort_values(["silhouette", "dimensions"], ascending=[False, True])
            .iloc[0]
        )
        k_dimension = int(primary_kmeans_row["dimensions"])
        k_clusters = int(primary_kmeans_row["clusters"])
        kmeans_labels = kmeans_models[(k_dimension, k_clusters)].labels_

        hdbscan_table, hdbscan_models = hdbscan_grid(
            scores,
            dimensions=dimensions,
            minimum_cluster_sizes=cluster_config["hdbscan_min_cluster_size"],
            minimum_samples=cluster_config["hdbscan_min_samples"],
        )
        eligible_hdbscan = hdbscan_table[
            (hdbscan_table["clusters"] >= 2) & (hdbscan_table["coverage"] >= 0.5)
        ]
        primary_hdbscan_row = (
            eligible_hdbscan.loc[eligible_hdbscan["selection_score"].idxmax()]
            if not eligible_hdbscan.empty
            else None
        )
        hdbscan_stability = np.nan
        if primary_hdbscan_row is not None:
            h_dimension = int(primary_hdbscan_row["dimensions"])
            h_size = int(primary_hdbscan_row["min_cluster_size"])
            h_samples = int(primary_hdbscan_row["min_samples"])
            hdbscan_stability = density_subsample_stability(
                scores[:, :h_dimension],
                lambda: HDBSCAN(min_cluster_size=h_size, min_samples=h_samples),
                seeds=stability_seeds,
                fraction=fraction,
            )
            hdbscan_labels = hdbscan_models[(h_dimension, h_size, h_samples)].labels_
        else:
            hdbscan_labels = np.full(len(values), -1, dtype=int)

        fitted_gmm_row = primary_gmm_row if primary_gmm_row is not None else diagnostic_gmm_row
        if fitted_gmm_row is not None:
            g_dimension = int(fitted_gmm_row["dimensions"])
            g_components = int(fitted_gmm_row["clusters"])
            g_covariance = str(fitted_gmm_row["covariance_type"])
            gmm = gmm_models[(g_dimension, g_components, g_covariance)]
            gmm_labels = gmm.predict(scores[:, :g_dimension])
            gmm_probability = gmm.predict_proba(scores[:, :g_dimension])
            medoids = cluster_medoids(scores[:, :g_dimension], gmm_labels)
            representatives = {
                component: int(np.argmax(gmm_probability[:, component]))
                for component in range(g_components)
            }
            gmm_weights = gmm.weights_
            gmm_means = gmm.means_
            gmm_covariances = gmm.covariances_
        else:
            g_dimension, g_components, g_covariance = 0, 0, "none"
            gmm_labels = np.full(len(values), -1, dtype=int)
            gmm_probability = np.empty((len(values), 0), dtype=float)
            medoids, representatives = {}, {}
            gmm_weights = np.empty(0, dtype=float)
            gmm_means = np.empty((0, 0), dtype=float)
            gmm_covariances = np.empty(0, dtype=float)

        for row in sensitivity:
            threshold = float(row["validity_threshold"])
            alternate_scores = fitted[threshold][2]
            if g_components > 1 and g_dimension <= alternate_scores.shape[1]:
                alternate_gmm = GaussianMixture(
                    n_components=g_components,
                    covariance_type=g_covariance,
                    reg_covar=1e-6,
                    n_init=3,
                    max_iter=500,
                    random_state=random_state,
                ).fit(alternate_scores[:, :g_dimension])
                row["gmm_partition_ari_vs_primary"] = float(
                    adjusted_rand_score(
                        gmm_labels,
                        alternate_gmm.predict(alternate_scores[:, :g_dimension]),
                    )
                )
            alternate_kmeans = KMeans(
                n_clusters=k_clusters,
                n_init=30,
                random_state=random_state,
            ).fit(alternate_scores[:, :k_dimension])
            row["kmeans_partition_ari_vs_primary"] = float(
                adjusted_rand_score(kmeans_labels, alternate_kmeans.labels_)
            )

        embedding_dimensions = min(int(config["embeddings"]["input_pca_dimensions"]), scores.shape[1])
        embedding_trials, embeddings, embedding_configs = _embedding_grid(
            scores[:, :embedding_dimensions],
            config["embeddings"],
            [int(seed) for seed in config["embeddings"]["seeds"]],
        )

    method_agreement: dict[str, float] = {}
    if g_components > 1:
        method_agreement["gmm_kmeans_ari"] = float(
            adjusted_rand_score(gmm_labels, kmeans_labels)
        )
    density_members = hdbscan_labels >= 0
    if density_members.sum() and len(np.unique(hdbscan_labels[density_members])) >= 2:
        method_agreement["hdbscan_kmeans_ari_nonnoise"] = float(
            adjusted_rand_score(
                hdbscan_labels[density_members], kmeans_labels[density_members]
            )
        )
        if g_components > 1:
            method_agreement["hdbscan_gmm_ari_nonnoise"] = float(
                adjusted_rand_score(
                    hdbscan_labels[density_members], gmm_labels[density_members]
                )
            )

    medoid_rows: list[dict[str, Any]] = []
    dataset_paths = np.asarray(artifact["dataset_paths"], dtype=object)
    dataset_names = np.asarray(artifact["y"], dtype=object)
    for cluster, index in medoids.items():
        medoid_rows.append(
            {
                "cluster": cluster,
                "role": "medoid",
                "dataset_index": index,
                "dataset": dataset_names[index],
                "dataset_path": dataset_paths[index],
                "posterior": float(gmm_probability[index, cluster]),
            }
        )
        representative = representatives[cluster]
        medoid_rows.append(
            {
                "cluster": cluster,
                "role": "highest-posterior member",
                "dataset_index": representative,
                "dataset": dataset_names[representative],
                "dataset_path": dataset_paths[representative],
                "posterior": float(gmm_probability[representative, cluster]),
            }
        )

    pd.DataFrame(sensitivity).to_csv(output_dir / "validity-pca-sensitivity.csv", index=False)
    pd.concat((gmm_table, kmeans_table, hdbscan_table), ignore_index=True).to_csv(
        output_dir / "cluster-model-grid.csv", index=False
    )
    gmm_winners.to_csv(output_dir / "gmm-dimension-winners.csv", index=False)
    kmeans_winners_table.to_csv(output_dir / "kmeans-dimension-winners.csv", index=False)
    embedding_trials.to_csv(output_dir / "embedding-grid.csv", index=False)
    pd.DataFrame(medoid_rows).to_csv(output_dir / "cluster-exemplars.csv", index=False)

    summary = {
        "feature_artifact": str(feature_path),
        "feature_artifact_sha256": _file_sha256(feature_path),
        "rows": len(values),
        "raw_features": values.shape[1],
        "primary_validity_threshold": primary_threshold,
        "retained_features": len(transform.keep_indices),
        "raw_row_validity": {
            "minimum": float(np.mean(np.isfinite(values), axis=1).min()),
            "median": float(np.median(np.mean(np.isfinite(values), axis=1))),
            "maximum": float(np.mean(np.isfinite(values), axis=1).max()),
        },
        "pca_cumulative_explained_variance": {
            str(dimension): float(np.cumsum(pca.explained_variance_ratio_)[min(dimension, len(pca.explained_variance_ratio_)) - 1])
            for dimension in (2, 5, 10, 20, 40, 80)
        },
        "primary_gmm": None if primary_gmm_row is None else primary_gmm_row.to_dict(),
        "diagnostic_gmm": None if diagnostic_gmm_row is None else diagnostic_gmm_row.to_dict(),
        "gmm_validated": primary_gmm_row is not None,
        "primary_kmeans": primary_kmeans_row.to_dict(),
        "primary_hdbscan": None if primary_hdbscan_row is None else primary_hdbscan_row.to_dict(),
        "primary_hdbscan_subsample_stability": hdbscan_stability,
        "hdbscan_validated": bool(
            primary_hdbscan_row is not None
            and np.isfinite(hdbscan_stability)
            and hdbscan_stability >= minimum_gmm_stability
        ),
        "embedding_configs": embedding_configs,
        "method_agreement": method_agreement,
        "selection_note": (
            "BIC selects k/covariance only within each fixed PCA dimension; "
            "cross-dimensional choice uses subsample stability with a smaller-dimension tie-break."
        ),
    }
    _atomic_json(output_dir / "atlas-summary.json", summary)
    _atomic_npz(
        output_dir / "atlas-results.npz",
        {
            "dataset": dataset_names,
            "dataset_paths": dataset_paths,
            "labels": artifact["labels"],
            "M": artifact["M"],
            "T": artifact["T"],
            "feature_keep_indices": transform.keep_indices,
            "feature_valid_fraction": transform.valid_fraction.astype(np.float32),
            "feature_impute_values": transform.impute_values.astype(np.float32),
            "feature_center": transform.center.astype(np.float32),
            "pca_scores": scores.astype(np.float32),
            "pca_components": pca.components_.astype(np.float32),
            "pca_explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
            "gmm_labels": gmm_labels,
            "gmm_probability": gmm_probability.astype(np.float32),
            "gmm_weights": gmm_weights.astype(np.float32),
            "gmm_means": gmm_means.astype(np.float32),
            "gmm_covariances": gmm_covariances.astype(np.float32),
            "gmm_dimension": g_dimension,
            "gmm_components": g_components,
            "gmm_covariance_type": g_covariance,
            "gmm_validated": primary_gmm_row is not None,
            "kmeans_labels": kmeans_labels,
            "kmeans_dimension": k_dimension,
            "kmeans_clusters": k_clusters,
            "hdbscan_labels": hdbscan_labels,
            "hdbscan_validated": bool(
                primary_hdbscan_row is not None
                and np.isfinite(hdbscan_stability)
                and hdbscan_stability >= minimum_gmm_stability
            ),
            "umap": embeddings["umap"].astype(np.float32),
            "tsne": embeddings["tsne"].astype(np.float32),
            "config_json": json.dumps(_json_value(config), sort_keys=True),
            "summary_json": json.dumps(_json_value(summary), sort_keys=True),
        },
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print(json.dumps(_json_value(run(args.config)), indent=2))


if __name__ == "__main__":
    main()
