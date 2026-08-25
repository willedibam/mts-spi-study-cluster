#!/usr/bin/env python3
"""Render publication-oriented figures from the fitted Zenodo corpus atlas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse, Polygon
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.stats import spearmanr
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.atlas_analysis import (  # noqa: E402
    cluster_medoids,
    cluster_tag_enrichment,
    load_unified_artifact,
)


def _style() -> None:
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png")
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def _hulls(ax: plt.Axes, coordinates: np.ndarray, labels: np.ndarray) -> None:
    for label in np.unique(labels):
        if label < 0:
            continue
        points = coordinates[labels == label]
        if len(points) < 3:
            continue
        try:
            vertices = ConvexHull(points).vertices
        except Exception:
            continue
        ax.add_patch(
            Polygon(
                points[vertices],
                closed=True,
                facecolor="0.45",
                edgecolor="0.25",
                linewidth=0.5,
                alpha=0.08,
                zorder=0,
            )
        )
        center = np.median(points, axis=0)
        ax.text(
            center[0],
            center[1],
            str(label),
            ha="center",
            va="center",
            color="0.25",
            fontsize=7,
            fontweight="bold",
            zorder=1,
        )


def plot_embedding_overview(
    results: dict[str, np.ndarray],
    output_dir: Path,
    *,
    highlight_tags: Sequence[str],
) -> None:
    pca = np.asarray(results["pca_scores"])[:, :2]
    coordinates = (("PCA", pca), ("UMAP", results["umap"]), ("t-SNE", results["tsne"]))
    validated_gmm = bool(np.asarray(results["gmm_validated"]).item())
    labels = results["gmm_labels"] if validated_gmm else results["kmeans_labels"]
    overlay_name = "validated GMM" if validated_gmm else "stable K-means sensitivity"
    tag_rows = [set(map(str, row)) for row in results["labels"]]
    palette = sns.color_palette("colorblind", n_colors=max(1, len(highlight_tags)))

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.25), constrained_layout=True)
    for ax, (name, xy) in zip(axes, coordinates):
        _hulls(ax, xy, labels)
        ax.scatter(xy[:, 0], xy[:, 1], s=8, c="0.12", alpha=0.62, linewidths=0, zorder=2)
        for color, tag in zip(palette, highlight_tags):
            member = np.asarray([tag in row for row in tag_rows])
            ax.scatter(
                xy[member, 0],
                xy[member, 1],
                s=18,
                color=color,
                edgecolor="white",
                linewidth=0.25,
                alpha=0.9,
                label=f"{tag} (n={member.sum()})",
                zorder=3,
            )
        ax.set_title(name)
        ax.set_xlabel(f"{name} 1")
        ax.set_ylabel(f"{name} 2")
        ax.text(
            0.01,
            0.99,
            f"faint hulls: {overlay_name}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            color="0.35",
        )
        sns.despine(ax=ax)
    if highlight_tags:
        handles, labels_text = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels_text, loc="outside lower center", ncol=min(3, len(handles)), frameon=False)
    fig.suptitle("SPI–SPI atlas: embeddings are views, not fitted cluster spaces", y=1.08)
    _save(fig, output_dir, "atlas-embedding-overview")


def _marginal_covariance(
    covariance: np.ndarray, covariance_type: str, component: int
) -> np.ndarray:
    if covariance_type == "full":
        return covariance[component, :2, :2]
    if covariance_type == "tied":
        return covariance[:2, :2]
    if covariance_type == "diag":
        return np.diag(covariance[component, :2])
    if covariance_type == "spherical":
        return np.eye(2) * covariance[component]
    raise ValueError(f"unknown covariance type: {covariance_type}")


def plot_gmm_diagnostics(
    results: dict[str, np.ndarray], model_grid: pd.DataFrame, output_dir: Path
) -> None:
    components = int(np.asarray(results["gmm_components"]).item())
    if components < 2:
        return
    dimension = int(np.asarray(results["gmm_dimension"]).item())
    covariance_type = str(np.asarray(results["gmm_covariance_type"]).item())
    scores = np.asarray(results["pca_scores"])
    labels = np.asarray(results["gmm_labels"])
    probability = np.asarray(results["gmm_probability"])
    means = np.asarray(results["gmm_means"])
    covariance = np.asarray(results["gmm_covariances"])
    palette = sns.color_palette("muted", n_colors=components)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), constrained_layout=True)
    grid = model_grid[
        (model_grid["method"] == "gmm") & (model_grid["dimensions"] == dimension)
    ]
    minimum = grid["bic"].min()
    for name, group in grid.groupby("covariance_type"):
        axes[0].plot(group["clusters"], group["bic"] - minimum, marker="o", ms=2.5, lw=1, label=name)
    axes[0].axvline(components, color="0.2", lw=0.8, ls="--")
    axes[0].set(xlabel="Mixture components, k", ylabel=r"$\Delta$BIC", title=f"Model selection in PCA-{dimension}")
    axes[0].legend(frameon=False, title="Covariance")
    axes[0].set_yscale("symlog", linthresh=10)
    axes[0].set_ylim(bottom=0)

    for component in range(components):
        member = labels == component
        axes[1].scatter(
            scores[member, 0], scores[member, 1], s=8, color=palette[component], alpha=0.5, linewidths=0
        )
        matrix = _marginal_covariance(covariance, covariance_type, component)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        for standard_deviations, alpha in ((1, 0.20), (2, 0.08)):
            axes[1].add_patch(
                Ellipse(
                    means[component, :2],
                    width=2 * standard_deviations * np.sqrt(max(eigenvalues[0], 0)),
                    height=2 * standard_deviations * np.sqrt(max(eigenvalues[1], 0)),
                    angle=angle,
                    facecolor=palette[component],
                    edgecolor=palette[component],
                    linewidth=0.8,
                    alpha=alpha,
                )
            )
        axes[1].text(means[component, 0], means[component, 1], str(component), fontsize=7, weight="bold")
    axes[1].set(xlabel="PC1", ylabel="PC2", title=f"Marginal {covariance_type} Gaussian components")

    axes[2].hist(probability.max(axis=1), bins=np.linspace(0, 1, 26), color="0.25", alpha=0.8)
    axes[2].set(xlabel="Maximum posterior membership", ylabel="Datasets", title="Assignment uncertainty")
    for ax in axes:
        sns.despine(ax=ax)
    status = "validated" if bool(np.asarray(results["gmm_validated"]).item()) else "diagnostic only; unstable"
    fig.suptitle(f"Gaussian mixture diagnostics ({status})", y=1.08)
    _save(fig, output_dir, "atlas-gmm-diagnostics")


def plot_missingness(
    feature: dict[str, np.ndarray], results: dict[str, np.ndarray], output_dir: Path
) -> None:
    values = np.asarray(feature["X"])
    row_validity = np.mean(np.isfinite(values), axis=1)
    feature_validity = np.mean(np.isfinite(values), axis=0)
    M = np.asarray(feature["M"], dtype=float)
    T = np.asarray(feature["T"], dtype=float)
    rho_m = spearmanr(row_validity, np.log(M)).statistic
    rho_t = spearmanr(row_validity, np.log(T)).statistic

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.8), constrained_layout=True)
    axes[0].hist(row_validity, bins=30, color="0.2")
    axes[0].axvline(np.median(row_validity), color="white", lw=1)
    axes[0].set(xlabel="Finite SPI-pair fraction per dataset", ylabel="Datasets")
    axes[1].hist(feature_validity, bins=30, color="0.2")
    for threshold in (0.90, 0.95, 1.00):
        axes[1].axvline(threshold, color="0.55", lw=0.8, ls="--")
    axes[1].set(xlabel="Finite dataset fraction per feature", ylabel="Features")
    scatter = axes[2].scatter(np.log10(T), row_validity, c=M, cmap="viridis", s=8, alpha=0.65, linewidths=0)
    axes[2].set(xlabel=r"$\log_{10}(T)$", ylabel="Finite SPI-pair fraction")
    axes[2].set_title(rf"Spearman $\rho_M={rho_m:.02f}$, $\rho_T={rho_t:.02f}$")
    fig.colorbar(scatter, ax=axes[2], label="Channels, M")
    for ax in axes:
        sns.despine(ax=ax)
    fig.suptitle("Estimator validity is structured missingness, not corpus non-completion", y=1.02)
    _save(fig, output_dir, "atlas-validity-diagnostics")


def _bin_columns(values: np.ndarray, maximum_columns: int = 2600) -> np.ndarray:
    width = int(np.ceil(values.shape[1] / maximum_columns))
    padding = (-values.shape[1]) % width
    if padding:
        values = np.pad(values, ((0, 0), (0, padding)), constant_values=np.nan)
    reshaped = values.reshape(values.shape[0], -1, width)
    finite = np.isfinite(reshaped)
    counts = finite.sum(axis=2)
    return np.divide(
        np.where(finite, reshaped, 0).sum(axis=2),
        counts,
        out=np.full(counts.shape, np.nan, dtype=np.float32),
        where=counts > 0,
    )


def plot_full_feature_matrix(
    feature: dict[str, np.ndarray], results: dict[str, np.ndarray], output_dir: Path
) -> None:
    values = np.asarray(feature["X"], dtype=np.float32)
    cluster_labels = (
        results["gmm_labels"]
        if bool(np.asarray(results["gmm_validated"]).item())
        else results["kmeans_labels"]
    )
    order = np.lexsort((results["pca_scores"][:, 0], cluster_labels))
    display = _bin_columns(values[order])
    cmap = plt.get_cmap("vlag").copy()
    cmap.set_bad("0.88")
    fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    image = ax.imshow(display, aspect="auto", interpolation="nearest", cmap=cmap, vmin=-1, vmax=1, rasterized=True)
    ax.set(
        xlabel=f"SPI-pair feature index (all {values.shape[1]:,}; adjacent columns binned for display)",
        ylabel=f"Datasets (all {values.shape[0]:,}; ordered by fitted cluster then PC1)",
        title="Full Pearson SPI–SPI meta-feature matrix",
    )
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    _save(fig, output_dir, "atlas-full-feature-matrix")


def _pair_matrix(feature: dict[str, np.ndarray], row: np.ndarray) -> np.ndarray:
    spi_order = [str(name) for name in feature["spi_order"]]
    lookup = {name: index for index, name in enumerate(spi_order)}
    matrix = np.full((len(spi_order), len(spi_order)), np.nan, dtype=np.float32)
    for value, first, second in zip(row, feature["feature_spi_a"], feature["feature_spi_b"]):
        i, j = lookup[str(first)], lookup[str(second)]
        matrix[i, j] = matrix[j, i] = value
    valid = np.isfinite(matrix).any(axis=1)
    matrix[np.flatnonzero(valid), np.flatnonzero(valid)] = 1.0
    return matrix


def plot_selected_feature_matrices(
    feature: dict[str, np.ndarray], results: dict[str, np.ndarray], exemplars: pd.DataFrame, output_dir: Path
) -> None:
    if "method" not in exemplars:
        exemplars = exemplars.assign(method="gmm")
    primary_method = (
        "gmm" if bool(np.asarray(results["gmm_validated"]).item()) else "kmeans"
    )
    selected = exemplars[
        (exemplars["method"] == primary_method) & (exemplars["role"] == "medoid")
    ]
    selected = selected.head(6)
    if selected.empty:
        return
    columns = min(3, len(selected))
    rows = int(np.ceil(len(selected) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.2 * columns + 0.7, 3.0 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    visible_axes: list[plt.Axes] = []
    for ax, (_, exemplar) in zip(axes.flat, selected.iterrows()):
        visible_axes.append(ax)
        index = int(exemplar["dataset_index"])
        matrix = _pair_matrix(feature, feature["X"][index])
        image = ax.imshow(matrix, cmap="vlag", vmin=-1, vmax=1, interpolation="nearest", rasterized=True)
        ax.set_title(f"Cluster {int(exemplar['cluster'])}: {exemplar['dataset']}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.flat[len(selected) :]:
        ax.axis("off")
    if image is not None:
        fig.colorbar(
            image,
            ax=visible_axes,
            shrink=0.75,
            pad=0.02,
            label="Pearson SPI–SPI similarity",
        )
    fig.suptitle(
        "Observed cluster medoids: complete 289 × 289 SPI similarity matrices",
        y=1.08,
    )
    _save(fig, output_dir, "atlas-medoid-feature-matrices")


def _complete_exemplars(results: dict[str, np.ndarray]) -> pd.DataFrame:
    """Reconstruct all method exemplars for artifacts made before that table existed."""

    scores = np.asarray(results["pca_scores"])
    datasets = np.asarray(results["dataset"], dtype=object)
    paths = np.asarray(results["dataset_paths"], dtype=object)
    rows: list[dict[str, object]] = []

    gmm_labels = np.asarray(results["gmm_labels"])
    probability = np.asarray(results["gmm_probability"])
    gmm_dimension = int(np.asarray(results["gmm_dimension"]).item())
    if gmm_dimension and probability.shape[1]:
        for cluster, index in cluster_medoids(scores[:, :gmm_dimension], gmm_labels).items():
            rows.append(
                {
                    "method": "gmm",
                    "cluster": cluster,
                    "role": "medoid",
                    "dataset_index": index,
                    "dataset": datasets[index],
                    "dataset_path": paths[index],
                    "posterior": float(probability[index, cluster]),
                }
            )
            representative = int(np.argmax(probability[:, cluster]))
            rows.append(
                {
                    "method": "gmm",
                    "cluster": cluster,
                    "role": "highest-posterior member",
                    "dataset_index": representative,
                    "dataset": datasets[representative],
                    "dataset_path": paths[representative],
                    "posterior": float(probability[representative, cluster]),
                }
            )

    for method, label_key, dimension in (
        (
            "kmeans",
            "kmeans_labels",
            int(np.asarray(results["kmeans_dimension"]).item()),
        ),
        (
            "hdbscan",
            "hdbscan_labels",
            int(
                (json.loads(str(np.asarray(results["summary_json"]).item())).get("primary_hdbscan") or {}).get(
                    "dimensions", 0
                )
            ),
        ),
    ):
        labels = np.asarray(results[label_key])
        if not dimension or not np.any(labels >= 0):
            continue
        for cluster, index in cluster_medoids(scores[:, :dimension], labels).items():
            rows.append(
                {
                    "method": method,
                    "cluster": cluster,
                    "role": "medoid",
                    "dataset_index": index,
                    "dataset": datasets[index],
                    "dataset_path": paths[index],
                    "posterior": np.nan,
                }
            )
    return pd.DataFrame(rows)


def run(
    feature_path: Path,
    results_path: Path,
    analysis_dir: Path,
    output_dir: Path,
    *,
    highlight_tags: Sequence[str],
) -> None:
    _style()
    feature = load_unified_artifact(feature_path)
    with np.load(results_path, allow_pickle=True) as archive:
        results = {name: archive[name] for name in archive.files}
    model_grid = pd.read_csv(analysis_dir / "cluster-model-grid.csv")
    exemplars = _complete_exemplars(results)
    exemplars.to_csv(analysis_dir / "cluster-exemplars.csv", index=False)
    enrichment_tables = [
        cluster_tag_enrichment(results["gmm_labels"], results["labels"].tolist(), method="gmm"),
        cluster_tag_enrichment(results["kmeans_labels"], results["labels"].tolist(), method="kmeans"),
    ]
    if np.any(results["hdbscan_labels"] >= 0):
        enrichment_tables.append(
            cluster_tag_enrichment(results["hdbscan_labels"], results["labels"].tolist(), method="hdbscan")
        )
    pd.concat(enrichment_tables, ignore_index=True).to_csv(analysis_dir / "cluster-tag-enrichment.csv", index=False)

    plot_embedding_overview(results, output_dir, highlight_tags=highlight_tags)
    plot_gmm_diagnostics(results, model_grid, output_dir)
    plot_missingness(feature, results, output_dir)
    plot_full_feature_matrix(feature, results, output_dir)
    plot_selected_feature_matrices(feature, results, exemplars, output_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--highlight-tag", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run(
        args.feature.resolve(),
        args.results.resolve(),
        args.analysis_dir.resolve(),
        args.output_dir.resolve(),
        highlight_tags=args.highlight_tag,
    )


if __name__ == "__main__":
    main()
