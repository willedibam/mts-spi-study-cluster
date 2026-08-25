#!/usr/bin/env python3
"""Compare the SPI--SPI atlas geometry with an aggregated Catch22 control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _neighbours(distances: np.ndarray, k: int) -> np.ndarray:
    order = np.argsort(distances, axis=1)
    return order[:, 1 : k + 1]


def _neighbour_overlap(first: np.ndarray, second: np.ndarray, k: int) -> float:
    left = _neighbours(first, k)
    right = _neighbours(second, k)
    return float(
        np.mean([len(set(a).intersection(b)) / k for a, b in zip(left, right)])
    )


def _label_jaccard(labels: list[set[str]], neighbours: np.ndarray) -> float:
    values: list[float] = []
    for index, row in enumerate(neighbours):
        for other in row:
            union = labels[index] | labels[other]
            if union:
                values.append(len(labels[index] & labels[other]) / len(union))
    return float(np.mean(values))


def _rho(first: np.ndarray, second: np.ndarray) -> float:
    value = spearmanr(first, second).statistic
    return float(value) if np.isfinite(value) else np.nan


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def run(
    feature_path: Path,
    catch22_path: Path,
    atlas_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    with np.load(feature_path, allow_pickle=True) as feature:
        spi_dataset = np.asarray(feature["y"], dtype=str)
        spi_labels = [set(map(str, row)) for row in feature["labels"]]
        M = np.asarray(feature["M"], dtype=float)
        T = np.asarray(feature["T"], dtype=float)
        row_validity = np.mean(np.isfinite(feature["X"]), axis=1)
    with np.load(atlas_path, allow_pickle=True) as atlas:
        atlas_dataset = np.asarray(atlas["dataset"], dtype=str)
        spi_scores = np.asarray(atlas["pca_scores"], dtype=float)
        spi_variance = np.asarray(atlas["pca_explained_variance_ratio"], dtype=float)
    with np.load(catch22_path, allow_pickle=True) as catch:
        catch_dataset = np.asarray(catch["dataset"], dtype=str)
        catch_values = np.asarray(catch["X"], dtype=float)

    if not np.array_equal(spi_dataset, atlas_dataset):
        raise ValueError("SPI feature and atlas dataset orders differ")
    if not np.array_equal(spi_dataset, catch_dataset):
        raise ValueError("SPI and Catch22 dataset orders differ")
    if not np.isfinite(catch_values).all():
        raise ValueError("Catch22 control contains non-finite values")

    spi_dimension = min(
        len(spi_variance), int(np.searchsorted(np.cumsum(spi_variance), 0.95) + 1)
    )
    scaled_catch = StandardScaler().fit_transform(catch_values)
    catch_pca = PCA(
        n_components=min(100, len(catch_values) - 1, catch_values.shape[1]),
        svd_solver="full",
    ).fit(scaled_catch)
    catch_dimension = min(
        catch_pca.n_components_,
        int(np.searchsorted(np.cumsum(catch_pca.explained_variance_ratio_), 0.95) + 1),
    )
    catch_scores = catch_pca.transform(scaled_catch)[:, :catch_dimension]
    spi_scores = spi_scores[:, :spi_dimension]

    spi_condensed = pdist(spi_scores)
    catch_condensed = pdist(catch_scores)
    spi_distances = squareform(spi_condensed)
    catch_distances = squareform(catch_condensed)
    log_m = np.log(M)
    log_t = np.log(T)
    delta_log_m = pdist(log_m[:, None], metric="cityblock")
    delta_log_t = pdist(log_t[:, None], metric="cityblock")
    delta_validity = pdist(row_validity[:, None], metric="cityblock")

    ks = [value for value in (15, 30, 60, 120) if value < len(spi_dataset)]
    neighbour_rows: list[dict[str, float | int]] = []
    for k in ks:
        spi_neighbours = _neighbours(spi_distances, k)
        catch_neighbours = _neighbours(catch_distances, k)
        neighbour_rows.append(
            {
                "k": k,
                "overlap": _neighbour_overlap(spi_distances, catch_distances, k),
                "random_overlap": k / (len(spi_dataset) - 1),
                "spi_label_jaccard": _label_jaccard(spi_labels, spi_neighbours),
                "catch22_label_jaccard": _label_jaccard(spi_labels, catch_neighbours),
            }
        )

    confounding = {
        "spi_delta_log_M": _rho(spi_condensed, delta_log_m),
        "spi_delta_log_T": _rho(spi_condensed, delta_log_t),
        "spi_delta_validity": _rho(spi_condensed, delta_validity),
        "catch22_delta_log_M": _rho(catch_condensed, delta_log_m),
        "catch22_delta_log_T": _rho(catch_condensed, delta_log_t),
    }
    result: dict[str, object] = {
        "n_datasets": len(spi_dataset),
        "spi_preprocessing": "95%-valid features, median imputation, mean centring, covariance PCA",
        "catch22_preprocessing": "per-feature z-score, covariance PCA",
        "spi_pca_dimensions_used": spi_dimension,
        "spi_variance_retained": float(spi_variance[:spi_dimension].sum()),
        "spi_variance_target_95pct_achieved": bool(
            spi_variance[:spi_dimension].sum() >= 0.95
        ),
        "catch22_pca_dimensions_used": catch_dimension,
        "catch22_variance_retained": float(
            catch_pca.explained_variance_ratio_[:catch_dimension].sum()
        ),
        "pairwise_distance_spearman": _rho(spi_condensed, catch_condensed),
        "neighbourhoods": neighbour_rows,
        "distance_confounding": confounding,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "representation-comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "sans-serif"})
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.05), constrained_layout=True)

    k_values = np.asarray([row["k"] for row in neighbour_rows])
    axes[0].plot(
        k_values,
        [row["overlap"] for row in neighbour_rows],
        "-o",
        label="observed",
    )
    axes[0].plot(
        k_values,
        [row["random_overlap"] for row in neighbour_rows],
        "--",
        color="0.45",
        label="random expectation",
    )
    axes[0].set(
        xlabel="Neighbours, k",
        ylabel="Mean neighbour overlap",
        title="Local geometry agreement",
        ylim=(0, 1),
    )
    axes[0].legend(frameon=False)

    rng = np.random.default_rng(1729)
    sample = rng.choice(len(spi_condensed), size=min(60_000, len(spi_condensed)), replace=False)
    x = spi_condensed[sample] / np.median(spi_condensed)
    y = catch_condensed[sample] / np.median(catch_condensed)
    axes[1].hexbin(x, y, gridsize=55, mincnt=1, cmap="Greys", bins="log")
    axes[1].set(
        xlabel="SPI–SPI distance / median",
        ylabel="Catch22 distance / median",
        title=f"Global distance agreement (ρ={result['pairwise_distance_spearman']:.2f})",
    )

    names = [r"$|\Delta\log M|$", r"$|\Delta\log T|$", "Δ validity"]
    spi_values = [
        confounding["spi_delta_log_M"],
        confounding["spi_delta_log_T"],
        confounding["spi_delta_validity"],
    ]
    catch_values_plot = [
        confounding["catch22_delta_log_M"],
        confounding["catch22_delta_log_T"],
        np.nan,
    ]
    positions = np.arange(len(names))
    width = 0.36
    axes[2].bar(positions - width / 2, spi_values, width, label="SPI–SPI")
    axes[2].bar(
        positions + width / 2,
        catch_values_plot,
        width,
        color="0.55",
        label="Catch22",
    )
    axes[2].text(positions[2] + width / 2, 0.015, "n/a", ha="center", va="bottom", color="0.4")
    axes[2].axhline(0, color="0.2", linewidth=0.6)
    axes[2].set(
        xticks=positions,
        xticklabels=names,
        ylabel="Spearman ρ with pairwise distance",
        title="Size and validity diagnostics",
    )
    axes[2].legend(frameon=False)
    for ax in axes:
        sns.despine(ax=ax)
    fig.suptitle("SPI–SPI geometry versus aggregated Catch22 control", y=1.06)
    _save(fig, output_dir, "atlas-catch22-comparison")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature", type=Path, required=True)
    parser.add_argument("--catch22", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(
        json.dumps(
            run(
                arguments.feature,
                arguments.catch22,
                arguments.atlas,
                arguments.output_dir,
            ),
            indent=2,
        )
    )
