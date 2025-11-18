from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from .plot_style import apply_plot_style
from .utils import ensure_dir, load_json, project_root

LOGGER = logging.getLogger(__name__)


def _mode_results_dir(mode: str) -> Path:
    base = ensure_dir(project_root() / "results")
    return ensure_dir(base / mode)


def _load_feature_archive(mode: str) -> dict[str, np.ndarray]:
    path = _mode_results_dir(mode) / f"spi_space_features_{mode}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature archive not found at {path}. Run spi_space_features first."
        )
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _build_pair_index(feature_pairs: np.ndarray) -> dict[tuple[int, int], int]:
    return {tuple(pair): idx for idx, pair in enumerate(feature_pairs)}


def _resolve_spi_selection(
    selection_path: Path,
    spi_names: list[str],
    classifier_name: str | None = None,
) -> list[int]:
    payload = load_json(selection_path)
    if "classifiers" in payload:
        entries = payload["classifiers"] or []
        if not entries:
            raise ValueError(f"No classifiers listed in {selection_path}")
        candidate = None
        if classifier_name:
            for entry in entries:
                if entry.get("classifier") == classifier_name:
                    candidate = entry
                    break
            if candidate is None:
                raise ValueError(
                    f"Classifier '{classifier_name}' not found in {selection_path}"
                )
        else:
            candidate = entries[0]
        payload = candidate
    indices: list[int] = []
    if "selected_spi_indices" in payload:
        indices = [int(idx) for idx in payload["selected_spi_indices"]]
    elif "indices" in payload:
        indices = [int(idx) for idx in payload["indices"]]
    elif "selected" in payload:
        indices = [int(idx) for idx in payload["selected"]]
    if not indices and "selected_spi_names" in payload:
        lookup = {name: i for i, name in enumerate(spi_names)}
        for name in payload["selected_spi_names"]:
            if name not in lookup:
                raise ValueError(f"SPI '{name}' not found in feature archive.")
            indices.append(lookup[name])
    if not indices:
        raise ValueError(
            f"Could not find 'selected_spi_indices' or 'selected_spi_names' in {selection_path}"
        )
    ordered = []
    seen = set()
    for idx in indices:
        if idx < 0 or idx >= len(spi_names):
            raise ValueError(f"SPI index {idx} is out of range.")
        if idx not in seen:
            ordered.append(idx)
            seen.add(idx)
    return ordered


def _restrict_features(
    X: np.ndarray,
    feature_pairs: np.ndarray,
    selection: list[int] | None,
) -> tuple[np.ndarray, list[int], list[tuple[int, int]]]:
    if selection is None:
        n_features = X.shape[1]
        pairs = [tuple(pair) for pair in feature_pairs]
        return X, list(range(n_features)), pairs
    selection = sorted(selection)
    pair_index = _build_pair_index(feature_pairs)
    selected_columns: list[int] = []
    selected_pairs: list[tuple[int, int]] = []
    for i_idx, spi_i in enumerate(selection):
        for spi_j in selection[i_idx + 1 :]:
            key = (spi_i, spi_j) if spi_i < spi_j else (spi_j, spi_i)
            if key not in pair_index:
                raise KeyError(f"Pair {key} is missing from the feature archive.")
            selected_columns.append(pair_index[key])
            selected_pairs.append(key)
    if not selected_columns:
        raise ValueError("SPI selection must contain at least two SPIs.")
    restricted = X[:, selected_columns]
    return restricted, selected_columns, selected_pairs


def _standardise_features(X: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def _run_umap(
    X: np.ndarray,
    *,
    n_neighbors: int,
    min_dist: float,
    n_components: int,
    metric: str,
    random_state: int,
) -> np.ndarray:
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(X)


def _build_dataframe(
    embedding: np.ndarray,
    archive: dict[str, np.ndarray],
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "x": embedding[:, 0],
            "y": embedding[:, 1] if embedding.shape[1] > 1 else 0.0,
            "dataset_path": archive["dataset_paths"],
            "mts_class": archive["mts_class"],
            "variant": archive["variant"],
            "M": archive["M"],
            "T": archive["T"],
        }
    )
    return df


def _size_mapping(values: Iterable[int]) -> dict[int, float]:
    unique = sorted(set(int(v) for v in values))
    base = 40.0
    step = 40.0
    return {val: base + i * step for i, val in enumerate(unique)}


def _plot_embedding(
    df: pd.DataFrame, title: str, prefix: str, figures_dir: Path
) -> Path:
    apply_plot_style()
    ensure_dir(figures_dir)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    classes = sorted(df["mts_class"].unique())
    palette = sns.color_palette("pastel", n_colors=len(classes))
    color_map = dict(zip(classes, palette))
    size_map = _size_mapping(df["M"])
    for cls in classes:
        cls_df = df[df["mts_class"] == cls]
        if cls_df.empty:
            continue
        ax.scatter(
            cls_df["x"],
            cls_df["y"],
            s=cls_df["M"].map(size_map),
            color=color_map[cls],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.85,
            label=cls,
        )
        if len(cls_df) >= 5:
            sns.kdeplot(
                x=cls_df["x"],
                y=cls_df["y"],
                levels=3,
                color=color_map[cls],
                linewidths=1.0,
                alpha=0.6,
                ax=ax,
            )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(title)
    legend_handles = [Line2D([0], [0], marker="o", color=color_map[c], linestyle="", markerfacecolor=color_map[c], label=c, markersize=8) for c in classes]
    legend1 = ax.legend(
        handles=legend_handles,
        title="MTS class",
        loc="upper right",
        frameon=False,
    )
    ax.add_artist(legend1)
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="gray",
            linestyle="",
            markersize=np.sqrt(size_map[m]) / 2,
            label=f"M={m}",
        )
        for m in sorted(size_map)
    ]
    ax.legend(
        handles=size_handles,
        title="Process size",
        loc="lower right",
        frameon=False,
    )
    ax.grid(False)
    fig_path = figures_dir / f"{prefix}.pdf"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    return fig_path


def _save_csv(df: pd.DataFrame, prefix: str, results_dir: Path) -> Path:
    ensure_dir(results_dir)
    csv_path = results_dir / f"{prefix}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _default_prefix(mode: str, selection: list[int] | None) -> str:
    if selection:
        return f"umap_{mode}_spisubset_k{len(selection)}"
    return f"umap_{mode}_allspis"


def _maybe_compute_silhouette(embedding: np.ndarray, labels: np.ndarray) -> float | None:
    unique_labels = np.unique(labels)
    if embedding.shape[0] < 3 or unique_labels.size < 2:
        return None
    try:
        score = silhouette_score(embedding, labels)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to compute silhouette score: %s", exc)
        return None
    return float(score)


def run_pipeline(args: argparse.Namespace) -> None:
    results_dir = _mode_results_dir(args.mode)
    archive = _load_feature_archive(args.mode)
    spi_names = archive["spi_names"].tolist()
    selection = (
        _resolve_spi_selection(
            Path(args.spi_selection_json), spi_names, args.selection_classifier
        )
        if args.spi_selection_json
        else None
    )
    X = np.array(archive["X"], dtype=float)
    feature_pairs = archive["feature_pairs"]
    X_subset, _, _ = _restrict_features(X, feature_pairs, selection)
    if X_subset.shape[1] == 0:
        raise ValueError("Selected feature subset is empty.")
    X_scaled = _standardise_features(X_subset)
    embedding = _run_umap(
        X_scaled,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=args.n_components,
        metric=args.metric,
        random_state=args.random_state,
    )
    df = _build_dataframe(embedding, archive)
    prefix = args.out_prefix or _default_prefix(args.mode, selection)
    csv_path = _save_csv(df, prefix, results_dir)
    figures_dir = ensure_dir(results_dir / "figures" / "spi_space")
    fig_path = _plot_embedding(
        df,
        title=f"UMAP ({args.mode}, {'subset' if selection else 'all SPIs'})",
        prefix=prefix,
        figures_dir=figures_dir,
    )
    silhouette = _maybe_compute_silhouette(embedding, archive["mts_class"])
    if silhouette is not None:
        LOGGER.info("Silhouette score by class: %.3f", silhouette)
    LOGGER.info("Saved embedding CSV to %s and figure to %s", csv_path, fig_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="UMAP embeddings of datasets in SPI–SPI feature space."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["dev", "full"],
        help="Dataset mode to analyse.",
    )
    parser.add_argument(
        "--spi-selection-json",
        type=str,
        default=None,
        help="JSON file containing selected SPI indices or names.",
    )
    parser.add_argument(
        "--selection-classifier",
        type=str,
        default=None,
        help="Classifier name to pull from the selection JSON (default: first entry).",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=12,
        help="UMAP n_neighbors parameter (default: 12).",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.75,
        help="UMAP min_dist parameter (default: 0.75).",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=2,
        help="Number of embedding dimensions (default: 2).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        help="UMAP distance metric (default: euclidean).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for UMAP (default: 42).",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default=None,
        help="Optional prefix for output files.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_pipeline(args)


if __name__ == "__main__":
    main()
