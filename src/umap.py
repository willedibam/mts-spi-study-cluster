from __future__ import annotations

import argparse
import colorsys
import fnmatch
import logging
import math
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from .plot_style import apply_plot_style
from .utils import ensure_dir, load_json, project_root

LOGGER = logging.getLogger(__name__)


def _results_dir(mode: str, space: str, category: str) -> Path:
    base = ensure_dir(project_root() / "results" / mode / space)
    return ensure_dir(base / category)


def _load_feature_archive(mode: str, space: str) -> dict[str, np.ndarray]:
    path = _results_dir(mode, space, "features") / f"spi_space_features_{mode}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature archive not found at {path}. Run 'python -m src.features' first."
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


def _resolve_spi_selection_txt(path: Path, spi_names: list[str]) -> list[int]:
    lookup = {name: idx for idx, name in enumerate(spi_names)}
    indices: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped not in lookup:
            raise ValueError(f"SPI '{stripped}' (from {path}) not found in archive.")
        idx = lookup[stripped]
        if idx not in indices:
            indices.append(idx)
    if not indices:
        raise ValueError(f"No valid SPI names found in {path}")
    return indices


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


def _restrict_spi_blocks(
    X: np.ndarray,
    spi_slices: np.ndarray,
    selection: list[int] | None,
) -> np.ndarray:
    if not selection:
        return X
    cols: list[int] = []
    for idx in selection:
        if idx < 0 or idx >= spi_slices.shape[0]:
            raise ValueError(f"SPI index {idx} is out of range for SPI space.")
        start, length = spi_slices[idx]
        start = int(start)
        length = int(length)
        cols.extend(range(start, start + length))
    if not cols:
        raise ValueError("Selection resulted in an empty SPI feature subset.")
    return X[:, cols]


def _restrict_spi_blocks(
    X: np.ndarray,
    spi_slices: np.ndarray,
    selection: list[int] | None,
) -> np.ndarray:
    if selection is None or not selection:
        return X
    cols: list[int] = []
    for idx in selection:
        if idx < 0 or idx >= spi_slices.shape[0]:
            raise ValueError(f"SPI index {idx} is out of range for SPI feature slices.")
        start, length = spi_slices[idx]
        cols.extend(range(int(start), int(start + length)))
    if not cols:
        raise ValueError("Selection resulted in an empty SPI feature subset.")
    return X[:, cols]


def _parse_multi_values(raw: str, caster) -> list:
    parts: list = []
    for chunk in str(raw).split(","):
        token = chunk.strip()
        if not token:
            continue
        parts.append(caster(token))
    if not parts:
        raise ValueError(f"No valid values found in '{raw}'")
    return parts


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


def _cluster_embedding(
    embedding: np.ndarray,
    *,
    algorithm: str,
    n_clusters: int,
    random_state: int,
) -> np.ndarray:
    n_samples = embedding.shape[0]
    if n_samples == 0:
        return np.array([])
    if n_clusters <= 0:
        n_clusters = max(2, int(math.sqrt(n_samples)))
    if algorithm == "kmeans":
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
        return model.fit_predict(embedding)
    raise ValueError(f"Unsupported clustering algorithm '{algorithm}'.")


def _export_clusters(
    *,
    df: pd.DataFrame,
    embedding: np.ndarray,
    export_dir: Path,
    algorithm: str,
    cluster_k: int,
    random_state: int,
) -> None:
    ensure_dir(export_dir)
    labels = _cluster_embedding(
        embedding,
        algorithm=algorithm,
        n_clusters=cluster_k,
        random_state=random_state,
    )
    if labels.size == 0:
        LOGGER.warning("No data points to cluster; skipping export.")
        return
    df_export = df.copy()
    df_export["cluster_id"] = labels
    master_csv = export_dir / "clusters.csv"
    df_export.to_csv(master_csv, index=False)
    LOGGER.info("Wrote cluster assignments to %s", master_csv)
    for cluster_id in sorted(set(labels)):
        subset = df_export[df_export["cluster_id"] == cluster_id]
        cluster_dir = ensure_dir(export_dir / f"cluster_{cluster_id:02d}")
        subset_path = cluster_dir / "members.csv"
        subset.to_csv(subset_path, index=False)
        figures_dir = ensure_dir(cluster_dir / "figures")
        for _, row in subset.iterrows():
            dataset_rel = Path(row["dataset_path"])
            dataset_abs = dataset_rel
            if not dataset_abs.is_absolute():
                dataset_abs = project_root() / dataset_rel
            heatmap_src = dataset_abs / "figures" / "mts_heatmap.png"
            if heatmap_src.exists():
                mts_class = str(row.get("mts_class") or "MTS")
                dest_name = f"{mts_class}-{dataset_abs.name}.png"
                dest = figures_dir / dest_name
                try:
                    shutil.copy2(heatmap_src, dest)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Failed to copy %s -> %s (%s)", heatmap_src, dest, exc)


def _build_dataframe(
    embedding: np.ndarray,
    dataset_paths: np.ndarray,
    mts_class: np.ndarray,
    variant: np.ndarray,
    M: np.ndarray,
    T: np.ndarray,
    display_class: np.ndarray,
    base_class: np.ndarray,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "x": embedding[:, 0],
            "y": embedding[:, 1] if embedding.shape[1] > 1 else 0.0,
            "dataset_path": dataset_paths,
            "mts_class": mts_class,
            "variant": variant,
            "M": M,
            "T": T,
            "display_class": display_class,
            "base_class": base_class,
        }
    )
    return df


def _size_mapping(values: Iterable[int]) -> dict[int, float]:
    unique = sorted(set(int(v) for v in values))
    if not unique:
        return {}
    base = 20.0
    minimum = max(1, unique[0])
    return {val: base * (val / minimum) for val in unique}


def _tonal_palette(base_color: tuple[float, float, float], count: int) -> list[tuple[float, float, float]]:
    if count <= 1:
        return [base_color]
    h, l, s = colorsys.rgb_to_hls(*base_color)
    light_vals = np.linspace(
        max(0.15, l - 0.25), min(0.9, l + 0.25), count
    )
    shades = []
    for light in light_vals:
        shades.append(colorsys.hls_to_rgb(h, light, min(1.0, s + 0.05)))
    return shades


def _build_variant_palette(df: pd.DataFrame) -> dict[str, tuple[float, float, float]]:
    display = df["display_class"].tolist()
    base = df["base_class"].tolist()
    base_order: list[str] = []
    for cls in base:
        if cls not in base_order:
            base_order.append(cls)
    groups: dict[str, list[str]] = {cls: [] for cls in base_order}
    for disp, base_cls in zip(display, base):
        group = groups.setdefault(base_cls, [])
        if disp not in group:
            group.append(disp)
    base_palette = sns.color_palette("pastel", n_colors=len(base_order))
    color_map: dict[str, tuple[float, float, float]] = {}
    for base_cls, base_color in zip(base_order, base_palette):
        variants = groups.get(base_cls, [])
        shades = _tonal_palette(base_color, max(1, len(variants)))
        for disp, shade in zip(variants, shades):
            color_map[disp] = shade
    return color_map


def _plot_embedding(
    df: pd.DataFrame,
    space: str,
    prefix: str,
    figures_dir: Path,
    n_neighbors: int,
    min_dist: float,
    silhouette: float | None,
    split_variants: bool,
    label_column: str,
) -> Path:
    apply_plot_style()
    ensure_dir(figures_dir)
    fig, ax = plt.subplots(figsize=(6, 6))
    if split_variants:
        color_lookup = _build_variant_palette(df)
        ordered_labels = []
        seen = set()
        for base_cls in df["base_class"]:
            if base_cls not in seen:
                variants = df.loc[df["base_class"] == base_cls, "display_class"].unique()
                for val in variants:
                    if val not in ordered_labels:
                        ordered_labels.append(val)
                seen.add(base_cls)
    else:
        ordered_labels = []
        for cls in df["base_class"]:
            if cls not in ordered_labels:
                ordered_labels.append(cls)
        palette = sns.color_palette("pastel", n_colors=len(ordered_labels))
        color_lookup = dict(zip(ordered_labels, palette))
    size_map = _size_mapping(df["M"])
    for label in ordered_labels:
        cls_df = df[df[label_column] == label]
        if cls_df.empty:
            continue
        color = color_lookup.get(label, sns.color_palette("pastel", 1)[0])
        ax.scatter(
            cls_df["x"],
            cls_df["y"],
            s=cls_df["M"].map(size_map),
            color=color,
            edgecolor="white",
            linewidth=0.4,
            alpha=0.9,
            label=label,
        )
        if len(cls_df) >= 5:
            sns.kdeplot(
                x=cls_df["x"],
                y=cls_df["y"],
                levels=4,
                color=color,
                alpha=0.25,
                fill=True,
                thresh=0.05,
                ax=ax,
            )
    ax.set_facecolor("white")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_box_aspect(1)
    sil_text = f"{silhouette:.3f}" if silhouette is not None else "NA"
    ax.set_title(f"features: {space} | nn={n_neighbors} | md={min_dist:.2f} | sil={sil_text}")
    ax.tick_params(axis="both", which="both", colors="black", width=0.8, length=4.0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=color_lookup.get(label, "gray"),
            linestyle="",
            markerfacecolor=color_lookup.get(label, "gray"),
            label=label,
            markersize=6,
        )
        for label in ordered_labels
    ]
    legend_title = "Class (variant)" if split_variants else "Class"
    legend1 = ax.legend(
        handles=legend_handles,
        title=legend_title,
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
            markerfacecolor="gray",
            markersize=np.sqrt(size_map[m]) / 2,
            label=f"M={m}",
        )
        for m in sorted(size_map)
    ]
    size_legend = ax.legend(
        handles=size_handles,
        title="Process size",
        loc="lower right",
        frameon=False,
    )
    ax.grid(False)
    fig_path = figures_dir / f"{prefix}.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    return fig_path


def _default_prefix(mode: str, space: str, selection: list[int] | None) -> str:
    base = f"umap_{mode}_{space}"
    if selection:
        return f"{base}_subset_k{len(selection)}"
    return f"{base}_allspis"


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
    archive = _load_feature_archive(args.mode, args.feature_space)
    space_in_archive = str(archive.get("space") or args.feature_space)
    if space_in_archive != args.feature_space:
        raise ValueError(
            f"Archive was generated for space '{space_in_archive}', but '--feature-space {args.feature_space}' was requested."
        )
    spi_names = archive["spi_names"].tolist()
    if args.spi_selection_json and args.spi_selection_txt:
        raise ValueError(
            "Specify at most one of --spi-selection-json or --spi-selection-txt."
        )
    if args.spi_selection_json:
        selection = _resolve_spi_selection(
            Path(args.spi_selection_json), spi_names, args.selection_classifier
        )
    elif args.spi_selection_txt:
        selection = _resolve_spi_selection_txt(Path(args.spi_selection_txt), spi_names)
    else:
        selection = None
    dataset_paths_all = np.array(archive["dataset_paths"], dtype=object)
    normalized_paths = np.array(
        [str(Path(p).as_posix()) for p in dataset_paths_all], dtype=object
    )
    mts_class_all = np.array(archive["mts_class"], dtype=object)
    variant_all = np.array(archive["variant"], dtype=object)
    M_all = np.array(archive["M"])
    T_all = np.array(archive["T"])
    base_classes_all = np.array(
        archive.get("base_class", archive["mts_class"]), dtype=object
    )
    display_classes_all = np.array(
        archive.get("display_class", archive["mts_class"]), dtype=object
    )
    if args.dataset_filter:
        pattern = args.dataset_filter
        mask_entries: list[bool] = []
        for rel, norm, disp, base in zip(
            dataset_paths_all, normalized_paths, display_classes_all, base_classes_all
        ):
            tail = Path(rel).name
            candidates = [
                norm,
                rel,
                rel.replace("\\", "/"),
                tail,
                str(disp),
                str(base),
            ]
            mask_entries.append(any(fnmatch.fnmatch(item, pattern) for item in candidates))
        mask = np.array(mask_entries, dtype=bool)
        if not mask.any():
            raise ValueError(
                f"No datasets matched dataset_filter '{args.dataset_filter}'."
            )
    else:
        mask = slice(None)
    dataset_paths = dataset_paths_all[mask]
    mts_class = mts_class_all[mask]
    variant = variant_all[mask]
    M = M_all[mask]
    T = T_all[mask]
    display_classes = display_classes_all[mask]
    base_classes = base_classes_all[mask]
    X = np.array(archive["X"], dtype=float)[mask]
    if args.feature_space == "spi-spi":
        feature_pairs = archive["feature_pairs"]
        X_subset, _, _ = _restrict_features(X, feature_pairs, selection)
    else:
        spi_feature_slices = archive.get("spi_feature_slices")
        if spi_feature_slices is None or spi_feature_slices.size == 0:
            raise ValueError(
                "SPI feature slices missing from archive; recompute with --space spi."
            )
        X_subset = _restrict_spi_blocks(X, spi_feature_slices, selection)
    if X_subset.shape[1] == 0:
        raise ValueError("Selected feature subset is empty.")
    X_scaled = _standardise_features(X_subset)
    n_neighbors_values = _parse_multi_values(args.n_neighbors, int)
    min_dist_values = _parse_multi_values(args.min_dist, float)
    prefix_base = args.out_prefix or _default_prefix(
        args.mode, args.feature_space, selection
    )
    figures_dir = _results_dir(args.mode, args.feature_space, "umap")
    cluster_counts = _parse_multi_values(args.cluster_count, int)
    if not cluster_counts:
        cluster_counts = [0]
    for n_neighbors in n_neighbors_values:
        for min_dist in min_dist_values:
            embedding = _run_umap(
                X_scaled,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=args.n_components,
                metric=args.metric,
                random_state=args.random_state,
            )
            df = _build_dataframe(
                embedding,
                dataset_paths,
                mts_class,
                variant,
                M,
                T,
                display_classes,
                base_classes,
            )
            min_dist_token = f"{min_dist:.2f}".replace(".", "p")
            suffix = f"{prefix_base}_nn{n_neighbors}_mindist{min_dist_token}"
            silhouette = _maybe_compute_silhouette(
                embedding,
                df["display_class"] if args.split_variants else df["base_class"],
            )
            fig_path = _plot_embedding(
                df,
                space=args.feature_space,
                prefix=suffix,
                figures_dir=figures_dir,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                silhouette=silhouette,
                split_variants=args.split_variants,
                label_column="display_class" if args.split_variants else "base_class",
            )
            if silhouette is not None:
                LOGGER.info(
                    "Silhouette score (n_neighbors=%d, min_dist=%.2f): %.3f",
                    n_neighbors,
                    min_dist,
                    silhouette,
                )
            LOGGER.info("Saved UMAP figure to %s", fig_path)
            if args.export_clusters:
                for cluster_k in cluster_counts:
                    k_token = "auto" if cluster_k <= 0 else f"k{cluster_k}"
                    export_dir = ensure_dir(Path(args.export_clusters) / suffix / k_token)
                    _export_clusters(
                        df=df,
                        embedding=embedding,
                        export_dir=export_dir,
                        algorithm=args.cluster_algorithm,
                        cluster_k=cluster_k,
                        random_state=args.random_state,
                    )


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
        "--feature-space",
        choices=["spi-spi", "spi"],
        default="spi-spi",
        help="Feature space to load for embeddings (default: spi-spi).",
    )
    parser.add_argument(
        "--spi-selection-txt",
        type=str,
        default=None,
        help="Plain-text file with SPI names (one per line) to use as subset.",
    )
    parser.add_argument(
        "--selection-classifier",
        type=str,
        default=None,
        help="Classifier name to pull from the selection JSON (default: first entry).",
    )
    parser.add_argument(
        "--n-neighbors",
        type=str,
        default="12",
        help="UMAP n_neighbors values (comma-separated for sweeps).",
    )
    parser.add_argument(
        "--min-dist",
        type=str,
        default="0.75",
        help="UMAP min_dist values (comma-separated for sweeps).",
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
        "--split-variants",
        action="store_true",
        help="Color each class variant separately (default groups by base class).",
    )
    parser.add_argument(
        "--export-clusters",
        type=str,
        help="Optional directory to store per-cluster exports (CSV + time series copies).",
    )
    parser.add_argument(
        "--cluster-algorithm",
        type=str,
        choices=["kmeans"],
        default="kmeans",
        help="Clustering algorithm to run when exporting clusters (default: kmeans).",
    )
    parser.add_argument(
        "--cluster-count",
        type=str,
        default="0",
        help="Comma-separated cluster counts (default: sqrt(n) heuristic when <=0).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--dataset-filter",
        type=str,
        default=None,
        help="Optional glob (matched against the dataset path, e.g. 'data/full/CML/*alpha1p75*') to limit which datasets are plotted.",
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
