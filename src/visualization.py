from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr, zscore
from sklearn.preprocessing import StandardScaler, robust_scale
from sklearn.decomposition import PCA
try:
    from sklearn.manifold import TSNE
except ImportError:
    TSNE = None
try:
    from umap import UMAP
except ImportError:
    UMAP = None

from .plot_style import apply_plot_style
from .spi_color import infer_spi_color_scale
from .utils import load_json


def plot_mpi_heatmap(
    mts_class: str,
    dataset_slug: str,
    spis: Iterable[str],
    *,
    base_dir: Path | str = Path("data") / "full",
    auto_scale: bool = True,
    center: float | None = None,
    cmap: str | None = None,
) -> None:
    """
    Plot heatmaps for one or more SPIs from a dataset's spi_mpis.npz.

    Args:
        mts_class: Dataset class name (e.g., "CML", "Kuramoto").
        dataset_slug: Directory slug (e.g., "M25_T1600_I0_<variant-slug>").
        spis: Iterable of SPI names to plot.
        base_dir: Base path to dataset directory (default: data/full).
        auto_scale: If True, infer vmin/vmax/center/cmap from data and SPI name.
        center: Optional manual center override.
        cmap: Optional manual cmap override.
    """
    apply_plot_style()
    dataset_dir = Path(base_dir) / mts_class / dataset_slug
    archive = dataset_dir / "spi_mpis.npz"
    if not archive.exists():
        raise FileNotFoundError(f"Missing archive: {archive}")
    npz = np.load(archive)
    labels_map: dict[str, list[str]] = {}
    meta_path = dataset_dir / "meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
            for entry in meta.get("pyspi", {}).get("spis", []):
                labels_map[str(entry.get("name"))] = entry.get("labels", [])
    spis = list(spis)
    if not spis:
        raise ValueError("No SPIs provided to plot.")

    n = len(spis)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    fig.suptitle(f"{mts_class}/{dataset_slug}", fontsize=12)

    for ax, spi in zip(axes.ravel(), spis):
        if spi not in npz:
            ax.axis("off")
            ax.set_title(f"{spi} (missing)")
            continue
        arr = npz[spi]
        vmin = vmax = None
        used_center = center
        used_cmap = cmap
        if auto_scale:
            scale = infer_spi_color_scale(
                spi,
                float(np.nanmin(arr)),
                float(np.nanmax(arr)),
                labels=labels_map.get(spi),
            )
            vmin, vmax, inferred_center, inferred_cmap = (
                scale.vmin,
                scale.vmax,
                scale.center,
                scale.cmap,
            )
            used_center = inferred_center if center is None else center
            used_cmap = inferred_cmap if cmap is None else cmap
        else:
            used_center = center
            used_cmap = cmap or "coolwarm"
        sns.heatmap(
            arr,
            cmap=used_cmap or "coolwarm",
            center=used_center,
            vmin=vmin,
            vmax=vmax,
            square=True,
            xticklabels=False,
            yticklabels=False,
            cbar=True,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
        ax.set_title(spi)

    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def plot_mts_heatmap(
    mts: np.ndarray,
    *,
    title: str | None = None,
    cmap: str = "icefire",
    vmin: float | None = -2,
    vmax: float | None = 2,
    figsize: tuple[float, float] = (9, 6),
) -> None:
    """
    Plot a multivariate time series heatmap (T x M -> displayed as M rows).

    Args:
        mts: Array shaped (T, M) or (M, T); if second dimension equals M, transpose for display.
        title: Optional title string.
        cmap: Colormap for display.
        vmin, vmax: Optional fixed bounds.
        figsize: Figure size.
    """
    apply_plot_style()
    data = np.asarray(mts)
    if data.ndim != 2:
        raise ValueError("mts must be 2D (T x M or M x T).")
    # Heuristic: if columns >> rows, assume shape (T, M) and transpose to (M, T)
    if data.shape[0] > data.shape[1]:
        data = data.T
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.pcolormesh(
        data,
        shading="flat",
        vmin=vmin,
        vmax=vmax,
        cmap=sns.color_palette(cmap, as_cmap=True),
    )
    ax.grid(False)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    fig.tight_layout()
    plt.show()


def scale_mts_heatmap(
    data_dir: str | Path,
    *,
    filename: str = "timeseries.npy",
) -> list[Path]:
    """
    Scale and save MxT heatmaps as SVGs.

    - If `data_dir/filename` exists, process just that file.
    - Otherwise, recursively find all `filename` matches under `data_dir`.
    - Layout is MxT (transpose if loaded as TxM).
    - Normalisation: robust_scale for Cauchy datasets; otherwise z-score per channel
      (robust_scale line is left as a commented alternative).
    - Color: icefire; vmin/vmax = [-2, 2], except Cauchy uses symmetric 0.1/99.9 pct.
    - Output: saved alongside each timeseries as `mts_heatmap_scaled.svg` and `.png`.
    """

    def _process(data_path: Path) -> Path:
        data = np.load(data_path).astype(float, copy=False)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {data.shape} for {data_path}")
        if data.shape[0] > data.shape[1]:
            data = data.T
        M, T = data.shape

        is_cauchy = "cauchy" in data_path.name.lower() or "cauchy" in data_path.parent.name.lower()
        if is_cauchy:
            scaled = robust_scale(data, axis=1)
        else:
            scaled = zscore(data, axis=1, nan_policy="omit")
            # scaled = robust_scale(data, axis=1)  # robust alternative
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

        if is_cauchy:
            lo, hi = np.percentile(scaled, [0.1, 99.9])
            bound = float(np.max(np.abs([lo, hi])))
            vmin, vmax = -bound, bound
        else:
            vmin, vmax = -2.0, 2.0

        base_M, base_T = 16.0, 1000.0
        base_fig = (8.0, 4.0)
        width = float(np.clip(base_fig[0] * (T / base_T), 4.0, 18.0))
        height = float(np.clip(base_fig[1] * (M / base_M), 2.0, 12.0))

        fig, ax = plt.subplots(figsize=(width, height), dpi=300)
        ax.pcolormesh(
            scaled,
            shading="flat",
            vmin=vmin,
            vmax=vmax,
            cmap=sns.color_palette("icefire", as_cmap=True),
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        fig.tight_layout(pad=0.05)
        base = data_path.with_name("mts_heatmap_scaled")
        svg_path = base.with_suffix(".svg")
        png_path = base.with_suffix(".png")
        fig.savefig(svg_path, format="svg", bbox_inches="tight", pad_inches=0)
        fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return svg_path

    apply_plot_style()
    root = Path(data_dir)
    targets: list[Path] = []
    if root.is_file() and root.name == filename:
        targets = [root]
    elif (root / filename).exists():
        targets = [root / filename]
    else:
        targets = list(root.rglob(filename))
    if not targets:
        raise FileNotFoundError(f"No {filename} found under {root}")
    return [_process(path) for path in sorted(targets)]


def _clean_legend(ax, hue: str, size_col: str) -> None:
    """
    Helper to remove size indicators and redundant titles from Seaborn legends.
    """
    handles, labels = ax.get_legend_handles_labels()
    
    if size_col in labels:     # 1. Truncate at size_col to remove size entries
        size_idx = labels.index(size_col)
        handles, labels = handles[:size_idx], labels[:size_idx]

    if labels and labels[0] == hue:     # 2. Remove redundant header label if it matches the hue title
        handles, labels = handles[1:], labels[1:]

    ax.legend(
        handles,
        labels,
        title=hue,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        ncol=1,
        fancybox=True,
        frameon=True
    )


def plot_spi_space_individual(
    dataset_path: str,
    spis: list[str],
) -> None:
    if len(spis) != 2:
        raise ValueError("Expects exactly two SPI names.")
    
    spi_x, spi_y = spis
    
    if any(ch in dataset_path for ch in "*?[]"):
        matches = sorted(Path(".").glob(dataset_path))
        if not matches:
            raise FileNotFoundError(
                f"No dataset matched pattern '{dataset_path}' in project root"
            )
        if len(matches) > 1:
            matched = ", ".join(p.name for p in matches)
            raise ValueError(
                f"Pattern '{dataset_path}' matched multiple datasets: {matched}"
            )
        dataset_dir = matches[0]
    else:
        dataset_dir = Path(dataset_path)

    archive = dataset_dir / "spi_mpis.npz"
    if not archive.exists():
        raise FileNotFoundError(f"Missing archive: {archive}")
    meta_path = dataset_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json: {meta_path}")
    meta = load_json(meta_path)
    spi_meta = meta.get("pyspi", {}).get("spis", [])
    directed_map = {
        entry.get("name"): bool(entry.get("directed", False))
        for entry in spi_meta
        if isinstance(entry, dict) and entry.get("name")
    }
    missing_meta = [name for name in (spi_x, spi_y) if name not in directed_map]
    if missing_meta:
        raise KeyError(f"Missing directed metadata for {missing_meta} in {meta_path}")
        
    with np.load(archive) as npz:
        if spi_x not in npz or spi_y not in npz:
            raise KeyError(f"Missing SPIs in archive")
        arr_x, arr_y = npz[spi_x], npz[spi_y]

    if arr_x.shape != arr_y.shape:
        raise ValueError("Shape mismatch")
    
    apply_plot_style()

    directed_x = directed_map[spi_x]
    directed_y = directed_map[spi_y]
    upper_mask = np.triu(np.ones(arr_x.shape, dtype=bool), k=1)
    lower_mask = np.tril(np.ones(arr_x.shape, dtype=bool), k=-1)

    def _extract(mask_is_lower: bool) -> tuple[np.ndarray, np.ndarray]:
        mask_x = lower_mask if (mask_is_lower and directed_x) else upper_mask
        mask_y = lower_mask if (mask_is_lower and directed_y) else upper_mask
        x_vals = arr_x[mask_x]
        y_vals = arr_y[mask_y]
        valid_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        return x_vals[valid_mask], y_vals[valid_mask]

    directions = [(r"$i \to j$", False)]
    if directed_x or directed_y:
        directions.append((r"$j \to i$", True))

    plotted = False
    for direction_label, use_lower in directions:
        x_vals, y_vals = _extract(use_lower)
        if x_vals.size == 0:
            continue
            
        rho, _ = spearmanr(x_vals, y_vals)
        g = sns.jointplot(
            x=x_vals,
            y=y_vals,
            kind="scatter",
            height=6,
            marginal_kws=dict(kde=True, fill=True),
            s=15,
            alpha=0.6,
            color="#1f77b4"
        )
        # sns.kdeplot(x=x_vals, y=y_vals, ax=g.ax_joint, levels=5, color="#1f77b4", alpha=0.5, linewidths=1 # the contours

        sns.regplot(x=x_vals, y=y_vals, ax=g.ax_joint, scatter=False, color="#d62728", ci=None)
        x_label = f"{spi_x} ({direction_label})" if directed_x else spi_x
        y_label = f"{spi_y} ({direction_label})" if directed_y else spi_y
        g.set_axis_labels(x_label, y_label)
        title_slug = "/".join(dataset_dir.parts[-2:])
        g.fig.suptitle(f"{title_slug}\n$\\rho = {rho:.2f}$", y=1.02)
    
        plotted = True

    if plotted:
        plt.show()


def plot_mts_corr_density(
    mts_class_paths: list[str],
    spi_pair: list[str],
    *,
    bw_adjust: float = 1.0,
    show_hist: bool = False,
    kde: bool = True,
    bins: int = 40,
) -> None:
    """
    Plot density of Spearman correlations between two SPI matrices across mts_classes.

    Args:
        mts_class_paths: List of class directories (e.g., ["data/full/CauchyNoise", "data/full/VAR_1"]).
        spi_pair: Two SPI names to compare (e.g., ["cov_EmpiricalCovariance", "mi_kraskov_NN-4"]).
                  For directed SPIs you can suffix with __ij or __ji; if a directed SPI is given
                  without a suffix, both directions are plotted.
        bw_adjust: Optional KDE bandwidth adjustment passed to seaborn.
        show_hist: If True, overlay per-class histograms (density-normalized).
        bins: Number of histogram bins when show_hist is True.
    """
    if len(spi_pair) != 2:
        raise ValueError("spi_pair must contain exactly two SPI names.")

    def _parse_token(token: str) -> tuple[str, str | None]:
        if token.endswith("__ij"):
            return token.rsplit("__", 1)[0], "ij"
        if token.endswith("__ji"):
            return token.rsplit("__", 1)[0], "ji"
        return token, None

    def _safe_zscore(vec: np.ndarray) -> np.ndarray:
        std = vec.std()
        if std < 1e-12 or not np.isfinite(std):
            return np.zeros_like(vec)
        return (vec - vec.mean()) / std

    def _vector_for(mat: np.ndarray, directed: bool, direction: str | None) -> np.ndarray:
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError(f"SPI matrix must be square, got shape={mat.shape}")
        if not directed:
            mat = 0.5 * (mat + mat.T)
            mask = np.triu(np.ones(mat.shape, dtype=bool), k=1)
            return mat[mask]
        if direction == "ji":
            mask = np.tril(np.ones(mat.shape, dtype=bool), k=-1)
        else:
            mask = np.triu(np.ones(mat.shape, dtype=bool), k=1)
        return mat[mask]

    apply_plot_style()

    spi_x_base, spi_x_dir_req = _parse_token(spi_pair[0])
    spi_y_base, spi_y_dir_req = _parse_token(spi_pair[1])

    def _collect(direction_choice_x: str | None, direction_choice_y: str | None) -> None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
        palette = sns.color_palette("tab10", len(mts_class_paths))
        plotted = False

        for idx, class_path in enumerate(mts_class_paths):
            color = palette[idx % len(palette)]
            class_dir = Path(class_path)
            if not class_dir.exists():
                raise FileNotFoundError(f"MTS class directory not found: {class_dir}")
            values: list[float] = []
            label = class_dir.name

            for dataset_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
                meta_path = dataset_dir / "meta.json"
                npz_path = dataset_dir / "spi_mpis.npz"
                if not meta_path.exists() or not npz_path.exists():
                    continue

                meta = load_json(meta_path)
                spi_meta = {
                    entry.get("name"): entry
                    for entry in meta.get("pyspi", {}).get("spis", [])
                    if isinstance(entry, dict) and entry.get("name")
                }
                if spi_x_base not in spi_meta or spi_y_base not in spi_meta:
                    continue
                directed_x = bool(spi_meta[spi_x_base].get("directed", False))
                directed_y = bool(spi_meta[spi_y_base].get("directed", False))

                with np.load(npz_path) as npz:
                    if spi_x_base not in npz or spi_y_base not in npz:
                        continue
                    vec_x = _vector_for(np.asarray(npz[spi_x_base], float), directed_x, direction_choice_x)
                    vec_y = _vector_for(np.asarray(npz[spi_y_base], float), directed_y, direction_choice_y)

                if vec_x.shape != vec_y.shape:
                    continue
                valid = np.isfinite(vec_x) & np.isfinite(vec_y)
                if not valid.any():
                    continue
                zx = _safe_zscore(vec_x[valid])
                zy = _safe_zscore(vec_y[valid])
                rho = spearmanr(zx, zy).correlation
                if np.isfinite(rho):
                    values.append(float(rho))

            if values:
                plotted = True
                if show_hist:
                    sns.histplot(
                        values,
                        bins=bins,
                        binrange=(-1, 1),
                        stat="density",
                        color=color,
                        element="step",
                        fill=True,
                        alpha=0.25,
                        ax=ax,
                        label=f"{label} (n={len(values)})",
                    )
                if kde:
                    sns.kdeplot(
                        values,
                        label=f"{label} (n={len(values)})",
                        ax=ax,
                        bw_adjust=bw_adjust,
                        clip=(-1, 1),
                        fill=False,
                        color=color,
                        alpha=0.6,
                    )

        dir_suffix_x = f"__{direction_choice_x}" if direction_choice_x else ""
        dir_suffix_y = f"__{direction_choice_y}" if direction_choice_y else ""
        ax.set_xlim(-1, 1)
        ax.set_xlabel(f"{spi_x_base}{dir_suffix_x} vs {spi_y_base}{dir_suffix_y}")
        ax.set_ylabel("Density")
        if plotted and ax.get_legend_handles_labels()[0]:
            ax.legend(title="mts_class")
        plt.tight_layout()
        if plotted:
            plt.show()

    # If a directed SPI is passed without suffix, plot both ij/ji; otherwise honor the suffix.
    directions_x = [spi_x_dir_req] if spi_x_dir_req else [None, "ij", "ji"]
    directions_y = [spi_y_dir_req] if spi_y_dir_req else [None, "ij", "ji"]

    seen: set[tuple[str | None, str | None]] = set()
    for dx in directions_x:
        for dy in directions_y:
            key = (dx, dy)
            if key in seen:
                continue
            seen.add(key)
            _collect(dx, dy)

def plot_pca(
    x: np.ndarray,
    meta_df,
    *,
    n_components: int = 2,
    random_state: int = 0,
    feature_space: str = "",
    hue: str = "mts_class",
    size_col: str = "M",
    sizes: tuple[int, int] = (20, 160),
    facecolor: str = "#282a36", #old: #2C2C34
) -> np.ndarray:
    """
    PCA embedding + scatter/KDE plot.
    """
    apply_plot_style()
    
    xs = StandardScaler().fit_transform(x)
    pca = PCA(n_components=n_components, random_state=random_state)
    embedding = pca.fit_transform(xs)
    
    var_ratios = pca.explained_variance_ratio_
    var_pc1 = var_ratios[0] 
    var_pc2 = var_ratios[1] 

    meta_df = meta_df.copy()
    meta_df["pca_x"] = embedding[:, 0]
    meta_df["pca_y"] = embedding[:, 1]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    
    sns.scatterplot(
        data=meta_df,
        x="pca_x",
        y="pca_y",
        hue=hue,
        palette="pastel",
        size=size_col,
        sizes=sizes,
        alpha=0.8,
        ax=ax,
        legend="full", 
    )
    
    # sns.kdeplot(
    #     data=meta_df,
    #     x="pca_x",
    #     y="pca_y",
    #     hue=hue,
    #     palette="pastel",
    #     levels=10,
    #     thresh=0.05,
    #     fill=True,
    #     alpha=0.5,
    #     ax=ax,
    #     legend=False
    # )
    
    _clean_legend(ax, hue, size_col)
    
    ax.set_title(f"PCA ({feature_space}) | Var: {var_pc1+var_pc2:.4f}")
    ax.set_xlabel(f"PC1 ({var_pc1:.4f})")
    ax.set_ylabel(f"PC2 ({var_pc2:.4f})")
    ax.set_box_aspect(1)
    ax.set_facecolor(facecolor)
    
    plt.tight_layout()
    plt.show()
    return embedding


def plot_umap(
    x: np.ndarray,
    meta_df,
    *,
    metric: str = "euclidean",
    n_neighbors: int = 7,
    min_dist: float = 0.5,
    random_state: int = 0,
    feature_space: str = "",
    hue: str = "mts_class",
    size_col: str = "M",
    sizes: tuple[int, int] = (20, 160),
    facecolor: str = "#282a36",
) -> np.ndarray:
    """
    UMAP embedding + scatter/KDE plot.
    """
    if UMAP is None:
        raise ImportError("umap-learn is required for plot_umap")

    apply_plot_style()
    
    xs = StandardScaler().fit_transform(x)
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=True,
    )
    embedding = reducer.fit_transform(xs)
    
    meta_df = meta_df.copy()
    meta_df["umap_x"] = embedding[:, 0]
    meta_df["umap_y"] = embedding[:, 1]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    
    sns.scatterplot(
        data=meta_df,
        x="umap_x",
        y="umap_y",
        hue=hue,
        palette="pastel",
        size=size_col,
        sizes=sizes,
        alpha=0.8,
        ax=ax,
        legend="full", 
    )
    
    sns.kdeplot(
        data=meta_df,
        x="umap_x",
        y="umap_y",
        hue=hue,
        palette="pastel",
        levels=10,
        thresh=0.05,
        fill=True,
        alpha=0.5,
        ax=ax,
        legend=False
    )
    
    _clean_legend(ax, hue, size_col)
    
    ax.set_title(f"UMAP ({feature_space}, metric={metric})")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_box_aspect(1)
    ax.set_facecolor(facecolor)
    
    plt.tight_layout()
    plt.show()
    return embedding


def plot_tsne(
    x: np.ndarray,
    meta_df,
    *,
    metric: str = "euclidean",
    perplexity: float = 30.0,
    random_state: int = 0,
    feature_space: str = "",
    hue: str = "mts_class",
    size_col: str = "M",
    sizes: tuple[int, int] = (20, 160),
    facecolor: str = "#282a36",
) -> np.ndarray:
    """
    t-SNE embedding + scatter/KDE plot.
    """
    if TSNE is None:
        raise ImportError("scikit-learn is required for plot_tsne")

    apply_plot_style()
    
    xs = StandardScaler().fit_transform(x)
    tsne = TSNE(
        n_components=2,
        metric=metric,
        random_state=random_state,
        init="pca",
        perplexity=perplexity,
        learning_rate="auto",
        n_jobs=-1,
        verbose=0,
    )
    embedding = tsne.fit_transform(xs)
    
    meta_df = meta_df.copy()
    meta_df["tsne_x"] = embedding[:, 0]
    meta_df["tsne_y"] = embedding[:, 1]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    
    sns.scatterplot(
        data=meta_df,
        x="tsne_x",
        y="tsne_y",
        hue=hue,
        palette="pastel",
        size=size_col,
        sizes=sizes,
        alpha=0.8,
        ax=ax,
        legend="full", 
    )
    
    sns.kdeplot(
        data=meta_df,
        x="tsne_x",
        y="tsne_y",
        hue=hue,
        palette="pastel",
        levels=10,
        thresh=0.05,
        fill=True,
        alpha=0.5,
        ax=ax,
        legend=False
    )
    
    _clean_legend(ax, hue, size_col)
    
    ax.set_title(f"t-SNE ({feature_space}, metric={metric}, perplexity={perplexity})")
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.set_box_aspect(1)
    ax.set_facecolor(facecolor)
    
    plt.tight_layout()
    plt.show()
    return embedding
