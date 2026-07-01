from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import json
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.offsetbox import AnchoredText
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
try:
    import plotly.graph_objects as _go
except ImportError:
    _go = None

from .plot_style import apply_plot_style
from .spi_color import infer_spi_color_scale
from .utils import load_json

DEFAULT_DPI = 600

def plot_mpi_heatmap(
    dataset_dir: Path | str,
    spi: str,
    *,
    save_dir: Path | str,
    cmap: str,
) -> None:
    """
    Plot a single SPI matrix heatmap for a dataset directory.
    """
    apply_plot_style()
    dataset_dir = Path(dataset_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    archive = dataset_dir / "spi_mpis.npz"
    if not archive.exists():
        raise FileNotFoundError(f"Missing archive: {archive}")

    meta_path = dataset_dir / "meta.json"
    labels = None
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
            for entry in meta.get("pyspi", {}).get("spis", []):
                if str(entry.get("name")) == spi:
                    labels = entry.get("labels")
                    break

    with np.load(archive) as npz:
        if spi not in npz:
            raise KeyError(f"Missing SPI '{spi}' in {archive}")
        arr = npz[spi]

    scale = infer_spi_color_scale(
        spi,
        float(np.nanmin(arr)),
        float(np.nanmax(arr)),
        labels=labels,
    )
    used_cmap = cmap or scale.cmap or "coolwarm"

    fig, ax = plt.subplots(figsize=(5, 5), dpi=DEFAULT_DPI)
    sns.heatmap(
        arr,
        cmap=used_cmap,
        center=scale.center,
        vmin=scale.vmin,
        vmax=scale.vmax,
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    fig.suptitle(f"{spi}", fontsize=20)
    fig.tight_layout()

    dataset_name = dataset_dir.name
    mts_class = dataset_dir.parent.name
    output_path = save_dir / f"{spi}_mpi_heatmap_{mts_class}_{dataset_name}.svg"
    fig.savefig(output_path, dpi=DEFAULT_DPI, transparent=True)
    plt.show()


def plot_unravel_mpi(
    dataset_dir: Path | str,
    spi: str,
    *,
    save_dir: Path | str,
    cmap: str,
) -> np.ndarray | dict[str, np.ndarray]:
    """
    Unravel an MPI matrix into a 1D "barcode" and save as SVG/PNG.

    - If the SPI is undirected, returns the lower-triangular off-diagonal entries.
    - If directed, returns a dict with "__ij" (upper) and "__ji" (lower) vectors and
      saves both barcodes.
    """
    apply_plot_style()
    dataset_dir = Path(dataset_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    archive = dataset_dir / "spi_mpis.npz"
    if not archive.exists():
        raise FileNotFoundError(f"Missing archive: {archive}")

    meta_path = dataset_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json: {meta_path}")
    meta = load_json(meta_path)
    spi_entries = {
        entry.get("name"): entry
        for entry in meta.get("pyspi", {}).get("spis", [])
        if isinstance(entry, dict) and entry.get("name")
    }
    if spi not in spi_entries:
        raise KeyError(f"Missing SPI metadata for '{spi}' in {meta_path}")
    labels = spi_entries[spi].get("labels")
    directed = bool(spi_entries[spi].get("directed", False))

    with np.load(archive) as npz:
        if spi not in npz:
            raise KeyError(f"Missing SPI '{spi}' in {archive}")
        mat = np.asarray(npz[spi], float)

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"SPI matrix must be square, got shape={mat.shape}")

    scale = infer_spi_color_scale(
        spi,
        float(np.nanmin(mat)),
        float(np.nanmax(mat)),
        labels=labels,
    )
    used_cmap = cmap or scale.cmap or "coolwarm"

    upper_mask = np.triu(np.ones(mat.shape, dtype=bool), k=1)
    lower_mask = np.tril(np.ones(mat.shape, dtype=bool), k=-1)

    mts_class = dataset_dir.parent.name
    dataset_name = dataset_dir.name

    def _plot_barcode(vec: np.ndarray, suffix: str) -> None:
        if vec.size == 0:
            return
        width = max(4.0, float(vec.size) / 5.0)
        fig, ax = plt.subplots(figsize=(width, 1), dpi=DEFAULT_DPI)
        sns.heatmap(
            vec[np.newaxis, :],
            cmap=used_cmap,
            center=scale.center,
            vmin=scale.vmin,
            vmax=scale.vmax,
            cbar=False,
            xticklabels=False,
            yticklabels=False,
            ax=ax,
        )
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False)
        fig.tight_layout(pad=0.05)
        filename = f"barcode_{mts_class}_{dataset_name}_{spi}{suffix}"
        fig.savefig(save_dir / f"{filename}.svg", dpi=DEFAULT_DPI, transparent=True)
        fig.savefig(save_dir / f"{filename}.png", dpi=DEFAULT_DPI, transparent=True)
        plt.show()

    if directed:
        vec_ij = mat[upper_mask]
        vec_ji = mat[lower_mask]
        _plot_barcode(vec_ij, "__ij")
        _plot_barcode(vec_ji, "__ji")
        return {"ij": vec_ij, "ji": vec_ji}

    vec = mat[lower_mask]
    _plot_barcode(vec, "")
    return vec


def plot_spi_spi_barcode(
    dataset_dir: Path | str,
    spis: list[str] | None = None,
    *,
    spi_index_lim: int | None = None,
    save_dir: Path | str,
    cmap: str,
) -> np.ndarray:
    """
    Compute Spearman correlations between flattened MPI barcodes of SPIs and plot as a barcode.
    """
    apply_plot_style()
    dataset_dir = Path(dataset_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    archive = dataset_dir / "spi_mpis.npz"
    if not archive.exists():
        raise FileNotFoundError(f"Missing archive: {archive}")

    meta_path = dataset_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json: {meta_path}")
    meta = load_json(meta_path)
    spi_entries = [
        entry
        for entry in meta.get("pyspi", {}).get("spis", [])
        if isinstance(entry, dict) and entry.get("name")
    ]
    available_names = [str(entry["name"]) for entry in spi_entries]

    if not spis:
        spis = available_names
    if spi_index_lim is not None:
        spis = spis[:spi_index_lim]
    if len(spis) < 2:
        raise ValueError("At least two SPIs are required to compute pairwise correlations.")

    entry_map = {str(entry["name"]): entry for entry in spi_entries}
    with np.load(archive) as npz:
        vectors: list[tuple[str, np.ndarray]] = []

        def _is_directed(entry: dict) -> bool:
            labels = entry.get("labels") or []
            directed_label = any(isinstance(lbl, str) and lbl.lower() == "directed" for lbl in labels)
            return bool(entry.get("directed", False) or directed_label)

        for name in spis:
            if name not in entry_map:
                raise KeyError(f"Missing SPI metadata for '{name}' in {meta_path}")
            entry = entry_map[name]
            if name not in npz:
                raise KeyError(f"Missing SPI '{name}' in {archive}")
            mat = np.asarray(npz[name], float)
            if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
                raise ValueError(f"SPI matrix must be square, got shape={mat.shape} for {name}")

            upper_mask = np.triu(np.ones(mat.shape, dtype=bool), k=1)
            lower_mask = np.tril(np.ones(mat.shape, dtype=bool), k=-1)

            if _is_directed(entry):
                vectors.append((f"{name}__ij", mat[upper_mask]))
                vectors.append((f"{name}__ji", mat[lower_mask]))
            else:
                vectors.append((name, mat[lower_mask]))

    if len(vectors) < 2:
        raise ValueError("Need at least two SPI barcodes to correlate.")

    corr_values: list[float] = []
    # Pairwise Spearman correlations
    for idx in range(len(vectors)):
        for jdx in range(idx + 1, len(vectors)):
            _, vec_a = vectors[idx]
            _, vec_b = vectors[jdx]
            valid = np.isfinite(vec_a) & np.isfinite(vec_b)
            if valid.sum() < 2:
                corr = np.nan
            else:
                corr = float(spearmanr(vec_a[valid], vec_b[valid]).correlation)
            corr_values.append(corr)

    corr_vec = np.asarray(corr_values, float)
    if corr_vec.size == 0:
        raise ValueError("No pairwise correlations could be computed.")

    mts_class = dataset_dir.parent.name
    dataset_name = dataset_dir.name

    width = max(4.0, float(corr_vec.size) / 5.0)
    fig, ax = plt.subplots(figsize=(width, 1), dpi=DEFAULT_DPI)
    sns.heatmap(
        corr_vec[np.newaxis, :],
        cmap=cmap,
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        cbar=False,
        xticklabels=False,
        yticklabels=False,
        ax=ax,
    )
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False)
    fig.tight_layout(pad=0.05)

    filename = f"spi_spi_barcode_{mts_class}_{dataset_name}"
    fig.savefig(save_dir / f"{filename}.svg", dpi=DEFAULT_DPI, transparent=True)
    fig.savefig(save_dir / f"{filename}.png", dpi=DEFAULT_DPI, transparent=True)
    # plt.close(fig)
    plt.show()

    return corr_vec


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
    fig, ax = plt.subplots(figsize=figsize, dpi=DEFAULT_DPI)
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
    dpi: int,
    cmap: str = "icefire",
    *,
    filename: str = "timeseries.npy",
    save_tag: str | None = None,
    delta: int | None = None,
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
    - If delta > 1, subsample time axis by that stride and suffix filenames with `_delta<delta>_scaled`.
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

        used_delta = max(0, int(delta or 0))
        if used_delta > 1:
            scaled = scaled[:, ::used_delta]

        base_M, base_T = 16.0, 1000.0
        base_fig = (8.0, 4.0)
        eff_T = scaled.shape[1]
        width = float(np.clip(base_fig[0] * (eff_T / base_T), 4.0, 18.0))
        height = float(np.clip(base_fig[1] * (M / base_M), 2.0, 12.0))

        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
        ax.pcolormesh(
            scaled,
            shading="flat",
            vmin=vmin,
            vmax=vmax,
            cmap=sns.color_palette(cmap, as_cmap=True),
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        fig.tight_layout(pad=0.05)
        base_stem = data_path.name.lower().rsplit(".", 1)[0]
        delta_suffix = f"_delta{used_delta}" if used_delta > 1 else ""
        base = data_path.with_name(f"{base_stem}{delta_suffix}_scaled")
        if save_tag:
            svg_path = base.with_name(f"{save_tag}_{base.name}").with_suffix(".svg")
            png_path = base.with_name(f"{save_tag}_{base.name}").with_suffix(".png")
        else:
            svg_path = base.with_suffix(".svg")
            png_path = base.with_suffix(".png")
        fig.savefig(svg_path, format="svg", dpi=dpi, bbox_inches="tight", pad_inches=0)
        fig.savefig(png_path, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
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


def plot_mts_channel(
    dataset_dir: Path | str,
    channel: int,
    *,
    clip: tuple[int, int] | None = None,
    save_dir: Path | str,
    cmap: str = "icefire",
    stems: bool = False,
) -> np.ndarray:
    """
    Plot a single channel from an MxT multivariate time series as a 1D barcode.
    Uses the same scaling as scale_mts_heatmap for consistency.

    Args:
        dataset_dir: Directory containing timeseries.npy or direct path to it.
        channel: Channel index (0-based).
        clip: Optional (start, end) indices to slice the time axis.
        save_dir: Directory to save outputs.
        cmap: Colormap name.
        stems: If True, render a stem plot instead of a heatmap barcode.
    """
    apply_plot_style()
    dataset_dir = Path(dataset_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data_path = dataset_dir / "timeseries.npy" if dataset_dir.is_dir() else dataset_dir
    if not data_path.exists():
        raise FileNotFoundError(f"Missing timeseries file: {data_path}")

    data = np.load(data_path).astype(float, copy=False)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape} for {data_path}")
    if data.shape[0] > data.shape[1]:
        data = data.T
    M, T = data.shape
    if channel < 0 or channel >= M:
        raise IndexError(f"Channel {channel} out of bounds for M={M}")

    is_cauchy = "cauchy" in data_path.name.lower() or "cauchy" in data_path.parent.name.lower()
    if is_cauchy:
        scaled = robust_scale(data, axis=1)
    else:
        scaled = zscore(data, axis=1, nan_policy="omit")
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

    if is_cauchy:
        lo, hi = np.percentile(scaled, [0.1, 99.9])
        bound = float(np.max(np.abs([lo, hi])))
        vmin, vmax = -bound, bound
    else:
        vmin, vmax = -2.0, 2.0

    start = end = None
    if clip is not None:
        if (
            not isinstance(clip, tuple)
            or len(clip) != 2
            or not all(isinstance(x, int) for x in clip)
        ):
            raise TypeError("clip must be a tuple of two ints (start, end).")
        start, end = clip
        if not (0 <= start < end <= T):
            raise ValueError(f"clip must satisfy 0 <= start < end <= {T}, got {clip}")
        channel_vec = np.asarray(scaled[channel, start:end], float)
    else:
        channel_vec = np.asarray(scaled[channel, :], float)

    effective_len = channel_vec.shape[0]
    width = float(np.clip(8.0 * (effective_len / 1000.0), 4.0, 18.0))
    if dataset_dir.is_dir():
        dataset_name = dataset_dir.name
        mts_class = dataset_dir.parent.name
    else:
        dataset_name = dataset_dir.stem
        mts_class = dataset_dir.parent.name

    clip_suffix = f"_clip-{clip[0]}-{clip[1]}" if clip is not None else ""
    base_name = f"channel-{channel}_{mts_class}_{dataset_name}{clip_suffix}"

    if stems:
        stem_name = f"stems_{base_name}"
        plot_stems(
            series=channel_vec,
            name=stem_name,
            output_dir=save_dir,
            cmap=cmap,
        )
    else:
        fig, ax = plt.subplots(figsize=(width, 1), dpi=DEFAULT_DPI)
        sns.heatmap(
            channel_vec[np.newaxis, :],
            cmap=sns.color_palette(cmap, as_cmap=True),
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            xticklabels=False,
            yticklabels=False,
            ax=ax,
        )
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False)
        fig.tight_layout(pad=0.05)

        fig.savefig(save_dir / f"{base_name}.svg", dpi=DEFAULT_DPI, transparent=True)
        fig.savefig(save_dir / f"{base_name}.png", dpi=DEFAULT_DPI, transparent=True)
        plt.show()

    return channel_vec


def plot_stems(
    series: np.ndarray | None = None,
    name: str = "stems",
    output_dir: Path | str = Path("."),
    *,
    dataset_dir: Path | str | None = None,
    channel: int = 0,
    clip: tuple[int, int] | None = None,
    cmap: str = "icefire",
    ax=None,
):
    """
    Plot a 1D series as a stem plot; can load a channel from a dataset_dir if series is not provided.
    Saves both SVG and PNG with the given name.
    """
    apply_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if series is None:
        if dataset_dir is None:
            raise ValueError("Provide either series or dataset_dir.")
        dataset_dir = Path(dataset_dir)
        data_path = dataset_dir / "timeseries.npy" if dataset_dir.is_dir() else dataset_dir
        if not data_path.exists():
            raise FileNotFoundError(f"Missing timeseries file: {data_path}")
        data = np.load(data_path).astype(float, copy=False)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {data.shape} for {data_path}")
        if data.shape[0] > data.shape[1]:
            data = data.T
        M, T = data.shape
        if channel < 0 or channel >= M:
            raise IndexError(f"Channel {channel} out of bounds for M={M}")
        is_cauchy = "cauchy" in data_path.name.lower() or "cauchy" in data_path.parent.name.lower()
        if is_cauchy:
            scaled = robust_scale(data, axis=1)
        else:
            scaled = zscore(data, axis=1, nan_policy="omit")
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
        if clip is not None:
            if (
                not isinstance(clip, tuple)
                or len(clip) != 2
                or not all(isinstance(x, int) for x in clip)
            ):
                raise TypeError("clip must be a tuple of two ints (start, end).")
            start, end = clip
            if not (0 <= start < end <= T):
                raise ValueError(f"clip must satisfy 0 <= start < end <= {T}, got {clip}")
            vals = np.asarray(scaled[channel, start:end], float)
        else:
            vals = np.asarray(scaled[channel, :], float)
    else:
        vals = np.asarray(series, dtype=float).reshape(-1)
        if clip is not None:
            if (
                not isinstance(clip, tuple)
                or len(clip) != 2
                or not all(isinstance(x, int) for x in clip)
            ):
                raise TypeError("clip must be a tuple of two ints (start, end).")
            start, end = clip
            if not (0 <= start < end <= vals.size):
                raise ValueError(f"clip must satisfy 0 <= start < end <= {vals.size}, got {clip}")
            vals = vals[start:end]

    xs = np.arange(vals.size)
    cmap_fn = sns.color_palette(cmap, as_cmap=True)
    colors = cmap_fn((vals - vals.min()) / (vals.ptp() or 1.0))
    if ax is None:
        width = float(np.clip(8.0 * (vals.size / 1000.0), 4.0, 18.0))
        fig, ax = plt.subplots(figsize=(width, 2.5), dpi=DEFAULT_DPI)
    else:
        fig = ax.figure
    ax.vlines(xs, 0.0, vals, colors=colors, linewidth=1.5)
    ax.hlines(0.0, xs[0] - 1, xs[-1] + 1, colors="black", linewidth=1.0)
    ax.scatter(xs, vals, c=colors, s=10, zorder=3, alpha=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.tick_params(left=False, bottom=False)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    svg_path = output_dir / f"{name}.svg"
    png_path = output_dir / f"{name}.png"
    fig.savefig(svg_path, transparent=True, dpi=DEFAULT_DPI)
    fig.savefig(png_path, transparent=True, dpi=DEFAULT_DPI)
    plt.close(fig)
    return fig, ax


def _scatter_size_kw(meta_df, size_col: str | None, s: float) -> dict:
    """Build seaborn size kwargs: scale around baseline s if size_col exists."""
    if size_col and size_col in meta_df.columns:
        return {"size": size_col, "sizes": (s * 0.4, s * 2.0)}
    return {"s": s}


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


def _best_corner_loc(x_vals: np.ndarray, y_vals: np.ndarray) -> tuple[float, float, str, str]:
    """Return (x_ax, y_ax, ha, va) in axes coords for the least-dense corner."""
    if x_vals.size < 2:
        return 0.97, 0.97, "right", "top"
    xr, yr = x_vals.ptp(), y_vals.ptp()
    if xr < 1e-12 or yr < 1e-12:
        return 0.97, 0.97, "right", "top"
    xn = (x_vals - x_vals.min()) / xr
    yn = (y_vals - y_vals.min()) / yr
    corners = [
        (0.03, 0.97, "left",  "top"),
        (0.97, 0.97, "right", "top"),
        (0.03, 0.03, "left",  "bottom"),
        (0.97, 0.03, "right", "bottom"),
    ]
    counts = [
        int(((xn < 0.25) & (yn > 0.75)).sum()),
        int(((xn > 0.75) & (yn > 0.75)).sum()),
        int(((xn < 0.25) & (yn < 0.25)).sum()),
        int(((xn > 0.75) & (yn < 0.25)).sum()),
    ]
    return corners[int(np.argmin(counts))]


def plot_spi_space_individual(
    dataset_path: str,
    spis: list[str],
    *,
    split_directed: bool = False,
    metric: str = "spearman",
    show_dataset: bool = False,
) -> None:
    """
    Scatter + marginal KDE of two SPI flattened vectors from a single dataset.

    Args:
        dataset_path: Path to dataset directory (supports glob patterns).
        spis: Exactly two SPI names.
        split_directed: If False (default), symmetrize matrices and plot once.
                        If True, respect directionality and plot i→j / j→i separately.
    """
    if len(spis) != 2:
        raise ValueError("Expects exactly two SPI names.")

    spi_x, spi_y = spis

    if any(ch in str(dataset_path) for ch in "*?[]"):
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
        arr_x = np.asarray(npz[spi_x], float)
        arr_y = np.asarray(npz[spi_y], float)

    if arr_x.shape != arr_y.shape:
        raise ValueError("Shape mismatch")

    apply_plot_style()

    directed_x = directed_map[spi_x] if split_directed else False
    directed_y = directed_map[spi_y] if split_directed else False

    # Symmetrize if not splitting
    if not split_directed:
        arr_x = 0.5 * (arr_x + arr_x.T)
        arr_y = 0.5 * (arr_y + arr_y.T)

    upper_mask = np.triu(np.ones(arr_x.shape, dtype=bool), k=1)
    lower_mask = np.tril(np.ones(arr_x.shape, dtype=bool), k=-1)

    def _extract(use_lower: bool) -> tuple[np.ndarray, np.ndarray]:
        mask_x = lower_mask if (use_lower and directed_x) else upper_mask
        mask_y = lower_mask if (use_lower and directed_y) else upper_mask
        x_vals = arr_x[mask_x]
        y_vals = arr_y[mask_y]
        valid_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        return x_vals[valid_mask], y_vals[valid_mask]

    directions: list[tuple[str, bool]] = [(r"$i \to j$", False)]
    if split_directed and (directed_x or directed_y):
        directions.append((r"$j \to i$", True))

    plotted = False
    for direction_label, use_lower in directions:
        x_vals, y_vals = _extract(use_lower)
        if x_vals.size == 0:
            continue

        if metric == "pearson":
            corr = float(np.corrcoef(x_vals, y_vals)[0, 1])
            corr_label = f"$r = {corr:.2f}$"
        else:
            corr, _ = spearmanr(x_vals, y_vals)
            corr_label = f"$\\rho = {corr:.2f}$"
        g = sns.jointplot(
            x=x_vals,
            y=y_vals,
            kind="scatter",
            height=6,
            marginal_kws=dict(kde=True, fill=True),
            s=15,
            alpha=0.6,
            color="#1f77b4",
        )
        sns.regplot(x=x_vals, y=y_vals, ax=g.ax_joint, scatter=False, color="#d62728", ci=None)
        x_label = f"{spi_x} ({direction_label})" if directed_x else spi_x
        y_label = f"{spi_y} ({direction_label})" if directed_y else spi_y
        g.set_axis_labels(x_label, y_label)
        title_slug = "/".join(dataset_dir.parts[-2:]) if show_dataset else dataset_dir.parts[-2] if show_dataset else dataset_dir.parts[-2]
        g.fig.suptitle(title_slug, y=1.02)
        at = AnchoredText(corr_label, loc="upper left", frameon=True, prop=dict(size=10), pad=0.3, borderpad=0.3)
        at.patch.set(facecolor="none", edgecolor="none", alpha=0.7)
        g.ax_joint.add_artist(at)

        plotted = True

    if plotted:
        plt.show()


def plot_spi_space_individual_embed(
    dataset_path: str,
    spis: list[str],
    ax,
    *,
    split_directed: bool = False,
    marginals: bool = True,
    marginal_kde: bool = True,
    main_spines: bool = True,
    metric: str = "spearman",
    show_dataset: bool = False,
) -> None:
    """
    Same visual output as plot_spi_space_individual, but embeds into an existing subplot axes.

    When marginals=True, the provided ax must be a subplot cell (created via plt.subplots or
    GridSpec). It will be removed and replaced by an inner 2x2 GridSpecFromSubplotSpec
    replicating the jointplot layout: histogram (+ optional KDE) on top (x) and right (y),
    scatter + regression line in the center.

    When marginals=False, the provided ax is used directly for the scatter plot only.

    NOTE: split_directed=True is not supported; only the symmetrised single-direction
    case is rendered. Use plot_spi_space_individual for directed multi-plot cases.

    Args:
        dataset_path: Path to dataset directory.
        spis: Exactly two SPI names.
        ax: Matplotlib subplot axes to embed into.
        split_directed: Ignored (symmetrised only); kept for API parity.
        marginals: If True, show marginal histograms on top and right.
        marginal_kde: If True, overlay a smoothed KDE on the marginal histograms.
        main_spines: If True (default), all four spines are shown on the scatter axes.
                     If False, top and right spines are hidden.
    """
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    if len(spis) != 2:
        raise ValueError("Expects exactly two SPI names.")

    spi_x, spi_y = spis

    if any(ch in str(dataset_path) for ch in "*?[]"):
        matches = sorted(Path(".").glob(str(dataset_path)))
        if not matches:
            raise FileNotFoundError(f"No dataset matched pattern '{dataset_path}'")
        if len(matches) > 1:
            raise ValueError(f"Pattern '{dataset_path}' matched multiple datasets")
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
            raise KeyError(f"Missing SPIs in archive: {archive}")
        arr_x = np.asarray(npz[spi_x], float)
        arr_y = np.asarray(npz[spi_y], float)

    arr_x = 0.5 * (arr_x + arr_x.T)
    arr_y = 0.5 * (arr_y + arr_y.T)
    upper_mask = np.triu(np.ones(arr_x.shape, dtype=bool), k=1)
    x_vals, y_vals = arr_x[upper_mask], arr_y[upper_mask]
    valid = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_vals, y_vals = x_vals[valid], y_vals[valid]

    if metric == "pearson":
        corr = float(np.corrcoef(x_vals, y_vals)[0, 1])
        corr_label = f"$r={corr:.2f}$"
    else:
        corr, _ = spearmanr(x_vals, y_vals)
        corr_label = f"$\\rho={corr:.2f}$"
    title_slug = "/".join(dataset_dir.parts[-2:]) if show_dataset else dataset_dir.parts[-2]

    if marginals:
        fig = ax.get_figure()
        ss = ax.get_subplotspec()
        ax.remove()

        inner = GridSpecFromSubplotSpec(
            2, 2, subplot_spec=ss,
            height_ratios=[1, 4], width_ratios=[4, 1],
            hspace=0.05, wspace=0.05,
        )
        ax_joint = fig.add_subplot(inner[1, 0])
        ax_marg_x = fig.add_subplot(inner[0, 0], sharex=ax_joint)
        ax_marg_y = fig.add_subplot(inner[1, 1], sharey=ax_joint)
        ax_corner = fig.add_subplot(inner[0, 1])
        ax_corner.axis("off")

        # Marginals: histogram + optional KDE
        sns.histplot(x=x_vals, ax=ax_marg_x, stat="density", kde=marginal_kde,
                     color="#1f77b4", fill=True, linewidth=0.8)
        sns.histplot(y=y_vals, ax=ax_marg_y, stat="density", kde=marginal_kde,
                     color="#1f77b4", fill=True, linewidth=0.8)

        # Top marginal: keep bottom spine only
        ax_marg_x.spines[["top", "left", "right"]].set_visible(False)
        ax_marg_x.tick_params(left=False, labelleft=False)
        plt.setp(ax_marg_x.get_xticklabels(), visible=False)
        ax_marg_x.set_xlabel("")
        ax_marg_x.set_ylabel("")

        # Right marginal: keep left spine only
        ax_marg_y.spines[["top", "right", "bottom"]].set_visible(False)
        ax_marg_y.tick_params(bottom=False, labelbottom=False)
        plt.setp(ax_marg_y.get_yticklabels(), visible=False)
        ax_marg_y.set_xlabel("")
        ax_marg_y.set_ylabel("")

        ax_scatter = ax_joint
    else:
        ax_scatter = ax

    # Scatter + regression
    ax_scatter.scatter(x_vals, y_vals, s=15, alpha=0.4, color="#1f77b4")
    sns.regplot(x=x_vals, y=y_vals, ax=ax_scatter, scatter=False, color="#d62728", ci=None)
    ax_scatter.set_xlabel(spi_x)
    ax_scatter.set_ylabel(spi_y)
    ax_scatter.set_title(title_slug)
    at = AnchoredText(corr_label, loc="upper left", frameon=True, prop=dict(size=15), pad=0.3, borderpad=0.3)
    at.patch.set(facecolor="none", edgecolor="gray", alpha=0.7)
    ax_scatter.add_artist(at)
    if not main_spines:
        ax_scatter.spines[["top", "right"]].set_visible(False)


def plot_spi_space_recursive(
    dataset_path: str,
    spis: list[str],
    *,
    split_directed: bool = False,
    formats: list[str] | None = None,
    dpi: int = 300,
    show: bool = False,
    show_dataset: bool = False,
) -> list[Path]:
    """
    Recursively plot SPI-SPI scatter plots for all datasets under a directory.

    Args:
        dataset_path: Parent directory containing M<M>_T<T>_I<I>_<variant> subdirectories,
                      each with an spi_mpis.npz file.
        spis: Exactly two SPI names to compare.
        split_directed: If False (default), symmetrize matrices and plot once.
                        If True, respect directionality and plot i→j / j→i separately.
        formats: List of output formats (e.g., ["png", "svg"]). Defaults to ["png"].
        dpi: Resolution for saved figures. Defaults to 300.
        show: If True, display each plot interactively. Defaults to False.

    Returns:
        List of paths to saved plot files.
    """
    if len(spis) != 2:
        raise ValueError("Expects exactly two SPI names.")

    spi_x, spi_y = spis
    formats = formats or ["png"]
    root = Path(dataset_path)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {root}")

    # Create output directory
    output_dir = root / f"plot_{spi_x}_{spi_y}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all subdirectories with spi_mpis.npz
    candidates = sorted(
        p.parent for p in root.rglob("spi_mpis.npz")
        if p.parent != root  # exclude root itself if it has the file
    )
    if not candidates:
        raise FileNotFoundError(f"No spi_mpis.npz files found under {root}")

    apply_plot_style()
    saved_paths: list[Path] = []

    for dataset_dir in candidates:
        archive = dataset_dir / "spi_mpis.npz"
        meta_path = dataset_dir / "meta.json"
        if not meta_path.exists():
            continue

        meta = load_json(meta_path)
        spi_meta = meta.get("pyspi", {}).get("spis", [])
        directed_map = {
            entry.get("name"): bool(entry.get("directed", False))
            for entry in spi_meta
            if isinstance(entry, dict) and entry.get("name")
        }
        if spi_x not in directed_map or spi_y not in directed_map:
            continue

        with np.load(archive) as npz:
            if spi_x not in npz or spi_y not in npz:
                continue
            arr_x = np.asarray(npz[spi_x], float)
            arr_y = np.asarray(npz[spi_y], float)

        if arr_x.shape != arr_y.shape:
            continue

        directed_x = directed_map[spi_x] if split_directed else False
        directed_y = directed_map[spi_y] if split_directed else False

        # Symmetrize if not splitting
        if not split_directed:
            arr_x = 0.5 * (arr_x + arr_x.T)
            arr_y = 0.5 * (arr_y + arr_y.T)

        upper_mask = np.triu(np.ones(arr_x.shape, dtype=bool), k=1)
        lower_mask = np.tril(np.ones(arr_x.shape, dtype=bool), k=-1)

        def _extract(use_lower: bool) -> tuple[np.ndarray, np.ndarray]:
            mask_x = lower_mask if (use_lower and directed_x) else upper_mask
            mask_y = lower_mask if (use_lower and directed_y) else upper_mask
            x_vals = arr_x[mask_x]
            y_vals = arr_y[mask_y]
            valid_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
            return x_vals[valid_mask], y_vals[valid_mask]

        directions: list[tuple[str, bool, str]] = [(r"$i \to j$", False, "")]
        if split_directed and (directed_x or directed_y):
            directions.append((r"$j \to i$", True, "__ji"))
            directions[0] = (r"$i \to j$", False, "__ij")

        base_name = dataset_dir.name

        for direction_label, use_lower, dir_suffix in directions:
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
                color="#1f77b4",
            )
            sns.regplot(x=x_vals, y=y_vals, ax=g.ax_joint, scatter=False, color="#d62728", ci=None)
            x_label = f"{spi_x} ({direction_label})" if directed_x else spi_x
            y_label = f"{spi_y} ({direction_label})" if directed_y else spi_y
            g.set_axis_labels(x_label, y_label)
            title_slug = "/".join(dataset_dir.parts[-2:]) if show_dataset else dataset_dir.parts[-2] if show_dataset else dataset_dir.parts[-2]
            g.fig.suptitle(title_slug, y=1.02)
            at = AnchoredText(f"$\\rho = {rho:.2f}$", loc="upper left", frameon=True, prop=dict(size=10), pad=0.3, borderpad=0.3)
            at.patch.set(facecolor="none", edgecolor="none", alpha=0.7)
            g.ax_joint.add_artist(at)

            for fmt in formats:
                out_path = output_dir / f"{base_name}{dir_suffix}.{fmt}"
                g.fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
                saved_paths.append(out_path)

            if show:
                plt.show()
            else:
                plt.close(g.fig)

    print(f"[INFO] Saved {len(saved_paths)} plots to {output_dir}")
    return saved_paths


def plot_mts_corr_density(
    mts_class_paths: list[str],
    spi_pair: list[str],
    *,
    split_directed: bool = False,
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
                  When split_directed=True, you can suffix with __ij or __ji to select a direction;
                  if omitted, both directions are plotted for directed SPIs.
        split_directed: If False (default), symmetrize matrices and produce one plot.
                        If True, respect directionality metadata and plot i→j / j→i separately.
        bw_adjust: Optional KDE bandwidth adjustment passed to seaborn.
        show_hist: If True, overlay per-class histograms (density-normalized).
        kde: If True (default), overlay KDE curves.
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

    def _vector_for(
        mat: np.ndarray, directed: bool, direction: str | None
    ) -> np.ndarray:
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
        fig, ax = plt.subplots(figsize=(6, 6), dpi=DEFAULT_DPI)
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
                directed_x = bool(spi_meta[spi_x_base].get("directed", False)) if split_directed else False
                directed_y = bool(spi_meta[spi_y_base].get("directed", False)) if split_directed else False

                with np.load(npz_path) as npz:
                    if spi_x_base not in npz or spi_y_base not in npz:
                        continue
                    vec_x = _vector_for(
                        np.asarray(npz[spi_x_base], float), directed_x, direction_choice_x
                    )
                    vec_y = _vector_for(
                        np.asarray(npz[spi_y_base], float), directed_y, direction_choice_y
                    )

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

    if split_directed:
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
    else:
        _collect(None, None)


def plot_pca(
    x: np.ndarray,
    meta_df,
    *,
    n_components: int = 2,
    random_state: int = 0,
    feature_space: str = "",
    hue: str = "mts_class",
    size_col: str | None = "M",
    s: float = 30,
    alpha: float = 0.8,
    kde: bool = False,
    ax=None,
    save: str | None = None,
) -> tuple[np.ndarray, plt.Axes]:
    """PCA embedding + scatter plot. Pass ax to embed in an existing figure.
    save: path to save figure (e.g. "pca.svg"). Only works when ax is None.
    """
    apply_plot_style()

    xs = StandardScaler().fit_transform(x)
    pca = PCA(n_components=n_components, random_state=random_state)
    embedding = pca.fit_transform(xs)
    var = pca.explained_variance_ratio_

    meta_df = meta_df.copy()
    meta_df["pca_x"] = embedding[:, 0]
    meta_df["pca_y"] = embedding[:, 1]

    own_fig = ax is None
    if own_fig:
        _, ax = plt.subplots(figsize=(8, 8), dpi=DEFAULT_DPI)

    sns.scatterplot(data=meta_df, x="pca_x", y="pca_y", hue=hue,
                    **_scatter_size_kw(meta_df, size_col, s), alpha=alpha,
                    edgecolor="face", ax=ax, legend="full")
    if kde:
        sns.kdeplot(data=meta_df, x="pca_x", y="pca_y", hue=hue,
                    levels=10, thresh=0.05, fill=True, alpha=0.3, ax=ax, legend=False)

    ax.legend(title=hue, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    ax.set_title(f"PCA ({feature_space}) | Var: {var[0]+var[1]:.4f}")
    ax.set_xlabel(f"PC1 ({var[0]:.4f})")
    ax.set_ylabel(f"PC2 ({var[1]:.4f})")
    ax.set_box_aspect(1)
    ax.patch.set_alpha(0)
    ax.get_figure().patch.set_alpha(0)

    if own_fig:
        plt.tight_layout()
        if save:
            ax.get_figure().savefig(save, bbox_inches="tight", transparent=True)
        plt.show()
    return embedding, ax


def plot_pca_dark(
    x: np.ndarray,
    meta_df,
    *,
    n_components: int = 2,
    random_state: int = 0,
    feature_space: str = "",
    hue: str = "mts_class",
    size_col: str | None = "M",
    s: float = 30,
    alpha: float = 0.8,
    facecolor: str = "#282a36",
    kde: bool = True,
    ax=None,
    save: str | None = None,
) -> tuple[np.ndarray, plt.Axes]:
    """PCA embedding + scatter/KDE plot (dark background). Pass ax to embed."""
    apply_plot_style()

    xs = StandardScaler().fit_transform(x)
    pca = PCA(n_components=n_components, random_state=random_state)
    embedding = pca.fit_transform(xs)
    var = pca.explained_variance_ratio_

    meta_df = meta_df.copy()
    meta_df["pca_x"] = embedding[:, 0]
    meta_df["pca_y"] = embedding[:, 1]

    own_fig = ax is None
    if own_fig:
        _, ax = plt.subplots(figsize=(8, 8), dpi=DEFAULT_DPI)

    sns.scatterplot(data=meta_df, x="pca_x", y="pca_y", hue=hue, palette="pastel",
                    **_scatter_size_kw(meta_df, size_col, s), alpha=alpha,
                    edgecolor="white", linewidth=0.3, ax=ax, legend="full")
    if kde:
        sns.kdeplot(data=meta_df, x="pca_x", y="pca_y", hue=hue, palette="pastel",
                    levels=10, thresh=0.05, fill=True, alpha=0.5, ax=ax, legend=False)

    ax.legend(title=hue, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    ax.set_title(f"PCA ({feature_space}) | Var: {var[0]+var[1]:.4f}")
    ax.set_xlabel(f"PC1 ({var[0]:.4f})")
    ax.set_ylabel(f"PC2 ({var[1]:.4f})")
    ax.set_box_aspect(1)
    ax.set_facecolor(facecolor)

    if own_fig:
        plt.tight_layout()
        if save:
            ax.get_figure().savefig(save, bbox_inches="tight")
        plt.show()
    return embedding, ax


def plot_umap(
    x: np.ndarray,
    meta_df,
    *,
    metric: str = "euclidean",
    scale: bool = True,
    n_neighbors: int = 7,
    min_dist: float = 0.5,
    random_state: int = 0,
    feature_space: str = "",
    hue: str = "mts_class",
    size_col: str | None = "M",
    s: float = 30,
    alpha: float = 0.8,
    kde: bool = False,
    ax=None,
    save: str | None = None,
) -> tuple[np.ndarray, plt.Axes]:
    """UMAP embedding + scatter plot. Pass ax to embed in an existing figure.
    save: path to save figure (e.g. "umap.svg"). Only works when ax is None.
    scale: if True (default), z-score feature columns before UMAP. Disable for
           bounded-correlation feature spaces where magnitude is meaningful.
    """
    if UMAP is None:
        raise ImportError("umap-learn is required for plot_umap")

    apply_plot_style()

    xs = StandardScaler().fit_transform(x) if scale else x
    embedding = UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist,
        metric=metric, random_state=random_state, verbose=False,
    ).fit_transform(xs)

    meta_df = meta_df.copy()
    meta_df["umap_x"] = embedding[:, 0]
    meta_df["umap_y"] = embedding[:, 1]

    own_fig = ax is None
    if own_fig:
        _, ax = plt.subplots(figsize=(8, 8), dpi=DEFAULT_DPI)

    sns.scatterplot(data=meta_df, x="umap_x", y="umap_y", hue=hue,
                    **_scatter_size_kw(meta_df, size_col, s), alpha=alpha,
                    edgecolor="face", ax=ax, legend="full")
    if kde:
        sns.kdeplot(data=meta_df, x="umap_x", y="umap_y", hue=hue,
                    levels=10, thresh=0.05, fill=True, alpha=0.3, ax=ax, legend=False)

    ax.legend(title=hue, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    ax.set_title(f"UMAP ({feature_space}, metric={metric}, scale={scale})")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_box_aspect(1)
    ax.patch.set_alpha(0)
    ax.get_figure().patch.set_alpha(0)

    if own_fig:
        plt.tight_layout()
        if save:
            ax.get_figure().savefig(save, bbox_inches="tight", transparent=True)
        plt.show()
    return embedding, ax


def plot_umap_dark(
    x: np.ndarray,
    meta_df,
    *,
    metric: str = "euclidean",
    scale: bool = True,
    n_neighbors: int = 7,
    min_dist: float = 0.5,
    random_state: int = 0,
    feature_space: str = "",
    hue: str = "mts_class",
    s: float = 30,
    size_col: str | None = "M",
    alpha: float = 0.8,
    facecolor: str = "#282a36",
    kde: bool = True,
    ax=None,
    save: str | None = None,
) -> tuple[np.ndarray, plt.Axes]:
    """UMAP embedding + scatter/KDE plot (dark background). Pass ax to embed."""
    if UMAP is None:
        raise ImportError("umap-learn is required for plot_umap_dark")

    apply_plot_style()

    xs = StandardScaler().fit_transform(x) if scale else x
    embedding = UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist,
        metric=metric, random_state=random_state, verbose=False,
    ).fit_transform(xs)

    meta_df = meta_df.copy()
    meta_df["umap_x"] = embedding[:, 0]
    meta_df["umap_y"] = embedding[:, 1]

    own_fig = ax is None
    if own_fig:
        _, ax = plt.subplots(figsize=(8, 8), dpi=DEFAULT_DPI)

    sns.scatterplot(data=meta_df, x="umap_x", y="umap_y", hue=hue, palette="pastel",
                    **_scatter_size_kw(meta_df, size_col, s), alpha=alpha,
                    edgecolor="white", linewidth=0.3, ax=ax, legend="full")
    if kde:
        sns.kdeplot(data=meta_df, x="umap_x", y="umap_y", hue=hue, palette="pastel",
                    levels=10, thresh=0.05, fill=True, alpha=0.5, ax=ax, legend=False)

    ax.legend(title=hue, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    ax.set_title(f"UMAP ({feature_space}, metric={metric}, scale={scale})")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_box_aspect(1)
    ax.set_facecolor(facecolor)

    if own_fig:
        plt.tight_layout()
        if save:
            ax.get_figure().savefig(save, bbox_inches="tight")
        plt.show()
    return embedding, ax


def plot_tsne(
    x: np.ndarray,
    meta_df,
    *,
    metric: str = "euclidean",
    scale: bool = True,
    perplexity: float = 30.0,
    random_state: int = 0,
    feature_space: str = "",
    hue: str = "mts_class",
    s: float = 30,
    size_col: str | None = "M",
    alpha: float = 0.8,
    kde: bool = False,
    ax=None,
    save: str | None = None,
) -> tuple[np.ndarray, plt.Axes]:
    """t-SNE embedding + scatter plot. Pass ax to embed in an existing figure.
    save: path to save figure (e.g. "tsne.svg"). Only works when ax is None.
    scale: if True (default), z-score feature columns before t-SNE.
    """
    if TSNE is None:
        raise ImportError("scikit-learn is required for plot_tsne")

    apply_plot_style()

    xs = StandardScaler().fit_transform(x) if scale else x
    embedding = TSNE(
        n_components=2, metric=metric, random_state=random_state,
        init="pca", perplexity=perplexity, learning_rate="auto", n_jobs=-1, verbose=0,
    ).fit_transform(xs)

    meta_df = meta_df.copy()
    meta_df["tsne_x"] = embedding[:, 0]
    meta_df["tsne_y"] = embedding[:, 1]

    own_fig = ax is None
    if own_fig:
        _, ax = plt.subplots(figsize=(8, 8), dpi=DEFAULT_DPI)

    sns.scatterplot(data=meta_df, x="tsne_x", y="tsne_y", hue=hue,
                    **_scatter_size_kw(meta_df, size_col, s), alpha=alpha,
                    edgecolor="face", ax=ax, legend="full")
    if kde:
        sns.kdeplot(data=meta_df, x="tsne_x", y="tsne_y", hue=hue,
                    levels=10, thresh=0.05, fill=True, alpha=0.3, ax=ax, legend=False)

    ax.legend(title=hue, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    ax.set_title(f"t-SNE ({feature_space}, metric={metric}, scale={scale}, perplexity={perplexity})")
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.set_box_aspect(1)
    ax.patch.set_alpha(0)
    ax.get_figure().patch.set_alpha(0)

    if own_fig:
        plt.tight_layout()
        if save:
            ax.get_figure().savefig(save, bbox_inches="tight", transparent=True)
        plt.show()
    return embedding, ax


def plot_tsne_dark(
    x: np.ndarray,
    meta_df,
    *,
    metric: str = "euclidean",
    scale: bool = True,
    perplexity: float = 30.0,
    random_state: int = 0,
    feature_space: str = "",
    hue: str = "mts_class",
    s: float = 30,
    size_col: str | None = "M",
    alpha: float = 0.8,
    facecolor: str = "#282a36",
    kde: bool = True,
    ax=None,
    save: str | None = None,
) -> tuple[np.ndarray, plt.Axes]:
    """t-SNE embedding + scatter/KDE plot (dark background). Pass ax to embed."""
    if TSNE is None:
        raise ImportError("scikit-learn is required for plot_tsne_dark")

    apply_plot_style()

    xs = StandardScaler().fit_transform(x) if scale else x
    embedding = TSNE(
        n_components=2, metric=metric, random_state=random_state,
        init="pca", perplexity=perplexity, learning_rate="auto", n_jobs=-1, verbose=0,
    ).fit_transform(xs)

    meta_df = meta_df.copy()
    meta_df["tsne_x"] = embedding[:, 0]
    meta_df["tsne_y"] = embedding[:, 1]

    own_fig = ax is None
    if own_fig:
        _, ax = plt.subplots(figsize=(8, 8), dpi=DEFAULT_DPI)

    sns.scatterplot(data=meta_df, x="tsne_x", y="tsne_y", hue=hue, palette="pastel",
                    **_scatter_size_kw(meta_df, size_col, s), alpha=alpha,
                    edgecolor="white", linewidth=0.3, ax=ax, legend="full")
    if kde:
        sns.kdeplot(data=meta_df, x="tsne_x", y="tsne_y", hue=hue, palette="pastel",
                    levels=10, thresh=0.05, fill=True, alpha=0.5, ax=ax, legend=False)

    ax.legend(title=hue, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    ax.set_title(f"t-SNE ({feature_space}, metric={metric}, scale={scale}, perplexity={perplexity})")
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.set_box_aspect(1)
    ax.set_facecolor(facecolor)

    if own_fig:
        plt.tight_layout()
        if save:
            ax.get_figure().savefig(save, bbox_inches="tight")
        plt.show()
    return embedding, ax



# Anchor colors for regime mixing: linear=peach, monotonic=lavender, non-monotonic=mint
_REGIME_ANCHORS = np.array([
    [1.0,  0.80, 0.55],   # linear       → peach
    [0.78, 0.65, 0.95],   # monotonic    → lavender
    [0.55, 0.90, 0.88],   # non-monotonic → mint
], dtype=np.float32)
_REGIME_LABELS = ["linear", "monotonic", "non-monotonic"]


def _regime_rgb(proportions: np.ndarray) -> np.ndarray:
    """Mix anchor colors by regime proportions. proportions: (N, 3) in [0,1]."""
    return np.clip(proportions @ _REGIME_ANCHORS, 0.0, 1.0).astype(np.float32)


def regime_colors(dataset_paths) -> np.ndarray:
    """
    Parse l/m/nm proportions from sin_mts dataset slugs.
    Returns (N, 3) array where columns are [p_linear, p_monotonic, p_non_monotonic].
    Use _regime_rgb() to convert to displayable colors.
    """
    pat = re.compile(r"_l-(\d+)_m-(\d+)_nm-(\d+)")
    out = []
    for path in dataset_paths:
        m = pat.search(str(path))
        if m is None:
            raise ValueError(f"Cannot parse l/m/nm from path: {path}")
        l, mo, nm = int(m.group(1)), int(m.group(2)), int(m.group(3))
        total = l + mo + nm
        out.append([l / total, mo / total, nm / total])
    return np.array(out, dtype=np.float32)


def _embedding(Xs: np.ndarray, method: str, random_state: int, **kwargs):
    """Compute 2D embedding. Returns (emb, xlabel, ylabel, title)."""
    if method == "pca":
        pca = PCA(n_components=2, random_state=random_state)
        emb = pca.fit_transform(Xs)
        var = pca.explained_variance_ratio_
        return emb, f"PC1 ({var[0]:.3f})", f"PC2 ({var[1]:.3f})", f"PCA | var={var[0]+var[1]:.3f}"
    if method == "umap":
        if UMAP is None:
            raise ImportError("umap-learn is required for UMAP")
        emb = UMAP(n_neighbors=kwargs.get("n_neighbors", 7), min_dist=kwargs.get("min_dist", 0.5),
                   metric=kwargs.get("metric", "euclidean"), random_state=random_state, verbose=False
                   ).fit_transform(Xs)
        return emb, "UMAP-1", "UMAP-2", f"UMAP (nn={kwargs.get('n_neighbors',7)}, d={kwargs.get('min_dist',0.5)})"
    if method == "tsne":
        if TSNE is None:
            raise ImportError("scikit-learn is required for t-SNE")
        p = kwargs.get("perplexity", 30.0)
        emb = TSNE(n_components=2, metric=kwargs.get("metric", "euclidean"), random_state=random_state,
                   init="pca", perplexity=p, learning_rate="auto").fit_transform(Xs)
        return emb, "t-SNE-1", "t-SNE-2", f"t-SNE (p={p})"
    raise ValueError(f"Unknown method '{method}'. Expected 'pca', 'umap', or 'tsne'.")


def _point_sizes(meta_df, size_col, sizes, n):
    if meta_df is not None and size_col and size_col in meta_df.columns:
        vals = meta_df[size_col].to_numpy(dtype=float)
        lo, hi = vals.min(), vals.max()
        norm = (vals - lo) / (hi - lo) if hi > lo else np.full(n, 0.5)
        return sizes[0] + norm * (sizes[1] - sizes[0])
    return sizes[0]


def plot_regime_embedding(
    X: np.ndarray,
    proportions: np.ndarray,
    *,
    method: str = "pca",
    meta_df=None,
    size_col: str | None = "M",
    sizes: tuple[float, float] = (20.0, 120.0),
    random_state: int = 0,
    n_neighbors: int = 7,
    min_dist: float = 0.5,
    perplexity: float = 30.0,
    metric: str = "euclidean",
    facecolor: str = "#282a36",
    alpha: float = 0.85,
    plotly: bool = False,
) -> np.ndarray:
    """
    Embed X and scatter-plot with per-point colors mixed from regime proportions.
    Anchors: linear=orange, monotonic=purple, non-monotonic=teal.

    Args:
        proportions: (N, 3) from regime_colors() — [p_linear, p_monotonic, p_non_monotonic].
        plotly: If True, render interactively with plotly instead of matplotlib.
    Returns:
        (N, 2) embedding array.
    """
    Xs = StandardScaler().fit_transform(X)
    emb, xlabel, ylabel, title = _embedding(
        Xs, method, random_state,
        n_neighbors=n_neighbors, min_dist=min_dist, perplexity=perplexity, metric=metric,
    )
    rgb = _regime_rgb(proportions)
    s = _point_sizes(meta_df, size_col, sizes, len(proportions))

    if plotly:
        if _go is None:
            raise ImportError("plotly is required for plotly=True")
        # plotly marker size is diameter; matplotlib s is area — convert via sqrt
        # marker_size = np.sqrt(s) if hasattr(s, "__len__") else float(np.sqrt(s))
        marker_size = s**(1/3) if hasattr(s, "__len__") else float(s**(1/3))
        colors = [f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})" for r, g, b in rgb]
        hover = [
            f"p_linear={p[0]:.2f}, p_mono={p[1]:.2f}, p_nm={p[2]:.2f}"
            for p in proportions
        ]
        fig = _go.Figure()
        fig.add_trace(_go.Scatter(
            x=emb[:, 0], y=emb[:, 1],
            mode="markers",
            marker=dict(color=colors, size=marker_size),
            text=hover,
            hoverinfo="text",
            showlegend=False,
        ))
        for label, color in zip(_REGIME_LABELS, _REGIME_ANCHORS):
            fig.add_trace(_go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(color=f"rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})", size=10),
                name=label,
            ))
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            plot_bgcolor=facecolor,
            paper_bgcolor=facecolor,
            font=dict(color="white"),
            legend_title_text="regime (pure)",
            width=700, height=700,
            yaxis=dict(scaleanchor="x", scaleratio=1),
        )
        fig.show()
        return emb

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 8), dpi=DEFAULT_DPI)
    ax.scatter(emb[:, 0], emb[:, 1], c=rgb, s=s, alpha=alpha, linewidths=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_box_aspect(1)
    ax.set_facecolor(facecolor)
    for label, color in zip(_REGIME_LABELS, _REGIME_ANCHORS):
        ax.scatter([], [], c=[color], s=50, label=label)
    ax.legend(title="regime (pure)", loc="upper left", bbox_to_anchor=(1, 1), frameon=True)
    plt.tight_layout()
    plt.show()
    return emb


def plot_regime_embedding_triple(
    X: np.ndarray,
    proportions: np.ndarray,
    *,
    method: str = "pca",
    meta_df=None,
    size_col: str | None = "M",
    sizes: tuple[float, float] = (20.0, 120.0),
    random_state: int = 0,
    n_neighbors: int = 7,
    min_dist: float = 0.5,
    perplexity: float = 30.0,
    metric: str = "euclidean",
    facecolor: str = "#282a36", # "#282a36"
    alpha: float = 0.85,
) -> np.ndarray:
    """
    Embed X and produce 3 subplots — one per regime proportion (p_linear, p_monotonic, p_non_monotonic).
    Each subplot uses a sequential colormap matched to the anchor color.

    Returns:
        (N, 2) embedding array.
    """
    apply_plot_style()
    Xs = StandardScaler().fit_transform(X)
    emb, xlabel, ylabel, title_base = _embedding(
        Xs, method, random_state,
        n_neighbors=n_neighbors, min_dist=min_dist, perplexity=perplexity, metric=metric,
    )
    s = _point_sizes(meta_df, size_col, sizes, len(proportions))
    cmaps = ["YlOrBr", "BuPu", "BuGn"]  # peach, lavender, mint

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), dpi=DEFAULT_DPI)
    for i, (ax, label, cmap) in enumerate(zip(axes, _REGIME_LABELS, cmaps)):
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=proportions[:, i],
                        cmap=cmap, vmin=0, vmax=1, s=s, alpha=alpha, linewidths=0)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{title_base} — p_{label}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel if i == 0 else "")
        ax.set_box_aspect(1)
        ax.set_facecolor(facecolor)
    plt.tight_layout()
    plt.show()
    return emb


def plot_ternary_regime(
    proportions: np.ndarray,
    *,
    meta_df=None,
    hue_col: str | None = "mts_class",
    facecolor: str = "#282a36",
    alpha: float = 0.85,
    s: float = 40.0,
    show_background: bool = True,
) -> None:
    """
    Plot datasets as points in the (p_linear, p_monotonic, p_non_monotonic) ternary space.

    Vertex layout (equilateral triangle):
        bottom-left  = linear
        bottom-right = monotonic
        top          = non-monotonic

    Args:
        proportions: (N, 3) from regime_colors().
        hue_col: Column in meta_df to color points by (categorical). None = regime mix color.
        show_background: Fill the triangle with the regime anchor color gradient.
    """
    apply_plot_style()

    # Barycentric → Cartesian: bottom-left=linear, bottom-right=monotonic, top=non-monotonic
    def _to_cart(prop):
        p_l, p_m, p_nm = prop[:, 0], prop[:, 1], prop[:, 2]
        x = p_m + 0.5 * p_nm
        y = (np.sqrt(3) / 2) * p_nm
        return x, y

    fig, ax = plt.subplots(figsize=(7, 7), dpi=DEFAULT_DPI)
    ax.set_aspect("equal")
    ax.set_facecolor(facecolor)
    ax.axis("off")

    # Triangle border
    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, np.sqrt(3) / 2, 0]
    ax.plot(tri_x, tri_y, color="white", linewidth=1.0, zorder=2)

    # Vertex labels
    offset = 0.05
    ax.text(-offset, -offset,         "linear",          ha="center", va="top",    color="white", fontsize=10)
    ax.text(1 + offset, -offset,      "monotonic",       ha="center", va="top",    color="white", fontsize=10)
    ax.text(0.5, np.sqrt(3)/2 + offset, "non-monotonic", ha="center", va="bottom", color="white", fontsize=10)

    if show_background:
        # Dense grid of barycentric points → colored by anchor mix
        n_grid = 80
        grid = []
        for i in range(n_grid + 1):
            for j in range(n_grid + 1 - i):
                k = n_grid - i - j
                grid.append([i / n_grid, j / n_grid, k / n_grid])
        grid = np.array(grid, dtype=np.float32)
        gx, gy = _to_cart(grid)
        gc = _regime_rgb(grid)
        ax.scatter(gx, gy, c=gc, s=6, alpha=0.5, linewidths=0, zorder=1)

    # Data points
    px, py = _to_cart(proportions)
    if meta_df is not None and hue_col and hue_col in meta_df.columns:
        classes = meta_df[hue_col].values
        unique = list(dict.fromkeys(classes))
        palette = sns.color_palette("tab10", len(unique))
        color_map = dict(zip(unique, palette))
        for cls in unique:
            mask = classes == cls
            ax.scatter(px[mask], py[mask], c=[color_map[cls]], s=s, alpha=alpha,
                       linewidths=0.3, edgecolors="white", zorder=3, label=cls)
        ax.legend(title=hue_col, loc="upper left", bbox_to_anchor=(1, 1), frameon=True)
    else:
        ax.scatter(px, py, c=_regime_rgb(proportions), s=s, alpha=alpha,
                   linewidths=0.3, edgecolors="white", zorder=3)

    ax.set_title("Regime composition (ternary)")
    plt.tight_layout()
    plt.show()


def plot_spi_corr_sweep(
    dataset_path: Path | str,
    spi_pair: list[str],
    *,
    folder_template: str,
    x_var: str,
    x_transform=None,
    metric: str = "spearman",
    hue: bool = True,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ax=None,
) -> pd.DataFrame:
    """
    Plot corr(spi_pair[0], spi_pair[1]) as a function of a swept parameter,
    walking all datasets under dataset_path/<class_dir>/<dataset_dir>.

    For each dataset dir whose name matches `folder_template`, parses named
    numeric variables (int or float) and uses x_var as the x-axis value.

    Args:
        dataset_path: Root dir, e.g. project_root() / "data/dtw_euclidean_vary-p".
        spi_pair: Exactly two SPI names, e.g. ["dtw", "pdist_euclidean"].
        folder_template: Template using <varname> placeholders (ints or floats),
            e.g. "w-<w>_p-<p>" matches folders like "M16_T100_I5_...w-16_p-0.1".
        x_var: Name of the parsed variable to use as the x-axis.
        x_transform: Optional callable(dict[str, float]) -> float to derive the
            x value from all parsed vars, e.g. for a proportion:
            lambda g: g["w"] / (g["w"] + g["m"]).
            If None, uses groups[x_var] directly.
        metric: "spearman" (default) or "pearson".
        hue: If True, color lines by the intermediate class subdirectory name.
        title: Axes title. If None, no title is set.
        xlabel: x-axis label. If None, no label is set.
        ylabel: y-axis label. If None, no label is set.
        ax: Matplotlib axes to draw on. If None, uses plt.gca().

    Returns:
        DataFrame with columns [x, correlation, mts_class].
    """
    if len(spi_pair) != 2:
        raise ValueError("spi_pair must contain exactly two SPI names.")
    if metric not in ("spearman", "pearson"):
        raise ValueError("metric must be 'spearman' or 'pearson'.")

    spi_x, spi_y = spi_pair
    root = Path(dataset_path)
    if not root.exists():
        raise FileNotFoundError(f"dataset_path not found: {root}")

    # Convert "w-<w>_p-<p>" -> regex "w-(?P<w>[\d.]+)_p-(?P<p>[\d.]+)"
    regex = re.sub(r"<(\w+)>", lambda m: f"(?P<{m.group(1)}>[\\d.]+)", folder_template)
    pat = re.compile(regex)

    rows: list[dict] = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for dataset_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            m = pat.search(f"{class_dir.name}/{dataset_dir.name}")
            if m is None:
                continue

            groups = {k: float(v) for k, v in m.groupdict().items()}
            if x_var not in groups:
                raise KeyError(f"x_var '{x_var}' not in template groups: {list(groups)}")

            x = x_transform(groups) if x_transform is not None else groups[x_var]

            npz_path = dataset_dir / "spi_mpis.npz"
            if not npz_path.exists():
                continue

            with np.load(npz_path) as npz:
                if spi_x not in npz or spi_y not in npz:
                    continue
                arr_x = np.asarray(npz[spi_x], float)
                arr_y = np.asarray(npz[spi_y], float)

            if arr_x.shape != arr_y.shape or arr_x.ndim != 2:
                continue

            arr_x = 0.5 * (arr_x + arr_x.T)
            arr_y = 0.5 * (arr_y + arr_y.T)
            mask = np.triu(np.ones(arr_x.shape, dtype=bool), k=1)
            vec_x, vec_y = arr_x[mask], arr_y[mask]
            valid = np.isfinite(vec_x) & np.isfinite(vec_y)
            if valid.sum() < 2:
                continue

            vec_x, vec_y = vec_x[valid], vec_y[valid]
            if metric == "spearman":
                corr = float(spearmanr(vec_x, vec_y).correlation)
            else:
                corr = float(np.corrcoef(vec_x, vec_y)[0, 1])

            if not np.isfinite(corr):
                continue

            rows.append({"x": x, "correlation": corr, "mts_class": class_dir.name})

    if not rows:
        raise ValueError(
            f"No matching datasets found under {root} with template '{folder_template}'"
        )

    df = pd.DataFrame(rows)

    if ax is None:
        ax = plt.gca()

    sns.lineplot(
        data=df,
        x="x",
        y="correlation",
        hue="mts_class" if hue else None,
        ax=ax,
        errorbar=("ci", 95),
        err_style="band",
    )
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    return df


def plot_spi_corr_meta(
    dataset_path: Path | str,
    spi_pairs: list[list[str]],
    *,
    x_key: str,
    x_transform=None,
    pair_labels: list[str] | None = None,
    metric: str = "spearman",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ax=None,
) -> pd.DataFrame:
    """
    Plot corr(spi_x, spi_y) vs a swept generator parameter for multiple SPI pairs,
    reading the x-axis value from meta.json["generator"]["params"][x_key].

    Walks dataset_path/<class_dir>/<dataset_dir>/, reads each meta.json for the
    generator parameter value, and computes pairwise SPI correlation for each pair.
    Multiple spi_pairs are overlaid as separate hued lines; confidence bands reflect
    variation across instances (dataset_dirs) sharing the same x value.

    This is the meta.json-aware successor to plot_spi_corr_sweep: no folder-name
    regex required, and multiple pairs can be overlaid in one call.

    Args:
        dataset_path: Root dir (parent of class dirs).
        spi_pairs: List of [spi_x, spi_y] pairs to overlay as separate lines.
        x_key: Key in meta["generator"]["params"] to use as x-axis, e.g. "A".
        x_transform: Optional callable(float) -> float applied to the raw x value.
        pair_labels: Display labels for each pair. If None, auto-generates "(spi_x, spi_y)".
        metric: "spearman" (default) or "pearson" — correlation between SPI matrix entries.
        title: Axes title.
        xlabel: x-axis label.
        ylabel: y-axis label.
        ax: Matplotlib axes. If None, uses plt.gca().

    Returns:
        DataFrame with columns [x, correlation, spi_pair].
    """
    if not spi_pairs:
        raise ValueError("spi_pairs must be non-empty.")
    if metric not in ("spearman", "pearson"):
        raise ValueError("metric must be 'spearman' or 'pearson'.")
    if pair_labels is not None and len(pair_labels) != len(spi_pairs):
        raise ValueError("pair_labels must have the same length as spi_pairs.")

    labels = pair_labels or [f"({p[0]}, {p[1]})" for p in spi_pairs]

    root = Path(dataset_path)
    if not root.exists():
        raise FileNotFoundError(f"dataset_path not found: {root}")

    rows: list[dict] = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for dataset_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            meta_path = dataset_dir / "meta.json"
            npz_path = dataset_dir / "spi_mpis.npz"
            if not meta_path.exists() or not npz_path.exists():
                continue

            meta = load_json(meta_path)
            gen_params = meta.get("generator", {}).get("params", {})
            if x_key not in gen_params:
                continue
            raw_x = float(gen_params[x_key])
            x = x_transform(raw_x) if x_transform is not None else raw_x

            with np.load(npz_path) as npz:
                for pair, label in zip(spi_pairs, labels):
                    spi_x, spi_y = pair
                    if spi_x not in npz or spi_y not in npz:
                        continue
                    arr_x = np.asarray(npz[spi_x], float)
                    arr_y = np.asarray(npz[spi_y], float)
                    if arr_x.shape != arr_y.shape or arr_x.ndim != 2:
                        continue

                    arr_x = 0.5 * (arr_x + arr_x.T)
                    arr_y = 0.5 * (arr_y + arr_y.T)
                    mask = np.triu(np.ones(arr_x.shape, dtype=bool), k=1)
                    vec_x, vec_y = arr_x[mask], arr_y[mask]
                    valid = np.isfinite(vec_x) & np.isfinite(vec_y)
                    if valid.sum() < 2:
                        continue
                    vec_x, vec_y = vec_x[valid], vec_y[valid]

                    if metric == "spearman":
                        corr = float(spearmanr(vec_x, vec_y).correlation)
                    else:
                        corr = float(np.corrcoef(vec_x, vec_y)[0, 1])

                    if not np.isfinite(corr):
                        continue
                    rows.append({"x": x, "correlation": corr, "spi_pair": label})

    if not rows:
        raise ValueError(
            f"No matching datasets found under {root} with x_key='{x_key}'"
        )

    df = pd.DataFrame(rows)

    if ax is None:
        ax = plt.gca()

    sns.lineplot(
        data=df,
        x="x",
        y="correlation",
        hue="spi_pair",
        ax=ax,
        errorbar=("ci", 95),
        err_style="band",
    )
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    return df


# ---------------------------------------------------------------------------
# Channel-type marker constants for plot_spi_space_individual_embed_properties
# ---------------------------------------------------------------------------
_CHANNEL_TAGS = ["linear", "monotonic", "non-monotonic"]
_CHANNEL_ABBREVIATIONS = {
    "linear": "L",
    "monotonic": "M",
    "non-monotonic": "NM",
}

_PAIR_MARKERS: dict[tuple[str, str], str] = {
    ("linear",       "linear"):        "o",
    ("linear",       "monotonic"):     "^",
    ("linear",       "non-monotonic"): "s",
    ("monotonic",    "monotonic"):     "*",
    ("monotonic",    "non-monotonic"): "P",
    ("non-monotonic","non-monotonic"): "X",
}

_MIXED_PAIR_COLORS: dict[tuple[str, str], str] = {
    ("linear",       "monotonic"):     "#0072B2",  # blue
    ("linear",       "non-monotonic"): "#D55E00",  # vermillion
    ("monotonic",    "non-monotonic"): "#009E73",  # bluish green
}

_SAME_TYPE_PAIR_COLOR = "#8a8a8a"


def _tag_channel(a: float, thresh_linear: float, thresh_monotonic: float) -> str:
    if a < thresh_linear:
        return "linear"
    if a < thresh_monotonic:
        return "monotonic"
    return "non-monotonic"


def _pair_key(tag_i: str, tag_j: str) -> tuple[str, str]:
    """Canonical (sorted) key for an unordered channel-type pair."""
    order = {t: i for i, t in enumerate(_CHANNEL_TAGS)}
    return (tag_i, tag_j) if order[tag_i] <= order[tag_j] else (tag_j, tag_i)


def _is_same_type_pair(pair_key: tuple[str, str]) -> bool:
    return pair_key[0] == pair_key[1]


def _pair_label(pair_key: tuple[str, str], *, count: int | None = None) -> str:
    prefix = "same" if _is_same_type_pair(pair_key) else "mixed"
    pair = "-".join(_CHANNEL_ABBREVIATIONS[tag] for tag in pair_key)
    label = f"{prefix} {pair}"
    return f"{label} (n={count})" if count is not None else label


def plot_spi_space_individual_embed_properties(
    dataset_path: str,
    spis: list[str],
    ax,
    *,
    thresh_linear: float = np.pi / 8,
    thresh_monotonic: float = np.pi / 2,
    color_mixed_pairs: bool = True,
    hollow_same_type_pairs: bool = True,
    same_type_color: str = _SAME_TYPE_PAIR_COLOR,
    mixed_pair_colors: Mapping[tuple[str, str], str] | None = None,
    label_pair_counts: bool = False,
    s: float = 25.0,
    alpha: float = 0.4,
    main_spines: bool = True,
    metric: str = "spearman",
    trend: bool = True,
    trend_color: str = "#333333",
    identity: bool = False,
    show_dataset: bool = False,
) -> None:
    """
    Like plot_spi_space_individual_embed (marginals=False), but markers encode
    the channel-type pair for each SPI value.

    Each upper-triangle entry (i,j) represents the SPI between channels i and j.
    Each channel is tagged {linear, monotonic, non-monotonic} from its a_value
    in meta.json (generator.a_values), using thresholds:
        a < thresh_linear     → linear
        a < thresh_monotonic  → monotonic
        else                  → non-monotonic

    Marker shapes per unordered pair:
        L–L   circle  'o'
        L–M   triangle '^'
        L–NM  square  's'
        M–M   star    '*'
        M–NM  filled+ 'P'
        NM–NM filled× 'X'

    The default visual hierarchy treats same-type channel pairs as context
    (neutral hollow markers) and mixed-type pairs as the comparison signal
    (filled colorblind-safe markers).

    Parameters
    ----------
    thresh_linear    : a threshold below which a channel is "linear" (default π/8)
    thresh_monotonic : a threshold below which a channel is "monotonic" (default π/2)
    color_mixed_pairs     : if True, mixed channel-type pairs get distinct colors
                            while same-type pairs are neutral grey; if False, all
                            pairs are grey
    hollow_same_type_pairs: if True, same-type channel pairs are drawn hollow
    same_type_color        : neutral color for same-type pairs
    mixed_pair_colors      : optional overrides for mixed-pair colors, keyed by
                             canonical channel-type tuples, e.g.
                             ("linear", "non-monotonic")
    label_pair_counts      : if True, append each pair-type count to legend labels
    trend                  : if True, draw a fitted regression line
    trend_color            : color for the trend line and correlation label frame
    identity               : if True, draw a y=x reference line. Only use this
                             when both axes are directly comparable and on the
                             same numeric scale.
    """
    if len(spis) != 2:
        raise ValueError("Expects exactly two SPI names.")

    spi_x, spi_y = spis
    dataset_dir = Path(dataset_path)

    archive = dataset_dir / "spi_mpis.npz"
    if not archive.exists():
        raise FileNotFoundError(f"Missing archive: {archive}")
    meta_path = dataset_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json: {meta_path}")

    meta = load_json(meta_path)
    a_values = meta.get("generator", {}).get("a_values")
    if a_values is None:
        raise KeyError(
            f"'a_values' not found in meta.json generator block for {dataset_dir}. "
            "Dataset must be generated (not cache-loaded) with sin_mts_smooth."
        )
    a_values = np.asarray(a_values, dtype=float)
    M = len(a_values)

    spi_meta = meta.get("pyspi", {}).get("spis", [])
    directed_map = {
        entry.get("name"): bool(entry.get("directed", False))
        for entry in spi_meta
        if isinstance(entry, dict) and entry.get("name")
    }
    for name in (spi_x, spi_y):
        if name not in directed_map:
            raise KeyError(f"Missing directed metadata for '{name}' in {meta_path}")

    with np.load(archive) as npz:
        if spi_x not in npz or spi_y not in npz:
            raise KeyError(f"Missing SPIs in archive: {archive}")
        arr_x = np.asarray(npz[spi_x], float)
        arr_y = np.asarray(npz[spi_y], float)

    arr_x = 0.5 * (arr_x + arr_x.T)
    arr_y = 0.5 * (arr_y + arr_y.T)

    tags = [_tag_channel(a, thresh_linear, thresh_monotonic) for a in a_values]
    upper_i, upper_j = np.triu_indices(M, k=1)

    x_vals = arr_x[upper_i, upper_j]
    y_vals = arr_y[upper_i, upper_j]
    valid = np.isfinite(x_vals) & np.isfinite(y_vals)
    upper_i, upper_j = upper_i[valid], upper_j[valid]
    x_vals, y_vals = x_vals[valid], y_vals[valid]

    if x_vals.size == 0:
        raise ValueError(f"No finite paired SPI values found in {archive}")

    if metric == "pearson":
        corr = float(np.corrcoef(x_vals, y_vals)[0, 1])
        corr_label = f"$r={corr:.2f}$"
    elif metric == "spearman":
        corr, _ = spearmanr(x_vals, y_vals)
        corr_label = f"$\\rho={corr:.2f}$"
    else:
        raise ValueError("metric must be either 'pearson' or 'spearman'")
    title_slug = "/".join(dataset_dir.parts[-2:]) if show_dataset else dataset_dir.parts[-2]

    apply_plot_style()

    pair_keys = [_pair_key(tags[i], tags[j]) for i, j in zip(upper_i, upper_j)]
    style_order = list(_PAIR_MARKERS.keys())
    mixed_colors = dict(_MIXED_PAIR_COLORS)
    if mixed_pair_colors is not None:
        mixed_colors.update(mixed_pair_colors)

    for key in style_order:
        marker = _PAIR_MARKERS[key]
        mask = np.array([pk == key for pk in pair_keys])
        if not mask.any():
            continue
        is_same_type = _is_same_type_pair(key)
        c = (
            same_type_color
            if is_same_type or not color_mixed_pairs
            else mixed_colors[key]
        )
        marker_style = (
            {"facecolors": "none", "edgecolors": c, "linewidths": 0.8}
            if is_same_type and hollow_same_type_pairs
            else {"facecolors": c, "edgecolors": c, "linewidths": 0.3}
        )
        ax.scatter(
            x_vals[mask], y_vals[mask],
            marker=marker, s=s, alpha=alpha,
            label=_pair_label(
                key,
                count=int(mask.sum()) if label_pair_counts else None,
            ),
            zorder=1 if is_same_type else 2,
            **marker_style,
        )

    if trend:
        sns.regplot(
            x=x_vals,
            y=y_vals,
            ax=ax,
            scatter=False,
            color=trend_color,
            line_kws={"lw": 1.0, "alpha": 0.8, "zorder": 3},
            ci=None,
        )
    if identity:
        lim = [min(x_vals.min(), y_vals.min()), max(x_vals.max(), y_vals.max())]
        ax.plot(lim, lim, color="gray", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xlabel(spi_x)
    ax.set_ylabel(spi_y)
    ax.set_title(title_slug)
    ax.set_box_aspect(1)
    at = AnchoredText(
        corr_label,
        loc="upper left",
        frameon=True,
        prop=dict(size=15),
        pad=0.3,
        borderpad=0.3,
    )
    at.patch.set(facecolor="none", edgecolor=trend_color, lw=0.7, alpha=0.7)
    ax.add_artist(at)
    if not main_spines:
        ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelbottom=True, labelleft=True)


def plot_spi_space_individual_embed_lags(
    dataset_path: str,
    spis: list[str],
    ax,
    *,
    lag_feature: str = "diff",
    cmap: str = "viridis",
    s: float = 25.0,
    alpha: float = 0.6,
    main_spines: bool = True,
    metric: str = "spearman",
    identity: bool = True,
    show_dataset: bool = False,
) -> None:
    """
    Like plot_spi_space_individual_embed (marginals=False), but markers are
    colored by a per-pair lag feature derived from generator.lags in meta.json.

    Parameters
    ----------
    lag_feature : "diff"  → |τ_i − τ_j|  (how misaligned the pair is)
                  "max"   → max(τ_i, τ_j) (absolute lag regime of the pair)
                  "mean"  → (τ_i + τ_j) / 2
    identity    : if True (default), draw a y=x reference line
    """
    if len(spis) != 2:
        raise ValueError("Expects exactly two SPI names.")
    if lag_feature not in ("diff", "max", "mean"):
        raise ValueError("lag_feature must be 'diff', 'max', or 'mean'.")

    spi_x, spi_y = spis
    dataset_dir = Path(dataset_path)

    archive = dataset_dir / "spi_mpis.npz"
    if not archive.exists():
        raise FileNotFoundError(f"Missing archive: {archive}")
    meta_path = dataset_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json: {meta_path}")

    meta = load_json(meta_path)
    lags = meta.get("generator", {}).get("lags")
    if lags is None:
        raise KeyError(
            f"'lags' not found in meta.json generator block for {dataset_dir}. "
            "Dataset must be generated with lagged_mts."
        )
    lags = np.asarray(lags, dtype=float)
    M = len(lags)

    spi_meta = meta.get("pyspi", {}).get("spis", [])
    directed_map = {
        entry.get("name"): bool(entry.get("directed", False))
        for entry in spi_meta
        if isinstance(entry, dict) and entry.get("name")
    }
    for name in (spi_x, spi_y):
        if name not in directed_map:
            raise KeyError(f"Missing directed metadata for '{name}' in {meta_path}")

    with np.load(archive) as npz:
        if spi_x not in npz or spi_y not in npz:
            raise KeyError(f"Missing SPIs in archive: {archive}")
        arr_x = np.asarray(npz[spi_x], float)
        arr_y = np.asarray(npz[spi_y], float)

    arr_x = 0.5 * (arr_x + arr_x.T)
    arr_y = 0.5 * (arr_y + arr_y.T)

    upper_i, upper_j = np.triu_indices(M, k=1)
    x_vals = arr_x[upper_i, upper_j]
    y_vals = arr_y[upper_i, upper_j]
    valid = np.isfinite(x_vals) & np.isfinite(y_vals)
    upper_i, upper_j = upper_i[valid], upper_j[valid]
    x_vals, y_vals = x_vals[valid], y_vals[valid]

    if lag_feature == "diff":
        lag_vals = np.abs(lags[upper_i] - lags[upper_j])
        cbar_label = r"$|\tau_i - \tau_j|$"
    elif lag_feature == "max":
        lag_vals = np.maximum(lags[upper_i], lags[upper_j])
        cbar_label = r"$\max(\tau_i, \tau_j)$"
    else:
        lag_vals = 0.5 * (lags[upper_i] + lags[upper_j])
        cbar_label = r"$(\tau_i + \tau_j)/2$"

    if metric == "pearson":
        corr = float(np.corrcoef(x_vals, y_vals)[0, 1])
        corr_label = f"$r={corr:.2f}$"
    else:
        corr, _ = spearmanr(x_vals, y_vals)
        corr_label = f"$\\rho={corr:.2f}$"
    title_slug = "/".join(dataset_dir.parts[-2:]) if show_dataset else dataset_dir.parts[-2]

    apply_plot_style()

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    sc = ax.scatter(
        x_vals, y_vals,
        c=lag_vals, cmap=cmap,
        vmin=0, vmax=lag_vals.max() if lag_vals.max() > 0 else 1,
        s=s, alpha=alpha, linewidths=0,
    )
    # inset_axes with bbox_transform=ax.transAxes anchors the colorbar to the
    # square axes box (post set_box_aspect), not the subplot cell — so its
    # height always matches the scatter plot exactly.
    cax = inset_axes(
        ax, width="5%", height="100%", loc="lower left",
        bbox_to_anchor=(1.05, 0.0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    plt.colorbar(sc, cax=cax, label=cbar_label)

    sns.regplot(x=x_vals, y=y_vals, ax=ax, scatter=False, color="red", line_kws={"lw": "1.0"}, ci=None)
    if identity:
        lim = [min(x_vals.min(), y_vals.min()), max(x_vals.max(), y_vals.max())]
        ax.plot(lim, lim, color="gray", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xlabel(spi_x)
    ax.set_ylabel(spi_y)
    ax.set_title(title_slug)
    ax.set_box_aspect(1)
    at = AnchoredText(corr_label, loc="upper left", frameon=True, prop=dict(size=15), pad=0.3, borderpad=0.3)
    at.patch.set(facecolor="none", edgecolor="red", lw=0.7, alpha=0.7)
    ax.add_artist(at)
    if not main_spines:
        ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelbottom=True, labelleft=True)


def _raincloud_half_violin(ax, data, pos, color, half_w=0.09, side="right"):
    """One-sided violin at x=pos. Skipped when degenerate (n<2 or ~0 variance)."""
    data = np.asarray(data, float)
    data = data[np.isfinite(data)]
    if len(data) < 2 or np.std(data) < 1e-9:
        return
    for b in ax.violinplot([data], positions=[pos], widths=half_w * 2,
                           showextrema=False)["bodies"]:
        v = b.get_paths()[0].vertices
        if side == "right":
            v[:, 0] = np.clip(v[:, 0], pos, np.inf)
        else:
            v[:, 0] = np.clip(v[:, 0], -np.inf, pos)
        b.set_facecolor(color)
        b.set_alpha(0.35)
        b.set_edgecolor("none")


def plot_raincloud(
    groups,
    labels=None,
    colors=None,
    positions=None,
    ax=None,
    point_hue=None,
    point_cmap="viridis",
    point_sizes=None,
    half_width=0.09,
    box_width=0.04,
    point_size=12,
    jitter=0.015,
    rng=0,
):
    """Raincloud (half-violin + box + jittered raw-point strip), one column per group.

    Returns ``ax`` (creates a figure only if ``ax is None``) so it embeds as a subplot.

    groups:      list of 1D arrays, one distribution per category.
    labels:      x tick labels (len == n groups).
    colors:      per-group colour; defaults to tab10.
    positions:   x positions; defaults to 0..n-1.
    point_hue:   optional list (parallel to groups) of per-point scalar arrays
                 (e.g. M) -> strip points coloured by ``point_cmap`` with a colourbar.
                 When None, points take the group colour.
    point_sizes: optional list (parallel to groups) of per-point sizes (e.g. M-scaled).
    n=10 caveat: the violin/KDE is illustrative at small n; the strip + box are the
                 honest summary (matches the r_rho_mi staggered raincloud).
    """
    n = len(groups)
    if ax is None:
        _, ax = plt.subplots(figsize=(1.6 * n + 2, 4.8), dpi=DEFAULT_DPI)
    if positions is None:
        positions = list(range(n))
    if colors is None:
        colors = list(sns.color_palette("tab10", n))
    rstate = np.random.default_rng(rng)

    norm = sc = None
    if point_hue is not None:
        allh = np.concatenate([np.asarray(h, float) for h in point_hue])
        norm = Normalize(vmin=np.nanmin(allh), vmax=np.nanmax(allh))

    for i, data in enumerate(groups):
        data = np.asarray(data, float)
        finite = np.isfinite(data)
        data = data[finite]
        pos, col = positions[i], colors[i]
        _raincloud_half_violin(ax, data, pos, col, half_w=half_width, side="right")
        if len(data):
            ax.boxplot([data], positions=[pos], widths=box_width, showfliers=False,
                       patch_artist=True,
                       boxprops=dict(facecolor="white", edgecolor=col, lw=1),
                       medianprops=dict(color=col, lw=1.4),
                       whiskerprops=dict(color=col), capprops=dict(color=col))
            jit = pos - box_width - 0.005 - np.abs(rstate.normal(0, jitter, len(data)))
            s = point_size
            if point_sizes is not None:
                s = np.asarray(point_sizes[i], float)[finite]
            if point_hue is not None:
                hv = np.asarray(point_hue[i], float)[finite]
                sc = ax.scatter(jit, data, s=s, c=hv, cmap=point_cmap, norm=norm,
                                alpha=0.8, linewidths=0, zorder=3)
            else:
                ax.scatter(jit, data, s=s, color=col, alpha=0.75, linewidths=0, zorder=3)
        if labels is None:
            ax.scatter([], [], color=col, label=str(i))

    ax.set_xticks(positions)
    if labels is not None:
        ax.set_xticklabels(labels)
    ax.set_xlim(min(positions) - 0.6, max(positions) + 0.4)
    ax.grid(True, axis="y", alpha=0.3)
    if point_hue is not None and sc is not None:
        ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="point hue")
    return ax
