from __future__ import annotations

from pathlib import Path

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import Normalize
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
    *,
    split_directed: bool = False,
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
        title_slug = "/".join(dataset_dir.parts[-2:])
        g.fig.suptitle(f"{title_slug}\n$\\rho = {rho:.2f}$", y=1.02)

        plotted = True

    if plotted:
        plt.show()

def plot_spi_space_recursive(
    dataset_path: str,
    spis: list[str],
    *,
    split_directed: bool = False,
    formats: list[str] | None = None,
    dpi: int = 300,
    show: bool = False,
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
            title_slug = "/".join(dataset_dir.parts[-2:])
            g.fig.suptitle(f"{title_slug}\n$\\rho = {rho:.2f}$", y=1.02)

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

    fig, ax = plt.subplots(figsize=(8, 8), dpi=DEFAULT_DPI)
    
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

    fig, ax = plt.subplots(figsize=(8, 8), dpi=DEFAULT_DPI)
    
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

    fig, ax = plt.subplots(figsize=(8, 8), dpi=DEFAULT_DPI)
    
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
