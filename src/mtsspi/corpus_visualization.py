"""Focused plots for corpus embeddings and multivariate time series."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import seaborn as sns


def process_by_time(values: np.ndarray) -> np.ndarray:
    """Canonicalize the repository's raw arrays to process x time."""

    data = np.asarray(values, dtype=np.float64).squeeze()
    if data.ndim != 2:
        raise ValueError("timeseries must be a two-dimensional array")
    return data.T if data.shape[0] > data.shape[1] else data


def scale_timeseries(values: np.ndarray, method: str) -> tuple[np.ndarray, float]:
    """Scale an MTS and return a symmetric display limit."""

    data = process_by_time(values)
    if method == "legacy":
        # Exact orientation used by old/plot_data.py: robust scaling was
        # inadvertently performed across processes at each time point.
        median = np.median(data, axis=0, keepdims=True)
        q25, q75 = np.quantile(data, (0.25, 0.75), axis=0, keepdims=True)
        scale = q75 - q25
        scale = np.where(scale > 0, scale, 1.0)
        transformed = (data - median) / scale
        limits = np.percentile(transformed, (5, 95))
        limit = float(max(limits))
    elif method == "zscore":
        location = np.mean(data, axis=1, keepdims=True)
        scale = np.std(data, axis=1, keepdims=True)
        transformed = (data - location) / np.where(scale > 0, scale, 1.0)
        limit = float(np.quantile(np.abs(transformed), 0.99))
    elif method == "robust":
        location = np.median(data, axis=1, keepdims=True)
        q25, q75 = np.quantile(data, (0.25, 0.75), axis=1, keepdims=True)
        scale = (q75 - q25) / 1.3489795003921634
        transformed = (data - location) / np.where(scale > 0, scale, 1.0)
        limit = float(np.quantile(np.abs(transformed), 0.99))
    else:
        raise ValueError(f"unknown heatmap scaling {method!r}")
    return transformed, max(abs(limit), np.finfo(float).eps)


def plot_mts_heatmap(
    values: np.ndarray,
    *,
    method: str = "zscore",
    ax: plt.Axes | None = None,
    title: str | None = None,
    colorbar: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot an MTS with a zero-centred diverging colour scale."""

    transformed, limit = scale_timeseries(values, method)
    if ax is None:
        width = min(10.0, max(4.0, transformed.shape[1] / 220.0))
        height = min(4.0, max(1.7, transformed.shape[0] / 6.0))
        fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    else:
        fig = ax.figure
    image = ax.imshow(
        transformed,
        aspect="auto",
        interpolation="nearest",
        cmap=sns.color_palette("icefire", as_cmap=True),
        vmin=-limit,
        vmax=limit,
        origin="lower",
        rasterized=True,
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Process")
    if title:
        ax.set_title(title)
    if colorbar:
        fig.colorbar(image, ax=ax, shrink=0.75, label="Scaled value")
    return fig, ax


def _confidence_ellipse(
    coordinates: np.ndarray,
    ax: plt.Axes,
    *,
    level: float = 0.80,
    alpha: float = 0.10,
) -> None:
    if len(coordinates) < 3:
        return
    covariance = np.cov(coordinates, rowvar=False)
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        return
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    # Two-dimensional chi-square quantile: F(x)=1-exp(-x/2).
    radius = np.sqrt(-2.0 * np.log(1.0 - level))
    width, height = 2 * radius * np.sqrt(eigenvalues)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    ax.add_patch(
        Ellipse(
            np.mean(coordinates, axis=0),
            width,
            height,
            angle=angle,
            facecolor="0.45",
            edgecolor="0.25",
            linewidth=0.6,
            alpha=alpha,
            zorder=0,
        )
    )


def plot_embedding(
    coordinates: np.ndarray,
    *,
    clusters: Sequence[int] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str = "Dimension 1",
    ylabel: str = "Dimension 2",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot every dataset neutrally and optional algorithmic cluster envelopes."""

    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape (n, 2)")
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.2, 4.0), constrained_layout=True)
    else:
        fig = ax.figure
    if clusters is not None:
        labels = np.asarray(clusters, dtype=int)
        if labels.shape != (len(coordinates),):
            raise ValueError("cluster labels do not match coordinates")
        for label in np.unique(labels):
            if label >= 0:
                _confidence_ellipse(coordinates[labels == label], ax)
    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=8,
        c="0.16",
        alpha=0.72,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    return fig, ax


def interactive_embedding_browser(
    coordinates: np.ndarray,
    names: Sequence[object],
    data_provider: Callable[[str], np.ndarray],
    *,
    title: str = "Dataset browser",
    scaling: str = "zscore",
):
    """Return a live Jupyter widget: click a point to display its raw MTS.

    This intentionally imports the optional notebook dependencies only when the
    browser is requested. The callback requires a live kernel.
    """

    try:
        import anywidget  # noqa: F401
        import ipywidgets as widgets
        import plotly.graph_objects as go
        from IPython.display import clear_output, display
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "install the 'interactive' extra to use the live browser"
        ) from error

    coordinates = np.asarray(coordinates, dtype=float)
    names = np.asarray(names).astype(str)
    figure = go.FigureWidget(
        go.Scattergl(
            x=coordinates[:, 0],
            y=coordinates[:, 1],
            mode="markers",
            customdata=names,
            hovertemplate="%{customdata}<extra></extra>",
            marker={"color": "#333333", "size": 6, "opacity": 0.75},
        )
    )
    figure.update_layout(
        title=title,
        template="simple_white",
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        width=650,
        height=520,
        showlegend=False,
    )
    output = widgets.Output(layout={"width": "650px"})

    def show_index(index: int) -> None:
        with output:
            clear_output(wait=True)
            fig, _ = plot_mts_heatmap(
                data_provider(names[index]),
                method=scaling,
                title=names[index],
                colorbar=True,
            )
            display(fig)
            plt.close(fig)

    def on_click(_trace, points, _state) -> None:
        if points.point_inds:
            show_index(int(points.point_inds[0]))

    figure.data[0].on_click(on_click)
    show_index(0)
    return widgets.VBox((figure, output))
