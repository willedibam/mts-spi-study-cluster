from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

_STYLE_INITIALISED = False


def apply_plot_style() -> None:
    global _STYLE_INITIALISED
    if _STYLE_INITIALISED:
        return
    plt.close("all")
    plt.rcdefaults()
    sns.reset_orig()
    sns.set_theme(
        style="white",
        rc={
            "text.usetex": True,
            "font.size": 12,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.bottom": True,     # Turn the tick ON
            "ytick.left": True,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        },
    )
    _STYLE_INITIALISED = True


def save_figure(
    fig: Figure,
    path: str | Path,
    *,
    dpi: int = 300,
    **savefig_kwargs,
) -> None:
    """
    Persist a Matplotlib figure as both PNG (primary path) and SVG.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=dpi, **savefig_kwargs)
    svg_path = dest.with_suffix(".svg")
    if svg_path == dest:
        png_path = dest.with_suffix(".png")
        fig.savefig(png_path, dpi=dpi, **savefig_kwargs)
        return
    fig.savefig(svg_path, dpi=dpi, **savefig_kwargs)
