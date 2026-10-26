"""Shared look of the case-study composites (2026-10-24).

Vocabulary from the method schematic (`docs/artwork/method_schematic/method_schematic.py`):
one pastel hue per statistic, its soft diverging map for z, ink and muted greys, and the
carpet-plus-trace MTS glyph. Data panels follow the inference composite
(`research/inference/benchmarks/composite_figures.py`): boxed, ticked axes with real values.

Colour roles never overlap: statistic hues mark statistics only (matrices, rugs, bars,
axis-label swatches); pair relations use saturated Okabe-Ito colours; z values use the
diverging map; ink and greys carry everything else.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb

INK, MUTED, HAIR, TRACE = "#2E3440", "#7B8190", "#DEDEDB", "#3b3b3b"
SAGE, LAVENDER, AMBER, ROSE, SKY = "#74B49B", "#9D8FCB", "#E0B060", "#D98FA5", "#86B3D9"
DIV = LinearSegmentedColormap.from_list("soft_div", ["#5D83C2", "#B9CBE6", "#F7F7F5", "#EDB9A6", "#CF7458"])
Z_NORM = Normalize(-1, 1)
DPI = 600


def shade(hue, f: float) -> tuple:
    return tuple(f * c for c in to_rgb(hue))


def tint(hue, f: float) -> tuple:
    return tuple(1 - f * (1 - c) for c in to_rgb(hue))


def seq(hue) -> LinearSegmentedColormap:
    """White -> hue -> deeper hue, for one statistic's matrix (white is zero dependence)."""
    return LinearSegmentedColormap.from_list(str(hue), ["#FFFFFF", tint(hue, 0.55), hue, shade(hue, 0.62)])


def style(live_text: bool = False):
    """LaTeX Computer Modern by default; ``live_text`` keeps SVG text editable (mathtext, no TeX)."""
    tex = not live_text and bool(shutil.which("latex") and shutil.which("dvipng"))
    mpl.rcParams.update({
        "text.usetex": tex, "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "font.family": "serif", "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm", "svg.fonttype": "none", "pdf.fonttype": 42,
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 2,
        "xtick.direction": "out", "ytick.direction": "out", "axes.edgecolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK, "text.color": INK,
        "axes.spines.top": True, "axes.spines.right": True, "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "figure.dpi": 180, "savefig.dpi": DPI,
    })


def axes_in(fig, x, y, w, h, **kw):
    """Axes placed in figure inches from the top-left corner."""
    W, H = fig.get_size_inches()
    return fig.add_axes((x / W, 1 - (y + h) / H, w / W, h / H), **kw)


def panel_title(fig, x, y, letter, head, sub=None):
    """Bold lowercase letter and bold title at (x, y) inches from the top-left; optional muted subtitle."""
    W, H = fig.get_size_inches()
    bold = (lambda s: rf"\textbf{{{s}}}") if mpl.rcParams["text.usetex"] else (lambda s: s)
    kw = dict(transform=fig.transFigure, va="baseline", fontsize=9,
              fontweight="normal" if mpl.rcParams["text.usetex"] else "bold")
    fig.text(x / W, 1 - y / H, bold(letter), **kw)
    fig.text((x + 0.15) / W, 1 - y / H, bold(head), **kw)
    if sub:
        fig.text((x + 0.15) / W, 1 - (y + 0.15) / H, sub, transform=fig.transFigure, va="baseline",
                 fontsize=8, color=MUTED)


def z_color(z: float) -> tuple:
    return DIV(Z_NORM(z))


def z_label(ax, z: float, x=0.05, y=0.93, ha="left", va="top", prefix="z", fontsize=8):
    """z in its own diverging colour: tinted fill, stronger edge from the same side, ink text."""
    fill = tint(z_color(z), 0.85)
    edge = shade(z_color(np.sign(z) * max(abs(z), 0.6)), 0.9) if abs(z) >= 0.05 else MUTED
    s = rf"${prefix} = {z:.2f}$".replace("-", "{-}") if z < 0 else rf"${prefix} = {z:.2f}$"
    return ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize, color=INK, zorder=6,
                   bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.12", fc=fill, ec=edge, lw=0.6))


def rugs(ax, x, y, cx, cy, frac=0.035, lw=0.35, alpha=0.5):
    """Edge-profile rugs inside the box: x statistic along the bottom, y statistic along the left."""
    tb = ax.get_xaxis_transform()
    tl = ax.get_yaxis_transform()
    ax.vlines(x, 0, frac, transform=tb, color=cx, lw=lw, alpha=alpha, zorder=1)
    ax.hlines(y, 0, frac, transform=tl, color=cy, lw=lw, alpha=alpha, zorder=1)


def carpet(ax, x: np.ndarray, cmap=None):
    """Channels x time carpet: robust-scaled per channel, icefire, limit at the 99th percentile of |x|."""
    from mtsspi.corpus_visualization import scale_timeseries
    from seaborn import color_palette
    scaled, limit = scale_timeseries(x, "robust")
    ax.imshow(scaled, aspect="auto", cmap=cmap or color_palette("icefire", as_cmap=True), vmin=-limit,
              vmax=limit, interpolation="nearest", rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5); s.set_color(INK)
    return limit


def bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def save(fig, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=DPI if ext != "png" else 300, bbox_inches="tight",
                    pad_inches=0.03)
    return out_dir / f"{stem}.png"
