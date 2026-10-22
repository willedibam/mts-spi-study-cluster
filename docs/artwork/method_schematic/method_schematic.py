"""Method schematic: construction of the SPI–SPI representation z (paper Fig. 1), 2026-10-22.

The panels run as a snake. Row 1, left to right: (a) a recording as two extracted traces,
a channel x time carpet and a coupling graph of the same channels; (b) the graph fanned out
into K planes, one weighted graph per statistic, and the K statistics as stacked M x M
matrices, with one entry's computation for i -> j and j -> i attached to the front matrix.
Row 2, right to left: (c) the off-diagonal entries of A^(a) and A^(b) drawn out into one
scatter, whose Pearson correlation is z_ab; (d) z_ab's place in the K x K agreement matrix,
whose strict upper triangle is drawn out into z.

Every value drawn is computed here from a small synthetic recording (directed VAR(1),
M = 6, T = 1000) with K = 5 numpy statistics; nothing is case-study data and no pyspi
runs. z is checked against the library contract (unified_ordered_v3). All panels are drawn
in figure inches on full-figure layers (one SVG group per layer), so connectors between
panels are plain lines.

    .venv/bin/python docs/artwork/method_schematic/method_schematic.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
from seaborn import color_palette

from mtsspi.corpus_visualization import scale_timeseries
from mtsspi.spi_spi_contract import UNIFIED_CONTRACT_VERSION, build_unified_feature_values

HERE = Path(__file__).resolve().parent
STAMP = "261022"
# Statistic order: a and b first, so z_ab is R[0, 1] and the first entry of z.
SPIS = {
    "lag1": "lag-1 cross-correlation corr(x_i(t), x_j(t+1)) (directed)",
    "granger": "linear Granger log-variance ratio, order 2 (directed)",
    "pearson": "Pearson correlation",
    "spearman": "Spearman correlation",
    "euclid": "Euclidean distance (z-scored series)",
}
SIMILARITY_SIGN = {"euclid": -1}  # distances are flipped only for drawing graph edges
# Channel labels are arbitrary (z is invariant to relabelling); this order puts the strongly
# asymmetric pair at the matrix corner, so i -> j is in the last column and j -> i in the last row.
CHANNEL_ORDER = [0, 1, 2, 4, 3, 5]
PAIR = (4, 5)  # highlighted channel pair (i, j), 0-based, after CHANNEL_ORDER
WINDOW = (200, 300)  # samples shown in the carpet and traces
# Irregular node layout in the unit square; i and j (nodes 4, 5) sit on top, under their traces.
LAYOUT = np.array([[0.04, 0.50], [0.30, 0.06], [0.97, 0.44], [0.68, 0.12], [0.27, 0.84], [0.77, 0.93]])

# Soft palette: a pastel hue for statistics a and b, quiet greys for the rest of the K, a
# pastel diverging map for R and z, a dark ink for the channel pair (i, j), and white cells
# with a hairline for entries that are never used (all diagonals).
HUES = ["#74B49B", "#9D8FCB"]
INK, MUTED, HAIR, CARD, NODE = "#2E3440", "#7B8190", "#DEDEDB", "#F3F3F1", "#AEB4BF"
DIV = LinearSegmentedColormap.from_list("soft_div", ["#5D83C2", "#B9CBE6", "#F7F7F5", "#EDB9A6", "#CF7458"])
MTS = color_palette("icefire", as_cmap=True)


def shade(hue: str, f: float) -> tuple:
    return tuple(f * c for c in to_rgb(hue))


def tint(hue: str, f: float) -> tuple:
    return tuple(1 - f * (1 - c) for c in to_rgb(hue))


def seq(hue: str) -> LinearSegmentedColormap:
    """Light tint -> hue -> deeper hue; the floor stays tinted so only unused cells are white."""
    return LinearSegmentedColormap.from_list(hue, [tint(hue, 0.22), hue, shade(hue, 0.72)])


def style():
    """docs/figure-style.md, typeset by LaTeX (Computer Modern)."""
    mpl.rcParams.update({
        "text.usetex": bool(shutil.which("latex") and shutil.which("dvipng")),
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "font.family": "serif", "font.serif": ["Computer Modern Roman"], "mathtext.fontset": "cm",
        "font.size": 8, "figure.facecolor": "white",
        "pdf.fonttype": 42, "figure.dpi": 180, "savefig.dpi": 600,
    })


# --- recording and statistics ---------------------------------------------------------------

def toy_recording(M: int = 6, T: int = 1000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Directed VAR(1) with heterogeneous lagged coupling and correlated innovations.

    Returns the z-scored recording (T x M) and the generator's coupling strength per unordered
    pair: |lagged coupling| in both directions plus |innovation correlation|, scaled to [0, 1].
    """
    rng = np.random.default_rng(seed)
    A = np.diag(rng.uniform(0.3, 0.6, M))
    W = rng.uniform(0.05, 0.45, (M, M)) * (rng.random((M, M)) < 0.6)
    np.fill_diagonal(W, 0)
    A = A + W  # A[t, s] > 0: s drives t at lag 1
    radius = max(abs(np.linalg.eigvals(A)))
    if radius > 0.9:
        A *= 0.9 / radius
    L = np.eye(M) + np.tril(rng.normal(0, 0.4, (M, M)) * (rng.random((M, M)) < 0.3), -1)
    burn = 500
    e = rng.multivariate_normal(np.zeros(M), L @ L.T, size=T + burn)
    x = np.zeros((T + burn, M))
    for t in range(1, T + burn):
        x[t] = A @ x[t - 1] + e[t]
    x = x[burn:]
    C = L @ L.T
    off = np.abs(A - np.diag(np.diag(A)))
    coupling = off + off.T + np.abs(C / np.sqrt(np.outer(np.diag(C), np.diag(C))) - np.eye(M))
    np.fill_diagonal(coupling, np.nan)
    order = np.ix_(CHANNEL_ORDER, CHANNEL_ORDER)
    return ((x - x.mean(0)) / x.std(0))[:, CHANNEL_ORDER], unit(coupling[order])


def _lag1(u, v):
    """corr(u_t, v_{t+1}): directed, u leads v by one sample."""
    return np.corrcoef(u[:-1], v[1:])[0, 1]


def _granger(x, i, j, p=2):
    """log(residual variance of j from its own past / from its own and i's past): i -> j."""
    n = len(x)
    lags = lambda v: np.column_stack([v[p - k - 1:n - k - 1] for k in range(p)])
    def resvar(y, X):
        X = np.column_stack([np.ones(len(y)), X])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return np.var(y - X @ beta)
    y = x[p:, j]
    return np.log(resvar(y, lags(x[:, j])) / resvar(y, np.column_stack([lags(x[:, j]), lags(x[:, i])])))


def statistics(x: np.ndarray) -> dict[str, np.ndarray]:
    """One M x M matrix per statistic; diagonals are NaN (never used)."""
    M = x.shape[1]
    ranks = np.argsort(np.argsort(x, 0), 0).astype(float)
    mats = {
        "pearson": np.corrcoef(x.T),
        "spearman": np.corrcoef(ranks.T),
        "euclid": np.sqrt(((x[:, :, None] - x[:, None, :]) ** 2).mean(0)),
        "lag1": np.zeros((M, M)), "granger": np.zeros((M, M)),
    }
    for i in range(M):
        for j in range(M):
            if i != j:
                mats["lag1"][i, j] = _lag1(x[:, i], x[:, j])
                mats["granger"][i, j] = _granger(x, i, j)
    for m in mats.values():
        np.fill_diagonal(m, np.nan)
    return {k: mats[k] for k in SPIS}


def construction(mats: dict[str, np.ndarray]) -> dict:
    M = next(iter(mats.values())).shape[0]
    mask = ~np.eye(M, dtype=bool)
    edges = [tuple(e) for e in np.argwhere(mask)]  # row-major, as the library contract
    P = np.vstack([m[mask] for m in mats.values()])
    R = np.corrcoef(P)
    z = R[np.triu_indices(len(P), 1)]
    z_lib, valid, _ = build_unified_feature_values(
        {k: np.nan_to_num(m) for k, m in mats.items()}, list(mats))
    assert valid.all() and np.allclose(z, z_lib, atol=1e-6), "z disagrees with the library contract"
    return {"M": M, "edges": edges, "P": P, "R": R, "z": z}


def unit(v: np.ndarray) -> np.ndarray:
    """Min-max to [0, 1] for colour only; z is blind to it."""
    return (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v))


def robust_scaled(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Channels x time, robust-scaled per channel; the order-parameter composite's heatmap glyph."""
    return scale_timeseries(x.T, "robust")


# --- drawing primitives (all coordinates in figure inches) ---------------------------------------

def layer(fig, gid: str, z: float = 0):
    """A transparent full-figure axes whose data units are inches; one SVG group per layer."""
    W, H = fig.get_size_inches()
    ax = fig.add_axes((0, 0, 1, 1), zorder=z)
    ax.set_gid(gid); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off"); ax.patch.set_alpha(0)
    return ax


def text(ax, x, y, s, size=8, color=INK, **kw):
    return ax.text(x, y, s, fontsize=size, color=color, **kw)


def title(ax, x, y, letter, head, *subs):
    text(ax, x, y, rf"\textbf{{{letter}}}", 9, va="baseline")
    text(ax, x + 0.14, y, head, 9, va="baseline")
    for k, sub in enumerate(subs, 1):
        text(ax, x + 0.14, y - 0.155 * k, sub, 8, MUTED, va="baseline")


def arrow(ax, p0, p1, rad=0.0, color=MUTED, lw=0.6, ls="-", head=True, alpha=1.0, z=5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>" if head else "-", mutation_scale=6, lw=lw,
                                 color=color, ls=ls, alpha=alpha, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def rounded(ax, x, y, w, h, fc, r=0.03, z=1, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, zorder=z,
                                **{"ec": "none", **kw}))


def cell(ax, x, y, s, fc, z=1, **kw):
    """A soft cell of pitch s with its lower-left corner at (x, y)."""
    g = 0.14 * s
    rounded(ax, x + g / 2, y + g / 2, s - g, s - g, fc, r=0.16 * s, z=z, **kw)


def unused(ax, x, y, s, z=1):
    cell(ax, x, y, s, "white", z=z, ec=HAIR, lw=0.4)


def ring(ax, x, y, s, ls="-", lw=0.8, z=3):
    cell(ax, x - 0.04 * s, y - 0.04 * s, 1.08 * s, "none", z=z, ec=INK, lw=lw, ls=ls)


def matrix(ax, A, hue, x0, y0, s, z=1, mark=False):
    """M x M statistic on a white card; row r at y0 + (M-1-r)s. Returns cell centres."""
    M = len(A)
    cmap, u = seq(hue), unit(A)
    pad = 0.12 * s
    rounded(ax, x0 - pad, y0 - pad, M * s + 2 * pad, M * s + 2 * pad, "white", r=0.45 * s, z=z, ec=HAIR, lw=0.5)
    centres = {}
    for r in range(M):
        for c in range(M):
            X, Y = x0 + c * s, y0 + (M - 1 - r) * s
            (unused(ax, X, Y, s, z + 0.1) if r == c else cell(ax, X, Y, s, cmap(u[r, c]), z + 0.1))
            centres[r, c] = (X + s / 2, Y + s / 2)
    if mark:
        i, j = PAIR
        ring(ax, x0 + j * s, y0 + (M - 1 - i) * s, s, z=z + 0.2)
        ring(ax, x0 + i * s, y0 + (M - 1 - j) * s, s, ls=(0, (1.5, 1)), z=z + 0.2)
    return centres


def graph(ax, pos, w, node, edge, r_node, lw_max, z=1, floor=0.12, rings=()):
    """Undirected weighted graph: edge width and opacity grow with w in [0, 1]; weak edges fade out."""
    M = len(pos)
    for a in range(M):
        for b in range(a + 1, M):
            if w[a, b] < floor:
                continue
            ax.plot(*zip(pos[a], pos[b]), color=edge, lw=0.2 + lw_max * w[a, b], alpha=0.1 + 0.8 * w[a, b],
                    solid_capstyle="round", zorder=z)
    for m in range(M):
        ax.add_patch(Circle(pos[m], r_node, fc=node, ec="white", lw=0.4, zorder=z + 0.1))
    for m in rings:
        ax.add_patch(Circle(pos[m], 1.55 * r_node, fc="none", ec=INK, lw=0.6, zorder=z + 0.2))


def edge_weights(A: np.ndarray, sign: int = 1) -> np.ndarray:
    """Symmetrised, min-max scaled strength of a statistic for drawing as an undirected graph."""
    u = unit(A) if sign > 0 else 1 - unit(A)
    w = (u + u.T) / 2
    return w ** 1.6  # emphasise strong edges so weak ones recede


def trace(ax, x0, y0, w, h, series, lw=0.6, color=INK, z=2):
    t = np.linspace(x0, x0 + w, len(series))
    ax.plot(t, y0 + h / 2 + series / (2.2 * np.abs(series).max()) * h, color=color, lw=lw, zorder=z)


def box_scatter(ax, x0, y0, w, h, xs, ys, pad=0.08):
    """Map data to a box; returns the mapping so other artists can address data coordinates."""
    lo_x, hi_x, lo_y, hi_y = xs.min(), xs.max(), ys.min(), ys.max()
    fx = lambda v: x0 + pad * w + (v - lo_x) / (hi_x - lo_x) * (1 - 2 * pad) * w
    fy = lambda v: y0 + pad * h + (v - lo_y) / (hi_y - lo_y) * (1 - 2 * pad) * h
    return fx, fy


# --- figure ------------------------------------------------------------------------------------

def figure(x, coupling, mats, c):
    M, edges, P, R, z = c["M"], c["edges"], c["P"], c["R"], c["z"]
    K = len(P)
    i, j = PAIR
    names = list(mats)
    fig = plt.figure(figsize=(7.09, 4.9))
    W, H = fig.get_size_inches()
    back = layer(fig, "lines_behind", 0)
    base = layer(fig, "panels", 1)
    front = layer(fig, "lines_front", 2)

    # (a) Recording: traces -> carpet -> coupling graph. ------------------------------------------
    title(base, 0.02, 4.75, "a", "Recording", r"$M$ channels $\times$ $T$ samples")
    scaled, limit = robust_scaled(x)
    t0, t1 = WINDOW
    cx, cy, cw, ch = 0.74, 2.9, 1.25, 0.96
    pitch, gap = ch / M, 0.022
    for m in range(M):
        y = cy + (M - 1 - m) * pitch
        clip = FancyBboxPatch((cx, y + gap), cw, pitch - 2 * gap, boxstyle="round,pad=0,rounding_size=0.025",
                              fc="none", ec="none", transform=base.transData)
        base.add_patch(clip)
        im = base.imshow(scaled[m:m + 1, t0:t1], cmap=MTS, vmin=-limit, vmax=limit, aspect="auto",
                         interpolation="nearest", extent=(cx, cx + cw, y + gap, y + pitch - gap), zorder=1)
        im.set_clip_path(clip)
        text(base, cx - 0.04, y + pitch / 2, f"${m + 1}$", 7, MUTED, ha="right", va="center")
    for m in PAIR:
        y = cy + (M - 1 - m) * pitch
        rounded(base, cx - 0.012, y + gap - 0.012, cw + 0.024, pitch - 2 * gap + 0.024, "none", r=0.03, z=2,
                ec=INK, lw=0.7)
    text(base, cx + cw / 2, cy - 0.05, "time", 8, MUTED, ha="center", va="top")
    for m, dy, rad in [(i, 0.2, 0.25), (j, -0.2, -0.25)]:
        y = cy + (M - 1 - m) * pitch + pitch / 2
        trace(base, 0.17, y + dy - 0.07, 0.36, 0.14, scaled[m, t0:t1], lw=0.5)
        text(base, 0.14, y + dy, "$x_i$" if m == i else "$x_j$", 8, ha="right", va="center")
        arrow(back, (0.55, y + dy), (cx - 0.13, y), rad=rad, lw=0.5, ls=(0, (2, 1.5)), head=False)

    gx, gy, gs = 2.15, 2.8, 1.12
    pos = np.column_stack([gx + LAYOUT[:, 0] * gs, gy + LAYOUT[:, 1] * gs])
    graph(base, pos, coupling ** 1.3, NODE, "#8C929E", 0.045, 1.6, z=1, floor=0.08, rings=PAIR)
    for m, lab in zip(PAIR, ("$x_i$", "$x_j$")):
        trace(base, pos[m, 0] - 0.16, pos[m, 1] + 0.07, 0.32, 0.12, scaled[m, t0:t1], lw=0.45)
    arrow(base, (cx + cw + 0.05, cy + ch / 2), (gx - 0.05, cy + ch / 2))
    text(base, gx + gs / 2, gy - 0.08, "nodes: channels; edges: coupling", 7, MUTED, ha="center", va="top")

    # (b) K statistics: the graph fanned out into K planes, and the same K as stacked matrices. ---
    title(base, 3.55, 4.75, "b", r"$K$ statistics", r"one weighted graph and one $M \times M$ matrix each")
    O = np.array([gx + gs + 0.16, gy + gs * 0.5])
    angles = np.deg2rad([62, 34, 0, -34, -62])  # middle three spread widest; outer two tuck behind
    who = [2, 3, 0, 1, 4]  # statistic drawn on each plane, top to bottom: a in the middle, b below it
    side, depth, shear, radius = 0.56, 0.3, 0.08, 1.0
    for k in sorted(range(K), key=lambda k: -abs(angles[k])):  # outer planes first, middle in front
        th = angles[k]
        er, et = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
        centre = O + radius * er
        u, v = depth * er + shear * et, side * et
        corners = [centre + sx * u / 2 + sy * v / 2 for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
        s_idx = who[k]
        coloured = s_idx < 2
        hue = HUES[s_idx] if coloured else None
        zk = 3 + 2 * (2 - abs(k - 2))
        base.add_patch(Polygon(corners, closed=True, fc=tint(hue, 0.13) if coloured else CARD,
                               ec=shade(hue, 0.85) if coloured else HAIR, lw=0.5, joinstyle="round", zorder=zk))
        local = centre + (LAYOUT[:, :1] - 0.5) * 0.82 * u + (LAYOUT[:, 1:] - 0.5) * 0.82 * v
        w = edge_weights(mats[names[s_idx]], SIMILARITY_SIGN.get(names[s_idx], 1))
        graph(base, local, w, shade(hue, 0.85) if coloured else NODE, shade(hue, 0.75) if coloured else "#9BA1AC",
              0.022, 0.9, z=zk + 0.2, floor=0.1)
        if coloured:
            text(base, *(centre + 0.62 * u + (0.3 * v if s_idx == 0 else 0 * v)), f"$A^{{({'ab'[s_idx]})}}$", 8,
                 shade(hue, 0.7), ha="left", va="center", zorder=zk + 1)
        arrow(back, O, centre - 0.5 * u, color=HAIR, lw=0.5, head=False)

    s = 0.14
    X0, Y0 = 5.42, 3.0  # front matrix A^(a); A^(b) and the grey cards step up-left behind it
    for d in reversed(range(2, K)):
        rounded(base, X0 - d * s - 0.12 * s, Y0 + d * s - 0.12 * s, M * s + 0.24 * s, M * s + 0.24 * s, CARD,
                r=0.45 * s, z=10 - d, ec=HAIR, lw=0.5)
    cb = matrix(base, mats[names[1]], HUES[1], X0 - s, Y0 + s, s, z=11)
    ca = matrix(base, mats[names[0]], HUES[0], X0, Y0, s, z=13, mark=True)
    text(base, X0 + 0.6 * s, Y0 - 0.06, "$A^{(a)}$", 8, shade(HUES[0], 0.7), ha="center", va="top", zorder=30)
    text(base, X0 - s - 0.06, Y0 + s + 0.35 * s, "$A^{(b)}$", 8, shade(HUES[1], 0.7), ha="right", va="center",
         zorder=30)
    text(base, X0 - (K - 1) * s - 0.06, Y0 + (K - 1) * s + M * s - 0.1, "$A^{(K)}$", 8, MUTED, ha="right",
         va="center", zorder=30)
    A_a = mats[names[0]]
    gx2, gsz = X0 + M * s + 0.16, 0.3
    for (src, dst), gy2, lab, ls in [((i, j), Y0 + 0.34, r"$i \to j$", "-"), ((j, i), Y0 - 0.22, r"$j \to i$", (0, (1.5, 1)))]:
        rounded(base, gx2, gy2, gsz, gsz, "white", r=0.02, z=14, ec=HAIR, lw=0.5)
        fx, fy = box_scatter(base, gx2, gy2, gsz, gsz, x[:-1, src], x[1:, dst], pad=0.06)
        base.scatter(fx(x[:-1, src]), fy(x[1:, dst]), s=0.3, color=HUES[0], lw=0, alpha=0.55, zorder=14.5,
                     rasterized=True)
        text(base, gx2 + gsz + 0.04, gy2 + gsz * 0.66, lab, 7, ha="left", va="center")
        text(base, gx2 + gsz + 0.04, gy2 + gsz * 0.3, f"${A_a[src, dst]:.2f}$", 7, shade(HUES[0], 0.6),
             ha="left", va="center")
        arrow(front, (gx2 - 0.01, gy2 + gsz / 2), ca[src, dst], lw=0.5, ls=ls, head=False, color=INK, z=20)

    # (c) z_ab: every off-diagonal entry of A^(a) and A^(b) drawn out into one scatter. -----------
    title(base, 4.22, 2.25, "c", r"$z_{ab}$", "correlation", "across pairs")
    sx0, sy0, sw = 4.98, 0.25, 1.45
    fx, fy = box_scatter(base, sx0, sy0, sw, sw, P[0], P[1])
    base.plot([sx0, sx0, sx0 + sw], [sy0 + sw, sy0, sy0], color=MUTED, lw=0.5, zorder=2)
    slope, icpt = np.polyfit(P[0], P[1], 1)
    xl = np.array([P[0].min(), P[0].max()])
    base.plot(fx(xl), fy(slope * xl + icpt), color=INK, lw=0.8, zorder=3)
    for k, (r_, c_) in enumerate(edges):
        pt = (fx(P[0, k]), fy(P[1, k]))
        hot = (r_, c_) in (PAIR, PAIR[::-1])
        base.add_patch(Circle(pt, 0.03 if hot else 0.022, fc=(INK if (r_, c_) == PAIR else "white") if hot else MUTED,
                              ec=INK if hot else "none", lw=0.6, zorder=4, alpha=1 if hot else 0.85))
        ls = "-" if (r_, c_) == PAIR else (0, (1.5, 1))
        arrow(front, ca[r_, c_], pt, rad=0.12, color=shade(HUES[0], 0.8) if hot else HUES[0], lw=0.6 if hot else 0.3,
              alpha=0.85 if hot else 0.12, ls=ls if hot else "-", head=False, z=1)
        arrow(back, cb[r_, c_], pt, rad=0.18, color=shade(HUES[1], 0.8) if hot else HUES[1], lw=0.6 if hot else 0.3,
              alpha=0.85 if hot else 0.16, ls=ls if hot else "-", head=False)
    text(base, sx0 + sw / 2, sy0 - 0.06, "$A^{(a)}_{ij}$", 8, shade(HUES[0], 0.7), ha="center", va="top")
    text(base, sx0 - 0.06, sy0 + sw / 2, "$A^{(b)}_{ij}$", 8, shade(HUES[1], 0.7), ha="right", va="center")
    text(base, sx0 + sw + 0.06, fy(slope * xl[1] + icpt), f"$z_{{ab}} = {R[0, 1]:.2f}$", 8, ha="left",
         va="center", zorder=5)

    # (d) R, with z_ab's cell; its strict upper triangle drawn out into z. --------------------------
    title(base, 0.02, 2.25, "d", r"Agreement matrix $R$", "upper triangle kept as $z$",
          r"$K(K-1)/2$ entries for any $M$, $T$")
    rs, rx, ry = 0.28, 2.1, 0.3
    norm = Normalize(-1, 1)
    zs, zx, zy = 0.14, 0.6, 0.3
    n = len(z)
    k = 0
    for r_ in range(K):
        for c_ in range(K):
            X, Y = rx + c_ * rs, ry + (K - 1 - r_) * rs
            if r_ == c_:
                unused(base, X, Y, rs, z=2)
                continue
            col = DIV(norm(R[r_, c_]))
            cell(base, X, Y, rs, col, z=2, alpha=1.0 if c_ > r_ else 0.16)
            if c_ > r_:
                ZY = zy + (n - 1 - k) * zs
                arrow(back, (X + rs * 0.2, Y + rs / 2), (zx + zs, ZY + zs / 2), rad=0.15, color=col, lw=0.5,
                      alpha=0.55, head=False)
                k += 1
    ring(base, rx + rs, ry + (K - 1) * rs, rs, lw=1.0)
    for kk, lab in [(0, "$a$"), (1, "$b$"), (K - 1, "$K$")]:
        color = shade(HUES[kk], 0.7) if kk < 2 else MUTED
        text(base, rx + (kk + 0.5) * rs, ry - 0.04, lab, 8, color, ha="center", va="top")
        text(base, rx + K * rs + 0.05, ry + (K - 1 - kk + 0.5) * rs, lab, 8, color, ha="left", va="center")
    for kk, v in enumerate(z):
        cell(base, zx, zy + (n - 1 - kk) * zs, zs, DIV(norm(v)), z=2)
    ring(base, zx, zy + (n - 1) * zs, zs, lw=0.9)
    text(base, zx + zs / 2, zy + n * zs + 0.03, "$z$", 8, ha="center", va="bottom")
    cax = fig.add_axes(((rx + K * rs + 0.25) / W, ry / H, 0.035 / W, 0.62 / H), zorder=3)
    cax.set_gid("d_colorbar")
    bar = fig.colorbar(plt.cm.ScalarMappable(norm, DIV), cax=cax, orientation="vertical", ticks=[-1, 0, 1])
    bar.outline.set_visible(False); cax.tick_params(length=0, pad=1.5, labelsize=7, colors=MUTED)
    # z_ab, read off panel c, is the (a, b) cell of R.
    target = (rx + 1.5 * rs, ry + K * rs + 0.01)
    base.add_patch(FancyArrowPatch((sx0 + 0.04, ry + K * rs + 0.14), target, arrowstyle="-|>", mutation_scale=6,
                                   lw=0.6, color=MUTED, zorder=5, shrinkA=0, shrinkB=0,
                                   connectionstyle="angle,angleA=180,angleB=90,rad=6"))
    return fig


def main():
    style()
    x, coupling = toy_recording()
    mats = statistics(x)
    c = construction(mats)
    fig = figure(x, coupling, mats, c)
    out = HERE / f"method_schematic_{STAMP}"
    for ext in ("svg", "pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=180 if ext == "png" else 600, bbox_inches="tight")
    summary = {
        "contract": UNIFIED_CONTRACT_VERSION, "M": c["M"], "T": len(x), "K": len(c["P"]),
        "statistics": SPIS, "pair_ij_1based": [PAIR[0] + 1, PAIR[1] + 1],
        "A_a_ij": float(mats["lag1"][PAIR]), "A_a_ji": float(mats["lag1"][PAIR[::-1]]),
        "z_ab": float(c["R"][0, 1]), "z": np.round(c["z"], 4).tolist(),
    }
    (HERE / f"method_schematic_{STAMP}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("statistics", "z")}, indent=1))


if __name__ == "__main__":
    main()
