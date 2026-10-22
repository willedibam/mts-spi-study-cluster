"""Method schematic: construction of the SPI–SPI representation z (paper Fig. 1), 2026-10-22.

The panels run as a snake. Row 1, left to right: (a) a recording as two extracted traces,
a channel x time carpet and a directed coupling graph of the same channels; (b) the graph
as K layers, one weighted graph per statistic (directed statistics drawn with one tapered
arc per direction), and the K statistics as stacked M x M matrices, with one entry's
computation for i -> j and j -> i attached to the front matrix. Row 2, right to left: (c)
the off-diagonal entries of A^(a) and A^(b) drawn out into one scatter, whose Pearson
correlation is z_ab; (d) z_ab's place in the K x K agreement matrix, one strict triangle of
which is drawn out into z; (e) schematic icons for what one z per recording supports.

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
DIRECTED = {"lag1", "granger"}
# Channel labels are arbitrary (z is invariant to relabelling); this order puts the strongly
# asymmetric pair at the matrix corner, so i -> j is in the last column and j -> i in the last row.
CHANNEL_ORDER = [0, 1, 2, 4, 3, 5]
PAIR = (4, 5)  # highlighted channel pair (i, j), 0-based, after CHANNEL_ORDER
EXTRA = 1  # one further channel annotated in panel a (channel 2), bottom node of the graph
WINDOW = (200, 300)  # samples shown in the carpet and traces
# Irregular node layout in the unit square; i and j (nodes 4, 5) sit on top, under their traces.
LAYOUT = np.array([[0.04, 0.50], [0.30, 0.06], [0.97, 0.44], [0.68, 0.12], [0.27, 0.84], [0.77, 0.93]])

# Soft palette: a pastel hue for statistics a and b, quiet greys for the rest of the K, a
# pastel diverging map for R and z, a dark ink for the channel pair (i, j), and white cells
# with a hairline for entries that are never used (all diagonals).
HUES = ["#74B49B", "#9D8FCB"]
FEATURE = "#C98A2E"  # z_ab itself: a warm ochre, unused elsewhere (fitted line, outlined value and cells)
PASTELS = ["#74B49B", "#9D8FCB", "#E0B060", "#D98FA5", "#86B3D9"]  # sage, lavender, amber, rose, sky
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

    Returns the z-scored recording (T x M) and the generator's coupling per ordered pair s -> t:
    |lagged coupling of t on s| plus |innovation correlation|, scaled to [0, 1].
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
    coupling = off.T + np.abs(C / np.sqrt(np.outer(np.diag(C), np.diag(C))) - np.eye(M))  # [s, t]: s -> t
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
    text(ax, x + 0.14, y, rf"\textbf{{{head}}}", 9, va="baseline")
    for k, sub in enumerate(subs, 1):
        text(ax, x + 0.14, y - 0.155 * k, sub, 8, MUTED, va="baseline")


def arrow(ax, p0, p1, rad=0.0, color=MUTED, lw=0.6, ls="-", head=True, alpha=1.0, z=5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>" if head else "-", mutation_scale=6, lw=lw,
                                 color=color, ls=ls, alpha=alpha, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def rounded(ax, x, y, w, h, fc, r=0.03, z=1, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, zorder=z,
                                **{"ec": "none", **kw}))


GAP = 0.1  # white space between cells, as a fraction of the pitch


def cell(ax, x, y, s, fc, z=1, **kw):
    """A soft cell of pitch s with its lower-left corner at (x, y)."""
    g = GAP * s
    rounded(ax, x + g / 2, y + g / 2, s - g, s - g, fc, r=0.16 * s, z=z, **kw)


def unused(ax, x, y, s, z=1):
    cell(ax, x, y, s, "white", z=z, ec=HAIR, lw=0.4)


def ring(ax, x, y, s, ls="-", lw=0.8, z=3, color=INK):
    cell(ax, x - 0.04 * s, y - 0.04 * s, 1.08 * s, "none", z=z, ec=color, lw=lw, ls=ls)


def card(ax, x0, y0, n, s, fc="white", z=1):
    pad = 0.12 * s
    rounded(ax, x0 - pad, y0 - pad, n * s + 2 * pad, n * s + 2 * pad, fc, r=0.45 * s, z=z, ec=HAIR, lw=0.5)


def matrix(ax, A, hue, x0, y0, s, z=1, mark=False):
    """M x M statistic on a white card; row r at y0 + (M-1-r)s. Returns cell centres."""
    M = len(A)
    cmap, u = seq(hue), unit(A)
    card(ax, x0, y0, M, s, z=z)
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


def tapered_arc(ax, p, q, width, color, alpha, bow=0.13, z=1, n=28):
    """Directed edge p -> q: a quadratic arc bowed to the left of travel, tapering from p to q.

    The two directions of one pair therefore bow apart, and the wide end marks the source.
    """
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    L = np.hypot(*d)
    ctrl = (p + q) / 2 + bow * L * np.array([-d[1], d[0]]) / L
    t = np.linspace(0, 1, n)[:, None]
    pts = (1 - t) ** 2 * p + 2 * (1 - t) * t * ctrl + t ** 2 * q
    tang = np.gradient(pts, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    normal = np.column_stack([-tang[:, 1], tang[:, 0]])
    half = (width / 2) * (1 - 0.72 * t)
    ax.add_patch(Polygon(np.vstack([pts + normal * half, (pts - normal * half)[::-1]]), closed=True, fc=color,
                         ec="none", alpha=alpha, zorder=z))


def graph(ax, pos, w, node, edge, r_node, width, z=1, floor=0.12, rings=(), directed=False):
    """Weighted graph; edge width (inches) and opacity grow with w in [0, 1], weak edges fade out.

    Directed: one tapered arc per ordered pair (w[s, t] for s -> t). Undirected: one line per pair.
    """
    M = len(pos)
    for a in range(M):
        for b in range(M):
            if a == b or (not directed and b < a) or w[a, b] < floor:
                continue
            wd, alpha = width * (0.25 + 0.75 * w[a, b]), 0.12 + 0.8 * w[a, b]
            if directed:
                tapered_arc(ax, pos[a], pos[b], wd, edge, alpha, z=z)
            else:
                ax.plot(*zip(pos[a], pos[b]), color=edge, lw=72 * wd * 0.7, alpha=alpha, solid_capstyle="round",
                        zorder=z)
    for m in range(M):
        ax.add_patch(Circle(pos[m], r_node, fc=node, ec="white", lw=0.4, zorder=z + 0.1))
    for m in rings:
        ax.add_patch(Circle(pos[m], 1.55 * r_node, fc="none", ec=INK, lw=0.6, zorder=z + 0.2))


def edge_weights(A: np.ndarray, sign: int = 1, directed: bool = False) -> np.ndarray:
    """Min-max scaled strength for drawing; symmetrised unless the statistic is directed."""
    u = unit(A) if sign > 0 else 1 - unit(A)
    if not directed:
        u = (u + u.T) / 2
    return u ** 1.6  # emphasise strong edges so weak ones recede


def trace(ax, x0, y0, w, h, series, lw=0.6, color=INK, z=2):
    t = np.linspace(x0, x0 + w, len(series))
    ax.plot(t, y0 + h / 2 + series / (2.2 * np.abs(series).max()) * h, color=color, lw=lw, zorder=z)


def brackets(ax, x0, y0, w, h, arm=0.05, lw=0.8, color=INK, z=3):
    """Square matrix brackets around the box (x0, y0, w, h)."""
    for side in (0, 1):
        xe, sgn = (x0, 1) if side == 0 else (x0 + w, -1)
        ax.plot([xe + sgn * arm, xe, xe, xe + sgn * arm], [y0, y0, y0 + h, y0 + h], color=color, lw=lw,
                solid_joinstyle="miter", zorder=z)


def box_scatter(x0, y0, w, h, xs, ys, pad=0.08):
    """Linear maps from data to a box in inches."""
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
    back = layer(fig, "lines_behind", 0)
    base = layer(fig, "panels", 1)
    front = layer(fig, "lines_front", 2)

    # (a) Recording: carpet (one card, rows are channels) -> directed coupling graph. -------------
    title(base, 0.02, 4.75, "a", "Recording", r"$M$ channels $\times$ $T$ samples")
    scaled, limit = robust_scaled(x)
    t0, t1 = WINDOW
    cx, cy, cw, ch = 0.46, 3.0, 1.5, 0.92
    pitch, gap = ch / M, 0.012
    rounded(base, cx - 0.03, cy - 0.03, cw + 0.06, ch + 0.06, "white", r=0.04, z=0.5, ec=HAIR, lw=0.5)
    for m in range(M):
        y = cy + (M - 1 - m) * pitch
        clip = FancyBboxPatch((cx, y + gap), cw, pitch - 2 * gap, boxstyle="round,pad=0,rounding_size=0.025",
                              fc="none", ec="none", transform=base.transData)
        base.add_patch(clip)
        im = base.imshow(scaled[m:m + 1, t0:t1], cmap=MTS, vmin=-limit, vmax=limit, aspect="auto",
                         interpolation="nearest", extent=(cx, cx + cw, y + gap, y + pitch - gap), zorder=1)
        im.set_clip_path(clip)
        text(base, cx - 0.06, y + pitch / 2, f"${m + 1}$", 7, MUTED, ha="right", va="center")
    text(base, cx + cw * 0.62, cy - 0.06, "time", 8, MUTED, ha="center", va="top")
    text(base, 0.07, cy + ch / 2, "channel", 8, MUTED, ha="center", va="center", rotation=90)
    row_y = lambda m: cy + (M - 1 - m) * pitch + pitch / 2
    # Annotated channels: the pair (i, j) in ink, one further channel (EXTRA) muted. Channel 2's trace sits
    # above-left of the carpet, i's and j's underneath; dashed leaders curve to each row's number.
    annotated = [(EXTRA, f"$x_{{{EXTRA + 1}}}$", MUTED, (0.02, cy + ch + 0.1), 0.35),
                 (i, "$x_i$", INK, (0.02, cy - 0.44), 0.15), (j, "$x_j$", INK, (0.5, cy - 0.44), 0.3)]
    for m, lab, col, (tx, ty), rad in annotated:
        y = cy + (M - 1 - m) * pitch
        rounded(base, cx - 0.012, y + gap - 0.012, cw + 0.024, pitch - 2 * gap + 0.024, "none", r=0.03, z=2,
                ec=col, lw=0.7)
        trace(base, tx, ty, 0.36, 0.13, scaled[m, t0:t1], lw=0.5, color=col)
        text(base, tx, ty + 0.15, lab, 8, col, ha="left", va="bottom")
        start = (tx + 0.38, ty + 0.065) if m == EXTRA else (tx + 0.34, ty + 0.15)
        arrow(back, start, (cx - 0.16, row_y(m)), rad=rad, lw=0.5, ls=(0, (2, 1.5)), head=False)

    gx, gy, gs = 2.18, 2.94, 0.98
    pos = np.column_stack([gx + LAYOUT[:, 0] * gs, gy + LAYOUT[:, 1] * gs])
    graph(base, pos, coupling ** 1.3, NODE, "#8C929E", 0.042, 0.026, z=1, floor=0.08, rings=PAIR, directed=True)
    base.add_patch(Circle(pos[EXTRA], 1.55 * 0.042, fc="none", ec=MUTED, lw=0.6, zorder=1.3))
    for m in PAIR:
        trace(base, pos[m, 0] - 0.15, pos[m, 1] + 0.065, 0.3, 0.11, scaled[m, t0:t1], lw=0.45)
    trace(base, pos[EXTRA, 0] - 0.15, pos[EXTRA, 1] - 0.18, 0.3, 0.11, scaled[EXTRA, t0:t1], lw=0.45, color=MUTED)
    arrow(base, (cx + cw + 0.07, cy + ch / 2), (gx - 0.06, cy + ch / 2))

    # (b) K statistics: layers of the same graph (a, b, one more, then dots), and the K matrices. --
    title(base, 3.22, 4.75, "b", r"$K$ statistics", r"one weighted graph and one $M \times M$ matrix each")
    O = np.array([gx + gs + 0.1, gy + gs * 0.5])
    LW, LH, SKEW, GAPL = 1.18, 0.38, 0.32, 0.14
    who = [0, 1, 2]  # a, b, then one representative of the rest; the dots stand for the remainder
    lx, top = 3.36, 4.36
    for k, s_idx in enumerate(who):
        o = np.array([lx, top - LH - k * (LH + GAPL)])
        u, v = np.array([LW, 0.0]), np.array([SKEW, LH])
        coloured = s_idx < 2
        hue = HUES[s_idx] if coloured else None
        base.add_patch(Polygon([o, o + u, o + u + v, o + v], closed=True, fc=tint(hue, 0.13) if coloured else CARD,
                               ec=shade(hue, 0.85) if coloured else HAIR, lw=0.5, joinstyle="round", zorder=3))
        local = o + (0.08 + 0.84 * LAYOUT[:, :1]) * u + (0.1 + 0.8 * LAYOUT[:, 1:]) * v
        name = names[s_idx]
        w = edge_weights(mats[name], SIMILARITY_SIGN.get(name, 1), name in DIRECTED)
        graph(base, local, w, shade(hue, 0.85) if coloured else NODE, shade(hue, 0.75) if coloured else "#9BA1AC",
              0.024, 0.018, z=3.2, floor=0.15, directed=name in DIRECTED)
        if coloured:
            text(base, *(o + u + v / 2 + np.array([0.06, 0])), f"$A^{{({'ab'[s_idx]})}}$", 8, shade(hue, 0.7),
                 ha="left", va="center", zorder=4)
        arrow(back, O, o + 0.5 * v + np.array([0.02, 0]), color=HAIR, lw=0.5, head=False)
    last = top - LH - (len(who) - 1) * (LH + GAPL)
    text(base, lx + (LW + SKEW) / 2, last - 0.06, r"$\vdots$", 10, MUTED, ha="center", va="top")

    s = 0.15
    Gx, Gy = 5.2, 2.84
    off = 0.45 * M * s  # A^(b) sits up-right; A^(a) covers its lower-left ~30%
    Bx, By = Gx + off, Gy + off
    for d in reversed(range(1, K - 1)):
        card(base, Bx + d * 0.08, By + d * 0.08, M, s, fc=CARD, z=6 - d * 0.5)
    cb = matrix(base, mats[names[1]], HUES[1], Bx, By, s, z=7, mark=True)
    ca = matrix(base, mats[names[0]], HUES[0], Gx, Gy, s, z=9, mark=True)
    text(base, Gx + M * s / 2, Gy - 0.06, "$A^{(a)}$", 8, shade(HUES[0], 0.7), ha="center", va="top", zorder=30)
    text(base, Bx - 0.05, By + M * s - 0.08, "$A^{(b)}$", 8, shade(HUES[1], 0.7), ha="right", va="center",
         zorder=30)
    text(base, Bx + (K - 2) * 0.08 + M * s, By + (K - 2) * 0.08 + M * s + 0.03, "$A^{(K)}$", 8, MUTED, ha="right",
         va="bottom", zorder=30)
    A_a = mats[names[0]]
    gsz, gxg = 0.29, Gx + M * s + 0.12
    for (src, dst), gyg, lab, ls in [((i, j), Gy + 0.03, r"$i \to j$", "-"), ((j, i), Gy - 0.34, r"$j \to i$", (0, (1.5, 1)))]:
        rounded(base, gxg, gyg, gsz, gsz, "white", r=0.02, z=14, ec=HAIR, lw=0.5)
        fx, fy = box_scatter(gxg, gyg, gsz, gsz, x[:-1, src], x[1:, dst], pad=0.06)
        base.scatter(fx(x[:-1, src]), fy(x[1:, dst]), s=0.3, color=HUES[0], lw=0, alpha=0.55, zorder=14.5,
                     rasterized=True)
        text(base, gxg + gsz + 0.04, gyg + gsz * 0.66, lab, 7, ha="left", va="center")
        text(base, gxg + gsz + 0.04, gyg + gsz * 0.3, f"${A_a[src, dst]:.2f}$", 7, shade(HUES[0], 0.6),
             ha="left", va="center")
        arrow(front, (gxg - 0.01, gyg + gsz / 2), ca[src, dst], lw=0.5, ls=ls, head=False, color=INK, z=20)

    # (c) z_ab: every off-diagonal entry of A^(a) and A^(b) drawn out into one boxed scatter. ------
    title(base, 4.78, 2.25, "c", r"$z_{ab}$")
    sx0, sy0, sw = 5.05, 0.28, 1.72
    rounded(base, sx0, sy0, sw, sw, "white", r=0.04, z=1.5, ec=MUTED, lw=0.5)
    fx, fy = box_scatter(sx0, sy0, sw, sw, P[0], P[1], pad=0.1)
    pts = np.column_stack([fx(P[0]), fy(P[1])])
    slope, icpt = np.polyfit(P[0], P[1], 1)
    xl = np.array([P[0].min(), P[0].max()])
    base.plot(fx(xl), fy(slope * xl + icpt), color=FEATURE, lw=1.4, zorder=3, solid_capstyle="round")
    # Rugs: each axis is one statistic's edge profile.
    for k in range(len(edges)):
        base.plot([pts[k, 0]] * 2, [sy0, sy0 + 0.055], color=shade(HUES[0], 0.85), lw=0.6, zorder=2)
        base.plot([sx0, sx0 + 0.055], [pts[k, 1]] * 2, color=shade(HUES[1], 0.85), lw=0.6, zorder=2)
    for k, (r_, c_) in enumerate(edges):
        pt = tuple(pts[k])
        hot = (r_, c_) in (PAIR, PAIR[::-1])
        base.add_patch(Circle(pt, 0.033 if hot else 0.024, fc=(INK if (r_, c_) == PAIR else "white") if hot else MUTED,
                              ec=INK if hot else "none", lw=0.6, zorder=4, alpha=1 if hot else 0.85))
        ls = "-" if (r_, c_) == PAIR else (0, (1.5, 1))
        style_a = dict(color=shade(HUES[0], 0.8) if hot else HUES[0], lw=0.6 if hot else 0.3,
                       alpha=0.85 if hot else 0.12, ls=ls if hot else "-", head=False)
        style_b = dict(style_a, color=shade(HUES[1], 0.8) if hot else HUES[1], alpha=0.85 if hot else 0.16)
        arrow(front, ca[r_, c_], pt, rad=0.12, z=1, **style_a)
        arrow(back, cb[r_, c_], pt, rad=0.18, **style_b)  # A^(b) sits behind, so its lines do too
    text(base, sx0 + sw / 2, sy0 - 0.06, "$A^{(a)}_{ij}$", 8, shade(HUES[0], 0.7), ha="center", va="top")
    text(base, sx0 - 0.06, sy0 + sw / 2, "$A^{(b)}_{ij}$", 8, shade(HUES[1], 0.7), ha="right", va="center")
    zlab = text(base, sx0 + 0.11, sy0 + sw - 0.09, f"$z_{{ab}} = {R[0, 1]:.2f}$", 8, ha="left", va="top", zorder=6)
    zlab.set_bbox(dict(boxstyle="round,pad=0.25,rounding_size=0.2", fc="white", ec=FEATURE, lw=1.0))

    # (d) R, with z_ab's cell; its strict lower triangle drawn out into z. --------------------------
    title(base, 2.38, 2.25, "d", r"Agreement matrix $R$")
    rs, rx, ry = 0.26, 3.0, 0.62
    norm = Normalize(-1, 1)
    zs, zx, zy = 0.15, 2.52, 0.42
    n = len(z)
    brackets(base, rx - 0.03, ry - 0.03, K * rs + 0.06, K * rs + 0.06)
    order = {pair: k for k, pair in enumerate((a, b) for a in range(K) for b in range(a + 1, K))}
    for r_ in range(K):
        for c_ in range(K):
            X, Y = rx + c_ * rs, ry + (K - 1 - r_) * rs
            if r_ == c_:
                unused(base, X, Y, rs, z=2)
                continue
            col = DIV(norm(R[r_, c_]))
            cell(base, X, Y, rs, col, z=2, alpha=1.0 if c_ < r_ else 0.16)
            if c_ < r_:  # kept entry (r, c) is z's coordinate for the pair (c, r)
                k = order[c_, r_]
                arrow(back, (X + rs * 0.2, Y + rs / 2), (zx + zs, zy + (n - 1 - k) * zs + zs / 2), rad=-0.12,
                      color=col, lw=0.5, alpha=0.55, head=False)
    ring(base, rx, ry + (K - 2) * rs, rs, lw=1.1, color=FEATURE)
    for kk, lab in [(0, "$a$"), (1, "$b$"), (K - 1, "$K$")]:
        color = shade(HUES[kk], 0.7) if kk < 2 else MUTED
        text(base, rx + (kk + 0.5) * rs, ry + K * rs + 0.05, lab, 8, color, ha="center", va="bottom")
        text(base, rx + K * rs + 0.07, ry + (K - 1 - kk + 0.5) * rs, lab, 8, color, ha="left", va="center")
    for kk, v in enumerate(z):
        cell(base, zx, zy + (n - 1 - kk) * zs, zs, DIV(norm(v)), z=2)
    ring(base, zx, zy + (n - 1) * zs, zs, lw=1.0, color=FEATURE)
    brackets(base, zx - 0.03, zy - 0.03, zs + 0.06, n * zs + 0.06, arm=0.035)
    text(base, zx + zs / 2, zy + n * zs + 0.06, "$z$", 8, ha="center", va="bottom")
    W, H = fig.get_size_inches()
    cax = fig.add_axes((rx / W, 0.4 / H, K * rs / W, 0.035 / H), zorder=3)
    cax.set_gid("d_colorbar")
    bar = fig.colorbar(plt.cm.ScalarMappable(norm, DIV), cax=cax, orientation="horizontal", ticks=[-1, 0, 1])
    bar.outline.set_visible(False); cax.tick_params(length=0, pad=1.5, labelsize=7, colors=MUTED)
    text(base, rx + K * rs / 2, 0.2, r"agreement $z_{ab}$", 7, MUTED, ha="center", va="top")

    # (e) What one z per recording supports: schematic icons, not data. --------------------------
    title(base, 0.02, 2.25, "e", "Statistical learning")
    uses(base)
    return fig


def uses(ax, sz=0.86):
    """Embedding (top left), unsupervised tracking (middle right), one coordinate (lower left)."""
    rng = np.random.default_rng(3)
    sage, lav, amber, rose, sky = PASTELS
    # Embedding: four clusters, each mixing recordings of three sizes M.
    x0, y0 = 0.1, 1.16
    rounded(ax, x0, y0, sz, sz, "white", r=0.03, z=2, ec=HAIR, lw=0.5)
    for c, (centre, hue) in enumerate(zip([(0.27, 0.71), (0.73, 0.75), (0.3, 0.27), (0.71, 0.31)],
                                          [shade(sage, 0.85), shade(lav, 0.9), shade(amber, 0.9), shade(rose, 0.9)])):
        p = np.array(centre) + rng.normal(0, 0.055, (9, 2))
        ax.scatter(x0 + p[:, 0] * sz, y0 + p[:, 1] * sz, s=np.repeat([1.8, 3.8, 6.5], 3), color=hue, lw=0,
                   alpha=0.95, zorder=3)
    # Tracking: PC1 of z (band = spread over replicates) follows an order parameter.
    x0, y0 = 1.14, 0.62
    rounded(ax, x0, y0, sz, sz, "white", r=0.03, z=2, ec=HAIR, lw=0.5)
    g = np.linspace(0, 1, 16)
    Q = 0.12 + 0.76 / (1 + np.exp(-(g - 0.5) / 0.08))
    q = 0.14 + 0.72 / (1 + np.exp(-(g - 0.56) / 0.06)) + rng.normal(0, 0.025, len(g))
    band = 0.05 + 0.06 * np.exp(-((g - 0.56) / 0.12) ** 2)
    X = x0 + (0.1 + 0.8 * g) * sz
    ax.fill_between(X, y0 + (q - band) * sz, y0 + (q + band) * sz, color=tint(sky, 0.45), lw=0, zorder=2.5)
    ax.plot(X, y0 + Q * sz, color=INK, lw=0.7, zorder=3)
    ax.plot(X, y0 + q * sz, color=shade(sky, 0.72), lw=1.0, zorder=3)
    ax.plot([x0 + 0.53 * sz] * 2, [y0 + 0.08 * sz, y0 + 0.92 * sz], color=MUTED, lw=0.4, ls=(0, (1.5, 1.5)),
            zorder=2)
    # One named coordinate z_ab separating two conditions (rainclouds).
    x0, y0 = 0.24, 0.08
    rounded(ax, x0, y0, sz, sz, "white", r=0.03, z=2, ec=HAIR, lw=0.5)
    for c, (cx, mu, hue) in enumerate([(0.3, 0.72, shade(rose, 0.9)), (0.68, 0.3, shade(sky, 0.85))]):
        vals = mu + rng.normal(0, 0.07, 12)
        yy = np.linspace(vals.min(), vals.max(), 30)
        dens = np.exp(-0.5 * ((yy - mu) / 0.07) ** 2)
        ax.fill_betweenx(y0 + yy * sz, x0 + (cx + 0.04) * sz, x0 + (cx + 0.04 + 0.17 * dens) * sz, color=hue,
                         alpha=0.45, lw=0, zorder=3)
        ax.scatter(x0 + (cx - 0.07 + rng.uniform(-0.04, 0.04, 12)) * sz, y0 + vals * sz, s=2.6, color=hue, lw=0,
                   zorder=3)
        ax.plot([x0 + (cx + 0.01) * sz] * 2, [y0 + (mu - 0.06) * sz, y0 + (mu + 0.06) * sz], color=hue, lw=1.3,
                solid_capstyle="round", zorder=3)


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
