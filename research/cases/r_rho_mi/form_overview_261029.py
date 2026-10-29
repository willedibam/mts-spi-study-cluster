"""Form case figure, overview layout (2026-10-29).

a  Dependency filters. A u block feeds each of three couplings (w, sinh 2w, w^2) through the strength key; beside
   each block's strongest and weakest rows, x_i against u shows the filter plus that strength's noise; then the
   4 linear / 4 monotone / 4 fold recording. A callout from the top u block shows the shared signal. Numbered
   bars mark the channel pairs shown in c.
b  Feature variation. z(r, rho) against z(rho, MI) on equal square axes, bottom level with a's heatmap, for every recording of the linear,
   monotone and non-monotone (fold) conditions, and of the all-three recording type shown in a (diamonds).
c  One silo per condition: a strong and a weak pair with r, rho, MI bars above their scatters, telescoped to their
   points in the condition's edge plane (bold z top-left). A rail beside each scatter repeats the pair's bar from
   a. Dotted lines join the non-monotone silo's top corners to its point in b (``--links``) and pair bars 3 and 4
   in a to their badges in c.

Every condition shares the latent and noise draws, so a channel with the same map is identical across recordings
and the marked pairs are the ones in c (asserted). Recording: replicate REP of `form_figure_261028.py`. MI is
pyspi's `mi_kraskov_NN-4` (KSG estimator 1, k = 4), plotted in nats against the Gaussian reference MI(rho); bars
show it on the correlation scale (Linfoot).

    python -m research.cases.r_rho_mi.form_overview_261029 [--live-text] [--panels]
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MPath

from research.cases.case_run import ROOT, ordered, z
from research.cases.composite_style import HAIR, INK, MUTED, axes_in, panel_title, save, shade, style, tint
from research.cases.r_rho_mi.form_figure_261028 import B, HUES, MAPS, M, REP, boxed, channel_carpet, ink, linfoot
from research.cases.r_rho_mi.form_ladder_261028 import COLOURS, LEVELS, PAIR_INK, make, relation, stats

OUT = ROOT / "results/cases/form-ladder-261028"
W, H = 7.2, 6.40
# (strong, weak) pairs per condition: block-1 channels with the lowest noise level, then the third
PAIRS = {"linear": ((0, 1), (2, 3)), "monotone": ((0, 4), (2, 6)), "fold": ((0, 8), (2, 10))}
NAMES = {"fold": "non-monotone"}           # display name where it differs from the map's key
CONDS = (("linear", "linear", ("r", "rho")), ("monotone", "monotone", ("r", "rho")), ("fold", "fold", ("rho", "mi")))
ZSER = (("z_r_rho", r"$z(r,\rho)$", INK, "o"), ("z_rho_mi", r"$z(\rho,\mathrm{MI})$", "#8A8F9C", "s"),
        ("z_r_mi", r"$z(r,\mathrm{MI})$", "#C5C8CE", "D"))


def _fig(x, y):
    return x / W, 1 - y / H


def ublock(fig, cx, cy, size, fontsize=8):
    """Rounded box labelled u, centred at (cx, cy) inches."""
    x0, y0 = _fig(cx - size / 2, cy + size / 2)
    fig.add_artist(FancyBboxPatch((x0, y0), size / W, size / H, boxstyle="round,pad=0,rounding_size=0.012",
                                  transform=fig.transFigure, fc="white", ec=INK, lw=0.7))
    fig.text(cx / W, 1 - cy / H, "$u$", ha="center", va="center", fontsize=fontsize, transform=fig.transFigure)


def badge(fig, x, y, num, colour, size=5.5):
    """Filled circle with a white number, centred at (x, y) inches: the pair's identity in a and c."""
    fig.text(x / W, 1 - y / H, str(num), ha="center", va="center", fontsize=size, color="white",
             transform=fig.transFigure, bbox=dict(boxstyle="circle,pad=0.18", fc=colour, ec="none"))


def flow(fig, a, b, colour="0.55", lw=0.6, zorder=-1):
    """Smooth curve with horizontal tangents from a to b (inches), drawn behind axes."""
    d = (b[0] - a[0]) / 2
    path = MPath([_fig(*p) for p in [a, (a[0] + d, a[1]), (b[0] - d, b[1]), b]], [MPath.MOVETO] + [MPath.CURVE4] * 3)
    fig.add_artist(PathPatch(path, transform=fig.transFigure, fill=False, ec=colour, lw=lw, zorder=zorder))


def system(fig, top, x, it, numbered):
    """u -> three couplings -> strength key -> x_i vs u per strength extreme -> heatmap; numbered bars mark the
    pairs in c. A callout from the top u block shows the shared signal. Returns each pair bar's lower end."""
    hy, hh = top + 0.22, A_HH
    sx, gx_out = 0.80, 0.94                         # strength key, strength-regime glyph column
    hx, hw = A_HX, A_HW
    row = hh / M
    t0, n = 300, int(110 * hw / 1.5)
    wgrid = np.linspace(-2.2, 2.2, 100)
    names = {"linear": "$w$", "monotone": r"$\sinh 2w$", "fold": "$w^2$"}
    arrow = dict(transform=fig.transFigure, arrowstyle="-|>", mutation_scale=5, color="0.55", lw=0.5)
    for k in range(3):
        kind = it.types[k * B]
        col = ink(kind)
        cy = hy + (k + 0.5) * hh / 3
        ublock(fig, 0.1, cy, 0.17, fontsize=6.5)
        gx = 0.34
        fig.add_artist(FancyArrowPatch(_fig(0.2, cy), _fig(gx - 0.02, cy), **arrow))
        fig.add_artist(FancyArrowPatch(_fig(gx + 0.31, cy), _fig(sx - 0.02, cy), **arrow))
        ga = axes_in(fig, gx, cy - 0.15, 0.3, 0.3)
        ga.plot(wgrid, MAPS[kind](wgrid), color=col, lw=1.0)
        ga.axis("off")
        ga.text(0.5, -0.04, names[kind], transform=ga.transAxes, ha="center", va="top", fontsize=6, color=col)
        # x against u for the block's strongest (upper) and weakest (lower) channel, each beside its two rows
        for c, r0 in ((k * B, k * B), (k * B + B - 1, k * B + 2)):
            g = axes_in(fig, gx_out, hy + r0 * row + 0.012, 2 * row - 0.024, 2 * row - 0.024)
            sel = slice(0, 2000, 3)
            g.scatter(it.latent[sel], x[sel, c], s=0.2, lw=0, color=col, alpha=0.6, rasterized=True)
            g.set(xlim=(-2.6, 2.6), ylim=np.quantile(x[:, c], [0.003, 0.997]))
            boxed(g, HAIR)
    fig.text((gx_out + row) / W, 1 - (hy - 0.06) / H, r"$x_i$ vs $u$", ha="center", va="baseline", fontsize=6,
             color=MUTED, transform=fig.transFigure)
    # callout: the top u block -> a small inset of the shared signal
    cy0 = hy + 0.5 * hh / 3
    ins = axes_in(fig, 0.3, top - 0.02, 0.45, 0.17)
    ins.plot(it.latent[t0:t0 + 45], color=INK, lw=0.55)
    ins.axis("off")
    fig.add_artist(FancyArrowPatch(_fig(0.1, cy0 - 0.09), _fig(0.28, top + 0.065), transform=fig.transFigure,
                                   arrowstyle="-", connectionstyle="angle3,angleA=90,angleB=0", color="0.55",
                                   lw=0.5))
    sk = axes_in(fig, sx, hy, 0.05, hh)
    sk.imshow((1 / np.sqrt(1 + it.noise_stds ** 2))[:, None], cmap="Greys", vmin=0.3, vmax=1.05, aspect="auto")
    boxed(sk, INK)
    fig.text((sx + 0.025) / W, 1 - (hy + hh + 0.12) / H, "strength", ha="center", va="baseline", fontsize=6,
             color=MUTED, transform=fig.transFigure)
    cp = axes_in(fig, hx, hy, hw, hh)
    channel_carpet(cp, x[t0:t0 + n])
    for b in (B - 0.5, 2 * B - 0.5):
        cp.axhline(b, color="white", lw=0.8)
    cp.text(0.5, -0.03, "time", transform=cp.transAxes, ha="center", va="top", fontsize=6, color=MUTED)
    # numbered pair bars: dots on the two channel rows, joined
    row_y = lambda c: hy + (c + 0.5) * row
    ends = {}
    for k, (num, i, j, kind) in enumerate(numbered):
        bx_ = hx + hw + 0.08 + 0.09 * k
        lo, hi = row_y(min(i, j)), row_y(max(i, j))
        fig.add_artist(plt.Line2D(*zip(_fig(bx_, lo), _fig(bx_, hi)), transform=fig.transFigure, color=ink(kind),
                                  lw=0.9, solid_capstyle="round"))
        fig.add_artist(plt.Line2D([_fig(bx_, lo)[0], _fig(bx_, hi)[0]], [_fig(bx_, lo)[1], _fig(bx_, hi)[1]],
                                  transform=fig.transFigure, ls="", marker="o", ms=2.8, mew=0, color=ink(kind)))
        badge(fig, bx_, hy - 0.1, num, ink(kind))
        ends[num] = (bx_, hi)
    return ends


def result(fig, left, top, w, h, frame, rep, mode="scatter", show_all=True):
    """The two rung features for every recording of the three single-coupling conditions.

    mode "scatter": z(r, rho) against z(rho, MI), one point per recording; "strip": both features per condition.
    Returns the axes and, per condition, the shown recording's position (the value its plane in c reports).
    """
    ax = axes_in(fig, left, top, w, h)
    ax.set_facecolor("none")                         # beams to c run up to the ringed points
    n = frame.groupby("level").size().min()
    shown = {}
    if mode == "scatter":
        for cond, kind, key in CONDS:
            f = frame[frame.level == cond]
            ax.scatter(f.z_r_rho, f.z_rho_mi, s=17, lw=0.4, ec="white", color=ink(kind), alpha=0.85, zorder=2)
            f0 = f[f.rep == rep].iloc[0]
            shown[cond] = (float(f0.z_r_rho), float(f0.z_rho_mi))
            ax.plot(*shown[cond], "o", ms=7, mfc="none", mec=ink(kind), mew=1.0, zorder=5)
        if show_all:                                 # the recording of a: every coupling at once
            f = frame[frame.level == "both"]
            ax.scatter(f.z_r_rho, f.z_rho_mi, s=14, lw=0.4, ec="white", color="0.55", alpha=0.85, zorder=2,
                       marker="D")
            f0 = f[f.rep == rep].iloc[0]
            ax.plot(f0.z_r_rho, f0.z_rho_mi, "D", ms=6, mfc="none", mec=INK, mew=0.9, zorder=5)
        labels = {"linear": (1.0, 0.91, "right", "top"), "monotone": (0.235, 0.962, "left", "bottom")}
        f = frame[frame.level == "fold"]                 # its label sits below, joined by a faint leader
        ax.annotate(NAMES["fold"], xy=(f.z_r_rho.mean(), f.z_rho_mi.mean()), xytext=(1.02, 0.47), ha="right",
                    va="top", fontsize=7, color=ink("fold"), zorder=6,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6),
                    arrowprops=dict(arrowstyle="-", color=ink("fold"), lw=0.5, alpha=0.6, shrinkA=0, shrinkB=5))
        if show_all:
            labels["all three, as in (a)"] = (0.9, 0.66, "right", "bottom")
        for cond, (tx, ty, ha, va) in labels.items():
            ax.text(tx, ty, NAMES.get(cond, cond), ha=ha, va=va, fontsize=7, zorder=6,
                    color=ink(cond) if cond in COLOURS else INK,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6))
        lim, ticks = (0.2, 1.03), [0.25, 0.5, 0.75, 1]
        ax.set(xlim=lim, ylim=lim, xticks=ticks, yticks=ticks)
        ax.set_xticklabels([".25", ".5", ".75", "1"]); ax.set_yticklabels([".25", ".5", ".75", "1"])
        ax.set_xlabel(r"$z(r,\rho)$", labelpad=1, fontsize=7)
        ax.set_ylabel(r"$z(\rho,\mathrm{MI})$", labelpad=2, fontsize=7)
        # what lowers each feature in this system
        cue = dict(transform=ax.transAxes, fontsize=6, color=MUTED, zorder=6,
                   bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.6))   # dotted links pass beneath
        ax.text(0.04, 0.03, r"$\leftarrow$ monotone curvature", ha="left", va="bottom", **cue)
        ax.text(0.03, 0.12, r"$\leftarrow$ non-monotone", ha="left", va="bottom", rotation=90, **cue)
    else:
        rng = np.random.default_rng(0)
        series = ZSER[:2]
        for s_, (col, lab, colour, mk) in enumerate(series):
            dx = (s_ - 0.5) * 0.24
            for k, (cond, *_) in enumerate(CONDS):
                v = frame[frame.level == cond][col].to_numpy()
                ax.scatter(k + dx + rng.uniform(-0.05, 0.05, len(v)), v, s=4, lw=0, color=colour, alpha=0.6, zorder=2)
            means = [frame[frame.level == c][col].mean() for c, *_ in CONDS]
            ax.plot(np.arange(3) + dx, means, ls="", marker=mk, ms=4.5, mfc=colour, mec=INK, mew=0.5, zorder=3,
                    label=lab)
        ax.set(xlim=(-0.5, 2.5), ylim=(0.0, 1.06), xticks=range(3), yticks=[0, 0.5, 1])
        ax.set_xticklabels([c for c, *_ in CONDS], fontsize=7)
        for tick, (cond, *_) in zip(ax.get_xticklabels(), CONDS):
            tick.set_color(INK if cond == "linear" else COLOURS[cond])
        ax.set_yticklabels(["0", ".5", "1"])
        ax.set_ylabel("feature value $z$", labelpad=1, fontsize=7)
        ax.legend(loc="lower left", fontsize=6.5, handletextpad=0.2, labelspacing=0.25, frameon=False,
                  borderaxespad=0.2)
        ax.text(0.99, 0.02, f"{n} recordings each", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5,
                color=MUTED)
        for k, (cond, kind, key) in enumerate(CONDS):
            col = "z_" + "_".join(key)
            s_ = [c for c, *_ in series].index(col)
            v = float(frame[(frame.level == cond) & (frame.rep == rep)][col].iloc[0])
            shown[cond] = (k + (s_ - 0.5) * 0.24, v)
            ax.plot(*shown[cond], "o", ms=6.5, mfc="none", mec=ink(kind), mew=1.0, zorder=5)
    ax.tick_params(labelsize=6.5, length=2, pad=1.5)
    return ax, shown


DOTS = (0, (1, 1.6))


def link(fig, src, xy, dst, corners, colour):
    """Thin dotted lines from corners of a plane in c to its point in b, beneath every axes."""
    to_fig = fig.transFigure.inverted()
    p = to_fig.transform(src.transData.transform(xy))
    for corner in corners:
        c = to_fig.transform(dst.transAxes.transform(corner))
        fig.add_artist(plt.Line2D([p[0], c[0]], [p[1], c[1]], transform=fig.transFigure, color=colour, lw=0.6,
                                  ls=DOTS, alpha=0.6, zorder=-2))


def drop(fig, a, b, colour, side=False):
    """Dotted curve from a pair bar in a to its badge in c (inches), beneath every axes. It leaves a downwards and
    meets b from above, or from the left when ``side`` (a badge far to the right, under a silo heading)."""
    if side:
        ctrl = [(a[0], a[1] + 0.6 * (b[1] - a[1])), (a[0] + 0.4 * (b[0] - a[0]), b[1])]
    else:
        d = b[1] - a[1]                               # stay vertical long enough to clear the bars below
        ctrl = [(a[0], a[1] + 0.85 * d), (b[0], b[1] - 0.4 * d)]
    path = MPath([_fig(*q) for q in [a, *ctrl, b]], [MPath.MOVETO] + [MPath.CURVE4] * 3)
    fig.add_artist(PathPatch(path, transform=fig.transFigure, fill=False, ec=colour, lw=0.6, ls=DOTS, alpha=0.6,
                             zorder=-2))


def zbox(fig, ax, value):
    """Neutral bold z label in the plane's top-left corner, drawn at figure level so the pair leaders pass beneath.
    The value, not a colour, carries the reading (z hues are kept for the SPI matrices)."""
    bold = r"\boldsymbol" if plt.rcParams["text.usetex"] else r"\mathbf"
    return fig.text(0.04, 0.97, rf"${bold}{{z = {value:.2f}}}$", transform=ax.transAxes, ha="left", va="top",
                    fontsize=6.5, color=INK, zorder=8,
                    bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.12", fc="#F3F3F1", ec="0.6", lw=0.5))


def rail(fig, x, top, h, i, j, colour):
    """Miniature of the pair bar in a: one grey segment per block (gaps echo the heatmap's block lines), the
    pair's two rows dotted and joined."""
    y = lambda c: top + (c + 0.5) * h / M
    to = lambda a, b: _fig(a, b)
    gap = 0.012
    for k in range(M // B):
        a0, a1 = top + k * B * h / M + gap, top + (k + 1) * B * h / M - gap
        fig.add_artist(plt.Line2D(*zip(to(x, a0), to(x, a1)), transform=fig.transFigure, color="0.8", lw=0.6))
    fig.add_artist(plt.Line2D(*zip(to(x, y(i)), to(x, y(j))), transform=fig.transFigure, color=colour, lw=0.9,
                              solid_capstyle="round"))
    fig.add_artist(plt.Line2D(*zip(to(x, y(i)), to(x, y(j))), transform=fig.transFigure, ls="", marker="o",
                              ms=2.4, mew=0, color=colour))


def bars(fig, left, top, w, vals):
    bx = axes_in(fig, left, top, w, BAR_H)
    bx.bar(range(3), np.clip(vals, 0, None), width=0.7, color=HUES, edgecolor=[shade(h, 0.7) for h in HUES], lw=0.4)
    for q, v in enumerate(vals):
        bx.text(q, max(v, 0) + 0.04, "0" if abs(v) < 0.06 else f"{v:.2f}".replace("0.", "."), ha="center",
                va="bottom", fontsize=5)
    bx.set(ylim=(0, 1.06), xlim=(-0.55, 2.55), yticks=[])
    bx.set_xticks(range(3), ["$r$", r"$\rho$", r"$\mathrm{MI}$"])
    bx.tick_params(axis="x", length=0, pad=1, labelsize=5.5)
    for s_ in ("top", "right", "left"):
        bx.spines[s_].set_visible(False)
    bx.spines["bottom"].set_linewidth(0.4)


def condition_column(fig, pl, top, x, m, it, kind, key, pairs, nums):
    """Two pairs side by side (bars over scatter), then the condition's edge plane with both pairs ringed."""
    gs, ps = GS, PS
    a, b_ = key
    ax = axes_in(fig, pl, top + PY, ps, ps)
    rel = ordered(relation(it.types))
    va, vb = ordered(m[a]), ordered(m[b_])
    for k in ("linear", "monotone", "fold"):
        sel = rel == k
        ax.scatter(va[sel], vb[sel], s=20, lw=0.4, ec="white", color=PAIR_INK[k], zorder=3 if k != "linear" else 2)
    if b_ == "mi":
        g = np.linspace(0, 0.995, 200)
        ax.plot(g, -0.5 * np.log(1 - (2 * np.sin(np.pi * g / 6)) ** 2), color=MUTED, lw=0.6, ls=(0, (2.5, 1.5)))
        ax.text(0.5, 0.45, "Gaussian", transform=ax.transAxes, ha="left", fontsize=6, color=MUTED)
        ax.set(xlim=(-0.08, 1.02), ylim=(-0.06, 1.42), yticks=[0, 0.5, 1])
        ax.set_ylabel(r"$\mathrm{MI}_{ij}$ (nats)", labelpad=2, fontsize=7)
    else:
        ax.plot([-0.15, 1.05], [-0.15, 1.05], color=HAIR, lw=0.6, zorder=0)
        ax.set(xlim=(-0.12, 1.02), ylim=(-0.12, 1.02), yticks=[0, 0.5, 1])
        ax.set_ylabel(r"$\rho_{ij}$", labelpad=3, fontsize=7)
    ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", ".5", "1"]); ax.set_yticklabels(["0", ".5", "1"])
    ax.tick_params(labelsize=6, length=2, pad=1)
    ax.set_xlabel(r"$r_{ij}$" if a == "r" else r"$\rho_{ij}$", labelpad=0, fontsize=7)
    zbox(fig, ax, z(m[a], m[b_]))
    for k, ((i, j), num, word) in enumerate(zip(pairs, nums, ("strong", "weak"))):
        gl = pl + k * (ps - gs)
        vals = [m["r"][i, j], m["rho"][i, j], float(linfoot(m["mi"][i, j]))]
        bars(fig, gl, top + GY - 0.09 - BAR_H, gs, vals)
        ga = axes_in(fig, gl, top + GY, gs, gs)
        rail(fig, gl - RAIL, top + GY, gs, i, j, ink(kind))
        badge(fig, *badge_at(gl, top), num, ink(kind))
        tex, strong = plt.rcParams["text.usetex"], word == "strong"
        ga.text(0.05, 0.95, rf"\textbf{{{word}}}" if tex and strong else word, transform=ga.transAxes, ha="left",
                va="top", fontsize=6, color=ink(kind), zorder=5, fontweight="bold" if strong and not tex else "normal",
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))
        ga.scatter(x[:, i], x[:, j], s=0.2, lw=0, color=shade(ink(kind), 0.85) if kind != "linear" else ink(kind),
                   alpha=0.5, rasterized=True)
        ga.set(xlim=np.quantile(x[:, i], [0.005, 0.995]), ylim=np.quantile(x[:, j], [0.005, 0.995]))
        boxed(ga, ink(kind))
        ga.text(0.5, -0.03, f"$x_{{{i + 1}}}$, $x_{{{j + 1}}}$", transform=ga.transAxes, ha="center", va="top",
                fontsize=5.5, color=MUTED)
        xy = (m[a][i, j], m[b_][i, j])
        ax.plot(*xy, "o", ms=7.5 if k == 0 else 6.5, mfc="none", mec=ink(kind), mew=0.8, zorder=7)
        fig.add_artist(ConnectionPatch(xyA=(0.5, -0.12), coordsA=ga.transAxes, xyB=xy, coordsB=ax.transData,
                                       color="0.6", lw=0.4, zorder=6))
    return ax


RAIL, BADGE = 0.075, 0.09                 # rail offset left of a pair scatter; its badge above the rail top
GS, PS = 0.72, 1.60                       # c: pair scatter and plane sizes (inches)
BAR_H, GY, PY = 0.36, 0.53, 1.40          # c, from the column top: bar height, pair scatter top, plane top
SLOT = 0.36                               # c: room either side of a column's plots (ylabel left; centred)
SILO = (0.31, 0.31, 0.18, 0.26)           # drawn outline's padding around the plots: left, right, top, bottom
R, GAP = 7.12, 0.08                       # right edge of c's slots (the first starts at 0, with a); slot gap
PITCH = 2 * SLOT + PS + GAP
PL = R - SLOT - PS - 2 * PITCH            # first plane's left edge
A_TOP, A_HX, A_HW, A_HH = 0.30, 1.32, 1.40, 1.56   # a: top, heatmap left, width, height (inches)
B_TOP = 0.36
BS = A_TOP + 0.22 + A_HH - B_TOP          # b: square axes side; its bottom on the heatmap's, right on c's plane
CTOP = 2.66                               # c: column top (silo outline top at CTOP - SILO[2])
BX = R - SLOT - BS


# alternative colours: couplings avoid the heatmap's blue and orange; the statistics become one indigo ramp,
# light to dark in order of tolerance (r < rho < MI), so the bar colours read as the ladder itself
PALETTES = {"current": None,
            "alt": dict(monotone="#13867A", fold="#C23B6E", hues=("#C9CDEB", "#8E97D3", "#4C559E"))}


def use_palette(name):
    global HUES
    p = PALETTES[name]
    if p is None:
        return
    for d in (PAIR_INK, COLOURS):
        d.update(monotone=p["monotone"], fold=p["fold"])
    HUES = p["hues"]


BADGES = "top"                            # "top": badge above each sub-column; "rail": on top of each rail
TOP_DY = 0.13                             # extra column drop that the "top" badge row needs


def badge_at(gl, top):
    """Badge centre (inches) for the pair whose scatter's left edge is gl, in a column starting at top."""
    return (gl - RAIL, top + GY - BADGE) if BADGES == "rail" else (gl + GS / 2, top - 0.06)


def silo(fig, pl, top, colour, bottom):
    """Rounded outline around one condition's column; returns its top-left and top-right corners (inches)."""
    x0, x1 = pl - SILO[0], pl + PS + SILO[1]
    y0, y1 = top - SILO[2], bottom + PY + PS + SILO[3]
    fig.add_artist(FancyBboxPatch(_fig(x0, y1), (x1 - x0) / W, (y1 - y0) / H, transform=fig.transFigure,
                                  boxstyle="round,pad=0,rounding_size=0.012", fc="none", ec=colour, lw=0.5,
                                  zorder=-3))
    return (x0, y0), (x1, y0)


def wedge(fig, src, xy, corners, colour):
    """Dotted lines from a silo's top corners (inches) to its point in b, beneath every axes."""
    p = fig.transFigure.inverted().transform(src.transData.transform(xy))
    for c in corners:
        c = _fig(*c)
        fig.add_artist(plt.Line2D([p[0], c[0]], [p[1], c[1]], transform=fig.transFigure, color=colour, lw=0.6,
                                  ls=DOTS, alpha=0.6, zorder=-2))


DROPS = (3, 4)                            # pairs whose bar in a is joined to its badge in c


def build(live_text=False, panel_b="scatter", links="fold", badges="top"):
    global BADGES, H
    BADGES = badges
    dy = TOP_DY if badges == "top" else 0.0
    H = 6.05 + dy
    style(live_text)
    fig = plt.figure(figsize=(W, H))
    frame = pd.read_csv(OUT / "z_by_recording.csv")
    xm, itm = make(LEVELS["both"], REP)
    numbered, num = [], 1
    for cond, kind, _ in CONDS:
        for i, j in PAIRS[cond]:
            numbered.append((num, i, j, kind)); num += 1
    panel_title(fig, 0, 0.18, "a", "Dependency filters")
    ends = system(fig, A_TOP, xm, itm, numbered)
    panel_title(fig, BX - 0.42, 0.18, "b", "Feature variation")
    if panel_b == "scatter":
        bax, shown = result(fig, BX, B_TOP, BS, BS, frame, REP, "scatter")
    else:
        bax, shown = result(fig, 3.9, 0.6, 3.2, 1.55, frame, REP, "strip")
    panel_title(fig, 0, 2.40, "c", "Channel pairs")
    for t in fig.texts[-2:]:                  # the bar-to-rail curve passes beneath the title
        t.set_bbox(dict(fc="white", ec="none", pad=0.5))
    for k, (cond, kind, key) in enumerate(CONDS):
        x, it = make(LEVELS[cond], REP)
        m = stats(x)
        for i, j in PAIRS[cond]:
            assert np.allclose(x[:, [i, j]], xm[:, [i, j]]), "marked pair must be identical across recordings"
        pl = PL + k * PITCH
        name = NAMES.get(cond, cond)
        head = rf"\textbf{{{name}}}" if plt.rcParams["text.usetex"] else name
        fig.text((pl + PS / 2) / W, 1 - (CTOP - 0.05) / H, head, fontsize=7.5, color=ink(kind), ha="center",
                 transform=fig.transFigure, va="baseline", fontweight="bold")
        pax = condition_column(fig, pl, CTOP + dy, x, m, it, kind, key, PAIRS[cond], (2 * k + 1, 2 * k + 2))
        corners = silo(fig, pl, CTOP, tint(ink(kind), 0.55), CTOP + dy)
        if links == "all" or links == cond:
            wedge(fig, bax, shown[cond], corners, ink(kind))
        for j, num in enumerate((2 * k + 1, 2 * k + 2)):
            if links != "none" and num in DROPS:      # pair bar in a -> its badge in c
                bx_, hi = ends[num]
                bx, by = badge_at(pl + j * (PS - GS), CTOP + dy)
                side = bx - bx_ > 1.0
                drop(fig, (bx_, hi + 0.05), (bx - 0.07, by) if side else (bx, by - 0.07), ink(kind), side)
    fig.bax = bax
    return fig


def export_panels(out=OUT, stem="form_overview", **kw):
    """Save a, b and c as separate files (svg, pdf, png), built without the cross-panel links."""
    from matplotlib.transforms import Bbox
    fig = build(**{**kw, "links": "none"})
    a_right = A_HX + A_HW + 0.08 + 0.09 * 5 + 0.1
    regions = {"a": (0, 0, a_right, A_TOP + 0.22 + A_HH + 0.18), "b": (BX - 0.5, 0, R, B_TOP + BS + 0.38),
               "c": (0, 2.25, R, H)}
    paths = []
    for name, (x0, y0, x1, y1) in regions.items():
        box = Bbox.from_extents(x0, H - y1, x1, H - y0)        # inches from the bottom-left
        fig.bax.set_visible(name != "c")                       # b's axis label reaches into c's strip
        for ext in ("svg", "pdf", "png"):
            path = out / f"{stem}_panel-{name}.{ext}"
            fig.savefig(path, bbox_inches=box, dpi=300 if ext == "png" else None)
        paths.append(out / f"{stem}_panel-{name}.png")
    plt.close(fig)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live-text", action="store_true")
    ap.add_argument("--panel-b", choices=("scatter", "strip"), default="scatter")
    ap.add_argument("--links", choices=("all", "linear", "monotone", "fold", "none"), default="fold")
    ap.add_argument("--palette", choices=tuple(PALETTES), default="current")
    ap.add_argument("--badges", choices=("rail", "top"), default="top")
    ap.add_argument("--panels", action="store_true", help="also save a, b and c as separate files")
    args = ap.parse_args(argv)
    use_palette(args.palette)
    stem = "form_overview" + ("" if (args.panel_b, args.links) == ("scatter", "fold") else
                              f"_{args.panel_b}_{args.links}") + ("" if args.palette == "current" else
                                                                  f"_{args.palette}") + ("" if args.badges == "top" else "_badges-rail") + \
        ("_live-text" if args.live_text else "")
    print(save(build(args.live_text, args.panel_b, args.links, args.badges), OUT, stem))
    if args.panels:
        for path in export_panels(stem=stem, live_text=args.live_text, panel_b=args.panel_b, badges=args.badges):
            print(path)


if __name__ == "__main__":
    main()
