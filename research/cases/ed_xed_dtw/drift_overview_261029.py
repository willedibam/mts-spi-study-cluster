"""Drift case figure (2026-10-29): ED, xED and DTW on the no-lag / lag / warping block system.

a  Delay structure. One smooth mother s (inset) read by three blocks of five channels,
   x_k(t) = s(t - d_k(t)) + noise (the model, above the recording); per block, each channel's delay d_k(t) over
   one warp cycle (no lag: 0; lag: flat at 1-5 samples; warping: waves of amplitude
   2-10 samples), coloured by size as in c, with the SPIs that forgive it; then the all-three recording
   (greyscale). Numbered bars mark the pairs in c.
b  Feature variation. z(ED, xED) against z(xED, DTW) for every recording of the four conditions. Lag lowers only
   the first, warping only the second.
c  One silo per condition: the pairs with the smaller and larger (bold) delay difference, left to right as their
   points lie in the plane, each as its ED / xED / DTW bars (an olive ramp, light to dark in order of tolerance) over
   d_j(t) - d_i(t) against time (true offset coloured by size, DTW's path estimated from the noisy pair in grey),
   telescoped to their points in the condition's edge plane (equal axes, dashed identity, bold z top-left;
   pairs shaded by lag difference on viridis or warp-amplitude difference on magma, with their distribution
   inset). Each condition is drawn in a mid hue of its map, here and in b. The warp is smooth by design (a slow
   sinusoidal lag); DTW's steps are its discrete path. ``--glyph map`` draws t_warp against t_orig over 16
   samples; ``--palette``, ``--heatmap``, ``--spi-hues`` and ``--no-dtw-path`` give the earlier variants.

Data: `drift_blocks_261029.py` (20 recordings per condition, z_by_recording.csv); the shown recording is
replicate REP, whose pyspi matrices are cached next to it.

    python -m research.cases.ed_xed_dtw.drift_overview_261029
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, FancyBboxPatch

from research.cases.case_run import ROOT, ordered, z
from research.cases.composite_style import (AMBER, HAIR, INK, MUTED, ROSE, SAGE, axes_in, panel_title, save, shade,
                                            style, tint)
from research.cases.ed_xed_dtw.drift_blocks_261029 import LEVELS, PAIR_INK, make, relation, stats
from research.cases.r_rho_mi.form_figure_261028 import boxed, channel_carpet

OUT = ROOT / "results/cases/drift-blocks-261029"
REP, M, B = 0, 15, 5
W, H = 7.2, 6.18
SPI_HUES = {"grey": ("#D2D2D2", "#9A9A9A", "#575757"),   # each a ramp, light to dark in order of tolerance;
            "form": (SAGE, ROSE, AMBER),                # olive (default) avoids viridis (lag), magma (warping)
                                                        # and the form figure's amber
            "warm": ("#E3C79F", "#C27E4E", "#7E3F24"),
            "ochre": ("#EEDDA8", "#CFA040", "#86621A"),
            "olive": ("#DADBAE", "#9CA257", "#566024"),
            "copper": ("#F2C9A2", "#C9824B", "#7A4425"),
            "slate": ("#D3D8DE", "#8796A8", "#3F4D60")}
SPI = {k: (name, h) for (k, name), h in zip((("ed", "ED"), ("xed", "xED"), ("dtw", "DTW")), SPI_HUES["olive"])}
HEATMAP = "grey"                                       # or "icefire"
DTW_PATH = True                                        # draw DTW's estimated path in the c glyphs
KIND = {"no lag": "aligned", "lag": "lagged", "warping": "warped"}
# condition -> (plane axes, (smaller, larger) pairs, all against block-1 channel 0); left to right in the order
# of their points in the plane, so the leaders do not cross
CONDS = {"no lag": (("ed", "xed"), ((0, 2), (0, 1))), "lag": (("ed", "xed"), ((0, 5), (0, 9))),
         "warping": (("xed", "dtw"), ((0, 10), (0, 14)))}
FORGIVE = {"aligned": ("ed", "xed", "dtw"), "lagged": ("xed", "dtw"), "warped": ("dtw",)}
KIND_NAME = {"aligned": "no lag", "lagged": "lag, 1–5 samples", "warped": "warping, ±2–10 samples"}
A_TOP, A_HX, A_HW, A_HH = 0.30, 2.10, 1.45, 1.56     # a: top, heatmap left, width, height (inches)
GS, PS = 0.72, 1.60                                    # c: pair glyph and plane sizes (inches)
BAR_H, GY, PY = 0.36, 0.53, 1.40                       # c, from the column top: bar height, glyph top, plane top
CTOP, DY = 2.66, 0.13                                  # c: heading baseline region; column top = CTOP + DY
WINDOW = (117.0, 133.0)                                # time maps: 16 samples about a warp zero crossing (t = 125)
GLYPH = "deviation"                                    # "deviation": delay vs t; "map": t_warp vs t_orig
SPAN, DEV = (0.0, 250.0), 11.0                         # deviation glyphs: one warp cycle; y range +- samples
DMAX = {"lagged": 5.0, "warped": 10.0}                 # largest lag / warp-amplitude difference
R, SLOT, GAP = 7.12, 0.36, 0.08
PITCH = 2 * SLOT + PS + GAP
PL = R - SLOT - PS - 2 * PITCH
B_TOP = 0.36
BS = A_TOP + 0.22 + A_HH - B_TOP
BX = R - SLOT - BS
SILO = (0.31, 0.31, 0.18, 0.26)                       # outline padding: left, right, top, bottom
DOTS = (0, (1, 1.6))


def _fig(x, y):
    return x / W, 1 - y / H


# "tied": lag sizes on viridis and warp sizes on magma, each condition drawn in a mid hue of its own map, so a
# silo, its glyphs, its plane and its point in b share a colour family; "vivid": the case script's colours with
# viridis for both; "clean": muted blue / plum with per-condition ramps; "steel", "mono": alternatives
_V, _M = plt.get_cmap("viridis"), plt.get_cmap("magma")
PALETTES = {"tied": {"aligned": "#8A8F99", "lagged": to_hex(_V(0.38)), "warped": to_hex(_M(0.42)),
                     "shade": {"lagged": ("viridis", 0.05, 0.85), "warped": ("magma", 0.12, 0.82)}},
            "clean": {"aligned": "#8A8F99", "lagged": "#3E6FA8", "warped": "#7E57A0", "shade": None},
            "vivid": {"aligned": "#7B8190", "lagged": PAIR_INK["lagged"], "warped": PAIR_INK["warped"],
                      "shade": "viridis"},
            # earthier pair, less saturated than Okabe-Ito
            "steel": {"aligned": "#8A8F99", "lagged": "#3A6A96", "warped": "#A9583A", "shade": "viridis"},
            # conditions in greys only: colour is left to the SPIs (bars) and to lag / warp size (viridis)
            "mono": {"aligned": "#A3A7AE", "lagged": "#5F6673", "warped": "#2B303A", "shade": "viridis"}}
PALETTE = "tied"
TRUE_INK, PATH_INK = INK, "0.68"                       # c glyphs: true offset (line), DTW path (steps), neutral


def ink(kind):
    return PALETTES[PALETTE][kind]


def shade_map(kind):
    """Colour map for shading pairs by lag or warp-amplitude difference."""
    name = PALETTES[PALETTE]["shade"]
    if isinstance(name, dict):
        cmap, lo, hi = name[kind]
        return LinearSegmentedColormap.from_list(f"{cmap}_{kind}", plt.get_cmap(cmap)(np.linspace(lo, hi, 64)))
    if name:
        return plt.get_cmap(name)
    return LinearSegmentedColormap.from_list(kind, [tint(ink(kind), 0.8), ink(kind), shade(ink(kind), 0.5)])


def degree_colour(kind, d):
    """Colour of a lag or warp-amplitude difference d, as in the planes; grey when there is none."""
    return ink("aligned") if kind == "aligned" or d == 0 else shade_map(kind)(d / DMAX[kind])


def badge(fig, x, y, num, colour, size=5.5):
    fig.text(x / W, 1 - y / H, str(num), ha="center", va="center", fontsize=size, color="white",
             transform=fig.transFigure, bbox=dict(boxstyle="circle,pad=0.18", fc=colour, ec="none"))


def zbox(fig, ax, value):
    bold = r"\boldsymbol" if plt.rcParams["text.usetex"] else r"\mathbf"
    return fig.text(0.04, 0.97, rf"${bold}{{z = {value:.2f}}}$", transform=ax.transAxes, ha="left", va="top",
                    fontsize=6.5, color=INK, zorder=8,
                    bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.12", fc="#F3F3F1", ec="0.6", lw=0.5))


def recording(cond, rep=REP):
    """The shown recording of a condition and its ED / xED / DTW matrices (pyspi, cached)."""
    x, it = make(LEVELS[cond], rep)
    path = OUT / f"mats_rep{rep}_{cond.replace(' ', '-')}.npz"
    if not path.exists():
        np.savez_compressed(path, **stats(x))
    return x, it, dict(np.load(path))


def warped_time(it, c, t_orig):
    """When channel c reads the mother instant t_orig: solve t - delay_c(t) = t_orig (t - delay is increasing)."""
    t = np.arange(it.delay.shape[0], dtype=float)
    return np.interp(t_orig, t - it.delay[:, c], t)


def dev_axes(ax, colour):
    ax.axhline(0, color="0.6", lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set(xlim=SPAN, ylim=(-DEV, DEV))
    boxed(ax, colour)


def tmap_axes(ax, colour):
    ax.plot(WINDOW, WINDOW, color=HAIR, lw=0.6, zorder=1)
    ax.set(xlim=WINDOW, ylim=WINDOW, aspect="equal")
    boxed(ax, colour)


def system(fig, top, x, it, numbered):
    """Mother inset -> per-block time maps with the SPIs that forgive them -> the all-three heatmap."""
    hy, hh = top + 0.22, A_HH
    row = hh / M
    arrow = dict(transform=fig.transFigure, arrowstyle="-|>", mutation_scale=5, color="0.55", lw=0.5)
    gx, gs = 0.30, 0.40
    s_ = np.linspace(*WINDOW, 200)
    for k, kind in enumerate(("aligned", "lagged", "warped")):
        cy = hy + (k + 0.5) * hh / 3
        ga = axes_in(fig, gx, cy - gs / 2 + 0.03, gs, gs)
        if GLYPH == "map":
            tmap_axes(ga, ink(kind))
            for c in range(k * B, (k + 1) * B):
                ga.plot(s_, warped_time(it, c, s_), color=ink(kind), lw=0.55, alpha=0.9, zorder=2)
        else:
            dev_axes(ga, ink(kind))
            so = np.linspace(*SPAN, 500)
            for c in range(k * B, (k + 1) * B):
                d = it.warps[c] if kind == "warped" else it.lags[c]
                ga.plot(so, warped_time(it, c, so) - so, color=degree_colour(kind, d), lw=0.8, zorder=2)
        ga.text(-0.08, 0.5, KIND_NAME[kind].split(",")[0], transform=ga.transAxes, ha="right", va="center",
                fontsize=6, color=ink(kind), rotation=90)
        if kind != "aligned":
            ga.text(0.5, -0.04, KIND_NAME[kind].split(", ")[1], transform=ga.transAxes, ha="center", va="top",
                    fontsize=5, color=MUTED)
        tx = gx + gs + 0.16
        for j, s in enumerate(("ed", "xed", "dtw")):
            ok = s in FORGIVE[kind]
            fig.text(tx / W, 1 - (cy - 0.10 + 0.11 * j) / H, SPI[s][0], fontsize=5.5, ha="left", va="center",
                     transform=fig.transFigure, color=INK if ok else "0.82",
                     fontweight="bold" if ok and not plt.rcParams["text.usetex"] else "normal")
        fig.add_artist(FancyArrowPatch(_fig(tx + 0.36, cy), _fig(A_HX - 0.05, cy), **arrow))
    head = r"$t_\mathrm{warp}$ vs $t_\mathrm{orig}$" if GLYPH == "map" else r"delay $d_k(t)$"
    fig.text((gx + gs / 2) / W, 1 - (hy - 0.05) / H, head, ha="center",
             va="baseline", fontsize=5.5, color=MUTED, transform=fig.transFigure)
    fig.text((gx + gs + 0.16) / W, 1 - (hy - 0.05) / H, "forgiven by", ha="left", va="baseline", fontsize=5.5,
             color=MUTED, transform=fig.transFigure)
    ins = axes_in(fig, 0.12, top - 0.07, 0.9, 0.15)          # the shared mother signal
    ins.plot(it.mother[::8][300:420], color=INK, lw=0.55)
    ins.axis("off")
    ins.text(1.02, 0.5, "mother $s$", transform=ins.transAxes, ha="left", va="center", fontsize=6, color=MUTED)
    ex = (A_HX + A_HW / 2) / W                                # the model, over the recording it generates
    fig.text(ex, 1 - (top + 0.005) / H, r"$x_k(t) = s\big(t - d_k(t)\big) + \varepsilon_k(t)$", ha="center",
             va="center", fontsize=7, color=INK, transform=fig.transFigure)
    fig.text(ex, 1 - (top + 0.15) / H, r"$d_k = 0$,  $\tau_k$  or  $A_k \sin(2\pi t/250)$", ha="center",
             va="center", fontsize=6, color=MUTED, transform=fig.transFigure)
    cp = axes_in(fig, A_HX, hy, A_HW, hh)
    if HEATMAP == "grey":
        y = x[0:150] / np.percentile(np.abs(x[0:150]), 99, axis=0)
        cp.imshow(y.T, aspect="auto", cmap="Greys_r", vmin=-1, vmax=1, interpolation="nearest", rasterized=True)
        cp.set_xticks([]); cp.set_yticks([])
        for s_ in cp.spines.values():
            s_.set_linewidth(0.5); s_.set_color(INK)
    else:
        channel_carpet(cp, x[0:150])
    for b in (B - 0.5, 2 * B - 0.5):
        cp.axhline(b, color="white", lw=0.8)
    cp.text(0.5, -0.03, "time", transform=cp.transAxes, ha="center", va="top", fontsize=6, color=MUTED)
    row_y = lambda c: hy + (c + 0.5) * row
    for k, (num, i, j, kind) in enumerate(numbered):
        bx_ = A_HX + A_HW + 0.08 + 0.09 * k
        lo, hi = row_y(min(i, j)), row_y(max(i, j))
        fig.add_artist(plt.Line2D(*zip(_fig(bx_, lo), _fig(bx_, hi)), transform=fig.transFigure, color=ink(kind),
                                  lw=0.9, solid_capstyle="round"))
        fig.add_artist(plt.Line2D([_fig(bx_, lo)[0], _fig(bx_, hi)[0]], [_fig(bx_, lo)[1], _fig(bx_, hi)[1]],
                                  transform=fig.transFigure, ls="", marker="o", ms=2.8, mew=0, color=ink(kind)))
        badge(fig, bx_, hy - 0.1, num, ink(kind))


def result(fig, left, top, side, frame):
    """z(ED, xED) against z(xED, DTW), one point per recording; returns the axes and each condition's shown point."""
    ax = axes_in(fig, left, top, side, side)
    ax.set_facecolor("none")
    colours = {c: ink(k) for c, k in KIND.items()} | {"all three": "0.55"}
    shown = {}
    for cond, col in colours.items():
        f = frame[frame.level == cond]
        mk = "D" if cond == "all three" else "o"
        ax.scatter(f.z_ed_xed, f.z_xed_dtw, s=14 if mk == "D" else 17, lw=0.4, ec="white", color=col, alpha=0.85,
                   marker=mk, zorder=2)
        f0 = f[f.rep == REP].iloc[0]
        shown[cond] = (float(f0.z_ed_xed), float(f0.z_xed_dtw))
        ax.plot(*shown[cond], mk, ms=6.5 if mk == "D" else 7, mfc="none", mec=col if mk == "o" else INK, mew=1.0,
                zorder=5)
    box = dict(fc="white", ec="none", alpha=0.85, pad=0.6)
    for cond, (tx, ty, ha, va) in {"no lag": (1.0, 0.95, "right", "top"), "lag": (0.27, 0.95, "left", "top"),
                                   "warping": (1.0, 0.49, "right", "bottom"),
                                   "all three": (0.62, 0.36, "center", "top")}.items():
        ax.text(tx, ty, "all three, as in (a)" if cond == "all three" else cond, ha=ha, va=va, fontsize=7,
                color=INK if cond == "all three" else colours[cond], zorder=6, bbox=box)
    lim, ticks = (0.15, 1.04), [0.25, 0.5, 0.75, 1]
    ax.set(xlim=lim, ylim=lim, xticks=ticks, yticks=ticks)
    ax.set_xticklabels([".25", ".5", ".75", "1"]); ax.set_yticklabels([".25", ".5", ".75", "1"])
    ax.set_xlabel(r"$z(\mathrm{ED},\mathrm{xED})$", labelpad=1, fontsize=7)
    ax.set_ylabel(r"$z(\mathrm{xED},\mathrm{DTW})$", labelpad=2, fontsize=7)
    cue = dict(transform=ax.transAxes, fontsize=6, color=MUTED, zorder=6, bbox=dict(fc="white", ec="none",
                                                                                      alpha=0.9, pad=0.6))
    ax.text(0.04, 0.03, r"$\leftarrow$ lag", ha="left", va="bottom", **cue)
    ax.text(0.03, 0.14, r"$\leftarrow$ warping", ha="left", va="bottom", rotation=90, **cue)
    ax.tick_params(labelsize=6.5, length=2, pad=1.5)
    return ax, shown


def bars(fig, left, top, vals):
    """ED, xED and DTW of one pair as bars (RMS distances of z-scored series: tall = the SPI is not fooled)."""
    bx = axes_in(fig, left, top, GS, BAR_H)
    hues = [SPI[s][1] for s in ("ed", "xed", "dtw")]
    bx.bar(range(3), vals, width=0.7, color=hues, edgecolor=[shade(h, 0.7) for h in hues], lw=0.4)
    for q, v in enumerate(vals):
        bx.text(q, v + 0.05, f"{v:.2f}".lstrip("0"), ha="center", va="bottom", fontsize=5)
    bx.set(ylim=(0, 1.45), xlim=(-0.55, 2.55), yticks=[])
    bx.set_xticks(range(3), [SPI[s][0] for s in ("ed", "xed", "dtw")])
    bx.tick_params(axis="x", length=0, pad=1, labelsize=5.5)
    for s_ in ("top", "right", "left"):
        bx.spines[s_].set_visible(False)
    bx.spines["bottom"].set_linewidth(0.4)


def degree(it, kind, i, j):
    """The pair's delay difference: lag difference, or warp-amplitude difference (samples)."""
    return abs(it.warps[i] - it.warps[j]) if kind == "warped" else abs(it.lags[i] - it.lags[j])


def dtw_path(a, b):
    """DTW's warping path between two series: matched sample indices (i, j), a staircase."""
    from dtaidistance import dtw
    return np.array(dtw.warping_path(np.ascontiguousarray(a, float), np.ascontiguousarray(b, float)))


def time_map(fig, left, top, x, it, i, j, kind, tag, bold, label=False):
    """Time in x_j against time in x_i at which both read the same instant of the mother (line), with DTW's
    path estimated from the noisy pair (steps), over WINDOW."""
    ga = axes_in(fig, left, top, GS, GS)
    path = dtw_path(x[:, i], x[:, j])
    if GLYPH == "map":
        tmap_axes(ga, ink(kind))
        s_ = np.linspace(*WINDOW, 200)
        keep = (path[:, 0] >= WINDOW[0] - 3) & (path[:, 0] <= WINDOW[1] + 3)
        ga.plot(*path[keep].T, color=PATH_INK, lw=0.6, zorder=2)
        ga.plot(warped_time(it, i, s_), warped_time(it, j, s_), color=TRUE_INK, lw=1.0, zorder=3,
                solid_capstyle="round")
    else:                                     # offset of x_j from x_i: true (line) and DTW's path (steps)
        dev_axes(ga, ink(kind))
        so = np.linspace(*SPAN, 500)
        ti = warped_time(it, i, so)
        keep = (path[:, 0] >= SPAN[0]) & (path[:, 0] <= SPAN[1])
        if DTW_PATH:
            ga.plot(path[keep, 0], path[keep, 1] - path[keep, 0], color=PATH_INK, lw=0.5, zorder=2)
        ga.plot(ti, warped_time(it, j, so) - ti, color=degree_colour(kind, degree(it, kind, i, j)), lw=1.2,
                zorder=3, solid_capstyle="round")
        if label:
            ga.set_yticks([-10, 0, 10]); ga.set_yticklabels(["−10", "0", "10"])
            ga.tick_params(axis="y", labelsize=5, length=1.5, pad=1, colors=MUTED)
            ga.set_ylabel(r"$d_j - d_i$", fontsize=5.5, color=MUTED)
            ga.yaxis.set_label_coords(-0.15, 0.5)            # tucked beside the narrow "0" tick, not past "−10"
    tex = plt.rcParams["text.usetex"]
    ga.text(0.05, 0.95, rf"\textbf{{{tag}}}" if tex and bold else tag, transform=ga.transAxes, ha="left", va="top",
            fontsize=6, color=ink(kind), zorder=5, fontweight="bold" if bold and not tex else "normal",
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))
    ga.text(0.5, -0.03, f"$x_{{{i + 1}}}$, $x_{{{j + 1}}}$", transform=ga.transAxes, ha="center", va="top",
            fontsize=5.5, color=MUTED)
    return ga


def column(fig, pl, cond, nums):
    """Two pairs side by side (bars over time map), then the condition's edge plane with both pairs ringed."""
    (a, b), pairs = CONDS[cond]
    kind, top = KIND[cond], CTOP + DY
    x, it, mats = recording(cond)
    ax = axes_in(fig, pl, top + PY, PS, PS)
    rel = ordered(relation(it.types))
    va, vb = ordered(mats[a]), ordered(mats[b])
    d = ordered(np.abs(it.warps[:, None] - it.warps[None]) if kind == "warped"
                else np.abs(it.lags[:, None] - it.lags[None]))
    plain = rel == "aligned"
    ax.scatter(va[plain], vb[plain], s=20, lw=0.4, ec="white", color=PAIR_INK["aligned"], zorder=2)
    if kind != "aligned":
        dmax = DMAX[kind]
        o = np.argsort(d[~plain])
        ax.scatter(va[~plain][o], vb[~plain][o], s=20, lw=0.4, ec="white", c=d[~plain][o], cmap=shade_map(kind),
                   vmin=0, vmax=dmax, zorder=3)
        # how the delayed pairs are spread over lag or warp-amplitude difference, in the same colours (pairs
        # with none are the grey points)
        hist = ax.inset_axes((0.06, 0.62, 0.30, 0.18))
        vals, counts = np.unique(np.round(d[~plain], 6), return_counts=True)
        step = np.diff(vals).min()
        hist.bar(vals, counts, width=0.75 * step, color=[degree_colour(kind, v) for v in vals], lw=0)
        hist.set_xticks([vals[0], dmax]); hist.set_xticklabels([f"{vals[0]:g}", f"{dmax:g}"]); hist.set_yticks([])
        hist.tick_params(axis="x", labelsize=5, length=1.5, pad=1, colors=MUTED)
        for side in ("top", "right", "left"):
            hist.spines[side].set_visible(False)
        hist.spines["bottom"].set_linewidth(0.4); hist.spines["bottom"].set_color(MUTED)
        hist.patch.set_alpha(0)
        sym = r"$|\Delta A|$" if kind == "warped" else r"$|\Delta\tau|$"
        hist.text(1.08, 0.0, sym, transform=hist.transAxes, ha="left", va="center", fontsize=5.5, color=MUTED)
    lim, ticks = (0.1, 1.25), [0.25, 0.5, 0.75, 1]       # equal axes: the identity runs at 45 degrees
    ax.plot(lim, lim, color="0.6", lw=0.6, ls=(0, (3, 2)), zorder=0)
    ax.set(xlim=lim, ylim=lim, xticks=ticks, yticks=ticks, aspect="equal")
    ax.set_xticklabels([".25", ".5", ".75", "1"]); ax.set_yticklabels([".25", ".5", ".75", "1"])
    ax.tick_params(labelsize=6, length=2, pad=1)
    ax.set_xlabel(SPI[a][0], labelpad=1, fontsize=7)
    ax.set_ylabel(SPI[b][0], labelpad=2, fontsize=7)
    zbox(fig, ax, z(mats[a], mats[b]))
    for k, ((i, j), num) in enumerate(zip(pairs, nums)):
        gl = pl + k * (PS - GS)
        dd = degree(it, kind, i, j)
        tag = (rf"$\pm${dd:g}" if kind == "warped" else rf"$\Delta\tau = {dd:g}$")
        bars(fig, gl, top + GY - 0.09 - BAR_H, [mats[s][i, j] for s in ("ed", "xed", "dtw")])
        ga = time_map(fig, gl, top + GY, x, it, i, j, kind, tag, bold=(k == 1 and kind != "aligned"), label=(k == 0))
        badge(fig, gl + GS / 2, top - 0.06, num, ink(kind))
        xy = (mats[a][i, j], mats[b][i, j])
        ax.plot(*xy, "o", ms=7.5 if k == 1 else 6.5, mfc="none", mec=ink(kind), mew=0.8, zorder=7)
        fig.add_artist(ConnectionPatch(xyA=(0.5, -0.12), coordsA=ga.transAxes, xyB=xy, coordsB=ax.transData,
                                       color="0.6", lw=0.4, zorder=6))
    return ax


def silo(fig, pl, colour):
    x0, x1 = pl - SILO[0], pl + PS + SILO[1]
    y0, y1 = CTOP - SILO[2], CTOP + DY + PY + PS + SILO[3]
    fig.add_artist(FancyBboxPatch(_fig(x0, y1), (x1 - x0) / W, (y1 - y0) / H, transform=fig.transFigure,
                                  boxstyle="round,pad=0,rounding_size=0.012", fc="none", ec=colour, lw=0.5,
                                  zorder=-3))
    return (x0, y0), (x1, y0)


def wedge(fig, src, xy, corners, colour):
    p = fig.transFigure.inverted().transform(src.transData.transform(xy))
    for c in corners:
        c = _fig(*c)
        fig.add_artist(plt.Line2D([p[0], c[0]], [p[1], c[1]], transform=fig.transFigure, color=colour, lw=0.6,
                                  ls=DOTS, alpha=0.6, zorder=-2))


def build(live_text=False, link="warping"):
    style(live_text)
    fig = plt.figure(figsize=(W, H))
    frame = pd.read_csv(OUT / "z_by_recording.csv")
    xm, itm, _ = recording("all three")
    numbered, num = [], 1
    for cond, (_, pairs) in CONDS.items():
        for i, j in pairs:
            numbered.append((num, i, j, KIND[cond])); num += 1
    panel_title(fig, 0, 0.18, "a", "Delay structure")
    system(fig, A_TOP, xm, itm, numbered)
    panel_title(fig, BX - 0.42, 0.18, "b", "Feature variation")
    bax, shown = result(fig, BX, B_TOP, BS, frame)
    panel_title(fig, 0, 2.40, "c", "Channel pairs")
    for k, cond in enumerate(CONDS):
        x, _, _ = recording(cond)
        assert np.allclose(x[:, 0], xm[:, 0]), "block-1 channels must be identical across recordings"
        pl = PL + k * PITCH
        head = rf"\textbf{{{cond}}}" if plt.rcParams["text.usetex"] else cond
        fig.text((pl + PS / 2) / W, 1 - (CTOP - 0.05) / H, head, fontsize=7.5, color=ink(KIND[cond]), ha="center",
                 transform=fig.transFigure, va="baseline", fontweight="bold")
        column(fig, pl, cond, (2 * k + 1, 2 * k + 2))
        corners = silo(fig, pl, tint(ink(KIND[cond]), 0.55))
        if link == cond:
            wedge(fig, bax, shown[cond], corners, ink(KIND[cond]))
    return fig


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live-text", action="store_true")
    ap.add_argument("--link", choices=(*CONDS, "none"), default="warping")
    ap.add_argument("--glyph", choices=("deviation", "map"), default="deviation")
    ap.add_argument("--palette", choices=tuple(PALETTES), default="tied")
    ap.add_argument("--heatmap", choices=("grey", "icefire"), default="grey")
    ap.add_argument("--spi-hues", choices=tuple(SPI_HUES), default="olive")
    ap.add_argument("--no-dtw-path", action="store_true")
    ap.add_argument("--stem", default=None, help="output name (default: built from the options)")
    args = ap.parse_args(argv)
    global GLYPH, PALETTE, HEATMAP, DTW_PATH
    GLYPH, PALETTE, HEATMAP, DTW_PATH = args.glyph, args.palette, args.heatmap, not args.no_dtw_path
    for (k, name), h in zip((("ed", "ED"), ("xed", "xED"), ("dtw", "DTW")), SPI_HUES[args.spi_hues]):
        SPI[k] = (name, h)
    stem = ("drift_overview" + ("" if GLYPH == "deviation" else "_map") + ("" if PALETTE == "tied" else f"_{PALETTE}")
            + ("_live-text" if args.live_text else ""))
    print(save(build(args.live_text, args.link), OUT, args.stem or stem))


if __name__ == "__main__":
    main()
