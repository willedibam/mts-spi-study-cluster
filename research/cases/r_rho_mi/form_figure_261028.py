"""Form case figure, draft 6 (2026-10-28): three rows that add one coupling at a time, and z-space.

Each row: one recording (a map glyph per block, strength key, a short heatmap strip, bars marking the two
extracted channel pairs) -> the two pairs (scatter beside r, rho, MI bars) -> that recording's edge plane.
Rows: all linear (r-rho), + monotone block (r-rho), + fold block (rho-MI). Right: every recording of the three
conditions in z-space, with the shown recordings joined a -> b -> c. Colour marks the pair's coupling.

The shown recording is replicate REP: among the 20, it is nearest the condition means in z (summed |dz| 0.05)
and its fold pair has a small positive r (fold pairs share a per-recording offset in r, -0.07 to +0.12 across
replicates; linear-monotone r ranges 0.22-0.67 because sinh is heavy-tailed). Pairs are picked by `pick`:
agreeing pairs with the highest (linear, strong) or ~0.5 (linear, weak) common value; the block 1 x block 2 pair
with the largest rho - r among pairs with rho, MI > .85 (present in rows b and c); the fold pair with the largest
MI and |r|, |rho| ~ 0, avoiding negative values.
MI is pyspi's `mi_kraskov_NN-4` (KSG estimator 1, k = 4), plotted in nats against the Gaussian reference
MI(rho); bars show it on the correlation scale (Linfoot).

    python -m research.cases.r_rho_mi.form_figure_261028 [--live-text]
"""
from __future__ import annotations

import argparse
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch, PathPatch
from matplotlib.path import Path as MPath
from seaborn import color_palette

from research.cases.case_run import ROOT, ordered, z
from research.cases.composite_style import (AMBER, HAIR, INK, LAVENDER, MUTED, SAGE, axes_in, panel_title, save,
                                            shade, style, z_label)
from research.cases.r_rho_mi.form_ladder_261028 import COLOURS, LEVELS, PAIR_INK, make, relation, stats

OUT = ROOT / "results/cases/form-ladder-261028"
HUES = (SAGE, LAVENDER, AMBER)
W, H = 7.2, 4.95
B, M, REP = 4, 12, 10
ROWS = [  # (condition, plane, head, wanted pairs)
    ("linear", ("r", "rho"), "All couplings linear", ("strong", "weak")),
    ("monotone", ("r", "rho"), r"Block 2 monotone, $\sinh 2w$", ("strong", "monotone")),
    ("both", ("rho", "mi"), r"Block 3 folded, $w^2$", ("monotone", "fold")),
]
MAPS = {"linear": lambda w: w, "monotone": lambda w: np.sinh(2 * w), "fold": lambda w: w ** 2}
# geometry (inches)
HX, HW, HH = 0.45, 0.95, 1.22                       # heatmap strip
GX, GS = 1.95, 0.5                                  # glyph scatter
PX, PS = 3.6, 1.2                                   # edge plane
ZX, ZS = 5.55, 1.5                                  # z-space


def linfoot(mi):
    return np.sqrt(1 - np.exp(-2 * np.clip(mi, 0, None)))


def ink(kind):
    return "0.45" if kind == "linear" else PAIR_INK[kind]


def pick(it, m, want):
    """Channel pair (i, j, kind) that shows `want` most cleanly in this recording."""
    t = it.types
    cand = []
    for i, j in itertools.combinations(range(M), 2):
        v = np.array([m["r"][i, j], m["rho"][i, j], float(linfoot(m["mi"][i, j]))])
        cand.append((i, j, {t[i], t[j]}, v))
    lin = [c for c in cand if c[2] == {"linear"}]
    if want == "strong":
        i, j, _, _ = max(lin, key=lambda c: c[3].mean() - 3 * np.ptp(c[3]))
        return i, j, "linear"
    if want == "weak":
        i, j, _, _ = min(lin, key=lambda c: abs(c[3].mean() - 0.5) + 3 * np.ptp(c[3]))
        return i, j, "linear"
    if want == "monotone":
        lm = [c for c in cand if c[0] < B <= c[1] < 2 * B and min(c[3][1:]) > 0.85]
        i, j, _, _ = max(lm, key=lambda c: c[3][1] - c[3][0])
        return i, j, "monotone"
    lf = [c for c in cand if c[2] == {"linear", "fold"}]
    i, j, _, _ = max(lf, key=lambda c: c[3][2] - 3 * abs(c[3][0]) - 3 * abs(c[3][1]) - 5 * (min(c[3][:2]) < -0.01))
    return i, j, "fold"


def channel_carpet(ax, x):
    """Channels x time, each channel scaled by its own 99th percentile of |x|."""
    y = x / np.percentile(np.abs(x), 99, axis=0)
    ax.imshow(y.T, aspect="auto", cmap=color_palette("icefire", as_cmap=True), vmin=-1, vmax=1,
              interpolation="nearest", rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5); s.set_color(INK)


def boxed(ax, colour=HAIR):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6); s.set_color(colour)


def _fig(x, y):
    return x / W, 1 - y / H


def curve(fig, a, b, colour, lw=0.8, halo=True):
    """Cubic curve with horizontal tangents from a to b (figure inches), over a white break."""
    d = (b[0] - a[0]) / 2
    path = MPath([_fig(*p) for p in [a, (a[0] + d, a[1]), (b[0] - d, b[1]), b]], [MPath.MOVETO] + [MPath.CURVE4] * 3)
    if halo:
        fig.add_artist(PathPatch(path, transform=fig.transFigure, fill=False, ec="white", lw=lw + 2.4))
    fig.add_artist(PathPatch(path, transform=fig.transFigure, fill=False, ec=colour, lw=lw))


def glyph(fig, left, top, x, i, j, m, kind):
    """Tick-free scatter of one channel pair with bars of r, rho and MI (correlation scale) beside it."""
    ga = axes_in(fig, left, top, GS, GS)
    ga.scatter(x[:, i], x[:, j], s=0.25, lw=0, color=shade(ink(kind), 0.85) if kind != "linear" else ink(kind),
               alpha=0.5, rasterized=True)
    ga.set(xlim=np.quantile(x[:, i], [0.005, 0.995]), ylim=np.quantile(x[:, j], [0.005, 0.995]))
    boxed(ga, ink(kind))
    ga.text(0.5, -0.04, f"$x_{{{i + 1}}}$, $x_{{{j + 1}}}$", transform=ga.transAxes, ha="center", va="top",
            fontsize=6, color=MUTED)
    bx = axes_in(fig, left + GS + 0.08, top + 0.07, 0.46, GS - 0.13)
    vals = [m["r"][i, j], m["rho"][i, j], float(linfoot(m["mi"][i, j]))]
    bx.bar(range(3), np.clip(vals, 0, None), width=0.7, color=HUES, edgecolor=[shade(h, 0.7) for h in HUES], lw=0.4)
    for q, v in enumerate(vals):
        bx.text(q, max(v, 0) + 0.04, "0" if abs(v) < 0.06 else f"{v:.2f}".replace("0.", "."), ha="center",
                va="bottom", fontsize=5)
    bx.set(ylim=(0, 1.12), xlim=(-0.55, 2.55), yticks=[])
    bx.set_xticks(range(3), ["$r$", r"$\rho$", "MI"])
    bx.tick_params(axis="x", length=0, pad=1, labelsize=5.5)
    for s in ("top", "right", "left"):
        bx.spines[s].set_visible(False)
    bx.spines["bottom"].set_linewidth(0.4)
    return bx


def recording(fig, y0, x, it, pairs):
    """Map glyph per block, strength key, heatmap strip and one bar per extracted pair; returns bar anchors."""
    t0, n = 300, 90
    cp = axes_in(fig, HX, y0, HW, HH)
    channel_carpet(cp, x[t0:t0 + n])
    for b in (B - 0.5, 2 * B - 0.5):
        cp.axhline(b, color="white", lw=0.8)
    wgrid = np.linspace(-2.2, 2.2, 100)
    for k in range(3):
        kind = it.types[k * B]
        ga = axes_in(fig, 0.02, y0 + k * HH / 3 + 0.06, 0.27, HH / 3 - 0.12)
        ga.plot(wgrid, MAPS[kind](wgrid), color=INK if kind == "linear" else COLOURS[kind], lw=0.9)
        ga.axis("off")
    sk = axes_in(fig, HX - 0.09, y0, 0.045, HH)
    sk.imshow((1 / np.sqrt(1 + it.noise_stds ** 2))[:, None], cmap="Greys", vmin=0.3, vmax=1.05, aspect="auto")
    boxed(sk, INK)
    row_y = lambda c: y0 + (c + 0.5) * HH / M
    anchors = []
    for k, (i, j, kind) in enumerate(pairs):
        bx_ = HX + HW + 0.09 + 0.12 * k
        lo, hi = row_y(min(i, j)), row_y(max(i, j))
        fig.add_artist(plt.Line2D(*zip(_fig(bx_, lo), _fig(bx_, hi)), transform=fig.transFigure, color=ink(kind),
                                  lw=0.9, solid_capstyle="round"))
        for yy in (lo, hi):
            fig.add_artist(plt.Line2D(*zip(_fig(HX + HW + 0.015, yy), _fig(bx_, yy)), transform=fig.transFigure,
                                      color=ink(kind), lw=0.5, alpha=0.6))
            fig.add_artist(plt.Line2D([_fig(bx_, yy)[0]], [_fig(bx_, yy)[1]], transform=fig.transFigure, ls="",
                                      marker="o", ms=2.6, mew=0, color=ink(kind)))
        anchors.append((bx_, (lo + hi) / 2))
    return anchors


def plane(fig, top, it, m, key):
    a, b_ = key
    ax = axes_in(fig, PX, top, PS, PS)
    rel = ordered(relation(it.types))
    va, vb = ordered(m[a]), ordered(m[b_])
    for kind in ("linear", "monotone", "fold"):
        sel = rel == kind
        ax.scatter(va[sel], vb[sel], s=4, lw=0, color=PAIR_INK[kind], zorder=3 if kind != "linear" else 2)
    if b_ == "mi":
        g = np.linspace(0, 0.995, 200)
        ax.plot(g, -0.5 * np.log(1 - (2 * np.sin(np.pi * g / 6)) ** 2), color=MUTED, lw=0.6,
                ls=(0, (2.5, 1.5)), zorder=1)
        ax.text(0.97, 0.58, "Gaussian", transform=ax.transAxes, ha="right", fontsize=6, color=MUTED)
        ax.set(xlim=(-0.08, 1.02), ylim=(-0.06, 1.42), yticks=[0, 0.5, 1])
        ax.set_ylabel(r"MI$_{ij}$ (nats)", labelpad=0, fontsize=7)
        zpos = dict(x=0.04, y=0.97, ha="left", va="top")
    else:
        ax.plot([-0.15, 1.05], [-0.15, 1.05], color=HAIR, lw=0.6, zorder=0)
        ax.set(xlim=(-0.12, 1.02), ylim=(-0.12, 1.02), yticks=[0, 0.5, 1])
        ax.set_ylabel(r"$\rho_{ij}$", labelpad=0, fontsize=7)
        zpos = dict(x=0.96, y=0.04, ha="right", va="bottom")
    ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", ".5", "1"])
    ax.set_yticklabels(["0", ".5", "1"])
    ax.tick_params(labelsize=6, length=2, pad=1)
    ax.set_xlabel(r"$r_{ij}$" if a == "r" else r"$\rho_{ij}$", labelpad=0, fontsize=7)
    z_label(ax, z(m[a], m[b_]), fontsize=6.5, **zpos)
    return ax


def zspace(fig, top, frame, shown):
    panel_title(fig, ZX - 0.42, top - 0.3, "d", "One point per recording")
    ax = axes_in(fig, ZX, top, ZS, ZS)
    for cond, *_ in ROWS:
        f = frame[frame.level == cond]
        ax.scatter(f.z_r_rho, f.z_rho_mi, s=7, lw=0, color=COLOURS[cond], alpha=0.55)
    pts = [shown[c] for c, *_ in ROWS]
    ax.plot(*zip(*pts), color=INK, lw=0.6, zorder=4)
    for letter, p in zip("abc", pts):
        ax.plot(*p, "o", ms=5, mfc="white", mec=INK, mew=0.7, zorder=5)
        ax.annotate(rf"\textbf{{{letter}}}" if plt.rcParams["text.usetex"] else letter, p, xytext=(-8, 4),
                    textcoords="offset points", fontsize=7, zorder=6,
                    fontweight="normal" if plt.rcParams["text.usetex"] else "bold")
    ax.set(xlim=(0.15, 1.05), ylim=(0.55, 1.0), xticks=[0.25, 0.5, 0.75, 1], yticks=[0.6, 0.8, 1])
    ax.set_xticklabels([".25", ".5", ".75", "1"]); ax.set_yticklabels([".6", ".8", "1"])
    ax.tick_params(labelsize=6, length=2, pad=1)
    ax.set_xlabel(r"$z(r,\rho)$", labelpad=0, fontsize=7)
    ax.set_ylabel(r"$z(\rho,\mathrm{MI})$", labelpad=0, fontsize=7)
    ax.text(0.03, 0.03, f"{frame.groupby('level').size().min()} recordings per row", transform=ax.transAxes,
            fontsize=5.5, color=MUTED)


def build(live_text=False):
    style(live_text)
    fig = plt.figure(figsize=(W, H))
    frame = pd.read_csv(OUT / "z_by_recording.csv")
    y0, rh = 0.1, 1.62
    shown = {}
    for r, (cond, key, head, wants) in enumerate(ROWS):
        x, it = make(LEVELS[cond], REP)
        m = stats(x)
        shown[cond] = (z(m["r"], m["rho"]), z(m["rho"], m["mi"]))
        pairs = [pick(it, m, w) for w in wants]
        top = y0 + r * rh
        panel_title(fig, 0, top + 0.1, "abc"[r], head)
        hy = top + 0.3
        anchors = recording(fig, hy, x, it, pairs)
        ax = plane(fig, hy - 0.04, it, m, key)
        a, b_ = key
        for k, ((i, j, kind), anchor) in enumerate(zip(pairs, anchors)):
            gtop = hy + (0.0 if k == 0 else HH - GS)
            bx = glyph(fig, GX, gtop, x, i, j, m, kind)
            curve(fig, anchor, (GX, gtop + GS / 2), ink(kind))
            xy = (m[a][i, j], m[b_][i, j])
            ax.plot(*xy, "o", ms=5, mfc="none", mec=ink(kind), mew=0.8, zorder=7)
            fig.add_artist(ConnectionPatch(xyA=(1.02, 0.5), coordsA=bx.transAxes, xyB=xy, coordsB=ax.transData,
                                           color="0.6", lw=0.4, zorder=6))
    zspace(fig, y0 + 0.42, frame, shown)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=3.4, mew=0, color=PAIR_INK[k]) for k in PAIR_INK]
    fig.legend(handles, ["both channels linear", "a monotone channel", "a fold channel"], loc="upper left",
               bbox_to_anchor=((ZX - 0.3) / W, 1 - (y0 + 0.42 + ZS + 0.45) / H), ncol=1, fontsize=6.5,
               handletextpad=0.1, labelspacing=0.3, frameon=False, title="pairs (b, c)", title_fontsize=6.5)
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live-text", action="store_true")
    args = ap.parse_args(argv)
    print(save(build(args.live_text), OUT, "form_figure" + ("_live-text" if args.live_text else "")))


if __name__ == "__main__":
    main()
