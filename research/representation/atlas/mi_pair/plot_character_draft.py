"""Draft redesign of fig:mi (v14): what was measured, the pattern effect in Δz, and its replication.

    python plot_character_draft.py <results/representation/atlas-mi-pair-261117> <output stem>

Reads the checksum-listed zns arrays (macaque), the mouse induction z arrays, the human propofol (PRIOS) per-patient
contrasts from the scorer summary, and the few raw windows it draws (figwin/windows.npz, copied from the Gadi banks by
fig_windows.py). (b) is, per state, the window whose Δz is nearest its state's median (PANEL_B), on common axes:
ketamine + medetomidine large (the only drug in all four macaques), wake, sleep, ketamine and propofol small;
medetomidine alone, the weakest, is left out. (a) is one real pair, GLYPH_PAIR: of the figwin windows' off-band
pairs, the one with the largest banked K - G among those where no sample of either channel lies beyond 5 robust SDs
(the pair of largest K - G overall, 78-90 in the same window, owes most of it to two brief transients: K - G .67,
.27 with them cut). Selected, not typical. Its normal-scored samples are drawn beside a synthetic Gaussian of the same
correlation; G and K are the banked values, the Gaussian's are computed on 4000 samples. Above (c), four of its dots drawn as their raw windows (stacked traces of every 4th
electrode): the median wake, sleep and propofol windows and the most extreme window, the propofol one being (b)'s.
(d) shares (c)'s axis, so subject and window effects compare directly. The segment-surrogate timescale panel of v11 is dropped (checkpoint fc81155e).
(a)'s brain is the provider's photograph of Chibi's left hemisphere with the real electrode positions (Neurotycho
ChibiMap.mat, from the 20110621 KTMD download, saved as figwin/brain_Chibi.npz); (d)'s icons are described in icon().

Presentation quantities not in the manuscript tables: the window residual (z_rec - z_surr minus its session's
eyes-closed wake mean) in units of the eyes-open wake window SD (as review/sleepmatch_ns.py), an exact rescaling,
drawn as the second axis of (c); and the recording pairs outside the surrogate's 95% band around the recording's own
G-K line, a count that a level shift of K cannot move. Its reference is the wake window's count, not 5% of the
pairs: the median wake window already has 36 of 496 outside.
"""
from argparse import ArgumentParser
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.path import Path as MPath
from matplotlib.transforms import blended_transform_factory
from matplotlib.patches import ConnectionPatch, Ellipse, Patch, PathPatch, Rectangle
import numpy as np
import pandas as pd
from scipy.interpolate import splev, splprep
from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening, gaussian_filter
from scipy.stats import norm, rankdata, t as tdist, ttest_1samp

from mtsspi.figures import figure_style, raincloud
from edges import ksg
from plot_character import read

DRUGS = ("KT", "KTMD", "MD", "PF")
STATES = ("awake_eo", "SLP") + DRUGS
NAME = {"awake_eo": "Wake", "SLP": "Sleep", "KT": "Ketamine", "KTMD": "Ket.\\,+\\,med.", "MD": "Medet.",
        "PF": "Propofol"}
SHORT = {"awake_eo": "Wake", "SLP": "Sleep", "KT": "Ket.", "KTMD": "Ket.\n+med.", "MD": "Medet.", "PF": "Prop."}
COLOR = {"awake_eo": "#8C969C", "awake_ec": "#8C969C", "SLP": "#3E5C76", "KT": "#C8553D", "KTMD": "#7B4B94",
         "MD": "#2E86AB", "PF": "#D9A21B", "ink": "#2B2B2B"}
MARK = {"George": "o", "Chibi": "s", "Kin2": "^", "Su": "D"}
GREY, LIGHT = "#78838A", "#D9DDE0"
MATCH = (.65, .88)                                       # delta-fraction overlap of drug and sleep windows
Q = 95                                                   # surrogate band
CALLOUT = (("20120726mdanesthesiaandsleepgeorgetoruyanagawa:awake_eo:03", "median"),     # dots of (c) drawn as
           ("20120711slpanesthesiaandsleepgeorgetoruyanagawa:sleep:06", "median"),       # their raw windows
           ("20110621ktmdanesthesiaandsleepchibitoruyanagawa:drug:00", "most extreme"),
           ("20120730pfanesthesiaandsleepchibitoruyanagawa:drug:08", "median"),
           ("20120719ktanesthesiaandsleepchibitoruyanagawa:drug:03", "median"),
           ("20120726mdanesthesiaandsleepgeorgetoruyanagawa:drug:05", "median"))
PANEL_B = (("KTMD", "20110114ktmdanesthesiaandsleepgeorgetoruyanagawa:drug:08", "Ketamine + medetomidine"),   # each
           ("awake_eo", CALLOUT[0][0], "Wake"), ("SLP", CALLOUT[1][0], "Sleep"),                        # state's
           ("KT", CALLOUT[4][0], "Ketamine"),                 # median-Δz
           ("PF", CALLOUT[3][0], "Propofol"))                                                         # window
GLYPH_PAIR = ("20110621ktmdanesthesiaandsleepchibitoruyanagawa:drug:02", 407)        # (a): largest K - G among
                                                         # the figwin windows' pairs that no transient touches


# ---------------------------------------------------------------- data
def tint(d):
    h = tdist.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean() - h, d.mean() + h


def bint(d, n=10000):
    m = np.random.default_rng(0).choice(d, (n, len(d))).mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def mouse_induction(root):
    """Per-mouse Δ_ind, as claar/claar_ind_sum.py (frozen Addendum A rule)."""
    A, I = np.load(root / "claar/claar_z_ns.npz"), np.load(root / "claar/claar_z_ind.npz")
    ra, ri = json.loads(str(A["rows"])), json.loads(str(I["rows"]))
    resA, resI = A["ns"][:, 0] - A["ns"][:, 1:].mean(1), I["ns"][:, 0] - I["ns"][:, 1:].mean(1)
    g = lambda R, k: np.array([r[k] for r in R])
    sA, stA, biA, dA = g(ra, "session"), g(ra, "state"), g(ra, "burst"), g(ra, "delta_fraction")
    sI, biI, dI = g(ri, "session"), g(ri, "burst"), g(ri, "delta_fraction")
    out = []
    for s in dict.fromkeys(sI):
        if s == "sub-569073_ses-20210416":
            continue
        a, i = (sA == s) & (stA == "awake"), sI == s
        t = i & (biI >= np.percentile(biA[a], 5)) & (dI >= np.percentile(dA[a], 95))
        if t.sum() >= 3:
            out.append(resI[t].mean() - resA[a].mean())
    return np.array(out)


def human_propofol(root, variant="prios-sfull"):
    """Per-patient Δ from the archived scorer summary (window arrays are on gdata, not local)."""
    m = re.search(r"\[car \(primary\)\] subject Δ \[([^\]]*)\]", (root / "human" / variant / "summary.txt").read_text())
    return np.array([float(x) for x in m.group(1).split()])


def band(edges, wid):
    """The recording's own G-K line, the half-width holding Q% of the surrogate's pairs around the surrogate's own
    line, and the recording pairs outside it. Centring on each own line keeps a level shift of K (to which z is
    blind) out of the count."""
    (g, k), (gs, ks) = edges[wid]
    cs = np.polyfit(gs, ks, 1); c = np.polyfit(g, k, 1)
    lim = np.percentile(np.abs(ks - np.polyval(cs, gs)), Q)
    return c, lim, np.abs(k - np.polyval(c, g)) > lim


def load(root):
    df, _ = read(root / "zns")
    edges, delta = {}, []
    for p in sorted((root / "zns").glob("zns_[AB]_*.npz")):
        with np.load(p) as d:
            edges.update(zip(d["id"], d["E"][:, 1])); delta.append(d["delta_fraction"])
    df["delta"] = np.concatenate(delta)
    df["group"] = np.where(df.state == "drug", df.agent, np.where(df.state == "sleep", "SLP", df.state))
    ref = df[df.state == "awake_ec"].groupby("unit").residual.mean()
    df["d"] = df.residual - df.unit.map(ref)                # window residual minus its session's eyes-closed wake
    df["sd"] = df.d / df[df.state == "awake_eo"].d.std(ddof=1)
    df["u_rec"], df["u_sur"] = 1 - df.real ** 2, 1 - df.null ** 2
    ex = 100 * (df.u_rec - df.u_sur)                     # extra % of K unexplained, recording over surrogate
    df["exd"] = ex - df.unit.map(ex[df.state == "awake_ec"].groupby(df.unit).mean())
    lv = pd.Series([np.mean(edges[w][0][1] - edges[w][1][1]) for w in df.id], index=df.index)
    df["lvl"] = lv - df.unit.map(lv[df.state == "awake_ec"].groupby(df.unit).mean())   # mean K over surrogate K
    return df, edges


# ---------------------------------------------------------------- glyphs
def normal_score(v):
    return norm.ppf(rankdata(v) / (len(v) + 1))


def gauss(r, n=4000, seed=3):
    """Normal-scored Gaussian sample whose correlation is exactly r."""
    rng = np.random.default_rng(seed); z = rng.standard_normal((n, 2)); z -= z.mean(0)
    u = z @ np.linalg.inv(np.linalg.cholesky(np.cov(z.T))).T @ np.linalg.cholesky([[1, r], [r, 1]]).T
    return np.column_stack([normal_score(u[:, 0]), normal_score(u[:, 1])])


def glyph(ax, u, color):
    """Joint samples (normal-scored), with the 2.45-SD ellipse of their correlation: the shape G assumes."""
    x, y = u.T
    ax.scatter(x, y, s=1.5, lw=0, color=color, alpha=.5, rasterized=True, zorder=2)
    r = np.corrcoef(x, y)[0, 1]; tt = np.linspace(0, 2 * np.pi, 200)
    e = np.linalg.cholesky([[1, r], [r, 1]]) @ np.vstack([np.cos(tt), np.sin(tt)]) * 2.45
    ax.plot(e[0], e[1], color=COLOR["ink"], lw=.55, ls=(0, (3, 1.6)), zorder=3)
    ax.set(xlim=(-3.4, 3.4), ylim=(-3.4, 3.4), aspect="equal", xticks=[], yticks=[])
    for sp in ax.spines.values():
        sp.set_visible(False)


def window_strip(ax, X, color, every=4):
    """A 10-s window as stacked traces of every 4th of its 32 electrodes, each standardised."""
    Z = (X - X.mean(0)) / X.std(0); t = np.arange(len(Z)) / 100
    for j, m in enumerate(range(0, Z.shape[1], every)):
        ax.plot(t, Z[:, m] * .28 - j, color=color, lw=.3, rasterized=True)
    ax.set(xlim=(0, 10), ylim=(-Z.shape[1] / every + .2, 1.2), xticks=[], yticks=[]); frame(ax, LIGHT, .6)


def frame(ax, color, lw):
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(color); s.set_linewidth(lw)


def panel_title(target, letter, concept, rest, sub=False):
    s = rf"\textbf{{{letter}}}\enspace\textbf{{{concept}}}" + (f": {rest}" if rest else "")
    (target.suptitle(s, x=0, ha="left", fontsize=8.5) if sub else target.set_title(s, loc="left", fontsize=8.5, pad=5))


def states_axis(ax):
    ax.set_xticks(range(len(STATES)), [SHORT[s] for s in STATES], fontsize=6.3)
    for tick, s in zip(ax.get_xticklabels(), STATES):
        tick.set_color(COLOR[s])
    ax.set_xlim(-.6, len(STATES) - .4); ax.tick_params(axis="x", length=0)


# ---------------------------------------------------------------- icons (cartoon brains, lateral view, front at right)
SHAPE = {"macaque": [(1, .36), (.9, .6), (.62, .74), (.3, .7), (.06, .52), (0, .32), (.12, .16), (.38, .13),
                     (.55, .2), (.66, .12), (.85, .14), (.97, .22)],
         "human": [(1, .4), (.9, .68), (.62, .82), (.3, .78), (.06, .6), (0, .36), (.1, .2), (.35, .16), (.52, .24),
                   (.64, .12), (.84, .14), (.97, .26)],
         "mouse": [(1, .3), (.93, .42), (.8, .5), (.5, .56), (.2, .52), (.04, .4), (0, .28), (.1, .18), (.45, .14),
                   (.8, .16), (.95, .2)]}
SULCI = {"macaque": [[(.7, .3), (.55, .36), (.38, .42)], [(.55, .72), (.5, .55), (.46, .4)]],
         "human": [[(.68, .3), (.52, .38), (.36, .44)], [(.56, .8), (.52, .6), (.47, .42)],
                   [(.78, .7), (.72, .55), (.8, .42)], [(.28, .7), (.22, .55), (.3, .4)]],
         "mouse": []}


def brain(ax, kind, grid, pick=None, pick_color=None, dot=3.5, fc="#EEF0F2"):
    """Cartoon brain with electrodes on a grid (clipped to the outline); pick joins two electrodes by an arc."""
    q = np.array(SHAPE[kind] + [SHAPE[kind][0]])
    tck, _ = splprep(q.T, s=0, per=True)
    xy = np.array(splev(np.linspace(0, 1, 200), tck)).T
    kw = {"fc": fc, "ec": GREY, "lw": .7}
    ax.add_patch(PathPatch(MPath(xy), zorder=1, **kw))
    if kind == "mouse":
        ax.add_patch(Ellipse((1.04, .3), .14, .12, zorder=0, **kw)); ax.add_patch(Ellipse((.02, .27), .14, .2, zorder=0, **kw))
    else:
        ax.add_patch(Ellipse((.16, .1), .26, .14, angle=-10, zorder=0, **kw))
    t = np.linspace(0, 1, 30)
    for sul in SULCI[kind]:
        c = np.polyfit([0, .5, 1], np.array(sul), 2)
        ax.plot(np.polyval(c[:, 0], t), np.polyval(c[:, 1], t), color=GREY, lw=.45, zorder=2)
    X, Y = np.meshgrid(np.linspace(*grid[0:2], grid[4]), np.linspace(*grid[2:4], grid[5]))
    pts = np.c_[X.ravel(), Y.ravel()]
    pts = pts[MPath(xy).contains_points(pts)]
    ax.scatter(*pts.T, s=dot, color=COLOR["ink"], lw=0, zorder=3)
    if pick is not None:
        a, b = pts[pick[0]], pts[pick[1]]
        ax.scatter(*np.array([a, b]).T, s=4 * dot, color=pick_color, ec="white", lw=.5, zorder=5)
        m = (a + b) / 2 + (0, .14); tt = t[:, None]
        ax.plot(*((1 - tt) ** 2 * a + 2 * (1 - tt) * tt * m + tt ** 2 * b).T, color=pick_color, lw=1, zorder=4)
    ax.set(xlim=(-.1, 1.12), ylim=(-.02, .86 if kind != "mouse" else .62), aspect="equal"); ax.axis("off")


def real_brain(ax, path, used, pick, pick_color):
    """The provider's lateral photograph of the animal's left hemisphere (Neurotycho <animal>Map.mat: image I, electrode
    X, Y in its pixels, channel i at index i - 1), cropped to the lateral view, greyed; the window's 32 electrodes in
    ink, the other lateral ones faint, the pair in pick_color."""
    with np.load(path) as d:
        I, X, Y = d["I"], d["X"], d["Y"]
    ys = np.sort(Y); lat = Y > ys[np.diff(ys).argmax()]                # the photo's lower view is the lateral one
    top, bot = int(Y[lat].min() - 70), int(min(Y[lat].max() + 150, I.shape[0]))
    left, right = int(max(X[lat].min() - 60, 0)), int(min(X[lat].max() + 50, I.shape[1]))
    g = I[top:bot, left:right].mean(2) / 255
    g = np.ma.masked_where(g > .86, g)                                  # the photo's pale background, made transparent
    ax.imshow(g, cmap="gray", vmin=-.35, vmax=1.15, extent=(left, right, bot, top), interpolation="lanczos", zorder=0)
    on = np.isin(np.arange(1, 129), used)
    ax.scatter(X[lat & ~on], Y[lat & ~on], s=1.2, color="white", lw=0, alpha=.8, zorder=2)
    ax.scatter(X[on], Y[on], s=3, color=COLOR["ink"], ec="white", lw=.25, zorder=3)
    a, b = (np.array([X[e - 1], Y[e - 1]]) for e in pick)
    ax.plot(*np.c_[a, b], color=pick_color, lw=1.2, zorder=4, solid_capstyle="round")
    ax.scatter(*np.c_[a, b], s=14, color=pick_color, ec="white", lw=.5, zorder=5)
    ax.set(xlim=(left, right), ylim=(bot, top), aspect="equal"); ax.axis("off")


VIEW = {"human": ((0, -1, 0), (0, 0, 1), (-1, 0, 0)), "macaque": ((0, -1, 0), (0, 0, 1), (-1, 0, 0)),    # lateral, front left
        "mouse": ((0, 0, 1), (-1, 0, 0), (0, -1, 0))}                                                  # dorsal, front up


def icon(ax, name, root, base=(.9, .91, .92)):
    """(d)'s brains, shaded template meshes (figwin/assets.npz, from fig_assets.py), electrodes as dots.
    human: fsaverage5 left pial, lateral, with every grid and strip electrode of the 7 PRIOS patients (MNI305, BIDS
    electrodes.tsv), both hemispheres projected onto one side. macaque: NMT v2.0 left gray surface, lateral, with Chibi's
    107 lateral electrodes moved from Neurotycho's photograph by a bounding-box fit (approximate; the few that land off
    the template are moved to its nearest vertex). mouse: Allen CCF 2017 whole-brain mesh, dorsal; the 30-electrode
    array is schematic (a grid over dorsal cortex), not the recorded positions."""
    with np.load(root / "figwin/assets.npz") as A:
        v, f = A[name + "_v"].astype(float), A[name + "_f"]
    P = v @ np.array(VIEW[name], float).T; T = P[f]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    L = np.array([-.35, .55, 1.]); sh = np.abs(n @ (L / np.linalg.norm(L)))   # two-sided Lambert, light upper left
    o = np.argsort(T[:, :, 2].mean(1)); c = np.clip(np.outer(.42 + .58 * sh[o], base), 0, 1)
    ax.add_collection(PolyCollection(T[o][:, :, :2], fc=c, ec=c, lw=.12, rasterized=True, zorder=1))
    lo, hi = P[:, :2].min(0), P[:, :2].max(0)
    if name == "human":
        E = np.load(root / "figwin/prios_electrodes_mni305.npy")[:, [1, 2]] * (-1, 1)
    elif name == "macaque":
        with np.load(root / "figwin/brain_Chibi.npz") as d:
            I, X, Y = d["I"], d["X"], d["Y"]
        ys = np.sort(Y); lat = Y > ys[np.diff(ys).argmax()]
        top = int(Y[lat].min() - 70); g = I[top:, :].mean(2)[::3, ::3] / 255
        rr, cc = np.nonzero(gaussian_filter(binary_fill_holes(binary_opening((g < .86) & (g > .02), iterations=2))
                                            .astype(float), 2) > .5)
        E = np.c_[lo[0] + (X[lat] - 3 * cc.min()) / (3 * np.ptp(cc)) * (hi - lo)[0],
                  hi[1] - (Y[lat] - top - 3 * rr.min()) / (3 * np.ptp(rr)) * (hi - lo)[1]]
    else:
        mid = P[:, 0].mean()
        gx, gy = np.meshgrid(mid + np.r_[-np.linspace(700, 4000, 3), np.linspace(700, 4000, 3)],
                             -np.linspace(3000, 8800, 5))
        E = np.c_[gx.ravel(), gy.ravel()]
    H, ex, ey = np.histogram2d(*P[:, :2].T, bins=160)                                # template's projected outline
    inside = binary_fill_holes(binary_closing(H > 0, iterations=3, border_value=0))
    ax.contour((ex[1:] + ex[:-1]) / 2, (ey[1:] + ey[:-1]) / 2, gaussian_filter(inside.astype(float), 1).T, [.5],
               colors=[GREY], linewidths=.3, zorder=2)
    ix = np.clip(np.searchsorted(ex, E[:, 0]) - 1, 0, 159); iy = np.clip(np.searchsorted(ey, E[:, 1]) - 1, 0, 159)
    off = ~inside[ix, iy]
    if off.any():
        E[off] = P[((P[None, :, :2] - E[off, None]) ** 2).sum(2).argmin(1), :2]
    ax.scatter(*E.T, s=.16 if name == "human" else .45, color=COLOR["ink"], lw=0, zorder=3)
    pad = .03 * (hi - lo).max()
    ax.set(xlim=(lo[0] - pad, hi[0] + pad), ylim=(lo[1] - pad, hi[1] + pad))
    ax.set_aspect("equal"); ax.axis("off")


def silhouette(ax, name, root):
    """PhyloPic silhouette (CC0; artists in fig_assets.py), facing left."""
    with np.load(root / "figwin/assets.npz") as A:
        V, C = A[name + "_sv"].astype(float), A[name + "_sc"]
    if name != "human":
        V[:, 0] *= -1
    ax.add_patch(PathPatch(MPath(V, C), fc=COLOR["ink"], ec="none"))
    ax.set(xlim=(V[:, 0].min(), V[:, 0].max()), ylim=(V[:, 1].min(), V[:, 1].max()))
    ax.set_aspect("equal"); ax.axis("off")


ICON = {"macaque": ((.22, .9, .24, .66, 7, 5), r"Macaque" + "\n" + "ECoG"), "mouse": ((.25, .85, .26, .46, 4, 2), "Mouse\nEEG"),
        "human": ((.45, .92, .28, .62, 5, 4), "Human\nECoG")}


# ---------------------------------------------------------------- figure
def draw(root, output):
    df, edges = load(root)
    chan = {}                                                # electrode numbers of each window's 32 zns channels
    for p in (root / "zns").glob("zns_[AB]_*.npz"):
        with np.load(p) as d:
            chan.update(zip(d["id"], d["chan"]))
    sess = df.groupby(["unit", "animal", "group"])[["d", "sd", "exd", "lvl", "delta"]].mean().reset_index()
    animal = sess[sess.group.isin(DRUGS)].groupby("animal").d.mean()
    t = ttest_1samp(animal.values, 0, alternative="less")

    figure_style()
    plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.3, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
                         "legend.fontsize": 6.5, "axes.linewidth": .7, "xtick.major.width": .7,
                         "ytick.major.width": .7, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
    FH = 6.8
    fig = plt.figure(figsize=(7.2, FH), layout="constrained")
    r1, r2 = fig.subfigures(2, 1, height_ratios=[3.44, FH - 3.44], hspace=.03)
    rng = np.random.default_rng(0)
    W, H = 7.2, 3.44                                         # row 1 size (in), for square axes

    # a: what we measured: two electrodes, their 10-s traces, and the two MI estimates of their joint samples
    def put(rect):
        ax = r1.add_axes(rect); ax.set_in_layout(False); return ax
    pc = COLOR["KTMD"]
    wid, pk = GLYPH_PAIR
    e1, e2 = (int(v[pk]) for v in np.triu_indices(32, 1))
    with np.load(root / "figwin/windows.npz") as fw:
        lab = list(fw[wid + "|chan"]); X = fw[wid][:, [lab.index(chan[wid][e]) for e in (e1, e2)]].astype(float)
    (gw, kw), (_, ksur) = edges[wid]
    real_brain(put([.02, .66, .21, .25]), root / "figwin/brain_Chibi.npz", [int(v) for v in chan[wid]],
               (int(chan[wid][e1]), int(chan[wid][e2])), pc)
    r1.text(.125, .665, "32 electrodes (5 medial, not shown), 496 pairs", ha="center", va="top", fontsize=5.8, color=GREY)
    tr = put([.02, .465, .21, .14])
    Xc = X - np.median(X, 0); Xc = Xc / np.abs(Xc).max(0) * .55          # whole 10 s, transients included
    for i_ in range(2):
        tr.plot(Xc[:, i_] + (1.1 - 1.1 * i_), color=pc, lw=.4)
        tr.text(-15, 1.1 - 1.1 * i_, rf"el.\,{chan[wid][(e1, e2)[i_]]}", ha="right", va="center", fontsize=5.4,
                color=pc)
    tr.text(1005, 1.8, r"ket.\,+\,med., 10 s", ha="right", va="bottom", fontsize=5.4, color=pc)
    tr.plot([800, 1000], [-.95] * 2, color=COLOR["ink"], lw=.8)
    tr.text(900, -1.05, "2 s", ha="center", va="top", fontsize=5.4)
    tr.set(xlim=(-30, 1005), ylim=(-1.2, 1.9)); tr.axis("off")
    u = np.column_stack([normal_score(X[:, 0]), normal_score(X[:, 1])])
    ug = gauss(np.corrcoef(u.T)[0, 1])
    Gg, Kg = -.5 * np.log(1 - np.corrcoef(ug.T)[0, 1] ** 2), ksg(ug[:, 0], ug[:, 1], 0)
    f2 = lambda v: f"{v:.2f}".lstrip("0")
    for x_, v, col, name, G_, K_ in ((.0, ug[np.random.default_rng(1).choice(len(ug), 1000, replace=False)],
                                      COLOR["ink"], "Gaussian, same $r$", Gg, Kg),
                                     (.13, u, pc, "this pair", gw[pk], kw[pk])):
        glyph(put([x_ + .01, .1, .1, .1 * W / H]), v, col)
        r1.text(x_ + .06, .095, name, ha="center", va="top", fontsize=5.6, color=col)
        kk = rf"\textbf{{{f2(K_)}}}" if col == pc else f2(K_)
        r1.text(x_ + .06, .06, rf"$G$ {f2(G_)}\quad $K$ " + kk, ha="center", va="top", fontsize=5.8, color=col)
    r1.text(.19, .02, f"its surrogate: $K$ {f2(ksur[pk])}", ha="center", va="top", fontsize=5, color=GREY)
    r1.text(.125, .44, r"$G$: MI if the pair were Gaussian" + "\n" + r"$K$: MI of any shape (kNN)", ha="center",
            va="top", fontsize=5.6, linespacing=1.25)
    print("a's pair:", wid, "electrodes", chan[wid][e1], chan[wid][e2], "banked G, K", gw[pk], kw[pk], "surrogate K",
          ksur[pk], "Gaussian G, K", Gg, Kg)

    # b: every pair of one window per state, each the state's median-Δz window; the largest drug effect drawn big
    full = 1.05 * max(v.max() for v in edges[PANEL_B[0][1]][0])
    view = (-.02, .42), (-.17, .52)                      # common to all five, so the windows compare directly
    f3 = lambda v: f"{v:.3f}".lstrip("0")
    m_ = .13; rects = [[.335, .11, .3, .3 * W / H]] + [[.7 + (q % 2) * (m_ + .035), .11 + (1 - q // 2) * (m_ * W / H + .075),
                                                       m_, m_ * W / H] for q in range(4)]
    for j, ((s_, wid, name), rect) in enumerate(zip(PANEL_B, rects)):
        ax = put(rect); big = j == 0; row = df[df.id == wid].iloc[0]
        ink = COLOR["ink"] if s_ == "awake_eo" else COLOR[s_]
        (g, k), _ = edges[wid]
        c, w, out = band(edges, wid)
        xx = np.linspace(-.05, 1.1 * full, 100)
        panes = ((ax, 1), (ax.inset_axes([.03, .6, .36, .37]), .35)) if big else ((ax, .5),)
        for a_, sc in panes:
            a_.fill_between(xx, np.polyval(c, xx) - w, np.polyval(c, xx) + w, color=LIGHT, lw=0, zorder=0)
            a_.plot(xx, np.polyval(c, xx), color="white", lw=.8 * sc + .2, zorder=1)
            a_.scatter(g[~out], k[~out], s=6 * sc, lw=.25 * sc ** .5, ec="white", color=GREY, alpha=.6, rasterized=True,
                       zorder=2)
            a_.scatter(g[out], k[out], s=13 * sc, lw=.45 * sc ** .5, ec="white", color=ink, zorder=3)
        ax.set(xlim=view[0], ylim=view[1])
        ax.set_xticks([0, .2, .4], ["0", ".2", ".4"]); ax.set_yticks([0, .2, .4], ["0", ".2", ".4"])
        if big:
            ov = panes[1][0]
            ov.set(xlim=(-.04, full), ylim=(-.17, full), xticks=[], yticks=[]); frame(ov, GREY, .6)
            ov.add_patch(Rectangle((view[0][0], view[1][0]), np.diff(view[0])[0], np.diff(view[1])[0], fill=False,
                                   ec=COLOR["ink"], lw=.6, zorder=4))
            ov.text(.95, .06, "all pairs", transform=ov.transAxes, ha="right", fontsize=5.2, color=GREY)
            ax.set_title(name, fontsize=8, color=ink, pad=2, fontweight="bold")
            ax.text(.97, .135, rf"$z$ {f3(row.real)}, its surrogate {f3(row.null)}", transform=ax.transAxes, ha="right",
                    fontsize=5.8, color=GREY)
            ax.text(.97, .03, rf"$\Delta z$ \textbf{{{row.sd:.1f}}} wake SDs".replace("-", "--"), transform=ax.transAxes,
                    ha="right", fontsize=10.5, color=ink)
            ax.set_xlabel(r"Gaussian $\mathrm{MI}$, $G$", fontsize=6.8, labelpad=1)
            ax.set_ylabel(r"kNN $\mathrm{MI}$, $K$", fontsize=6.8, labelpad=1)
        else:
            ax.tick_params(labelsize=5.4, pad=1.2, labelleft=j % 2 == 1, labelbottom=j > 2)
            ax.set_title(name, fontsize=6.8, color=ink, pad=1.5, fontweight="bold")
            ax.text(.96, .05, rf"$\Delta z$ \textbf{{{row.sd:+.1f}}} SD".replace("-", "--"), transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8.8, color=ink)
    dot = lambda c_, ms: Line2D([], [], ls="", marker="o", ms=ms, color=c_)
    r1.legend([Patch(color=LIGHT), dot(GREY, 2.5), tuple(dot(COLOR[c_], 3) for c_ in ("ink", "SLP", "KTMD", "KT", "PF"))],
              [r"surrogate's 95\% scatter about the line", "pair", "pair off the band"], loc="center",
              bbox_to_anchor=(.66, .895), ncol=3, frameon=False, fontsize=6, handlelength=3.2, handletextpad=.5,
              columnspacing=1.4, handler_map={tuple: HandlerTuple(ndivide=None, pad=.15)})
    r1.text(.66, .85, r"$z$: correlation of $G$ and $K$ across the 496 pairs; the surrogate keeps every $G$ and is "
            r"Gaussian, so only $K$ moves", ha="center", va="center", fontsize=5.9, color=GREY)
    for x_, l_, c_, r_ in ((.0, "a", "One pair", ""), (.29, "b", "Every pair", "one 10-s window per state, each its state's median")):
        r1.text(x_, .975, rf"\textbf{{{l_}}}\enspace\textbf{{{c_}}}" + (f": {r_}" if r_ else ""), fontsize=8.5,
                va="top")

    # c-d: the pattern effect and its replication
    left = r2
    gs = left.add_gridspec(2, 2, height_ratios=[.22, 1], width_ratios=[2.25, .75], hspace=.04, wspace=.05)
    ax_c = left.add_subplot(gs[1, 0]); ax_f = left.add_subplot(gs[1, 1], sharey=ax_c)
    above = blended_transform_factory(ax_c.transData, ax_c.transAxes)   # strips sit over their own state column
    strips = [ax_c.inset_axes([STATES.index(df.loc[df.id == wid, "group"].iloc[0]) - .45, 1.07, .9, .16],
                              transform=above) for wid, _ in CALLOUT]
    for sax in strips:
        sax.set_in_layout(False)                            # row 0 holds them; the layout must not shrink (c)
    head_d = left.add_subplot(gs[0, 1]); head_d.axis("off")
    pos = {}                                                 # window id -> its dot in (c)
    win = df[df.group.isin(STATES)]
    sdw = df[df.state == "awake_eo"].d.std(ddof=1)          # eyes-open wake window SD: the second axis is exact
    top = .031
    num = lambda v, f: "0" if round(v, 3) == 0 else (f"{v:{f}}".replace("0.", ".") if f.endswith("3f") else f"{v:{f}}")
    for i, s in enumerate(STATES):
        w = win[win.group == s]; c = COLOR[s]; drug = s in DRUGS; ink = COLOR["ink"] if drug else GREY
        n0 = len(ax_c.collections)
        raincloud(ax_c, w.d.values, i, c, rng, width=.8, box=True, point_sizes=1.5, point_kw={"alpha": .35})
        dots = next(c_ for c_ in ax_c.collections[n0:] if isinstance(c_, PathCollection))
        pos.update(zip(w.id, map(tuple, dots.get_offsets())))
        m3 = num(w.d.mean(), ".3f")
        m3 = m3.replace("-", "--")
        mask = dict(fc="white", ec="none", pad=.4)
        ax_c.text(i, top, rf"\textbf{{{m3}}}" if drug else m3, ha="center", va="top", fontsize=6.6, color=ink,
                  bbox=mask)
        ax_c.text(i, top - .0047, f"{w.sd.mean():.1f}".replace("-", "--"), ha="center", va="top", fontsize=5.8,
                  color=ink, bbox=mask)
    for y_, lab in ((top, r"mean $\Delta z$"), (top - .0047, "wake SDs")):
        ax_c.text(-.72, y_, lab, ha="right", va="top", fontsize=6.3, color=GREY, clip_on=False)
    ax_c.axhline(0, color=GREY, lw=.6, ls="--", zorder=0)
    ax_c.set_ylim(-.074, top + .001); ax_c.set_yticks([-.07, -.06, -.05, -.04, -.03, -.02, -.01, 0],
                                                      ["$-.07$", "", "$-.05$", "", "$-.03$", "", "$-.01$", "0"])
    ax_c.set_ylabel(r"$\Delta z$, each 10-s window", y=.42)
    panel_title(left, "c", "Pattern", r"$z$ falls under every anaesthetic, not in sleep", sub=True)
    states_axis(ax_c)
    assert {w_ for _, w_, _ in PANEL_B[1:]} <= {w_ for w_, _ in CALLOUT}       # (b)'s small windows are drawn above (c)
    with np.load(root / "figwin/windows.npz") as fw:        # raw windows of the called-out dots (fig_windows.py)
        for sax, (wid, note) in zip(strips, CALLOUT):
            r = df[df.id == wid].iloc[0]; s = r.group; lab = list(fw[wid + "|chan"])
            X = fw[wid][:, [lab.index(v) for v in chan[wid]]].astype(float)
            window_strip(sax, X, COLOR[s])
            sax.set_title(rf"\textbf{{{SHORT[s].replace(chr(10), '')}}}" + "\n" + note, fontsize=5.6,
                          color=COLOR[s], pad=1.5, linespacing=1.1)
            sax.set_xlabel(f"{r.sd:+.1f} SD".replace("-", "--"), fontsize=5.6, labelpad=1.5)
            x_, y_ = pos[wid]
            ax_c.scatter(x_, y_, s=16, facecolor="none", edgecolor=COLOR["ink"], lw=.6, zorder=4)
            con = ConnectionPatch((x_, top - .0105), (x_, y_), coordsA=ax_c.transData, coordsB=ax_c.transData,
                                  color=GREY, lw=.45, shrinkB=2.2, zorder=0, clip_on=False)
            con.set_in_layout(False); ax_c.add_artist(con)
        strips[0].set_ylabel("8 el.\n10 s", fontsize=5, color=GREY, labelpad=1.5, linespacing=1.05)

    sets = [("macaque", animal.values, tint), ("mouse", mouse_induction(root), bint),
            ("human", human_propofol(root), tint)]
    for i, (name, d, ci) in enumerate(sets):
        if i == 0:
            for an, v in animal.items():
                ax_f.scatter(i - .14 + rng.uniform(-.05, .05), v, marker=MARK[an], s=13, facecolor="white",
                             edgecolor=COLOR["ink"], lw=.8, zorder=3)
        else:
            ax_f.scatter(i - .14 + rng.uniform(-.07, .07, len(d)), d, s=11, facecolor="white",
                         edgecolor=COLOR["ink"], lw=.8, zorder=3)
        lo, hi = ci(d)
        ax_f.plot([i + .16] * 2, [lo, hi], color=COLOR["ink"], lw=1.2)
        ax_f.scatter(i + .16, d.mean(), s=20, color=COLOR["ink"], zorder=3)
        ax_f.text(i, top, rf"\textbf{{{num(d.mean(), '.3f').replace('-', '--')}}}", ha="center", va="top",
                  fontsize=6.6, color=COLOR["ink"])
        ax_f.text(i, top - .0047, f"{d.mean() / sdw:.1f}".replace("-", "--"), ha="center", va="top", fontsize=5.8,
                  color=COLOR["ink"])
        ax_f.text(i, -.036, rf"\textbf{{{(d < 0).sum()}/{len(d)}}}" + "\nbelow 0", ha="center", va="top",
                  fontsize=5.6, linespacing=1.1)
        bw, bh = {"mouse": (.5, .026), "macaque": (.84, .024), "human": (.84, .026)}[name]
        icon(ax_f.inset_axes([i - bw / 2, -.0715, bw, bh], transform=ax_f.transData), name, root)
        sw = {"mouse": .48, "macaque": .56, "human": .26}[name]           # silhouettes as column heads, above the axes
        sil = ax_f.inset_axes([i - sw / 2, 1.04, sw, .125], transform=blended_transform_factory(ax_f.transData, ax_f.transAxes))
        sil.set_anchor("S"); silhouette(sil, name, root)
    ax_f.text(0, -.0245, f"t(3) = {t.statistic:.1f}".replace("-", "--") + "\n" + f"p = {t.pvalue:.3f}".replace("0.", "."),
              ha="center", va="top", fontsize=5.4, color=GREY, linespacing=1.15)
    ax_f.axhline(0, color=GREY, lw=.6, ls="--", zorder=0)
    ax_f.set_xticks(range(3), [ICON[n][1] for n, _, _ in sets], fontsize=6.3); ax_f.set_xlim(-.5, 2.5)
    ax_f.tick_params(axis="x", length=0); ax_f.tick_params(axis="y", labelleft=False)
    sec = ax_f.secondary_yaxis("right", functions=(lambda v: v / sdw, lambda v: v * sdw))
    sec.set_yticks(range(-16, 1, 4)); sec.set_ylabel("in wake SDs", labelpad=1)
    head_d.set_title(r"\textbf{d}\enspace\textbf{Replication}" + "\nper subject, same axis", loc="left",
                     fontsize=8.5, pad=1.5, linespacing=1.2)

    for ext in ("pdf", "png"):
        fig.savefig(output.with_suffix("." + ext), dpi=600 if ext == "png" else None)
    plt.close(fig)
    print(f"animal means {np.round(animal.values, 4)}, t3 {t.statistic:.2f}, p {t.pvalue:.4f}")
    for name, d, ci in sets:
        print(name.replace("\n", " "), len(d), round(d.mean(), 4), np.round(ci(d), 4))
    print("b windows (z, surrogate, off band)", [(s_, round(df[df.id == w_].real.iloc[0], 4), round(df[df.id == w_].null.iloc[0], 4),
                                                  int(band(edges, w_)[2].sum())) for s_, w_, _ in PANEL_B])
    print(sdw, win.groupby("group")[["d", "sd", "lvl"]].mean().round(4).to_dict())


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path); p.add_argument("output", type=Path)
    a = p.parse_args()
    draw(a.root, a.output)
