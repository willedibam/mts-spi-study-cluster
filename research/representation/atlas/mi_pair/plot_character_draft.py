"""Draft redesign of fig:mi (v14): what was measured, the pattern effect in Δz, and its replication.

    python plot_character_draft.py <results/representation/atlas-mi-pair-261117> <output stem>
    (the figure as agreed: python plot_character_draft.py results/representation/atlas-mi-pair-261117
     results/representation/atlas-mi-pair-261117/mi-figure; writes .svg, .pdf and .png)

Reads the checksum-listed zns arrays (macaque), the mouse induction z arrays, the human propofol (PRIOS) per-patient
contrasts from the scorer summary, and the few raw windows it draws (figwin/windows.npz, copied from the Gadi banks by
fig_windows.py). (b) is, per state, the window whose Δz is nearest its state's median over all animals (PANEL_B), on
common axes: ketamine + medetomidine large (the only drug in all four macaques; George's, any animal), wake, sleep,
ketamine and propofol small, each Chibi's window nearest that median (the one animal recorded in sleep and under both,
so the four differ by state alone; George's wake and sleep windows, nearest overall, have higher typical G);
medetomidine alone, the weakest, is left out. (a) is one real pair (its two traces boxed either side of the brain, each with a fan to its electrode; G, K and K_surr inside the glyph tiles), GLYPH_PAIR: of the figwin windows' off-band
pairs, the one with the largest banked K - G among those where no sample of either channel lies beyond 5 robust SDs
(the pair of largest K - G overall, 78-90 in the same window, owes most of it to two brief transients: K - G .67,
.27 with them cut). Of those clean off-band pairs it is also first by K over its surrogate's K (.48, next .25) and
by share beyond the 99% contour (3.7%, next 2.7%); fig_candidates.py draws the leading candidates. "Clean" is per pair: the window's
two bursts (1.9 and 8.4 s, on the contiguous patch 78-90-102-114-118, opposite polarity on 78 and 90; artefact or
physiology unknown) reach 74 and 86 at about 3 robust SDs. Under the transient test's window-wide cut it keeps K - G
.43 of .63 (shifted-mask control .66; K by ksg with no Theiler window), third of the window's off-band pairs and level
with 78-90 (.41). The Gaussian tile's K_surr is its own joint IAAFT surrogate, as the bank's (edges.py). Selected, not typical. Its normal-scored samples are drawn beside a synthetic Gaussian of the same
correlation; G and K are the banked values, the Gaussian's are computed on 4000 samples. Above (c), four of its dots drawn as their raw windows (stacked traces of every 4th
electrode): the median wake, sleep and propofol windows and the most extreme window, the propofol one being (b)'s.
(d) shares (c)'s axis, so subject and window effects compare directly. Its strips are one subject's wake and anaesthetised windows chosen for contrast (fig_traces.py; TRACE_MAC), each channel standardised (G and K are blind to amplitude). (a)'s glyphs (glyph()): by default graded, samples between the same-r Gaussian's 95 and 99% contours in a tint, those beyond 99% in the pair's colour; --glyph dots colours only those beyond 99%; --glyph dark colours every sample by its Gaussian distance (inferno) on a dark tile, --glyph hex bins them; the shape of the two electrodes' difference is drawn below them (--no-diff drops it). The segment-surrogate timescale panel of v11 is dropped (checkpoint fc81155e).
(a)'s brain is the provider's photograph of Chibi's left hemisphere with the real electrode positions (Neurotycho
ChibiMap.mat, from the 20110621 KTMD download, saved as figwin/brain_Chibi.npz; George's GeorgeMap.mat, from the
20110112 KTMD "Spatial Map ECoG array" download, as figwin/brain_George.npz); (d)'s icons are described in icon().

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
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.path import Path as MPath
from matplotlib.transforms import blended_transform_factory
from matplotlib.patches import ConnectionPatch, Ellipse, FancyBboxPatch, Patch, PathPatch, Polygon, Rectangle
import numpy as np
import pandas as pd
from scipy.interpolate import splev, splprep
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, binary_opening, gaussian_filter
from scipy.stats import chi2, gaussian_kde, kurtosis, norm, rankdata, t as tdist, ttest_1samp

from matplotlib import patheffects

from mtsspi.figures import figure_style, raincloud
from mtsspi.surrogates import multivariate_iaaft
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
CALLOUT = (("20120717slpanesthesiaandsleepchibitoruyanagawa:awake_eo:08", "median"),     # dots of (c) drawn as
           ("20120706slpanesthesiaandsleepchibitoruyanagawa:sleep:15", "median"),       # their raw windows
           ("20110621ktmdanesthesiaandsleepchibitoruyanagawa:drug:00", "most extreme"),
           ("20120730pfanesthesiaandsleepchibitoruyanagawa:drug:08", "median"),
           ("20120719ktanesthesiaandsleepchibitoruyanagawa:drug:03", "median"),
           ("20120726mdanesthesiaandsleepgeorgetoruyanagawa:drug:05", "median"))
PANEL_B = (("KTMD", "20110114ktmdanesthesiaandsleepgeorgetoruyanagawa:drug:08", "Ketamine + medetomidine"),   # each
           ("awake_eo", CALLOUT[0][0], "Wake"), ("SLP", CALLOUT[1][0], "Sleep"),                        # state's
           ("KT", CALLOUT[4][0], "Ketamine"),                 # median-Δz
           ("PF", CALLOUT[3][0], "Propofol"))                                                         # window
TRACE_MAC = ("20110622ktmdanesthesiaandsleepchibitoruyanagawa:awake_eo:01",   # (d)'s macaque wake and anaesthetised
             "20110622ktmdanesthesiaandsleepchibitoruyanagawa:drug:03")       # windows: fig_traces.py macaque
GLYPH_PAIR = ("20110621ktmdanesthesiaandsleepchibitoruyanagawa:drug:02", 407)        # (a): largest K - G among
                                                         # the figwin windows' pairs that no transient touches


# ---------------------------------------------------------------- data
def tint(d):
    h = tdist.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean() - h, d.mean() + h


def bint(d, n=10000):
    m = np.random.default_rng(0).choice(d, (n, len(d))).mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def mouse_induction(root, detail=False):
    """Per-mouse Δ_ind, as claar/claar_ind_sum.py (frozen Addendum A rule). detail: also, per mouse, its session and
    the row masks of its awake (claar_z_ns) and qualifying induction (claar_z_ind) windows, and the residuals."""
    A, I = np.load(root / "claar/claar_z_ns.npz"), np.load(root / "claar/claar_z_ind.npz")
    ra, ri = json.loads(str(A["rows"])), json.loads(str(I["rows"]))
    resA, resI = A["ns"][:, 0] - A["ns"][:, 1:].mean(1), I["ns"][:, 0] - I["ns"][:, 1:].mean(1)
    g = lambda R, k: np.array([r[k] for r in R])
    sA, stA, biA, dA = g(ra, "session"), g(ra, "state"), g(ra, "burst"), g(ra, "delta_fraction")
    sI, biI, dI = g(ri, "session"), g(ri, "burst"), g(ri, "delta_fraction")
    out, info = [], []
    for s in dict.fromkeys(sI):
        if s == "sub-569073_ses-20210416":
            continue
        a, i = (sA == s) & (stA == "awake"), sI == s
        t = i & (biI >= np.percentile(biA[a], 5)) & (dI >= np.percentile(dA[a], 95))
        if t.sum() >= 3:
            out.append(resI[t].mean() - resA[a].mean()); info.append((s, a, t))
    return (np.array(out), info, resA, resI) if detail else np.array(out)


def mouse_wake_sd(root):
    """SD of the awake window residuals about their session's awake mean, every session: the mouse's own wake SD."""
    A = np.load(root / "claar/claar_z_ns.npz"); ra = json.loads(str(A["rows"]))
    res = A["ns"][:, 0] - A["ns"][:, 1:].mean(1)
    s, st = np.array([r["session"] for r in ra]), np.array([r["state"] for r in ra])
    return np.concatenate([res[(s == x) & (st == "awake")] - res[(s == x) & (st == "awake")].mean()
                           for x in dict.fromkeys(s[st == "awake"])]).std(ddof=1)


def human_propofol(root, variant="prios-sfull", detail=False):
    """Per-patient Δ (CAR, primary) from human/<variant>/z.npz (copied from gdata; score.py's Z and rows), checked
    against the archived scorer summary. detail: also the subjects, window residuals and states, and the patients'
    own wake SD (awake window residuals about their patient's awake mean)."""
    d = np.load(root / "human" / variant / "z.npz"); rows = json.loads(str(d["rows"]))
    subj, st = np.array([r["subject"] for r in rows]), np.array([r["state"] for r in rows])
    res = d["Z"][:, 0, 0] - d["Z"][:, 0, 1:].mean(1)
    keep = list(np.unique(subj))                              # all 7 pass score.py's ≥ 20-window inclusion
    D = np.array([res[(subj == s) & (st == "anaes")].mean() - res[(subj == s) & (st == "awake")].mean() for s in keep])
    m = re.search(r"\[car \(primary\)\] subject Δ \[([^\]]*)\]", (root / "human" / variant / "summary.txt").read_text())
    assert np.allclose(D.round(4), [float(x) for x in m.group(1).split()]), "z.npz disagrees with summary.txt"
    if not detail:
        return D
    sd = np.concatenate([res[(subj == s) & (st == "awake")] - res[(subj == s) & (st == "awake")].mean()
                         for s in keep]).std(ddof=1)
    return D, keep, subj, st, res, sd


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


def delta_scatter(ax, d, floor=-13, s=3.5, fit=True):
    """One animal's windows (d, rows of load()), Δz against delta fraction (prepare.delta_fraction: 0.5-4 Hz share of
    0.5-40 Hz power). Wake and recovery grey, sleep and drugs in their colours, windows below floor drawn at it as
    triangles. Shaded: the delta range that both its sleep and its drug windows reach. fit: the least-squares line of
    Δz on delta over its eyes-open, eyes-closed and sleep windows (the no-drug windows of review/sleepmatch_ns.py's
    regression, fitted here per animal), solid over their delta range and dashed beyond it, where it is extrapolated."""
    nd = d[d.state.isin(("awake_eo", "awake_ec", "recovery_ec"))]; sl, dr = d[d.group == "SLP"], d[d.state == "drug"]
    if len(sl) and len(dr):
        lo, hi = max(sl.delta.min(), dr.delta.min()), min(sl.delta.max(), dr.delta.max())
        if hi > lo:
            ax.axvspan(lo, hi, color="#E9ECEE", zorder=0, lw=0)
    ax.axhline(0, color=GREY, lw=.5, ls=(0, (3, 1.6)), zorder=1)
    if fit:
        f = d[d.group.isin(("awake_eo", "SLP")) | (d.state == "awake_ec")]; b = np.polyfit(f.delta, f.sd, 1)
        for (x0, x1), ls in (((f.delta.min(), f.delta.max()), "-"), ((f.delta.max(), 1), (0, (2, 1.4)))):
            ax.plot([x0, x1], np.polyval(b, [x0, x1]), color=COLOR["ink"], lw=.6, ls=ls, zorder=4)
    for e, c, a in [(nd, COLOR["awake_eo"], .35)] + [(d[d.group == g], COLOR[g], .8) for g in ("SLP", "MD", "PF", "KT", "KTMD")]:
        lo_ = e.sd < floor
        ax.scatter(e.delta[~lo_], e.sd[~lo_], s=s, lw=0, color=c, alpha=a, zorder=3 if c != COLOR["awake_eo"] else 2,
                   rasterized=True)
        ax.scatter(e.delta[lo_], np.full(lo_.sum(), floor), s=s * 1.6, marker="v", lw=0, color=c, alpha=a, zorder=3,
                   clip_on=False)
    ax.set_xlim(.15, 1); ax.set_ylim(floor - .5, 3.5)


# ---------------------------------------------------------------- glyphs
def normal_score(v):
    return norm.ppf(rankdata(v) / (len(v) + 1))


def gauss(r, n=4000, seed=3):
    """Normal-scored Gaussian sample whose correlation is exactly r."""
    rng = np.random.default_rng(seed); z = rng.standard_normal((n, 2)); z -= z.mean(0)
    u = z @ np.linalg.inv(np.linalg.cholesky(np.cov(z.T))).T @ np.linalg.cholesky([[1, r], [r, 1]]).T
    return np.column_stack([normal_score(u[:, 0]), normal_score(u[:, 1])])


DARK = "#101015"


def glyph(ax, u, color, style="graded", key=False):
    """Joint samples (normal-scored) of one pair, each point one 10-ms sample, x one electrode and y the other, under the
    dashed probability contours of the Gaussian of the same r (the shape G assumes). graded (default): samples inside
    its 95% contour light grey, those between its 95 and 99% contours a tint of color, those beyond 99% (the wings)
    color; contours 50, 95 and 99%. dots: only the samples beyond 99% coloured; contours 50, 90, 99%. dark: every
    sample coloured by its Gaussian distance q (inferno, log scale, floored so the core stays visible), the far ones
    larger. hex: the samples hex-binned (log count). key draws the graded colour key in the empty lower-right corner."""
    x, y = u.T; r = np.corrcoef(x, y)[0, 1]; E = 3.4; dark = style == "dark"
    q = (x ** 2 - 2 * r * x * y + y ** 2) / (1 - r ** 2); far = q > chi2.ppf(.99, 2); mid = (q > chi2.ppf(.95, 2)) & ~far
    lev = [.5, .95, .99] if style == "graded" else [.5, .9, .99]
    if style in ("dots", "graded"):
        core = ~(far | mid) if style == "graded" else ~far
        ax.scatter(x[core], y[core], s=2.2, lw=.15, ec="white", color="#A7AFB5", alpha=.6, rasterized=True, zorder=2)
        if style == "graded":
            ax.scatter(x[mid], y[mid], s=3.4, lw=.2, ec="white", color=mix(color, .42), rasterized=True, zorder=4)
        ax.scatter(x[far], y[far], s=4.6, lw=.2, ec="white", color=color, rasterized=True, zorder=4)
    elif dark:
        ax.add_patch(Rectangle((-E, -E), 2 * E, 2 * E, color=DARK, lw=0, zorder=0)); o = np.argsort(q)
        c = plt.get_cmap("inferno")(.18 + .82 * np.clip(np.log1p(q[o]) / np.log1p(chi2.ppf(.999, 2)), 0, 1))
        ax.scatter(x[o], y[o], c=c, s=np.where(far[o], 4, 1.8), lw=0, rasterized=True, zorder=2)
    else:
        cm = LinearSegmentedColormap.from_list("g", [mix(color, .18), color, mix(color, 1.6, COLOR["ink"])])
        ax.hexbin(x, y, gridsize=30, extent=(-E, E, -E, E), cmap=cm, bins="log", mincnt=1, lw=0, rasterized=True,
                  zorder=2)
    if key:                                              # lower-right corner: empty for r > 0
        for j, (c_, s_, l_) in enumerate(((mix(color, .42), 3.4, r"beyond 95\%"), (color, 4.6, r"beyond 99\%"))):
            ax.scatter(-.05, -2.4 - .5 * j, s=s_ * 1.6, lw=.25, ec="white", color=c_, zorder=5)
            ax.text(.2, -2.4 - .5 * j, l_, ha="left", va="center", fontsize=4.8, color=GREY)
    g = np.linspace(-E, E, 200); GX, GY = np.meshgrid(g, g)
    ax.contour(GX, GY, (GX ** 2 - 2 * r * GX * GY + GY ** 2) / (1 - r ** 2), levels=sorted(chi2.ppf(lev, 2)),
               colors=["#6E7380" if dark else COLOR["ink"]], linewidths=.5, linestyles=[(0, (3, 1.6))], zorder=3)
    ax.set(xlim=(-E, E), ylim=(-E, E), aspect="equal", xticks=[], yticks=[])
    for sp in ax.spines.values():
        sp.set_visible(False)


def rbox(ax, size, color, lw=.6, fc="none", pad=.0, rad=.045):
    """A rounded frame around an axes of size (w, h) inches (pad and corner radius in inches), as the method
    schematic's boxes."""
    w, h = size
    p_ = FancyBboxPatch((-pad / w, -pad / h), 1 + 2 * pad / w, 1 + 2 * pad / h, boxstyle=f"round,pad=0,rounding_size={rad / w}",
                        mutation_aspect=w / h, transform=ax.transAxes, fc=fc, ec=color, lw=lw, clip_on=False, zorder=-5)
    ax.add_patch(p_); return p_


def mix(color, f, other="white"):
    """color moved toward other: f < 1 gives a tint (f of the way from other), f > 1 goes past color toward other."""
    c, o = np.array(to_rgb(color)), np.array(to_rgb(other))
    return tuple(o + f * (c - o)) if f <= 1 else tuple(c + (f - 1) * (o - c))


def window_strip(ax, X, color, every=4, lw=.3, scale=None):
    """A window (100 Hz) as stacked traces of every 4th of its electrodes, each standardised, or all divided by one
    scale (a shared amplitude scale across windows)."""
    Z = (X - X.mean(0)) / (X.std(0) if scale is None else scale); t = np.arange(len(Z)) / 100
    n = len(range(0, Z.shape[1], every))
    for j, m in enumerate(range(0, Z.shape[1], every)):
        ax.plot(t, Z[:, m] * .28 - j, color=color, lw=lw, rasterized=True)
    ax.set(xlim=(0, t[-1]), ylim=(-n + .2, 1.2) if scale is None else (-n + .45, .55), xticks=[], yticks=[])
    frame(ax, LIGHT, .6)


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
    return a, b


VIEW = {"human": ((0, -1, 0), (0, 0, 1), (-1, 0, 0)), "macaque": ((0, -1, 0), (0, 0, 1), (-1, 0, 0)),    # lateral, front left
        "mouse": ((0, 0, 1), (-1, 0, 0), (0, -1, 0))}                                                  # dorsal, front up


def icon(ax, name, root, base=(.9, .91, .92), style="mesh"):
    """(d)'s brains, electrodes as dots. style "mesh": shaded template meshes (figwin/assets.npz, from fig_assets.py).
    style "map": region maps, each region a flat grey tone with white borders: macaque, the same mesh coloured by
    CHARM level 2; mouse, the Allen CCF top view of isocortex areas, mirrored to both hemispheres; human, ggseg's 2D
    Desikan-Killiany lateral view (electrodes placed by a bounding-box fit from the fsaverage projection, approximate).
    human: fsaverage5 left pial, lateral, with every grid and strip electrode of the 7 PRIOS patients (MNI305, BIDS
    electrodes.tsv), both hemispheres projected onto one side. macaque: NMT v2.0 left gray surface, lateral, with Chibi's
    107 lateral electrodes moved from Neurotycho's photograph by a bounding-box fit (approximate; the few that land off
    the template are moved to its nearest vertex). mouse: Allen CCF 2017 whole-brain mesh, dorsal; the 30-electrode
    array is schematic (a grid over dorsal cortex), not the recorded positions."""
    with np.load(root / "figwin/assets.npz") as A:
        As = {k: A[k] for k in A if k.startswith(name)}
    v, f = As[name + "_v"].astype(float), As[name + "_f"]
    P = v @ np.array(VIEW[name], float).T; T = P[f]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    L = np.array([-.35, .55, 1.]); sh = np.abs(n @ (L / np.linalg.norm(L)))   # two-sided Lambert, light upper left
    o = np.argsort(T[:, :, 2].mean(1))
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
    tone = lambda k: .73 + .21 * ((np.asarray(k) * 7919) % 5) / 4               # region -> one of five grey tones
    if style == "map" and name == "mouse":
        M = As["mouse_map"]; M = np.hstack([M, M[:, ::-1]]); h_, w_ = M.shape
        img = np.clip(np.dstack([tone(M) * b for b in base]) / max(base), 0, 1)
        img[binary_dilation((np.diff(M, axis=0, prepend=M[:1]) != 0) | (np.diff(M, axis=1, prepend=M[:, :1]) != 0),
                            iterations=3) | (M == 0)] = 1                        # white borders, white outside
        ext = (-w_ * 10, w_ * 10, -h_ * 20, 0)                                   # µm, 20-µm pixels; x 0 at the midline
        ax.imshow(img, extent=ext, interpolation="antialiased", zorder=1)
        ax.contour(np.linspace(ext[0], ext[1], w_), np.linspace(0, ext[2], h_), gaussian_filter((M > 0).astype(float), 1),
                   [.5], colors=[GREY], linewidths=.3, zorder=2)
        E = E - (mid, 0); r_, c_ = np.nonzero(M); lo = np.array([ext[0], -r_.max() * 20.]); hi = np.array([ext[1], -r_.min() * 20.])
        q = np.c_[c_ * 20. - w_ * 10, -r_ * 20.]; ii = np.clip(((E[:, 0] + w_ * 10) / 20).astype(int), 0, w_ - 1)
        off = M[np.clip((-E[:, 1] / 20).astype(int), 0, h_ - 1), ii] == 0       # grid points off the cortex map
        E[off] = q[((q[None] - E[off, None]) ** 2).sum(2).argmin(1)]
    elif style == "map" and name == "human":
        V, C, S = As["human_map_v"] * (1, -1), As["human_map_c"], list(As["human_map_s"]) + [len(As["human_map_c"])]
        for k, (a, b) in enumerate(zip(S[:-1], S[1:])):
            ax.add_patch(PathPatch(MPath(V[a:b], C[a:b]), fc=np.clip(tone(k) * np.array(base) / max(base), 0, 1),
                                   ec="white", lw=.35, zorder=1))
        O = As["human_map_o"] * (1, -1)
        ax.add_patch(PathPatch(MPath(O, As["human_map_oc"]), fc="none", ec=GREY, lw=.3, zorder=2))
        e_ = MPath(O, As["human_map_oc"]).get_extents(); mlo, mhi = np.array([e_.x0, e_.y0]), np.array([e_.x1, e_.y1])
        E = mlo + (E - lo) / (hi - lo) * (mhi - mlo); lo, hi = mlo, mhi
        inside = MPath(O, As["human_map_oc"]).contains_points(E)
        on = np.concatenate(MPath(O, As["human_map_oc"]).to_polygons()); on = on + .03 * (on.mean(0) - on)  # just inside
        E[~inside] = on[((on[None] - E[~inside, None]) ** 2).sum(2).argmin(1)]
    else:
        if style == "map":                                                        # macaque: CHARM level 2 on the mesh
            lab = As["macaque_lab"][1][f]; fl = tone(lab[:, 0]); fl[(lab != lab[:, :1]).any(1)] = 1.06
            c = np.clip(np.outer(fl[o] * (.9 + .1 * sh[o]), base) / max(base), 0, 1)
        else:
            c = np.clip(np.outer(.42 + .58 * sh[o], base), 0, 1)
        ax.add_collection(PolyCollection(T[o][:, :, :2], fc=c, ec=c, lw=.12, rasterized=True, zorder=1))
        H, ex, ey = np.histogram2d(*P[:, :2].T, bins=160)                            # template's projected outline
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


FACTS = {"macaque": "4 anaesthetics\n{n} animals", "mouse": "isoflurane\n{n} mice", "human": "propofol\n{n} patients"}
ICON = {"macaque": ((.22, .9, .24, .66, 7, 5), r"Macaque" + "\n" + "ECoG"), "mouse": ((.25, .85, .26, .46, 4, 2), "Mouse\nEEG"),
        "human": ((.45, .92, .28, .62, 5, 4), "Human\nECoG")}


# ---------------------------------------------------------------- figure
def draw(root, output, icons="mesh", style="dots", diff=False, labels="name", link=True, btop="none", banimal=None,
         ctop="traces"):
    df, edges = load(root)
    PANEL_B = globals()["PANEL_B"]
    if banimal:                                              # same rule, one animal: its window nearest each state's
        an_, _, part = banimal.partition(":")                # pooled median Δz; "<animal>:small" keeps the big one
        PANEL_B = tuple((s_, (lambda w: w.loc[(w.sd - df[df.group == s_].sd.median()).abs().idxmin(), "id"])(
            df[(df.group == s_) & (df.animal == an_)]) if (j_ or part != "small") else w0, n_)
            for j_, (s_, w0, n_) in enumerate(PANEL_B))
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
    W, H = 7.2, 3.44                                         # row 1 size (in), for square axes; rects are in its fractions
    EX = {"resid": .4, "time": .45, "clock": .45}.get(btop, 0)                       # extra row-1 height (in) for a strip above (b)
    H1 = H + EX; FH = 6.8 + EX
    fig = plt.figure(figsize=(7.2, FH), layout="constrained")
    r1, r2 = fig.subfigures(2, 1, height_ratios=[H1, FH - H1], hspace=.03)
    rng = np.random.default_rng(0)
    Y = lambda y, s=0: (y * H + s) / H1                      # a row-1 fraction, raised by s inches, in the taller row

    # a: what we measured: two electrodes, their 10-s traces, and the two MI estimates of their joint samples
    def put(rect, s=0):
        ax = r1.add_axes([rect[0], Y(rect[1], s), rect[2], rect[3] * H / H1]); ax.set_in_layout(False); return ax
    pc = COLOR["KTMD"]
    wid, pk = GLYPH_PAIR
    e1, e2 = (int(v[pk]) for v in np.triu_indices(32, 1))
    with np.load(root / "figwin/windows.npz") as fw:
        lab = list(fw[wid + "|chan"]); X = fw[wid][:, [lab.index(chan[wid][e]) for e in (e1, e2)]].astype(float)
    (gw, kw), (_, ksur) = edges[wid]
    AW = .31                                                 # (a)'s width (row fraction)
    bx = put([.08, .695, .15, .25], EX)
    pts = real_brain(bx, root / f"figwin/brain_{'Chibi' if 'chibi' in wid else 'George'}.npz",
                     [int(v) for v in chan[wid]], (int(chan[wid][e1]), int(chan[wid][e2])), pc)
    Xc = X - np.median(X, 0); Xc = Xc / np.abs(Xc).max(0) * .62          # whole 10 s, transients included
    tb = (.07, .1); fans = []                               # each trace in a box beside the brain, a fan to its electrode
    for side, i_ in zip((0, 1), np.argsort([pts[0][0], pts[1][0]])):     # the electrode further left on the left
        rect = [0 if side == 0 else AW - tb[0], .77, *tb]; ax_ = put(rect, EX)
        ax_.plot(Xc[:, i_], color=pc, lw=.4); ax_.set(xlim=(-25, 1025), ylim=(-.75, .75)); ax_.axis("off")
        rbox(ax_, (tb[0] * W, tb[1] * H), LIGHT, lw=.7, fc="white")
        ax_.text(.5, 1.06, rf"el.\,{chan[wid][(e1, e2)[i_]]}", transform=ax_.transAxes, ha="center", va="bottom",
                 fontsize=5.6, color=pc)
        fans.append((pts[i_], ax_, side))
        if side == 1:
            ax_.plot([800, 1000], [-1.0] * 2, color=COLOR["ink"], lw=.8, clip_on=False)
            ax_.text(900, -1.08, "2 s", ha="center", va="top", fontsize=5.2)
    r1.text(.155, Y(.705, EX), r"ket.\,+\,med., 10 s", ha="center", va="top", fontsize=5.4, color=pc)
    u = np.column_stack([normal_score(X[:, 0]), normal_score(X[:, 1])])
    ug = gauss(np.corrcoef(u.T)[0, 1])
    Gg, Kg = -.5 * np.log(1 - np.corrcoef(ug.T)[0, 1] ** 2), ksg(ug[:, 0], ug[:, 1], 0)
    sg = multivariate_iaaft(ug, np.random.default_rng(0), final="spectrum")[0]      # its joint IAAFT surrogate, as
    sg = (sg - sg.mean(0)) / sg.std(0); Ksg = ksg(sg[:, 0], sg[:, 1], 0)            # the bank's (edges.py)
    f2 = lambda v: f"{v:.2f}".lstrip("0")
    gw_, gap = .14, .015                                     # glyph tile width and the gap between the two tiles
    gy = .31 if diff else .25                                # glyph row; with diff, the difference strip goes below
    us = ug[np.random.default_rng(1).choice(len(ug), len(u), replace=False)]   # drawn: as many samples as the pair,
    us = np.column_stack([normal_score(us[:, 0]), normal_score(us[:, 1])])      # re-scored, so the margins match
    L = ((r"\textbf{Gaussian MI}", r"\textbf{kNN MI}", r"\textbf{surrogate}") if labels == "name" else
         (r"\textbf{Gaussian MI} $G$", r"\textbf{kNN MI} $K$", r"\textbf{surrogate} $K_\mathrm{s}$"))
    halo = [patheffects.withStroke(linewidth=1.6, foreground="white")]
    rend = fig.canvas.get_renderer()
    for x_, v, col, name, vals in ((.0, us, COLOR["ink"], r"Gaussian, same $r$", (f2(Gg), f2(Kg), f2(Ksg))),
                                   (gw_ + gap, u, pc, "this pair (non-Gaussian)",
                                    (f2(gw[pk]), rf"\textbf{{{f2(kw[pk])}}}", f2(ksur[pk])))):
        tile = [x_ + .015, gy, gw_, gw_ * W / H]; ga = put(tile, .8 * EX); glyph(ga, v, col, style, key=col != pc)
        rbox(ga, (gw_ * W, gw_ * W), "#B9C0C5" if col != pc else mix(pc, .7), lw=.7)
        r1.text(x_ + .015 + gw_ / 2, Y(gy + gw_ * W / H + .018, .8 * EX), rf"\textbf{{{name}}}", ha="center", va="bottom",
                fontsize=6, color=col)
        keys = [ga.text(0, .955 - .075 * j, l_, transform=ga.transAxes, ha="right", va="top", fontsize=4.8,
                        color=GREY if j == 2 else col, zorder=6, path_effects=halo) for j, l_ in enumerate(L)]
        kx = .04 + max(t_.get_window_extent(rend).width for t_ in keys) / (gw_ * W * fig.dpi)
        for j, (t_, v_) in enumerate(zip(keys, vals)):       # keys right-aligned, each value just after its key
            t_.set_x(kx)
            ga.text(kx + .03, .955 - .075 * j, v_, transform=ga.transAxes, ha="left", va="top", fontsize=4.8,
                    color=GREY if j == 2 else col, zorder=6, path_effects=halo)
    if diff:                                                 # same r fixes the spread of the difference, not its shape
        fr = put([.015, .03, gw_ * 2 + gap, .235], .66 * EX); fr.axis("off"); rbox(fr, ((gw_ * 2 + gap) * W, .235 * H), LIGHT, lw=.7)
        dx = put([.035, .085, gw_ * 2 + gap - .04, .14], .66 * EX); dd = (u[:, 0] - u[:, 1]) / (u[:, 0] - u[:, 1]).std(); xs = np.linspace(-6, 6, 300)
        dx.fill_between(xs, 1e-4, norm.pdf(xs), color=LIGHT, lw=0)
        dx.plot(xs, norm.pdf(xs), color=COLOR["ink"], lw=.5, ls=(0, (3, 1.6)))
        dx.plot(xs, gaussian_kde(dd, bw_method=.15)(xs), color=pc, lw=.9)
        dx.set(yscale="log", ylim=(1.5e-3, 1.5), xlim=(-6, 6), yticks=[], xticks=[])
        for sp in ("left", "right", "top"):
            dx.spines[sp].set_visible(False)
        dx.spines["bottom"].set_color(GREY); dx.spines["bottom"].set_linewidth(.5)
        dx.text(.5, -.1, "el. difference, in its SD (log density)", transform=dx.transAxes, ha="center", va="top",
                fontsize=4.8, color=GREY)
        dx.text(.99, 1.1, rf"kurtosis {kurtosis(dd):.1f}", transform=dx.transAxes, ha="right", va="top", fontsize=5, color=pc)
        dx.text(.01, 1.1, "Gaussian 0", transform=dx.transAxes, ha="left", va="top", fontsize=5, color=COLOR["ink"])
    print("a's pair:", wid, "electrodes", chan[wid][e1], chan[wid][e2], "banked G, K", gw[pk], kw[pk], "surrogate K",
          ksur[pk], "Gaussian G, K, K_surr", Gg, Kg, Ksg)

    # b: every pair of one window per state, each the state's median-Δz window; the largest drug effect drawn big
    full = 1.05 * max(v.max() for v in edges[PANEL_B[0][1]][0])
    view = (-.02, .42), (-.17, .52)                      # common to all five, so the windows compare directly
    f3 = lambda v: f"{v:.3f}".lstrip("0")
    m_ = .124; rects = [[.365, .11, .3, .3 * W / H]] + [[.715 + (q % 2) * (m_ + .032), .11 + (1 - q // 2) * (m_ * W / H + .075),
                                                       m_, m_ * W / H] for q in range(4)]
    if btop == "big":                                    # still square, larger: the smalls drop their y labels (same
        bi, si, g1, g2, gv = 2.288, 1.014, .12, .1, .26  # view as the big one) and the gaps close; sizes in inches
        rects = [[.365, .11, bi / W, bi / H]] + [[.365 + (bi + g1 + (q % 2) * (si + g2)) / W, .11 + (1 - q // 2) * (si + gv) / H,
                                                 si / W, si / H] for q in range(4)]
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
            a_.scatter(g[out], k[out], s=26 * sc, lw=.5 * sc ** .5, ec="white", color=ink, zorder=3)
        ax.set(xlim=view[0], ylim=view[1])
        ax.set_xticks([0, .2, .4], ["0", ".2", ".4"]); ax.set_yticks([0, .2, .4], ["0", ".2", ".4"])
        if big:
            ov = panes[1][0]
            ov.set(xlim=(-.04, full), ylim=(-.17, full), xticks=[], yticks=[]); frame(ov, GREY, .6)
            ov.add_patch(Rectangle((view[0][0], view[1][0]), np.diff(view[0])[0], np.diff(view[1])[0], fill=False,
                                   ec=COLOR["ink"], lw=.6, zorder=4))
            ov.text(.95, .06, "all pairs", transform=ov.transAxes, ha="right", fontsize=5.2, color=GREY)
            tx = ax
            if btop == "gmarg":                              # G of the pairs in and off the band, each its own share
                tx = put([rect[0], rect[1] + rect[3] + .012, rect[2], .27 / H]); bins = np.arange(-.02, .4201, .01)
                tx.hist(g[~out], bins, density=True, color=LIGHT, lw=0)
                tx.hist(g[out], bins, density=True, histtype="step", color=ink, lw=.9)
                tx.set(xlim=view[0], yticks=[]); tx.tick_params(labelbottom=False, length=1.5)
                for sp in ("left", "right", "top"):
                    tx.spines[sp].set_visible(False)
                tx.text(.99, .9, "share of pairs by $G$:", transform=tx.transAxes, ha="right", va="top", fontsize=5, color=GREY)
                tx.text(.99, .62, "in the band", transform=tx.transAxes, ha="right", va="top", fontsize=5, color=GREY)
                tx.text(.99, .34, "off it", transform=tx.transAxes, ha="right", va="top", fontsize=5, color=ink)
            tx.set_title(rf"\textbf{{{name}}}", fontsize=8, color=ink, pad=2)
            ax.text(.97, .2, r"$z$ = corr($G$, $K$)", transform=ax.transAxes, ha="right", fontsize=6, color=GREY)
            ax.text(.97, .135, rf"$z$ {f3(row.real)}, its surrogate {f3(row.null)}", transform=ax.transAxes, ha="right",
                    fontsize=6, color=GREY)
            ax.text(.97, .03, rf"$\Delta z$ \textbf{{{row.sd:.1f}}} wake SDs".replace("-", "--"), transform=ax.transAxes,
                    ha="right", fontsize=10.5, color=ink)
            ax.set_xlabel(r"Gaussian $\mathrm{MI}$, $G$", fontsize=6.8, labelpad=1)
            ax.set_ylabel(r"kNN $\mathrm{MI}$, $K$", fontsize=6.8, labelpad=1)
        else:
            ax.tick_params(labelsize=5.4, pad=1.2, labelleft=j % 2 == 1 and btop != "big", labelbottom=j > 2)
            ax.set_title(rf"\textbf{{{name}}}", fontsize=6.8, color=ink, pad=1.5)
            ax.text(.96, .05, rf"$\Delta z$ \textbf{{{row.sd:+.1f}}} SD".replace("-", "--"), transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8.8, color=ink)
    dot = lambda c_, ms: Line2D([], [], ls="", marker="o", ms=ms, color=c_)
    r1.legend([Patch(color=LIGHT), dot(GREY, 2.5), tuple(dot(COLOR[c_], 3) for c_ in ("ink", "SLP", "KTMD", "KT", "PF"))],
              [r"surrogate's 95\% scatter about the line", "pair", "pair off the band"], loc="center",
              bbox_to_anchor=(.855, .835) if btop == "gmarg" else (.685, Y(.895, EX)), ncol=1 if btop == "gmarg" else 3,
              frameon=False, fontsize=6, handlelength=3.2, handletextpad=.5,
              columnspacing=1.4, handler_map={tuple: HandlerTuple(ndivide=None, pad=.15)})
    if btop == "resid":                                  # per state, the share of pairs at least x from the line
        rx = put([.365, .852, .63, .32 / H]); rx.axvspan(0, 1, color=LIGHT, lw=0, zorder=0)
        sur = []
        for s_, wid, name in PANEL_B:
            (g, k), (gs, ks) = edges[wid]; c, w, out = band(edges, wid)
            r = np.sort(np.abs(k - np.polyval(c, g)) / w)[::-1]
            rx.plot(r, np.arange(1, len(r) + 1) / len(r), color=COLOR["ink"] if s_ == "awake_eo" else COLOR[s_],
                    lw=1.1 if s_ == "KTMD" else .8, zorder=3 if s_ == "KTMD" else 2)
            sur.append(np.abs(ks - np.polyval(np.polyfit(gs, ks, 1), gs)) / w)
        r = np.sort(np.concatenate(sur))[::-1]
        rx.plot(r, np.arange(1, len(r) + 1) / len(r), color=GREY, lw=.7, ls=(0, (3, 1.6)), zorder=1)
        rx.set(yscale="log", ylim=(1.5e-3, 1), xlim=(0, 3.2))
        rx.set_yticks([.01, .1, 1], [r"1\%", r"10\%", r"100\%"]); rx.set_yticks([], minor=True)
        rx.set_xticks([0, 1, 2, 3], ["0", "band edge", "2", "3"])
        rx.tick_params(labelsize=5.4, pad=1.2, length=1.5)
        for sp in ("right", "top"):
            rx.spines[sp].set_visible(False)
        rx.set_ylabel("pairs at least\nthis far", fontsize=5.4, labelpad=1, linespacing=1)
        rx.text(.995, .95, "distance from the line, in band half-widths", transform=rx.transAxes, ha="right", va="top",
                fontsize=5.4, color=GREY)
        rx.text(.995, .68, "dashed: the five surrogates", transform=rx.transAxes, ha="right", va="top", fontsize=5, color=GREY)
    if btop in ("time", "clock"):                        # Δz of every window of each (b) window's session. time: state
        clock = btop == "clock"                          # slots, windows evenly spread in each. clock: within a slot, real
        SEG = ("awake_eo", "awake_ec", None, "recovery_ec")  # seconds (figwin/start_s.csv, from the bank manifests);
        sv = pd.read_csv(root / "figwin/start_s.csv").drop_duplicates("id").set_index("id")   # slots, and the separate
        df["t"], df["rec"] = df.id.map(sv.start_s) + 5, df.id.map(sv.session)   # recordings in one, joined by breaks
        tl = put([.365, .795, .3, .42 / H]); tr = put([.715, .795, .28, .42 / H]); tr.sharey(tl)
        sel = lambda d, st: d[d.state == st] if st else d[d.state.isin(("drug", "sleep"))]
        def xs(e, brk):                                  # positions within a slot (from 0) and the slot's length
            if not clock:
                return (np.arange(len(e)) + .5) / len(e), 1., []
            x, o, cuts = [], 0., []
            for _, g in e.groupby("rec", sort=True):     # recordings in order, each from its first window
                x.append(g.t.values - g.t.min() + 5 + o); o = x[-1][-1] + 5; cuts.append(o); o += brk
            return np.concatenate(x), o - brk, cuts[:-1]
        # sleep's wake windows were recorded after sleep (its recordings 2-3), so its wake slots are left empty
        rows = {tl: [(df[df.id == PANEL_B[0][1]].unit.iloc[0], COLOR["KTMD"], PANEL_B[0][1], SEG)],
                tr: [(df[df.id == w_].unit.iloc[0], COLOR["ink"] if s_ == "awake_eo" else COLOR[s_], w_,
                      SEG[:2] if s_ == "awake_eo" else (None,) if s_ == "SLP" else SEG) for s_, w_, _ in PANEL_B[1:]]}
        for ax, rr in rows.items():
            brk = 150 if clock else 0                    # a recording break inside a slot, in seconds
            L_ = [max([xs(sel(df[df.unit == u], st), brk)[1] for u, _, _, sg in rr if st in sg and len(sel(df[df.unit == u], st))]
                      or [0]) for st in SEG]
            gap = .07 * sum(L_); x0 = np.concatenate([[0], np.cumsum(np.array(L_) + gap)])[:4]
            for u, col, ring, sg in rr:
                d = df[df.unit == u]
                for q, st in enumerate(SEG):
                    e = sel(d, st)
                    if st not in sg or not len(e):
                        continue
                    e = e.sort_values(["rec", "t"]) if clock else e
                    x, _, cuts = xs(e, brk); x = x0[q] + x
                    c_ = col if (st is None or ax is tr) else COLOR["awake_eo"]
                    fade = ax is tr and st is not None and col != COLOR["ink"]     # right: the drug or sleep slot leads
                    big = ax is tl
                    cut = np.flatnonzero(e.rec.values[1:] != e.rec.values[:-1]) + 1 if clock else []
                    for xi_, ei in zip(np.split(x, cut), np.split(e.sd.values, cut)):   # no line across a recording break
                        ax.plot(xi_, ei, color=c_, lw=.7 if big else .6, marker="o", ms=2.2 if big else 1.8, mec="white",
                                mew=.25, zorder=2, alpha=.3 if fade else 1)
                    for xi, (_, r_) in zip(x, e.iterrows()):
                        if r_.id == ring:
                            ax.plot(xi, r_.sd, "o", ms=5, mfc="none", mec=c_, mew=.8, zorder=3)
            third = "ket.\\,+\\,med." if ax is tl else "drug or sleep"
            bt = blended_transform_factory(ax.transData, ax.transAxes)
            ax.axhline(0, color=GREY, lw=.5, ls=(0, (3, 1.6)), zorder=0)
            for q, lab in enumerate(("eyes open", "eyes closed", third, "recovery")):
                ax.text(x0[q] + L_[q] / 2, 1.02, lab, transform=bt, ha="center", va="bottom", fontsize=5, color=GREY)
                ax.plot([x0[q], x0[q] + L_[q]], [0, 0], transform=bt, color=COLOR["ink"], lw=.7, clip_on=False,
                        solid_capstyle="butt")           # the bottom spine, broken between slots
            ax.set(xlim=(-.01 * x0[-1], x0[-1] + L_[-1] * 1.01), xticks=[])
            for sp in ("right", "top", "bottom"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(labelsize=5.4, pad=1.2, length=1.5)
            if not clock and ax is tl:                   # time: each slot's real span under it
                for q, st in enumerate(SEG):
                    e = sel(df[df.unit == rr[0][0]], st)
                    ax.text(x0[q] + L_[q] / 2, .03, f"{(e.t.max() - e.t.min() + 10) / 60:.0f} min", transform=bt,
                            ha="center", va="bottom", fontsize=4.8, color=GREY)
            if clock:                                    # a 10-min bar under the last slot
                ax.plot([x0[-1] + L_[-1] - 600, x0[-1] + L_[-1]], [-.1] * 2, transform=bt, color=COLOR["ink"], lw=.8,
                        clip_on=False, solid_capstyle="butt")
                ax.text(x0[-1] + L_[-1] - 300, -.14, "10 min", transform=bt, ha="center", va="top", fontsize=5)
        tl.set(ylim=(-13, 3.5), yticks=[0, -5, -10], yticklabels=["0", "--5", "--10"])
        tr.tick_params(labelleft=False)
        tl.set_ylabel("$\\Delta z$, wake SDs", fontsize=5.4, labelpad=1)
    small = df[df.id.isin([w_ for _, w_, _ in PANEL_B[1:]])].animal.unique()
    for x_, l_, c_, r_ in ((.0, "a", "One pair", ""), (.325, "b", "Every pair", "one 10-s window per state, each its state's median"
                                                                      + (f", the small four {small[0]}'s" if len(small) == 1 else ""))):
        r1.text(x_, Y(.975, EX), rf"\textbf{{{l_}}}\enspace\textbf{{{c_}}}" + (f": {r_}" if r_ else ""), fontsize=8.5,
                va="top")

    # c-d: the pattern effect and its replication
    left = r2
    gs = left.add_gridspec(2, 2, height_ratios=[.4, 1], width_ratios=[1.92, 1.08], hspace=.04, wspace=.05)
    ax_c = left.add_subplot(gs[1, 0]); ax_f = left.add_subplot(gs[1, 1], sharey=ax_c)
    above = blended_transform_factory(ax_c.transData, ax_c.transAxes)   # strips sit over their own state column
    strips = [ax_c.inset_axes([STATES.index(df.loc[df.id == wid, "group"].iloc[0]) - .45, 1.07, .9, .16],
                              transform=above) for wid, _ in CALLOUT] if ctop == "traces" else []
    if ctop == "delta":                                  # every animal: Δz against delta fraction (sleep: Chibi, George)
        for q, an in enumerate(("Chibi", "George", "Kin2", "Su")):
            dax = ax_c.inset_axes([-.45 + 1.5 * q, 1.12, 1.32, .36], transform=above); dax.set_in_layout(False)
            delta_scatter(dax, df[df.animal == an], s=2)
            dax.set_xticks([.2, .6, 1], [".2", ".6", "1"]); dax.set_yticks([0, -5, -10], ["0", "--5", "--10"])
            dax.tick_params(labelsize=5.2, pad=1, length=1.5, labelleft=q == 0)
            dax.text(.04, .05, an, transform=dax.transAxes, fontsize=6, color=COLOR["ink"], fontweight="bold")
            if q == 0:
                dax.set_ylabel("$\\Delta z$, wake SDs", fontsize=5.4, labelpad=1)
            if q == 3:
                dax.text(1.04, -.03, "delta\nfraction", transform=dax.transAxes, ha="left", va="top", fontsize=5,
                         color=GREY, linespacing=1)
                dax.text(1, 1.05, "grey: wake, recovery.  Line: wake and sleep fit, dashed where extrapolated.  "
                         "Shaded: delta reached by both sleep and drug", transform=dax.transAxes, ha="right",
                         va="bottom", fontsize=4.8, color=GREY)
    for sax in strips:
        sax.set_in_layout(False)                            # row 0 holds them; the layout must not shrink (c)
    head_d = left.add_subplot(gs[0, 1]); head_d.axis("off")
    pos = {}                                                 # window id -> its dot in (c)
    win = df[df.group.isin(STATES)]
    sdw = df[df.state == "awake_eo"].d.std(ddof=1)          # eyes-open wake window SD: the second axis is exact
    top = .027                                               # (d)'s traces sit above its axes, not in it
    num = lambda v, f: "0" if round(v, 3) == 0 else (f"{v:{f}}".replace("0.", ".") if f.endswith("3f") else f"{v:{f}}")
    for i, s in enumerate(STATES):
        w = win[win.group == s]; c = COLOR[s]; drug = s in DRUGS; ink = COLOR["ink"] if drug else GREY
        n0 = len(ax_c.collections)
        raincloud(ax_c, w.d.values, i, c, rng, width=.64, box=True, point_sizes=2.3, point_kw={"alpha": .35})
        dots = next(c_ for c_ in ax_c.collections[n0:] if isinstance(c_, PathCollection))
        pos.update(zip(w.id, map(tuple, dots.get_offsets())))
        m1 = f"{w.sd.mean():+.1f}".replace("-", "--") if abs(w.sd.mean()) >= .05 else "0"
        ax_c.text(i, top, rf"\textbf{{{m1}}}" if drug else m1, ha="center", va="top", fontsize=7.2, color=ink,
                  bbox=dict(fc="white", ec="none", pad=.4))
    for x_, ha, s_ in ((-.76, "right", "mean\n" + r"$\Delta z$"), (-.66, "left", "wake\nSDs")):   # either side of the
        ax_c.text(x_, top, s_, ha=ha, va="top", fontsize=6, color=GREY, clip_on=False,
                  linespacing=1.1).set_in_layout(False)  # axis;
                                                          # out of the layout, so the margin text does not push (c) right
    ax_c.axhline(0, color=GREY, lw=.6, ls="--", zorder=0)
    ax_c.set_ylim(-.074, top + .001)                        # ticks in eyes-open wake window SDs, the unit of every
    ks = np.arange(0, -.0741 / sdw, -5)                      # number printed in (b)-(d): an exact rescaling of Δz
    ax_c.set_yticks(ks * sdw, [f"{k:.0f}".replace("-", "--") for k in ks]); ax_c.tick_params(axis="y", labelsize=5.8, pad=1.5)
    ax_c.set_ylabel(r"$\Delta z$ per 10-s window, wake SDs", y=.42, labelpad=2)
    panel_title(left, "c", "Pattern", r"$z$ falls under every anaesthetic, not in sleep", sub=True)
    states_axis(ax_c); ax_c.set_xlim(-.7, None)              # room for 'wake SDs' left of Wake's mean
    assert ctop != "traces" or {w_ for _, w_, _ in PANEL_B[1:]} <= {w_ for w_, _ in CALLOUT}   # (b)'s small windows
    with np.load(root / "figwin/windows.npz") as fw:        # raw windows of the called-out dots (fig_windows.py)
        for sax, (wid, note) in zip(strips, CALLOUT):     # are drawn above (c)
            r = df[df.id == wid].iloc[0]; s = r.group; lab = list(fw[wid + "|chan"])
            X = fw[wid][:, [lab.index(v) for v in chan[wid]]].astype(float)
            window_strip(sax, X, COLOR[s])
            sax.set_title(rf"\textbf{{{SHORT[s].replace(chr(10), '')}}}" + "\n" + note, fontsize=5.6,
                          color=COLOR[s], pad=1.5, linespacing=1.1)
            sax.set_xlabel(f"{r.sd:+.1f} SD".replace("-", "--"), fontsize=5.6, labelpad=1.5)
            x_, y_ = pos[wid]
            ax_c.scatter(x_, y_, s=16, facecolor="none", edgecolor=COLOR["ink"], lw=.6, zorder=4)
            if not link:
                continue
            con = ConnectionPatch((x_, top - .0068), (x_, y_), coordsA=ax_c.transData, coordsB=ax_c.transData,
                                  color=GREY, lw=.45, shrinkB=2.2, zorder=0, clip_on=False)
            con.set_in_layout(False); ax_c.add_artist(con)
        if strips: strips[0].set_ylabel("8 el.\n10 s", fontsize=5, color=GREY, labelpad=1.5, linespacing=1.05)

    hum = human_propofol(root, detail=True)
    sets = [("macaque", animal.values, tint), ("mouse", mouse_induction(root), bint), ("human", hum[0], tint)]
    own = {"macaque": sdw, "mouse": mouse_wake_sd(root), "human": hum[-1]}     # each dataset's own wake window SD
    pb = {s_: w_ for s_, w_, _ in PANEL_B}
    with np.load(root / "figwin/windows.npz") as fw, np.load(root / "figwin/traces_d.npz") as td:
        mac = {k: fw[w_][:, [list(fw[w_ + "|chan"]).index(v) for v in chan[w_]]] for k, w_ in zip(("wake", "anaes"), TRACE_MAC)}
        TR = {"macaque": mac, "mouse": {k: td["mouse:" + k] for k in ("wake", "anaes")},
              "human": {k: td["human:" + k] for k in ("wake", "anaes")}}
    TCOL = {"macaque": COLOR["KTMD"], "mouse": COLOR["ink"], "human": COLOR["PF"]}
    hd = blended_transform_factory(ax_f.transData, ax_f.transAxes)
    TA, TW = (1.1, 1.29), (1.305, 1.495)                    # (d)'s anaesthetised and wake strips, in axes heights
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
        ax_f.text(i, top, rf"\textbf{{{d.mean() / own[name]:+.1f}}}".replace("-", "--"), ha="center", va="top",
                  fontsize=7.2, color=COLOR["ink"])
        for k, (y0, col) in (("wake", (TW[0], COLOR["awake_eo"])), ("anaes", (TA[0], TCOL[name]))):   # above the axes,
            X = TR[name][k].astype(float)                                     # as (c)'s strips
            sx = ax_f.inset_axes([i - .45, y0, .9, TW[1] - TW[0]], transform=hd); sx.set_in_layout(False)
            window_strip(sx, X, col, every=max(1, X.shape[1] // 5), lw=.25)   # each channel standardised, as G and K
                                                                              # see it: amplitude does not move them
        ax_f.text(i + .45, TW[1] + .004, f"{len(TR[name]['anaes']) // 100} s", transform=hd, ha="right", va="bottom",
                  fontsize=4.6, color=GREY).set_in_layout(False)
        ax_f.text(i, 1.012, FACTS[name].format(n=len(d)), transform=blended_transform_factory(ax_f.transData, ax_f.transAxes),
                  ha="center", va="bottom", fontsize=5.4, color=GREY, linespacing=1.3)
        ax_f.text(i, -.036, rf"\textbf{{{(d < 0).sum()}/{len(d)}}}" + "\nbelow 0", ha="center", va="top",
                  fontsize=5.6, linespacing=1.1)
        bw, bh = {"mouse": (.5, .026), "macaque": (.84, .024), "human": (.84, .026)}[name]
        icon(ax_f.inset_axes([i - bw / 2, -.0715, bw, bh], transform=ax_f.transData), name, root, style=icons)
        sw = {"mouse": .48, "macaque": .56, "human": .26}[name]           # silhouettes as column heads, under the means
        sil = ax_f.inset_axes([i - sw / 2, .0025, sw, .0125], transform=ax_f.transData); sil.set_in_layout(False)
        sil.set_anchor("S"); silhouette(sil, name, root)
    ax_f.text(0, -.0245, f"t(3) = {t.statistic:.1f}".replace("-", "--") + "\n" + f"p = {t.pvalue:.3f}".replace("0.", "."),
              ha="center", va="top", fontsize=5.4, color=GREY, linespacing=1.15)
    ax_f.axhline(0, color=GREY, lw=.6, ls="--", zorder=0)
    ax_f.set_xticks(range(3), [ICON[n][1] for n, _, _ in sets], fontsize=6.3); ax_f.set_xlim(-.5, 2.5)
    ax_f.tick_params(axis="x", length=0); ax_f.tick_params(axis="y", labelleft=False)
    ax_f.text(2.53, top, r"mean $\Delta z$," + "\nown SDs", ha="left", va="top", fontsize=5.6, color=GREY,
              linespacing=1.1, clip_on=False)
    for (lo_, hi_), s_, c_ in ((TW, "wake", COLOR["awake_eo"]), (TA, "anaes.", COLOR["ink"])):
        ax_f.text(2.53, (lo_ + hi_) / 2, s_, transform=hd, ha="left", va="center", fontsize=5.2, color=c_,
                  clip_on=False).set_in_layout(False)
    left.text(0, .99, r"\textbf{d}\enspace\textbf{Generalisation}" + "\nacross species, recordings and agents",
              transform=blended_transform_factory(ax_f.transAxes, left.transSubfigure), ha="left", va="top",
              fontsize=8.5, linespacing=1.2)              # level with (c)'s suptitle

    fig.canvas.draw()                                        # (a)'s fans need the settled layout
    for e_, ax_, side in fans:                               # over the photograph, under its electrodes
        edge = bx.transData.inverted().transform(ax_.transAxes.transform([(1 - side, .04), (1 - side, .96)]))
        bx.add_patch(Polygon([e_, *edge], closed=True, fc=LIGHT, ec="none", alpha=.45, zorder=1, clip_on=False))
    for ext in ("svg", "pdf", "png"):
        fig.savefig(output.with_suffix("." + ext), dpi=180 if ext == "png" else None)
    plt.close(fig)
    print(f"animal means {np.round(animal.values, 4)}, t3 {t.statistic:.2f}, p {t.pvalue:.4f}")
    for name, d, ci in sets:
        print(name.replace("\n", " "), len(d), round(d.mean(), 4), np.round(ci(d), 4), "own wake SD", round(own[name], 5),
              "->", round(d.mean() / own[name], 2), "macaque units", round(d.mean() / sdw, 2))
    print("b windows (z, surrogate, off band)", [(s_, round(df[df.id == w_].real.iloc[0], 4), round(df[df.id == w_].null.iloc[0], 4),
                                                  int(band(edges, w_)[2].sum())) for s_, w_, _ in PANEL_B])
    print(sdw, win.groupby("group")[["d", "sd", "lvl"]].mean().round(4).to_dict())


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--icons", choices=("mesh", "map"), default="mesh", help="(d)'s brains: shaded meshes or region maps")
    p.add_argument("--glyph", choices=("graded", "dots", "hex", "dark"), default="graded", help="(a)'s glyphs")
    p.add_argument("--no-diff", dest="diff", action="store_false", help="(a): drop the shape of the two electrodes' difference")
    p.add_argument("--labels", choices=("symbol", "name"), default="name",
                   help="(a)'s in-tile keys: 'Gaussian MI G' (symbol) or 'Gaussian MI' (name)")
    p.add_argument("--no-link", action="store_true", help="(c): circle the called-out dots without a line to the strip")
    p.add_argument("--b-top", choices=("none", "gmarg", "resid", "big", "time", "clock"), default="time",
                   help="(b)'s top band: as is, a G marginal over the big window, per-state residual curves, larger scatters, Δz over each session (state slots; clock: real seconds within them)")
    p.add_argument("--b-animal", default=None, help="(b): each state's median-Δz window from this animal only (e.g. Chibi)")
    p.add_argument("--c-top", choices=("traces", "delta"), default="traces",
                   help="(c)'s top row: raw windows of called-out dots, or Δz against delta fraction for the animals with sleep")
    a = p.parse_args()
    draw(a.root, a.output, a.icons, a.glyph, a.diff, a.labels, not a.no_link, a.b_top, a.b_animal, a.c_top)
