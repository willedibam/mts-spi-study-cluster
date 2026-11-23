"""Draft redesign of fig:mi (v11): what was measured, the pattern effect in Δz, replication, and its timescale.

    python plot_character_draft.py <results/representation/atlas-mi-pair-261117> <output stem>

Reads the checksum-listed zns arrays (macaque), the segment-surrogate score and calibration, the mouse induction
z arrays, the human propofol (PRIOS) per-patient contrasts from the scorer summary, and the few raw windows it draws
(figwin/windows.npz, copied from the Gadi banks by fig_windows.py). The traces in (a) are the real pair ringed in
(b): the propofol window's off-band pair of largest G inside (b)'s view in both windows. Above (c), four of its dots
drawn as their raw windows (stacked traces of every 4th electrode): the median wake, sleep and propofol windows and
the most extreme window, the propofol one being (b)'s. (d) shares (c)'s axis, so subject and window effects compare
directly. The glyphs in (a) are synthetic (fixed seed): a coupling that switches on and off between "seconds" (the
calibration's switch form) against a Gaussian at the same correlation, with G and K computed on the drawn samples.
The brains are cartoons; the electrodes picked on (a)'s brain are not the real pair's positions.

Presentation quantities not in the manuscript tables: the window residual (z_rec - z_surr minus its session's
eyes-closed wake mean) in units of the eyes-open wake window SD (as review/sleepmatch_ns.py), an exact rescaling,
drawn as the second axis of (c); and the recording pairs outside the surrogate's 95% band around the recording's own
G-K line (chance 5%), a count that a level shift of K cannot move.
"""
from argparse import ArgumentParser
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.path import Path as MPath
from matplotlib.patches import ConnectionPatch, Ellipse, Patch, PathPatch, Rectangle
import numpy as np
import pandas as pd
from scipy.interpolate import splev, splprep
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
         "MD": "#2E86AB", "PF": "#D9A21B", "gauss": "#3E5C76", "mix": "#C8553D", "ink": "#2B2B2B"}
MARK = {"George": "o", "Chibi": "s", "Kin2": "^", "Su": "D"}
GREY, LIGHT = "#78838A", "#D9DDE0"
SEG = {"N0": 10, "N2 L500": 5, "N2 L200": 2, "N2 L100": 1, "N2 L50": .5}
MATCH = (.65, .88)                                       # delta-fraction overlap of drug and sleep windows
Q = 95                                                   # surrogate band
CALLOUT = (("20120726mdanesthesiaandsleepgeorgetoruyanagawa:awake_eo:03", "median"),     # dots of (c) drawn as
           ("20120711slpanesthesiaandsleepgeorgetoruyanagawa:sleep:06", "median"),       # their raw windows
           ("20110621ktmdanesthesiaandsleepchibitoruyanagawa:drug:00", "most extreme"),
           ("20120730pfanesthesiaandsleepchibitoruyanagawa:drug:04", "window in b"))


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


def pair(kind, n=1000, blocks=10, hi=.95, seed=3):
    """Normal-scored samples. "mix": blocks ("seconds") alternate between coupled (r = hi) and uncoupled, the
    calibration's switch form; "gauss": one Gaussian at the mixture's pooled correlation, so both have the same G."""
    rng = np.random.default_rng(seed); on = np.repeat(np.arange(blocks) % 2 == 0, n // blocks)
    u = np.empty((n, 2))
    for b, r in ((on, hi), (~on, 0.)):
        u[b] = rng.multivariate_normal([0, 0], [[1, r], [r, 1]], b.sum())
    u = np.column_stack([normal_score(u[:, 0]), normal_score(u[:, 1])])
    if kind == "gauss":
        r = np.corrcoef(u.T)[0, 1]; z = rng.standard_normal((n, 2)); z -= z.mean(0)   # sample correlation exactly r
        u = z @ np.linalg.inv(np.linalg.cholesky(np.cov(z.T))).T @ np.linalg.cholesky([[1, r], [r, 1]]).T
        u = np.column_stack([normal_score(u[:, 0]), normal_score(u[:, 1])]); on = np.ones(n, bool)
    return u, on


def glyph(ax, kind):
    """Draws 1000 of 4000 samples with the ellipse G sees; returns (G, K) on all 4000 (K: the KSG estimator of
    edges.py; at n = 1000 its small-sample bias alone put the Gaussian's K .03 above its G)."""
    u, on = pair(kind, n=4000, blocks=40)
    G, K = -.5 * np.log(1 - np.corrcoef(u.T)[0, 1] ** 2), ksg(u[:, 0], u[:, 1], 0)
    keep = np.random.default_rng(1).choice(len(u), 1000, replace=False); u, on = u[keep], on[keep]; x, y = u.T
    shade = np.where(on, COLOR[kind], "#EBC3B9" if kind == "mix" else COLOR[kind])
    ax.scatter(x, y, s=1.4, lw=0, color=shade, alpha=.6, rasterized=True)
    r = np.corrcoef(x, y)[0, 1]; tt = np.linspace(0, 2 * np.pi, 200)
    e = np.linalg.cholesky([[1, r], [r, 1]]) @ np.vstack([np.cos(tt), np.sin(tt)]) * 2.45
    ax.plot(e[0], e[1], color=COLOR["ink"], lw=.6)
    ax.set(xlim=(-3.3, 3.3), ylim=(-3.3, 3.3), aspect="equal", xticks=[], yticks=[])
    frame(ax, LIGHT, .8)
    return G, K


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
    ns = json.loads((root / "nonstat/nonstat-score.json").read_text())["nulls"]
    cal = np.load(root / "nonstat/calib1.npz")

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
    pc = COLOR["PF"]
    brain(put([.02, .66, .21, .25]), "macaque", ICON["macaque"][0], pick=(8, 23), pick_color=pc)
    r1.text(.125, .665, "32 electrodes, 496 pairs", ha="center", va="top", fontsize=5.8, color=GREY)
    tr = put([.02, .465, .21, .14])                          # filled after (b), with the ringed pair's traces
    for x_, kind, name in ((.0, "gauss", "Gaussian"), (.13, "mix", "mixture over seconds")):
        G_, K_ = glyph(put([x_ + .015, .085, .09, .09 * W / H]), kind)
        r1.text(x_ + .06, .075, name + "\n" + rf"$G$ {G_:.2f}, $K$ {K_:.2f}".replace("0.", "."), ha="center",
                va="top", fontsize=5.5, color=COLOR[kind], linespacing=1.15)
    for y_, lab, c_ in ((.2, "coupled s", COLOR["mix"]), (.17, "uncoupled s", "#E2A797")):
        r1.text(.238, y_, lab, ha="left", va="center", fontsize=4.8, color=c_)
    r1.text(.125, .44, r"$G$: MI if the pair were Gaussian" + "\n" + r"$K$: MI of any shape (kNN)", ha="center",
            va="top", fontsize=5.6, linespacing=1.25)

    # b: every pair of one wake and one propofol window, square, zoomed on the dense pairs
    sample = df[df.unit == min(df.loc[df.agent == "PF", "unit"])]
    reps = []
    for state in ("awake_ec", "drug"):
        rows = sample[sample.state == state]
        reps.append(rows.loc[(rows.residual - rows.residual.median()).abs().idxmin()])
    full = 1.05 * max(v.max() for rep in reps for v in edges[rep.id][0])
    view = (-.015, .25), (-.14, .36)                     # holds 472 and 478 of the 496 pairs
    side = .29; paxes = []; names, colors = ("Wake, eyes closed", "Propofol"), (COLOR["ink"], COLOR["PF"])
    for j, (rep, name, ink) in enumerate(zip(reps, names, colors)):
        ax = put([.325 + j * .345, .11, side, side * W / H]); paxes.append(ax)
        (g, k), _ = edges[rep.id]
        c, w, out = band(edges, rep.id)
        ov = ax.inset_axes([.03, .6, .36, .37])
        xx = np.linspace(-.05, 1.1 * full, 100)
        for a_, sc in ((ax, 1), (ov, .35)):
            a_.fill_between(xx, np.polyval(c, xx) - w, np.polyval(c, xx) + w, color=LIGHT, lw=0, zorder=0)
            a_.plot(xx, np.polyval(c, xx), color="white", lw=.8 * sc + .2, zorder=1)
            a_.scatter(g[~out], k[~out], s=5 * sc, lw=0, color=GREY, alpha=.4, rasterized=True, zorder=2)
            a_.scatter(g[out], k[out], s=11 * sc, lw=0, color=ink, alpha=.95, zorder=3)
        ov.set(xlim=(-.04, full), ylim=(-.16, full), xticks=[], yticks=[])
        frame(ov, GREY, .6)
        ov.add_patch(Rectangle((view[0][0], view[1][0]), np.diff(view[0])[0], np.diff(view[1])[0], fill=False,
                               ec=COLOR["ink"], lw=.6, zorder=4))
        ov.text(.95, .06, "all pairs", transform=ov.transAxes, ha="right", fontsize=5.2, color=GREY)
        ax.set(xlim=view[0], ylim=view[1])
        ax.set_xticks([0, .1, .2], ["0", ".1", ".2"]); ax.set_yticks([0, .1, .2, .3], ["0", ".1", ".2", ".3"])
        ax.set_title(name, fontsize=8, color=GREY if j == 0 else ink, pad=2, fontweight="bold")
        f3 = lambda v: f"{v:.3f}".lstrip("0")
        ax.text(.97, .16, rf"$z=$\,\textbf{{{f3(rep.real)}}}", transform=ax.transAxes, ha="right", fontsize=7.5,
                color=ink)
        ax.text(.97, .085, rf"its surrogate: {f3(rep.null)}", transform=ax.transAxes, ha="right", fontsize=5.8,
                color=GREY)
        ax.text(.97, .025, f"{out.sum()} pairs off the band (chance {round(len(g) * (100 - Q) / 100)})",
                transform=ax.transAxes, ha="right", fontsize=5.4, color=GREY)
        ax.set_xlabel(r"Gaussian $\mathrm{MI}$, $G$", fontsize=6.8, labelpad=1)
        ax.set_ylabel(r"kNN $\mathrm{MI}$, $K$" if j == 0 else "", fontsize=6.8, labelpad=1)
    # a's traces: the propofol window's off-band pair of largest G inside b's view (both windows), ringed in b
    g, k = edges[reps[1].id][0]; out = band(edges, reps[1].id)[2]
    inview = np.all([(e_[0] < view[0][1]) & (e_[1] < view[1][1]) for e_ in (edges[r_.id][0] for r_ in reps)], 0)
    pk = np.flatnonzero(out & inview)
    pk = pk[np.argmax(g[pk])]
    assert all(np.array_equal(chan[reps[0].id], chan[r_.id]) for r_ in reps)   # same electrodes in both windows
    for ax, rep in zip(paxes, reps):
        ax.scatter(*np.array(edges[rep.id][0])[:, pk], s=34, facecolor="none", edgecolor=COLOR["ink"], lw=.7, zorder=5)
    e1, e2 = (int(v[pk]) for v in np.triu_indices(32, 1))
    with np.load(root / "figwin/windows.npz") as fw:
        lab = list(fw[reps[1].id + "|chan"]); X = fw[reps[1].id][:, [lab.index(chan[reps[1].id][e]) for e in (e1, e2)]]
    X = (X - X.mean(0)) / X.std(0)
    for i_ in range(2):
        tr.plot(X[:, i_] / 5 + (1.1 - 1.1 * i_), color=pc, lw=.4)
        tr.text(-15, 1.1 - 1.1 * i_, "xy"[i_], ha="right", va="center", fontsize=6.3, color=pc, style="italic")
    tr.plot([800, 1000], [-.95] * 2, color=COLOR["ink"], lw=.8)
    tr.text(900, -1.05, "2 s", ha="center", va="top", fontsize=5.4)
    tr.set(xlim=(-30, 1005), ylim=(-1.2, 1.9)); tr.axis("off")
    print("a's pair: electrodes", chan[reps[1].id][e1], chan[reps[1].id][e2], "G, K propofol", g[pk], k[pk],
          "wake", np.array(edges[reps[0].id][0])[:, pk])
    dot = lambda c_, ms: Line2D([], [], ls="", marker="o", ms=ms, color=c_)
    r1.legend([Patch(color=LIGHT), dot(GREY, 2.5), (dot(COLOR["ink"], 3), dot(COLOR["PF"], 3))],
              [r"surrogate's 95\% scatter about the line", "pair", "pair off the band"], loc="center",
              bbox_to_anchor=(.66, .895), ncol=3, frameon=False, fontsize=6, handlelength=1.6, handletextpad=.4,
              columnspacing=1.4, handler_map={tuple: HandlerTuple(ndivide=None, pad=.3)})
    r1.text(.66, .85, r"$z$: correlation of $G$ and $K$ across the 496 pairs; the surrogate keeps every $G$ and is "
            r"Gaussian, so only $K$ moves", ha="center", va="center", fontsize=5.9, color=GREY)
    for x_, l_, c_, r_ in ((.0, "a", "One pair", ""), (.29, "b", "Every pair", "one 10-s window, wake vs propofol")):
        r1.text(x_, .975, rf"\textbf{{{l_}}}\enspace\textbf{{{c_}}}" + (f": {r_}" if r_ else ""), fontsize=8.5,
                va="top")

    # c-e: the pattern effect, its replication, its timescale
    left, right = r2.subfigures(1, 2, width_ratios=[2.3, 1], wspace=.02)
    gs = left.add_gridspec(2, 2, height_ratios=[.2, 1], width_ratios=[1.4, .62], hspace=.04, wspace=.05)
    sg = gs[0, 0].subgridspec(1, len(CALLOUT), wspace=.1)
    strips = [left.add_subplot(sg[0, j]) for j in range(len(CALLOUT))]
    ax_c = left.add_subplot(gs[1, 0]); ax_f = left.add_subplot(gs[1, 1], sharey=ax_c)
    head_d = left.add_subplot(gs[0, 1]); head_d.axis("off")
    ax_h = right.subplots()
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
    for y_, lab in ((top, r"$\Delta z$"), (top - .0047, "SD")):
        ax_c.text(-.72, y_, lab, ha="right", va="top", fontsize=6.3, color=GREY, clip_on=False)
    ax_c.axhline(0, color=GREY, lw=.6, ls="--", zorder=0)
    ax_c.set_ylim(-.074, top + .001); ax_c.set_yticks([-.07, -.06, -.05, -.04, -.03, -.02, -.01, 0],
                                                      ["$-.07$", "", "$-.05$", "", "$-.03$", "", "$-.01$", "0"])
    ax_c.set_ylabel(r"$\Delta z$, net of surrogate (each 10-s window)")
    panel_title(left, "c", "Pattern", r"$z$ falls under every anaesthetic, not in sleep", sub=True)
    states_axis(ax_c)
    assert CALLOUT[-1][0] == reps[1].id
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
            con = ConnectionPatch((.5, -.34), (x_, y_), coordsA=sax.transAxes, coordsB=ax_c.transData,
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
        sc_ = {"mouse": .55, "macaque": .75, "human": .95}[name]
        ic = ax_f.inset_axes([i - .42 * sc_, -.07, .84 * sc_, .025 * sc_], transform=ax_f.transData)
        brain(ic, name, ICON[name][0], dot=1.2 if name != "mouse" else 2)
    ax_f.text(0, -.0245, f"t(3) = {t.statistic:.1f}".replace("-", "--") + "\n" + f"p = {t.pvalue:.3f}".replace("0.", "."),
              ha="center", va="top", fontsize=5.4, color=GREY, linespacing=1.15)
    ax_f.axhline(0, color=GREY, lw=.6, ls="--", zorder=0)
    ax_f.set_xticks(range(3), [ICON[n][1] for n, _, _ in sets], fontsize=6.3); ax_f.set_xlim(-.5, 2.5)
    ax_f.tick_params(axis="x", length=0); ax_f.tick_params(axis="y", labelleft=False)
    sec = ax_f.secondary_yaxis("right", functions=(lambda v: v / sdw, lambda v: v * sdw))
    sec.set_yticks(range(-16, 1, 4)); sec.set_ylabel("in wake SDs", labelpad=1)
    head_d.set_title(r"\textbf{d}\enspace\textbf{Replication}" + "\nper subject, same axis", loc="left",
                     fontsize=8.5, pad=1.5, linespacing=1.2)

    ax = ax_h
    keys = list(SEG); x = np.arange(len(keys))
    pooled = np.array([ns[k]["pooled"] for k in keys]) / ns["N0"]["pooled"]
    an = np.array([ns[k]["animals"] for k in keys]); an = an / an[0]
    for q, an_name in enumerate(animal.index):              # same animal order as the score (Chibi, George, Kin2, Su)
        ax.plot(x, an[:, q], color=LIGHT, lw=.8, zorder=1)
        ax.scatter(x, an[:, q], marker=MARK[an_name], s=11, facecolor="white", edgecolor=GREY, lw=.7, zorder=2)
    ax.plot(x, pooled, color=COLOR["ink"], lw=1.6, zorder=3)
    ax.scatter(x, pooled, s=20, color=COLOR["ink"], zorder=4)
    ax.legend([Line2D([], [], color=COLOR["ink"], lw=1.6, marker="o", ms=4),
               Line2D([], [], color=LIGHT, lw=.8, marker="o", ms=3.5, mfc="white", mec=GREY)],
              ["pooled", "per macaque"], loc="lower left", frameon=False, fontsize=5.6, handlelength=1.8)
    rel = lambda gname: cal[gname].mean(0)[3] / cal[gname].mean(0)[0]
    tv = [rel(g) for g in ("gain_local", "gain_cmfloor", "switch", "burst_slow")]
    xs_ = 5.2                                              # simulated references at 1 s, labelled directly
    ax.plot([xs_ - .25, xs_ + .25], [rel("quadratic")] * 2, color=COLOR["gauss"], lw=1.6)
    ax.text(xs_, rel("quadratic") - .04, "fixed\ncurve", fontsize=5.4, ha="center", va="top",
            color=COLOR["gauss"], linespacing=1.05)
    ax.add_patch(Rectangle((xs_ - .25, min(tv)), .5, max(tv) - min(tv), color=COLOR["mix"], alpha=.5, lw=0))
    ax.text(xs_, min(tv) - .04, "varying\nover s", fontsize=5.4, ha="center", va="top", color=COLOR["mix"],
            linespacing=1.05)
    ax.text(xs_, -.045, "simul.\n1 s", fontsize=5.4, ha="center", va="top", color=GREY, linespacing=1.05, clip_on=False)
    ax.plot([3.1, xs_ - .3], [pooled[3]] * 2, color=COLOR["ink"], lw=.5, ls=":", zorder=0)   # 1 s vs simulated
    ax.axvline(4.6, color=LIGHT, lw=.6)
    ax.axhline(1, color=GREY, lw=.6, ls="--", zorder=0)
    ax.set_xticks(x, ["10", "5", "2", "1", ".5"]); ax.tick_params(axis="x", pad=13, length=0)
    ax.set_ylim(0, 1.32); ax.set_xlim(-.5, 5.6)
    ax.set_yticks([0, .5, 1], ["0", "50\\%", "100\\%"])
    ax.set_ylabel("drug change retained", labelpad=1)
    for i, k in enumerate(keys):                       # segmentation glyph under each bar
        nseg = int(round(10 / SEG[k]))
        for j in range(nseg):
            ax.add_patch(Rectangle((i - .3 + j * .6 / nseg, -.075), .6 / nseg * .8, .04, color=GREY, lw=0,
                                   clip_on=False))
    ax.set_xlabel("surrogate refitted within segments (s)", labelpad=2)
    panel_title(ax, "e", "Timescale", "about a second")

    for ext in ("pdf", "png"):
        fig.savefig(output.with_suffix("." + ext), dpi=600 if ext == "png" else None)
    plt.close(fig)
    print(f"animal means {np.round(animal.values, 4)}, t3 {t.statistic:.2f}, p {t.pvalue:.4f}")
    for name, d, ci in sets:
        print(name.replace("\n", " "), len(d), round(d.mean(), 4), np.round(ci(d), 4))
    print("example windows z", [(round(r.real, 4), round(r.null, 4)) for r in reps], "outside", [int(band(edges, r.id)[2].sum()) for r in reps])
    print(sdw, win.groupby("group")[["d", "sd", "lvl"]].mean().round(4).to_dict())


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path); p.add_argument("output", type=Path)
    a = p.parse_args()
    draw(a.root, a.output)
