"""One-page case figure, first mock-up (2026-10-26): form ladder over timing ladder.

Each row reads left to right: construction -> single pairs -> edge-profile planes (one recording)
-> z across recordings. Data and helpers come from the two single-case composites
(`r_rho_mi/figures/r_rho_mi_composite_261024.py`, `r_xcorr_p/timing_composite_261025.py`),
which keep the panels moved to the SI (matrices, strength spread, narrow bands). Runs no pyspi.

    python -m research.cases.cases_page_261026 [--form sigmoid|exp1|exp1.5] [--live-text]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from mtsspi.generators import band_spectrum, generate_band_delayed_mts
from research.cases.composite_style import (HAIR, INK, MUTED, SKY, TRACE, axes_in, bare, carpet, panel_title, rugs,
                                            save, shade, style, tint, z_label)
from research.cases.r_rho_mi.figures import r_rho_mi_composite_261024 as F
from research.cases.r_xcorr_p import timing_composite_261025 as Tm
from research.cases.r_xcorr_p import timing_triplet as tt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/cases/cases-page-261026"
W, H = 7.09, 7.55
ROW_B = 3.72                      # top of the timing row, inches
LEFT_W = 2.6                      # construction + pairs block
PX0, PS, PG = 2.98, 0.8, 0.07     # planes: left edge, size, gap
ZX, ZW = 5.86, 1.18               # z panel
GREY = "#B4B7BD"
TITLE_INK = (shade(GREY, 0.62), shade("#0072B2", 0.9), shade("#CC79A7", 0.8))   # reference, first, second departure


def zfmt(v):
    return "1" if v >= 0.995 else (f"{abs(v):.2f}".lstrip("0") if abs(v) >= 0.005 else "0")


def row_header(fig, y, head, sub):
    bold = (lambda s: rf"\textbf{{{s}}}") if plt.rcParams["text.usetex"] else (lambda s: s)
    fig.text(0, 1 - y / H, bold(head), fontsize=9.5, va="baseline", transform=fig.transFigure)
    fig.text(1.42 / W, 1 - y / H, sub, fontsize=8, color=MUTED, va="baseline", transform=fig.transFigure)
    fig.add_artist(Line2D([0, 1], [1 - (y + 0.08) / H] * 2, color=HAIR, lw=0.6, transform=fig.transFigure))


def bars(ax, vals, cols, labels, title):
    ax.bar(range(3), np.clip(vals, 0, None), width=0.62, color=cols, edgecolor=[shade(c, 0.7) for c in cols], lw=0.5)
    for k, v in enumerate(vals):
        ax.text(k, max(v, 0) + 0.04 * ax.get_ylim()[1], zfmt(v), ha="center", va="bottom", fontsize=6.5)
    ax.axhline(0, color=INK, lw=0.5)
    ax.set_xlim(-0.55, 2.55)
    ax.set_xticks(range(3), labels)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=1.2, labelsize=7)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=7.5, pad=1, loc="left")


def plane_axes(fig, y, r, c):
    return axes_in(fig, PX0 + c * (PS + PG), y + r * (PS + 0.36), PS, PS)


def finish_plane(ax, r, c, ticks, labs, xl, yl, col_title, col_colour=INK):
    ax.set(xticks=ticks, yticks=ticks if yl else ticks)
    ax.set_xticklabels(labs)
    ax.set_yticklabels(labs if c == 0 else [])
    ax.tick_params(labelsize=6.5)
    if c == 0:
        ax.set_ylabel(yl, labelpad=1)
    if c == 1:
        ax.set_xlabel(xl, labelpad=1)
    if r == 0:
        ax.text(0.5, 1.05, col_title, transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, color=col_colour)


def z_dots(fig, y, frame_by_fam, specs, cats, n_label):
    """frame_by_fam: {key: [array per category]}; specs: (key, dx, marker, ls, colour, label)."""
    ax = axes_in(fig, ZX, y, ZW, 1.62)
    rng = np.random.default_rng(1)
    for key, dx, mk, ls, col, _ in specs:
        means = []
        for k, v in enumerate(frame_by_fam[key]):
            ax.scatter(k + dx + rng.uniform(-0.045, 0.045, len(v)), v, s=2.2, lw=0, color=MUTED, alpha=0.55, zorder=2)
            means.append(np.mean(v))
        ax.plot(np.arange(len(cats)) + dx, means, color=col, lw=0.9, ls=ls, marker=mk, ms=3, mew=0, zorder=3)
    ax.set(xlim=(-0.4, len(cats) - 0.6), ylim=(0, 1.05), xticks=range(len(cats)), yticks=[0, 0.5, 1])
    ax.set_xticklabels(cats, fontsize=7)
    ax.set_yticklabels(["0", ".5", "1"])
    ax.set_ylabel("$z$", labelpad=1)
    ax.text(0.97, 0.03, n_label, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color=MUTED)
    handles = [Line2D([], [], color=col, lw=0.9, ls=ls, marker=mk, ms=3, mew=0) for _, _, mk, ls, col, _ in specs]
    fig.legend(handles, [s[-1] for s in specs], loc="upper left", bbox_to_anchor=((ZX - 0.05) / W, 1 - (y + 1.88) / H),
               fontsize=7.5, handlelength=2.2, handletextpad=0.4, labelspacing=0.3)
    return ax


# --------------------------------------------------------------------------- form row

EXP_DIR = ROOT / "results/cases/r-rho-mi-exp-261027"
FORMS = {"sigmoid": None, "exp1": 1.0, "exp1.5": 1.5}


def form_source(form):
    """Instance-0 recordings per construction, the z table and the M map, for the archive or an exp-map run."""
    s = FORMS[form]
    if s is None:
        recs = {}
        for cls in F.CONS:
            x, mother, meta, m = F.load(cls, F.INSTANCE)
            recs[cls] = (x, mother, meta, m)
        table = cached_z_table()
        return dict(recs=recs, table=table, m_map=lambda u: 1 / (1 + np.exp(-np.pi * u)), zoom=True, n=F.N_INST,
                    note=r"M $=\sigma(\pi u)$")
    keys = {F.R: F.R, F.RHO: F.RHO, F.MI: F.MI.replace("-", "_")}
    recs = {}
    for cls, cons in zip(F.CONS, ("L", "L+M", "L+M+Q")):
        tag = f"{cons.replace('+', '')}_s{0 if cons == 'L' else s:g}"
        d = np.load(EXP_DIR / f"example_{tag}.npz")
        meta = {"generator": {"types": [str(t) for t in d["types"]], "noise_stds": d["noise_stds"]}}
        recs[cls] = (d["x"].astype(float), d["u"].astype(float), meta, {k: d[v] for k, v in keys.items()})
    frame = pd.read_csv(EXP_DIR / "z_by_recording.csv")
    frame = frame[(frame.scale == 5) & ((frame.s == s) | (frame.construction == "L"))]
    frame = frame.assign(cls=frame.construction.map(dict(zip(("L", "L+M", "L+M+Q"), F.CONS))))
    return dict(recs=recs, table=frame, m_map=lambda u: np.exp(s * u), zoom=False,
                n=int(frame.groupby("cls").size().min()), note=rf"M $=e^{{{"" if s == 1 else f"{s:g}"}u}}$")


def form_row(fig, y, form="sigmoid"):
    src = form_source(form)
    row_header(fig, y + 0.14, "Functional form", r"$r$, $\rho$, MI: linear, then monotone, then any dependence")
    x3, mother, meta3, m3 = src["recs"]["iter3_LMNM"]
    types = [F.FAMILY[s] for s in meta3["generator"]["types"]]

    # a construction
    panel_title(fig, 0, y + 0.44, "a", "Construction", f"one latent $u$; L, M, Q maps; {src['note']}")
    t0, n, cl, cw, ch = 400, 160, 0.62, LEFT_W - 0.62, 0.8
    tr = axes_in(fig, cl, y + 0.66, cw, 0.2)
    tr.plot(mother[t0:t0 + n], color=TRACE, lw=0.6)
    bare(tr)
    tr.set_xlim(0, n - 1)
    tr.text(-0.02, 0.5, "$u$", transform=tr.transAxes, ha="right", va="center", fontsize=8)
    cp = axes_in(fig, cl, y + 0.9, cw, ch)
    carpet(cp, x3[t0:t0 + n].T)
    bounds = np.flatnonzero(np.diff([ord(c) for c in types])) + 0.5
    for b in bounds:
        cp.axhline(b, color="white", lw=0.7)
    u = np.linspace(-2.2, 2.2, 200)
    maps = {"L": u, "M": src["m_map"](u), "Q": u ** 2}
    edges, rh = np.r_[-0.5, bounds, len(types) - 0.5], ch / len(types)
    for fam, lo, hi in zip("LMQ", edges[:-1], edges[1:]):
        h = (hi - lo) * rh
        g = min(0.24, 0.85 * h)
        ga = axes_in(fig, cl - 0.06 - g, y + 0.9 + lo * rh + 0.5 * (h - g), g, g)
        ga.plot(u, maps[fam], color=INK, lw=0.8)
        bare(ga)
        ga.text(-0.18, 0.5, fam, transform=ga.transAxes, ha="right", va="center", fontsize=7.5)
    cp.text(1, -0.04, f"{n} samples", transform=cp.transAxes, ha="right", va="top", fontsize=6.5, color=MUTED)

    # b pairs
    panel_title(fig, 0, y + 1.98, "b", "Channel pairs", r"bars: $r$, $\rho$, MI as $r_I=\sqrt{1-e^{-2I}}$")
    pairs = F.example_pairs(meta3, m3)[:3]
    words = {"L–L": ("linear", "linear"), "L–M": ("monotone", "monotone"), "L–Q": ("non-monotone", "non-monotone")}
    sq, pitch = 0.66, (LEFT_W - 0.12 - 0.66) / 2
    for k, (name, i, j) in enumerate(pairs):
        left = 0.12 + k * pitch
        bx = axes_in(fig, left, y + 2.38, sq, 0.24)
        bx.set_ylim(-0.03, 1.45)
        bars(bx, [m3[F.R][i, j], m3[F.RHO][i, j], float(F.linfoot(m3[F.MI][i, j]))],
             [F.HUE[F.R], F.HUE[F.RHO], F.HUE[F.MI]], ["$r$", r"$\rho$", "$r_I$"], words[name][0])
        ax = axes_in(fig, left, y + 2.78, sq, sq)
        ax.scatter(x3[:, i], x3[:, j], s=0.3, lw=0, color=F.REL[words[name][1]][0], alpha=0.45, rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"$x_{{{i + 1}}}$ ({name[0]})", fontsize=7, labelpad=1)
        ax.set_ylabel(f"$x_{{{j + 1}}}$ ({name[-1]})", fontsize=7, labelpad=1)

    # c planes
    panel_title(fig, PX0 - 0.3, y + 0.44, "c", "Edge-profile planes", "one recording; each point a channel pair")
    recs = {cls: (m, F.relations(meta["generator"]["types"])[0]) for cls, (_, _, meta, m) in src["recs"].items()}
    mi_max = max(float(np.max(F.upper(m[F.MI]))) for m, _ in recs.values())
    rows = [(F.R, F.RHO, "$r_{ij}$", r"$\rho_{ij}$"), (F.RHO, F.MI, r"$\rho_{ij}$", "$I_{ij}$ (nats)")]
    py = y + 0.84
    zs = {}
    for r, (kx, ky, xl, yl) in enumerate(rows):
        for c, (cls, (m, rel)) in enumerate(recs.items()):
            ax = plane_axes(fig, py, r, c)
            vx, vy = F.upper(m[kx]), F.upper(m[ky])
            if ky == F.MI:
                grid = np.linspace(0, 0.9995, 300)
                ax.plot(grid, F.gaussian_mi_of_rho(grid), color=MUTED, lw=0.6, ls=(0, (2.5, 1.5)), zorder=1)
                ax.set(xlim=(-0.06, 1.06), ylim=(-0.15, mi_max * 1.08))
            else:
                ax.plot([-0.06, 1.06], [-0.06, 1.06], color=HAIR, lw=0.6, zorder=0)
                ax.set(xlim=(-0.06, 1.06), ylim=(-0.06, 1.06))
            for key in ("linear", "monotone", "non-monotone"):
                sel = rel == key
                ax.scatter(vx[sel], vy[sel], s=3, lw=0, color=F.REL[key][0], zorder=3 if key != "linear" else 2)
            rugs(ax, vx, vy, shade(F.HUE[kx], 0.85), shade(F.HUE[ky], 0.85))
            zs[(r, cls)] = F.z(vx, vy)
            z_label(ax, zs[(r, cls)], x=0.06, y=0.94, fontsize=7)
            if ky == F.MI:
                ax.set(xticks=[0, 1], yticks=[0, 4])
                ax.set_xticklabels(["0", "1"])
                ax.set_yticklabels(["0", "4"] if c == 0 else [])
                ax.tick_params(labelsize=6.5)
                if c == 0:
                    ax.set_ylabel(yl, labelpad=1)
                    ax.text(0.5, 0.42, "Gaussian", transform=ax.transAxes, fontsize=6.5, color=MUTED, ha="center")
                if c == 1:
                    ax.set_xlabel(xl, labelpad=1)
            else:
                finish_plane(ax, r, c, [0, 1], ["0", "1"], xl, yl, F.CONS[cls].replace(" ", ""), TITLE_INK[c])
                if src["zoom"]:
                    zoom(ax, vx, vy, rel)
    handles = [Line2D([], [], ls="", marker="o", ms=3.2, mew=0, color=F.REL[k][0]) for k in F.REL]
    fig.legend(handles, ["same family", "monotone", "non-monotone"], loc="upper left", ncol=3,
               bbox_to_anchor=((PX0 - 0.12) / W, 1 - (py + 2 * PS + 0.36 + 0.3) / H), fontsize=7.5,
               handletextpad=0.1, columnspacing=0.8)

    # d z across recordings
    panel_title(fig, ZX - 0.35, y + 0.44, "d", "Across recordings")
    table = src["table"]
    data = {k: [table[table.cls == cls][k].to_numpy() for cls in F.CONS] for k in ("z_r_rho", "z_rho_mi")}
    z_dots(fig, y + 0.84, data,
           [("z_r_rho", -0.1, "o", "-", INK, r"$z(r,\rho)$"), ("z_rho_mi", 0.1, "s", (0, (3, 1.5)), INK, r"$z(\rho,\mathrm{MI})$")],
           ["L", "L+M", "L+M+Q"], f"{src['n']} each")
    return {"pairs": [(n, i + 1, j + 1) for n, i, j in pairs], "z_instance": {f"{r}|{c}": v for (r, c), v in zs.items()},
            "z_mean": table.groupby("cls")[["z_r_rho", "z_rho_mi"]].mean().round(4).to_dict()}


def zoom(ax, vx, vy, rel, lo=0.875):
    ax.add_patch(Rectangle((lo, lo), 1.0 - lo + 0.012, 1.0 - lo + 0.012, fill=False, lw=0.45, ec=MUTED, zorder=4))
    ins = ax.inset_axes([0.42, 0.05, 0.53, 0.53])
    ins.plot([lo, 1.005], [lo, 1.005], color=HAIR, lw=0.5, zorder=0)
    keep = (vx > lo - 0.01) & (vy > lo - 0.01)
    for key in ("linear", "monotone"):
        sel = keep & (rel == key)
        ins.scatter(vx[sel], vy[sel], s=1.6, lw=0, color=F.REL[key][0], zorder=3 if key == "monotone" else 2)
    ins.set(xlim=(lo, 1.005), ylim=(lo, 1.005), xticks=[], yticks=[])
    for s in ins.spines.values():
        s.set_linewidth(0.45); s.set_color(MUTED)


def cached_z_table() -> pd.DataFrame:
    path = OUT / "form_z_by_recording.csv"
    if path.exists():
        return pd.read_csv(path)
    OUT.mkdir(parents=True, exist_ok=True)
    table = F.z_table()
    table.to_csv(path, index=False)
    return table


# --------------------------------------------------------------------------- timing row

def timing_row(fig, y):
    row_header(fig, y + 0.14, "Temporal alignment",
               r"$r \le \max_h r(h) \le P$: no realignment, then one delay, then any phase per frequency")
    frame, mats, pop, _ = Tm.compute(False)
    it = Tm.internals()
    chosen = Tm.choose_pairs(it, mats)

    # e construction
    panel_title(fig, 0, y + 0.44, "e", "Construction", "two bands; each reaches channel $i$ after delay $\\tau_i$")
    ax = axes_in(fig, 0.12, y + 0.8, 1.0, 0.7)
    f = np.linspace(0, 0.5, 600)
    for (c, w), var, lab in zip(it.bands, it.band_var, ("slow", "fast")):
        s = band_spectrum(f, c, w, var)
        ax.fill_between(f, s, color=tint(MUTED, 0.35), lw=0)
        ax.plot(f, s, color=INK, lw=0.7)
        ax.text(c + (0.05 if c == 0 else 0), s.max() * (0.85 if c == 0 else 1.08), lab, fontsize=7, ha="left" if c == 0 else "center")
    ax.axhline(0, color=INK, lw=0.5)
    ax.set(xlim=(0, 0.5), xticks=[0, 0.25, 0.5], yticks=[])
    ax.set_xticklabels(["0", ".25", ".5"])
    ax.tick_params(labelsize=6.5)
    ax.set_xlabel("frequency", labelpad=1)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    dm = axes_in(fig, 1.84, y + 0.8, 0.7, 0.7)
    for k, (word, col) in enumerate(zip(("sync", "delay", "dispersive"), TITLE_INK)):
        fig.text((1.42 + 0.36 * k + (0.1 if k == 2 else 0)) / W, 1 - (y + 0.74) / H, word, color=col, fontsize=7,
                 transform=fig.transFigure, va="baseline")
    lag = Tm.PRM["lag_range"]
    for fam, col, mk, size in (("synchronous", GREY, "o", 14), ("delayed", Tm.DELAY, "o", 7), ("dispersive", Tm.DISP, "o", 7)):
        _, itf = generate_band_delayed_mts(**{**Tm.PRM, "T": 1}, family=fam, rng=0, return_internals=True)
        dm.scatter(itf.delays[0], itf.delays[1], s=size, lw=0, color=col, zorder=3 if fam != "dispersive" else 2,
                   alpha=0.9)
    dm.plot([-1, lag + 1], [-1, lag + 1], color=HAIR, lw=0.6, zorder=0)
    dm.set(xlim=(-1, lag + 1), ylim=(-1, lag + 1), xticks=[0, lag], yticks=[0, lag])
    dm.tick_params(labelsize=6.5)
    dm.set_xlabel(r"slow delay $\tau^{\mathrm{s}}_i$", labelpad=1, fontsize=7)
    dm.set_ylabel(r"fast delay $\tau^{\mathrm{f}}_i$", labelpad=1, fontsize=7)

    # f pairs
    panel_title(fig, 0, y + 1.98, "f", "Channel pairs", r"cross-correlogram $r_{ij}(h)$; bars: $r$, $\max_h$, $P$")
    show = [chosen[0], chosen[1], chosen[3]]
    names = ["sync", "delay", "dispersive"]
    sq, pitch = 0.66, (LEFT_W - 0.12 - 0.66) / 2
    pairs = []
    for k, (fam, (i, j)) in enumerate(show):
        x = generate_band_delayed_mts(**Tm.PRM, family=fam, rng=tt.recording_rng(fam, Tm.REP))
        hs, cc = Tm.correlogram(x, i, j)
        r0, xm, pv = (Tm.stat(mats, fam, key)[i, j] for key in (Tm.R, Tm.XC, Tm.P))
        left = 0.12 + k * pitch
        bx = axes_in(fig, left, y + 2.38, sq, 0.24)
        bx.set_ylim(0, 0.85)
        bars(bx, [r0, xm, pv], [Tm.HUE[Tm.R], Tm.HUE[Tm.XC], Tm.HUE[Tm.P]], ["$r$", "max", "$P$"], names[k])
        ax = axes_in(fig, left, y + 2.78, sq, sq)
        ax.axhline(0, color=HAIR, lw=0.5, zorder=0)
        ax.axvline(0, color=HAIR, lw=0.5, zorder=0)
        ax.plot(hs, cc, color=INK, lw=0.6)
        ax.axhline(pv, color=shade(SKY, 0.8), lw=0.6, ls=(0, (2, 1.2)))
        ax.plot(0, r0, "o", ms=2.6, color=shade(Tm.HUE[Tm.R], 0.8), zorder=4)
        ax.plot(hs[np.argmax(cc)], cc.max(), "o", ms=2.6, color=shade(Tm.HUE[Tm.XC], 0.85), zorder=4)
        ax.set(xlim=(-30, 30), ylim=(-0.45, 0.75), xticks=[-30, 0, 30], yticks=[0, 0.5])
        ax.set_xticklabels(["$-30$", "0", "30"])
        ax.set_yticklabels(["0", ".5"] if k == 0 else [])
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel("lag $h$", fontsize=7, labelpad=1)
        pairs.append((fam, i + 1, j + 1))

    # g planes
    panel_title(fig, PX0 - 0.3, y + 0.44, "g", "Edge-profile planes", "one recording; each point a channel pair")
    slow = it.delays[0]
    dtau = F.upper(np.abs(slow[:, None] - slow[None, :]).astype(float))
    mism = F.upper(tt.band_mismatch(it))
    cm_delay = LinearSegmentedColormap.from_list("d", [tint(Tm.DELAY, 0.2), Tm.DELAY, shade(Tm.DELAY, 0.6)])
    cm_disp = LinearSegmentedColormap.from_list("q", [tint(Tm.DISP, 0.2), Tm.DISP, shade(Tm.DISP, 0.6)])
    rows = [(Tm.R, Tm.XC, "$r_{ij}$", r"$\max_h r_{ij}(h)$"), (Tm.XC, Tm.P, r"$\max_h r_{ij}(h)$", "$P_{ij}$")]
    lo = min(-0.05, min(float(F.upper(Tm.stat(mats, fam, Tm.R)).min()) for fam in Tm.FAM) - 0.03)
    py = y + 0.84
    handles, zs = {}, {}
    for r, (kx, ky, xl, yl) in enumerate(rows):
        for c, fam in enumerate(Tm.FAM):
            ax = plane_axes(fig, py, r, c)
            vx, vy = F.upper(Tm.stat(mats, fam, kx)), F.upper(Tm.stat(mats, fam, ky))
            ax.plot([-1, 1], [-1, 1], color=HAIR, lw=0.6, zorder=0)
            if fam == "synchronous":
                ax.scatter(vx, vy, s=3, lw=0, color=GREY, zorder=2)
            elif r == 0:
                o = np.argsort(dtau)
                handles["d"] = ax.scatter(vx[o], vy[o], s=3, lw=0, c=dtau[o], cmap=cm_delay, vmin=0, vmax=lag, zorder=2)
            else:
                cval = mism if fam == "dispersive" else np.zeros_like(mism)
                o = np.argsort(cval)
                handles["q"] = ax.scatter(vx[o], vy[o], s=3, lw=0, c=cval[o], cmap=cm_disp, vmin=0,
                                          vmax=float(mism.max()), zorder=2)
            rugs(ax, vx, vy, shade(Tm.HUE[kx], 0.85), shade(Tm.HUE[ky], 0.85))
            zs[(r, fam)] = F.z(vx, vy)
            z_label(ax, zs[(r, fam)], x=0.06, y=0.94, fontsize=7)
            ax.set(xlim=(lo if kx == Tm.R else -0.02, 0.72), ylim=(-0.02, 0.72))
            finish_plane(ax, r, c, [0, 0.6], ["0", ".6"], xl, yl, {"synchronous": "sync", "delayed": "delay",
                                                                    "dispersive": "dispersive"}[fam], TITLE_INK[c])
    cb_y = py + 2 * PS + 0.36 + 0.42
    for k, (key, lab, ticks) in enumerate((("d", r"$|\tau^{\mathrm{s}}_i-\tau^{\mathrm{s}}_j|$", [0, lag]),
                                           ("q", r"$|\Delta^{\mathrm{s}}_{ij}-\Delta^{\mathrm{f}}_{ij}|$",
                                            [0, int(mism.max())]))):
        cax = axes_in(fig, PX0 + 0.05 + k * 1.35, cb_y, 1.0, 0.05)
        cb = fig.colorbar(handles[key], cax=cax, orientation="horizontal")
        cb.set_ticks(ticks)
        cb.ax.tick_params(labelsize=6.5, length=2, pad=1)
        cb.outline.set_linewidth(0.4)
        cb.ax.set_title(lab, fontsize=7, pad=2)

    # h z across recordings
    panel_title(fig, ZX - 0.35, y + 0.44, "h", "Across recordings")
    get = lambda name: [frame[(frame.pair == name) & (frame.family == fam)].z.to_numpy() for fam in Tm.FAM]
    data = {"rx": get(tt.pair_name(Tm.R, Tm.XC)), "xp": get(tt.pair_name(Tm.XC, Tm.P)),
            "xc": get(tt.pair_name(Tm.XC, Tm.CM))}
    z_dots(fig, y + 0.84, data,
           [("rx", -0.14, "o", "-", INK, r"$z(r,\max)$"), ("xp", 0.0, "s", (0, (3, 1.5)), INK, r"$z(\max,P)$"),
            ("xc", 0.14, "D", (0, (1, 1.2)), MUTED, r"$z(\max,\mathrm{coh})$")],
           ["sync", "delay", "disp."], f"{Tm.N_REP} each")
    return {"pairs": pairs, "z_instance": {f"{r}|{c}": v for (r, c), v in zs.items()},
            "z_mean": {f"{k}|{fam}": float(np.mean(v)) for k, vs in data.items() for fam, v in zip(Tm.FAM, vs)}}


def build(live_text=False, form="sigmoid"):
    style(live_text)
    fig = plt.figure(figsize=(W, H))
    summary = {"form_source": form, "form": form_row(fig, 0.0, form), "timing": timing_row(fig, ROW_B)}
    return fig, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live-text", action="store_true")
    ap.add_argument("--form", choices=FORMS, default="exp1", help="form-case source: archive sigmoid or exp-map run")
    args = ap.parse_args(argv)
    fig, summary = build(args.live_text, args.form)
    stem = "cases_page" + ("" if args.form == "sigmoid" else f"_{args.form}") + ("_live-text" if args.live_text else "")
    print(save(fig, OUT, stem))
    (OUT / f"{stem}.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
