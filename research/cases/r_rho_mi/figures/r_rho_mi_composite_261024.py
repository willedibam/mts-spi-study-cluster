"""r–ρ–MI case composite, first pass (2026-10-24).

Reads the archived 2026-09-24 corpus (pyspi r7, beta = pi; `configs/260924_beta.yaml`) and the
strength-spread sweep from `research/cases/r_rho_mi/strength_spread_261024.py`; runs no pyspi.
Every z is the representation's coordinate: Pearson correlation of two edge profiles over the
upper triangle (equal to the ordered-entry z for these symmetric statistics), with MI in nats.
The Linfoot scale appears only in panel c's bars.

    python -m research.cases.r_rho_mi.figures.r_rho_mi_composite_261024 [--live-text]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from research.cases.composite_style import (AMBER, HAIR, INK, LAVENDER, MUTED, SAGE, TRACE, axes_in, bare,
                                            carpet, panel_title, rugs, save, seq, shade, style, z_label)

ROOT = Path(__file__).resolve().parents[4]
ARCHIVE = ROOT / "data/cases/r-rho-mi-260924/data"
OUT = ROOT / "results/cases/r-rho-mi-261024"
R, RHO, MI = "cov_EmpiricalCovariance", "spearmanr", "mi_kraskov_NN-4"
HUE = {R: SAGE, RHO: LAVENDER, MI: AMBER}
BETA, INSTANCE, N_INST = "beta-pi", 0, 50
CONS = {"iter1_L": "L", "iter2_LM": "L + M", "iter3_LMNM": "L + M + Q"}
FAMILY = {"linear": "L", "monotonic": "M", "non-monotonic": "Q"}
REL = {"linear": ("#B4B7BD", "same family (linear)"), "monotone": ("#0072B2", "L–M (monotone)"),
       "non-monotone": ("#CC79A7", "L–Q, M–Q (non-monotone)")}
W, H = 7.09, 7.97


# --------------------------------------------------------------------------- data

def load(cls: str, inst: int):
    name = f"M32_T2000_I{inst}" + ("" if cls == "iter1_L" else f"_{BETA}")
    path = ARCHIVE / cls / name
    meta = json.loads((path / "meta.json").read_text())
    with np.load(path / "spi_mpis.npz") as bank:
        mats = {k: bank[k] for k in (R, RHO, MI)}
    return np.load(path / "timeseries.npy"), np.load(path / "mother.npy"), meta, mats


def upper(m):
    return m[np.triu_indices(m.shape[0], 1)]


def z(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def relations(types):
    """Generating relation per upper-triangle pair: same family linear, L–M monotone, any Q-cross non-monotone."""
    t = np.asarray([FAMILY[s] for s in types])
    iu = np.triu_indices(len(t), 1)
    a, b = t[iu[0]], t[iu[1]]
    return np.where(a == b, "linear", np.where((a == "Q") | (b == "Q"), "non-monotone", "monotone")), a, b


def z_table() -> pd.DataFrame:
    rows = []
    for cls in CONS:
        for inst in range(N_INST):
            _, _, meta, m = load(cls, inst)
            r, rho, mi = (upper(m[k]) for k in (R, RHO, MI))
            rel, _, _ = relations(meta["generator"]["types"])
            lm = rel == "monotone"
            rows.append({"cls": cls, "inst": inst, "z_r_rho": z(r, rho), "z_rho_mi": z(rho, mi), "z_r_mi": z(r, mi),
                         "z_r_rho_LM": z(r[lm], rho[lm]) if lm.sum() > 2 else np.nan})
    return pd.DataFrame(rows)


def linfoot(mi):
    return np.sqrt(1 - np.exp(-2 * np.clip(mi, 0, None)))


def gaussian_mi_of_rho(rho):
    """MI of a bivariate Gaussian whose Spearman correlation is rho (r = 2 sin(pi rho / 6))."""
    r = 2 * np.sin(np.pi * np.asarray(rho) / 6)
    return -0.5 * np.log1p(-np.clip(r, -0.999999, 0.999999) ** 2)


def example_pairs(meta, m):
    """For L–L, L–M, L–Q, M–Q: the pair nearest its type's median (r, rho, MI)."""
    types = [FAMILY[s] for s in meta["generator"]["types"]]
    iu = np.triu_indices(len(types), 1)
    v = np.stack([upper(m[k]) for k in (R, RHO, MI)], 1)
    out = []
    for want in ("LL", "LM", "LQ", "MQ"):
        idx = np.flatnonzero([types[i] + types[j] == want for i, j in zip(*iu)])
        med = np.median(v[idx], 0)
        k = idx[np.argmin(np.sum(((v[idx] - med) / (v.std(0) + 1e-12)) ** 2, 1))]
        out.append((want[0] + "–" + want[1], int(iu[0][k]), int(iu[1][k])))
    return out


# --------------------------------------------------------------------------- panels

def panel_construction(fig, x, mother, meta, y0=0.25):
    panel_title(fig, 0.0, y0, "a", "Construction", "one latent $u$, three channel maps, independent noise")
    t0, n = 400, 160
    left, width = 1.0, 2.25
    tr = axes_in(fig, left, y0 + 0.32, width, 0.36)
    tr.plot(mother[t0:t0 + n], color=TRACE, lw=0.7)
    bare(tr)
    tr.set_xlim(0, n - 1)
    tr.text(-0.03, 0.5, "$u(t)$", transform=tr.transAxes, ha="right", va="center", fontsize=8)
    cp = axes_in(fig, left, y0 + 0.74, width, 1.28)
    carpet(cp, x[t0:t0 + n].T)
    types = [FAMILY[s] for s in meta["generator"]["types"]]
    bounds = np.flatnonzero(np.diff([ord(c) for c in types])) + 0.5
    for b in bounds:
        cp.axhline(b, color="white", lw=0.8)
    cp.text(1.0, -0.035, f"{n} samples", transform=cp.transAxes, ha="right", va="top", fontsize=7, color=MUTED)
    M = len(types)
    edges = np.r_[-0.5, bounds, M - 0.5]
    u = np.linspace(-2.2, 2.2, 200)
    maps = {"L": (u, "$u$"), "M": (1 / (1 + np.exp(-np.pi * u)), r"$\sigma(\pi u)$"), "Q": (u ** 2, "$u^2$")}
    row_h = 1.28 / M
    for fam, lo, hi in zip("LMQ", edges[:-1], edges[1:]):
        h = (hi - lo) * row_h
        g = min(0.30, 0.86 * h)
        gy = y0 + 0.74 + lo * row_h + 0.5 * (h - g)
        ga = axes_in(fig, left - 0.08 - g, gy, g, g)
        ga.plot(u, maps[fam][0], color=INK, lw=0.9)
        ga.axhline(maps[fam][0].min(), color=HAIR, lw=0.4, zorder=0)
        bare(ga)
        ga.text(-0.12, 0.5, f"{fam} {maps[fam][1]}", transform=ga.transAxes, ha="right", va="center", fontsize=8)
    sd = np.asarray(meta["generator"]["noise_stds"])
    cp.text(0.0, -0.035, f"noise SD {sd.min():.2f}–{sd.max():.2f}", transform=cp.transAxes, ha="left", va="top",
            fontsize=7, color=MUTED)


def panel_matrices(fig, meta, m, pairs, y0=0.25):
    panel_title(fig, 3.45, y0, "b", "Edge profiles", "one L + M + Q recording; channels ordered L, M, Q")
    types = [FAMILY[s] for s in meta["generator"]["types"]]
    M = len(types)
    bounds = np.flatnonzero(np.diff([ord(c) for c in types])) + 0.5
    centres = [np.mean([i for i, t in enumerate(types) if t == f]) for f in "LMQ"]
    size, gap, x0 = 1.08, 0.13, 3.72
    titles = {R: "Pearson $r$", RHO: r"Spearman $\rho$", MI: "KSG MI (nats)"}
    for k, key in enumerate((R, RHO, MI)):
        ax = axes_in(fig, x0 + k * (size + gap), y0 + 0.5, size, size)
        mat = m[key].astype(float).copy()
        np.fill_diagonal(mat, np.nan)
        vmax = 1.0 if key != MI else float(np.ceil(np.nanmax(mat)))
        cmap = seq(HUE[key]).copy()
        cmap.set_bad(HAIR)
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        for b in bounds:
            ax.axhline(b, color="white", lw=0.7)
            ax.axvline(b, color="white", lw=0.7)
        for n, (_, i, j) in enumerate(pairs, 1):
            ax.plot(j, i, "o", ms=4.2, mfc="none", mec=INK, mew=0.6)
            if k == 0:
                ax.annotate(str(n), (j, i), xytext=(2.5, 2.5), textcoords="offset points", fontsize=6.5,
                            color=INK, ha="left", va="bottom")
        ax.set_xticks(centres, ["L", "M", "Q"])
        ax.set_yticks(centres, ["L", "M", "Q"] if k == 0 else [])
        ax.tick_params(length=0, pad=1.5)
        ax.xaxis.tick_top()
        ax.set_title(titles[key], pad=10, fontsize=8)
        for s in ax.spines.values():
            s.set_linewidth(0.5)
        cax = axes_in(fig, x0 + k * (size + gap) + 0.1 * size, y0 + 0.5 + size + 0.07, 0.8 * size, 0.045)
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_ticks([0, vmax / 2, vmax] if key != MI else [0, vmax])
        cb.ax.tick_params(labelsize=7, length=2, pad=1)
        cb.outline.set_linewidth(0.4)
    neg = float(np.nanmin(np.where(np.eye(M, dtype=bool), np.nan, m[R])))
    return neg


def panel_pairs(fig, x, m, pairs, y0=2.5):
    panel_title(fig, 0.0, y0, "c", "Pair relations",
                r"four pairs of that recording; bars give $r$, $\rho$ and $r_I=\sqrt{1-e^{-2I}}$, MI on a correlation scale")
    rel_word = {"L–L": "linear", "L–M": "monotone", "L–Q": "non-monotone", "M–Q": "non-monotone"}
    rel_key = {"L–L": "linear", "L–M": "monotone", "L–Q": "non-monotone", "M–Q": "non-monotone"}
    sq, bh, x0 = 0.82, 0.3, 0.3
    pitch = (W - 0.15 - x0 - sq) / (len(pairs) - 1)
    for n, (name, i, j) in enumerate(pairs):
        left = x0 + n * pitch
        bx = axes_in(fig, left, y0 + 0.44, sq, bh)
        vals = [m[R][i, j], m[RHO][i, j], float(linfoot(m[MI][i, j]))]
        cols = [HUE[R], HUE[RHO], HUE[MI]]
        bx.bar(range(3), vals, width=0.62, color=cols, edgecolor=[shade(c, 0.7) for c in cols], lw=0.5)
        for k, v in enumerate(vals):
            lab = "1" if v >= 0.995 else (f"{abs(v):.2f}".lstrip("0") if abs(v) >= 0.005 else "0")
            bx.text(k, max(v, 0) + 0.04, lab, ha="center", va="bottom", fontsize=6.5, color=INK)
        bx.axhline(0, color=INK, lw=0.5)
        bx.set_ylim(-0.03, 1.45)
        bx.set_xlim(-0.55, 2.55)
        bx.set_xticks(range(3), ["$r$", r"$\rho$", "$r_I$"])
        bx.set_yticks([])
        bx.tick_params(axis="x", length=0, pad=1.5)
        for sp in ("top", "right", "left"):
            bx.spines[sp].set_visible(False)
        bx.set_title(f"{n + 1} {name}, {rel_word[name]}", fontsize=8, pad=2, loc="left")
        ax = axes_in(fig, left, y0 + 0.44 + bh + 0.17, sq, sq)
        ax.scatter(x[:, i], x[:, j], s=0.4, lw=0, color=REL[rel_key[name]][0], alpha=0.45, rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
        ax.set_xlabel(f"$x_{{{i + 1}}}$", fontsize=7.5, labelpad=1)
        ax.set_ylabel(f"$x_{{{j + 1}}}$", fontsize=7.5, labelpad=1)


def panel_planes(fig, recs, y0=4.50):
    panel_title(fig, 0.0, y0, "d", "Edge-profile planes", "one recording per construction; each point is a channel pair")
    size, gap, x0 = 1.08, 0.13, 0.62
    top = y0 + 0.42
    rows = [(R, RHO, "Pearson $r_{ij}$", r"Spearman $\rho_{ij}$"),
            (RHO, MI, r"Spearman $\rho_{ij}$", "MI $I_{ij}$ (nats)")]
    mi_max = max(float(np.max(upper(m[MI]))) for _, _, m, _ in recs.values())
    for rr, (kx, ky, lx, ly) in enumerate(rows):
        for cc, (cls, (x, meta, m, rel)) in enumerate(recs.items()):
            ax = axes_in(fig, x0 + cc * (size + gap), top + rr * (size + 0.42), size, size)
            vx, vy = upper(m[kx]), upper(m[ky])
            if ky == MI:
                grid = np.linspace(0, 0.9995, 300)
                ax.plot(grid, gaussian_mi_of_rho(grid), color=MUTED, lw=0.7, ls=(0, (2.5, 1.5)), zorder=1)
                if cc == 0:
                    ax.text(0.62, 0.55, "Gaussian", fontsize=7, color=MUTED, ha="center", va="bottom")
                ax.set(xlim=(-0.05, 1.04), ylim=(-0.1, mi_max * 1.06))
            else:
                ax.plot([-0.05, 1.04], [-0.05, 1.04], color=HAIR, lw=0.7, zorder=0)
                ax.set(xlim=(-0.05, 1.04), ylim=(-0.05, 1.04))
            for key in ("linear", "monotone", "non-monotone"):
                sel = rel == key
                ax.scatter(vx[sel], vy[sel], s=4, lw=0, color=REL[key][0], zorder=3 if key != "linear" else 2)
            rugs(ax, vx, vy, shade(HUE[kx], 0.85), shade(HUE[ky], 0.85))
            z_label(ax, z(vx, vy))
            ax.set_xticks([0, 0.5, 1], ["0", ".5", "1"])
            if ky == MI:
                ax.set_yticks([0, 2, 4] if mi_max > 4 else [0, 1, 2, 3])
            else:
                ax.set_yticks([0, 0.5, 1], ["0", ".5", "1"])
            if cc:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(ly)
            if cc == 1:
                ax.set_xlabel(lx)
            if rr == 0:
                ax.text(0.5, 1.06, CONS[cls], transform=ax.transAxes, ha="center", va="bottom", fontsize=8)
                zoom(ax, vx, vy, rel)


def zoom(ax, vx, vy, rel, lo=0.875):
    """Magnify the near-ceiling corner, where the monotone offset lives; same scale in every column."""
    ax.add_patch(Rectangle((lo, lo), 1.0 - lo + 0.01, 1.0 - lo + 0.01, fill=False, lw=0.5, ec=MUTED, zorder=4))
    ins = ax.inset_axes([0.47, 0.06, 0.48, 0.48])
    ins.plot([lo, 1.005], [lo, 1.005], color=HAIR, lw=0.6, zorder=0)
    keep = (vx > lo - 0.01) & (vy > lo - 0.01)
    for key in ("linear", "monotone"):
        sel = keep & (rel == key)
        ins.scatter(vx[sel], vy[sel], s=2.2, lw=0, color=REL[key][0], zorder=3 if key == "monotone" else 2)
    ins.set(xlim=(lo, 1.005), ylim=(lo, 1.005), xticks=[], yticks=[])
    for s in ins.spines.values():
        s.set_linewidth(0.5); s.set_color(MUTED)
    ins.text(0.96, 0.04, f"{lo:.3g}–1", transform=ins.transAxes, fontsize=6.5, color=MUTED, ha="right", va="bottom")


def panel_z(fig, table, y0=4.50):
    panel_title(fig, 4.55, y0, "e", "Agreement across recordings", f"{N_INST} recordings per construction")
    ax = axes_in(fig, 4.95, y0 + 0.42, 1.9, 1.08)
    rng = np.random.default_rng(1)
    specs = [("z_r_rho", -0.13, "o", "-", r"$z(r,\rho)$"), ("z_rho_mi", 0.13, "s", (0, (3, 1.5)), r"$z(\rho,\mathrm{MI})$")]
    for key, dx, mk, ls, lab in specs:
        means = []
        for k, cls in enumerate(CONS):
            v = table[table.cls == cls][key].to_numpy()
            ax.scatter(k + dx + rng.uniform(-0.05, 0.05, len(v)), v, s=2.5, lw=0, color=MUTED, alpha=0.6, zorder=2)
            ax.scatter(k + dx, v[INSTANCE], s=12, marker=mk, facecolor="none", edgecolor=INK, lw=0.6, zorder=4)
            means.append(v.mean())
        ax.plot(np.arange(3) + dx, means, color=INK, lw=0.9, ls=ls, marker=mk, ms=3.2, mfc=INK, mew=0, zorder=3)
        ax.text(2 + dx + 0.18, means[-1], lab, fontsize=7.5, va="center", ha="left")
    ax.set(xlim=(-0.4, 2.4), ylim=(0, 1.06), xticks=range(3), yticks=[0, 0.5, 1])
    ax.set_xticklabels(["L", "L+M", "L+M+Q"])
    ax.set_yticklabels(["0", ".5", "1"])
    ax.set_ylabel("$z$")
    lm = table[table.cls == "iter2_LM"].z_r_rho_LM.mean()
    return lm


def panel_spread(fig, sweep: pd.DataFrame | None, y0=6.37):
    panel_title(fig, 4.55, y0, "f", "Strength spread", "per-channel noise multiplier; 20 recordings each")
    ax = axes_in(fig, 4.95, y0 + 0.42, 1.9, 0.98)
    if sweep is None:
        ax.text(0.5, 0.5, "sweep pending", transform=ax.transAxes, ha="center", va="center", color=MUTED)
        return
    specs = [("L+M", "z_r_rho", INK, "o", "-", r"$z(r,\rho)$, L+M"),
             ("L+M+Q", "z_rho_mi", INK, "s", (0, (3, 1.5)), r"$z(\rho,\mathrm{MI})$, L+M+Q"),
             ("L", "z_r_rho", "#A3A8B2", "o", "-", None), ("L", "z_rho_mi", "#A3A8B2", "s", (0, (3, 1.5)), None)]
    for cons, key, col, mk, ls, lab in specs:
        g = sweep[sweep.construction == cons].groupby("scale")[key]
        s = np.asarray(g.median().index, float)
        ax.fill_between(s, g.quantile(0.1), g.quantile(0.9), color=col, alpha=0.12, lw=0)
        ax.plot(s, g.median(), color=col, lw=0.9, ls=ls, marker=mk, ms=3, mew=0)
        if lab:
            ax.text(s[-1] * 1.15, g.median().iloc[-1], lab, fontsize=7, va="center", ha="left")
    ax.text(s[-1] * 1.15, sweep[(sweep.construction == "L")].groupby("scale").z_r_rho.median().iloc[-1], "L",
            fontsize=7, color="#A3A8B2", va="center", ha="left")
    ax.axvline(5, color=MUTED, lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set(xlim=(0.85, 60), ylim=(-0.05, 1.06), yticks=[0, 0.5, 1], xticks=[1, 5, 50])
    ax.set_xticklabels(["1", "5", "50"])
    ax.set_yticklabels(["0", ".5", "1"])
    ax.set_xlabel(r"mean noise multiplier $\mathrm{E}[\kappa]$")
    ax.set_ylabel("$z$")


def legend(fig, y):
    handles = [Line2D([], [], ls="", marker="o", ms=3.5, mew=0, color=c) for c, _ in REL.values()]
    fig.legend(handles, [lab for _, lab in REL.values()], loc="upper left", bbox_to_anchor=(0.62 / W, 1 - y / H),
               ncol=3, handletextpad=0.2, columnspacing=1.0, fontsize=7.5)


def build(live_text=False):
    style(live_text)
    recs = {}
    for cls in CONS:
        x, mother, meta, m = load(cls, INSTANCE)
        recs[cls] = (x, meta, m, relations(meta["generator"]["types"])[0])
    x3, mother3, meta3, m3 = load("iter3_LMNM", INSTANCE)
    pairs = example_pairs(meta3, m3)
    table = z_table()
    sweep_path = OUT / "strength_spread.csv"
    sweep = pd.read_csv(sweep_path) if sweep_path.exists() else None

    fig = plt.figure(figsize=(W, H))
    panel_construction(fig, x3, mother3, meta3)
    neg = panel_matrices(fig, meta3, m3, pairs)
    panel_pairs(fig, x3, m3, pairs)
    panel_planes(fig, recs)
    lm = panel_z(fig, table)
    panel_spread(fig, sweep)
    legend(fig, 7.82)
    summary = {"archive": str(ARCHIVE.relative_to(ROOT)), "beta": BETA, "instance": INSTANCE,
               "pairs": [(n, i + 1, j + 1) for n, i, j in pairs], "min_offdiag_r_in_b": neg,
               "z_mean": table.groupby("cls")[["z_r_rho", "z_rho_mi", "z_r_mi"]].mean().round(4).to_dict(),
               "z_sd": table.groupby("cls")[["z_r_rho", "z_rho_mi", "z_r_mi"]].std().round(4).to_dict(),
               "z_r_rho_within_LM_pairs_L+M": round(float(lm), 4),
               "z_instance": {cls: {"z_r_rho": z(upper(m[R]), upper(m[RHO])), "z_rho_mi": z(upper(m[RHO]), upper(m[MI]))}
                              for cls, (_, _, m, _) in recs.items()},
               "sweep": str(sweep_path.relative_to(ROOT)) if sweep is not None else None}
    return fig, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live-text", action="store_true", help="editable SVG text (mathtext, no LaTeX)")
    args = ap.parse_args(argv)
    fig, summary = build(args.live_text)
    stem = "r_rho_mi_composite" + ("_live-text" if args.live_text else "")
    print(save(fig, OUT, stem))
    (OUT / "r_rho_mi_composite.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
