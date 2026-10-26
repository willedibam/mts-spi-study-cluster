"""Timing-ladder composite, first pass (2026-10-25): r, max cross-correlation, phase-aligned P.

Recomputes the 2026-10-20 timing-triplet setting (M = 20, T = 20,000, delay range 15; 20
recordings per family) through ``timing_triplet.measure``: pyspi r, ``xcorr_max_sig-True`` and
``cohmag`` from ``spis/timing_triplet.yaml``, plus the bespoke Welch P. Results are cached in
``results/cases/timing-261025/``. P is not a pyspi statistic; ``cohmag`` (mean magnitude-squared
coherence) is the library's nearest coordinate and is reported beside it. Every z is over the upper
triangle, equal to the ordered-entry z for these symmetric statistics.

    python -m research.cases.timing_triplet.timing_composite_261025 [--live-text] [--recompute]
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

from mtsspi.generators import band_spectrum, generate_band_delayed_mts
from research.cases.composite_style import (HAIR, INK, MUTED, ROSE, SAGE, SKY, axes_in, bare, panel_title, rugs,
                                            save, seq, shade, style, tint, z_label)
from research.cases.timing_triplet import timing_triplet as tt

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/cases/timing-261025"
PRM = dict(M=20, T=20_000, lag_range=15)
N_REP, REP = 20, 0
R, XC, P, CM = tt.R, tt.XC, tt.P, tt.CM
HUE = {R: SAGE, XC: ROSE, P: SKY}
FAM = {"synchronous": "synchronous", "delayed": "pure delay", "dispersive": "dispersive"}
DELAY = "#0072B2"   # pair delay difference (as 'monotone' in the form case: the simple departure)
DISP = "#CC79A7"    # band-delay mismatch (as 'non-monotone': the departure the middle rung cannot absorb)
KAPPAS = (1, 0.5, 0.25, 0.1, 0.05, 0.02)
W, H = 7.09, 8.05


# --------------------------------------------------------------------------- data

def compute(recompute=False):
    cache = OUT / "timing_cache.npz"
    if cache.exists() and not recompute:
        d = np.load(cache, allow_pickle=False)
        frame = pd.read_csv(OUT / "z_by_recording.csv")
        mats = {f: d[f] for f in FAM}
        pop = {f: {k: d[f"pop_{f}_{k}"] for k in ("r", "xcorr", "P", "cohmag")} for f in FAM}
        nb = pd.read_csv(OUT / "narrowband.csv")
        return frame, mats, pop, nb
    OUT.mkdir(parents=True, exist_ok=True)
    frame, mats = tt.measure(tt.band_makers(PRM), N_REP)
    pop = {f: tt.population(PRM, f) for f in FAM}
    nb = tt.narrowband_scan(PRM, KAPPAS)
    frame.to_csv(OUT / "z_by_recording.csv", index=False)
    nb.to_csv(OUT / "narrowband.csv", index=False)
    np.savez_compressed(cache, **{f: mats[f] for f in FAM},
                        **{f"pop_{f}_{k}": pop[f][k] for f in FAM for k in ("r", "xcorr", "P", "cohmag")})
    return compute(False)


def internals():
    _, it = generate_band_delayed_mts(**{**PRM, "T": 1}, family="dispersive", rng=0, return_internals=True)
    return it


def upper(m):
    return m[np.triu_indices(m.shape[0], 1)]


def z(a, b):
    return float(np.corrcoef(upper(a), upper(b))[0, 1])


def stat(mats, fam, key, rep=REP):
    return mats[fam][rep, tt.STATS.index(key)]


def band_components(x, bands):
    """Split each channel into its slow and fast band by FFT masking (the bands do not overlap)."""
    f = np.fft.rfftfreq(len(x))
    X = np.fft.rfft(x, axis=0)
    out = []
    for c, w in bands:
        lo, hi = (0.0, w) if c == 0 else (c - w, c + w)
        out.append(np.fft.irfft(X * ((f >= lo) & (f <= hi))[:, None], n=len(x), axis=0))
    return out


def choose_pairs(it, mats):
    """(sync, delay, dispersive) share one pair with a large slow delay difference and band mismatch;
    a second dispersive pair has a large delay difference but no mismatch."""
    slow, fast = it.delays
    ds = np.abs(slow[:, None] - slow[None, :])
    mm = tt.band_mismatch(it)
    lam = np.outer(it.loadings[0], it.loadings[0]) * np.outer(it.loadings[1], it.loadings[1])
    iu = np.triu_indices(len(slow), 1)
    score = (ds * (mm >= 8) * lam)[iu]
    k = int(np.argmax(score))
    a = (int(iu[0][k]), int(iu[1][k]))
    score0 = (ds * (mm == 0) * lam)[iu]
    k0 = int(np.argmax(score0))
    b = (int(iu[0][k0]), int(iu[1][k0]))
    return [("synchronous", a), ("delayed", a), ("dispersive", b), ("dispersive", a)]


def correlogram(x, i, j, lag=30):
    a = (x[:, i] - x[:, i].mean()) / x[:, i].std()
    b = (x[:, j] - x[:, j].mean()) / x[:, j].std()
    T = len(a)
    hs = np.arange(-lag, lag + 1)
    return hs, np.array([np.dot(a[max(0, -h):T - max(0, h)], b[max(0, h):T - max(0, -h)]) / T for h in hs])


# --------------------------------------------------------------------------- panels

def panel_construction(fig, it, pair, y0=0.25):
    panel_title(fig, 0.0, y0, "a", "Construction",
                "two bands, each a common drive reaching every channel at its own delay; spectra identical across families")
    # band spectra: shared by all families
    ax = axes_in(fig, 0.3, y0 + 0.5, 1.15, 1.08)
    f = np.linspace(0, 0.5, 600)
    for (c, w), var in zip(it.bands, it.band_var):
        ax.fill_between(f, band_spectrum(f, c, w, var), color=tint(MUTED, 0.35), lw=0)
        ax.plot(f, band_spectrum(f, c, w, var), color=INK, lw=0.8)
    ax.axhline(0, color=INK, lw=0.5)
    ax.text(0.02, 0.92 * ax.get_ylim()[1], "slow", fontsize=7.5, color=INK, va="top")
    ax.text(0.25, 0.42 * ax.get_ylim()[1], "fast", fontsize=7.5, color=INK, ha="center")
    ax.set(xlim=(0, 0.5), xticks=[0, 0.25, 0.5], yticks=[], xlabel="frequency (cycles/sample)")
    ax.set_xticklabels(["0", ".25", ".5"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title("band spectra", fontsize=8, pad=3)
    # per-band cross-correlograms of one pair in each family, same process noise
    i, j = pair
    left, cw, gap, rh = 1.85, 1.62, 0.12, 0.46
    for c, fam in enumerate(FAM):
        x = generate_band_delayed_mts(**PRM, family=fam, rng=np.random.default_rng(7))
        comps = band_components(x, it.bands)
        _, itf = generate_band_delayed_mts(**{**PRM, "T": 1}, family=fam, rng=0, return_internals=True)
        for r_, (band, comp) in enumerate(zip(("slow", "fast"), comps)):
            ax = axes_in(fig, left + c * (cw + gap), y0 + 0.5 + r_ * (rh + 0.16), cw, rh)
            hs, cc = correlogram(comp, i, j, lag=24)
            d = int(itf.delays[r_][j] - itf.delays[r_][i])
            ax.axvline(0, color=HAIR, lw=0.5, zorder=0)
            ax.axhline(0, color=HAIR, lw=0.5, zorder=0)
            ax.plot(hs, cc, color=INK, lw=0.8)
            ax.axvline(d, color=DISP if (fam == "dispersive" and r_ == 1) else DELAY, lw=0.7, ls=(0, (2, 1.2)), zorder=1)
            ax.set(xlim=(-24, 24), ylim=(-0.65, 0.8), yticks=[])
            ax.set_xticks([-20, 0, 20])
            if r_ == 0:
                ax.set_xticklabels([])
                ax.set_title(FAM[fam], fontsize=8, pad=3)
            else:
                ax.set_xlabel("lag $h$", fontsize=7.5, labelpad=1)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.text(0.98, 0.95, rf"$\Delta = {d}$", transform=ax.transAxes, ha="right", va="top", fontsize=7.5)
            if c == 0:
                ax.text(-0.04, 0.5, band, transform=ax.transAxes, ha="right", va="center", fontsize=7.5)
    fig.text(left / W, 1 - (y0 + 0.5 + 2 * rh + 0.16 + 0.3) / H,
             f"band-limited cross-correlogram of channels {i + 1} and {j + 1}; dashed: generating delay difference $\\Delta$",
             fontsize=7, color=MUTED, transform=fig.transFigure, va="top")


def panel_matrices(fig, mats, order, y0=2.38):
    panel_title(fig, 0.0, y0, "b", "Edge profiles", "one recording per family; channels ordered by slow-band delay")
    size, gap, x0 = 0.66, 0.08, 0.62
    titles = {R: "Pearson $r$", XC: "max. xcorr", P: "aligned $P$"}
    lo = min(float(np.nanmin(stat(mats, f, R))) for f in FAM)
    for rr, fam in enumerate(FAM):
        for cc, key in enumerate((R, XC, P)):
            ax = axes_in(fig, x0 + cc * (size + gap), y0 + 0.45 + rr * (size + gap), size, size)
            m = stat(mats, fam, key)[np.ix_(order, order)].astype(float)
            np.fill_diagonal(m, np.nan)
            cmap = seq(HUE[key]).copy()
            cmap.set_bad(HAIR)
            ax.imshow(np.clip(m, 0, None), cmap=cmap, vmin=0, vmax=0.7, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.5)
            if rr == 0:
                ax.set_title(titles[key], fontsize=8, pad=3)
            if cc == 0:
                ax.set_ylabel(FAM[fam], fontsize=7.5, labelpad=2)
    return lo


def panel_pairs(fig, mats, chosen, y0=2.38):
    panel_title(fig, 3.35, y0, "c", "Pair relations", "cross-correlogram $r_{ij}(h)$ of four pairs; bars give $r$, max., $P$")
    sq_w, sq_h, bh, x0 = 0.72, 0.62, 0.3, 3.62
    pitch = (W - 0.12 - x0 - sq_w) / 3
    names = ["sync", "delay", "disp., matched", "disp., mismatched"]
    for n, (fam, (i, j)) in enumerate(chosen):
        x = generate_band_delayed_mts(**PRM, family=fam, rng=tt.recording_rng(fam, REP))
        hs, c = correlogram(x, i, j)
        r0, xm, pv = (stat(mats, fam, k)[i, j] for k in (R, XC, P))
        left = x0 + n * pitch
        bx = axes_in(fig, left, y0 + 0.52, sq_w, bh)
        vals, cols = [r0, xm, pv], [HUE[R], HUE[XC], HUE[P]]
        bx.bar(range(3), np.clip(vals, 0, None), width=0.62, color=cols, edgecolor=[shade(q, 0.7) for q in cols], lw=0.5)
        for k, v in enumerate(vals):
            bx.text(k, max(v, 0) + 0.03, f"{v:.2f}".replace("0.", ".", 1) if v >= 0 else f"{v:.2f}".replace("-0.", "-."),
                    ha="center", va="bottom", fontsize=6.5)
        bx.axhline(0, color=INK, lw=0.5)
        bx.set(ylim=(0, 0.85), xlim=(-0.55, 2.55), yticks=[])
        bx.set_xticks(range(3), ["$r$", "max", "$P$"])
        bx.tick_params(axis="x", length=0, pad=1.5)
        for s in ("top", "right", "left"):
            bx.spines[s].set_visible(False)
        bx.set_title(names[n].replace("\n", " "), fontsize=7.5, pad=2, loc="left")
        ax = axes_in(fig, left, y0 + 0.52 + bh + 0.17, sq_w, sq_h)
        ax.axhline(0, color=HAIR, lw=0.5, zorder=0)
        ax.axvline(0, color=HAIR, lw=0.5, zorder=0)
        ax.plot(hs, c, color=INK, lw=0.7)
        ax.axhline(pv, color=shade(SKY, 0.8), lw=0.7, ls=(0, (2, 1.2)))
        ax.plot(0, r0, "o", ms=3.2, color=shade(SAGE, 0.8), zorder=4)
        ax.plot(hs[np.argmax(c)], c.max(), "o", ms=3.2, color=shade(ROSE, 0.85), zorder=4)
        ax.set(xlim=(-30, 30), ylim=(-0.45, 0.75), xticks=[-30, 0, 30], yticks=[0, 0.5] if n == 0 else [])
        if n:
            ax.set_yticklabels([])
        ax.set_xlabel("lag $h$", fontsize=7.5, labelpad=1)
        ax.text(0.97, 0.04, f"({i + 1}, {j + 1})", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
                color=MUTED)


def panel_planes(fig, mats, it, y0=5.08):
    panel_title(fig, 0.0, y0, "d", "Edge-profile planes", "one recording per family; each point is a channel pair")
    size, gap, x0 = 1.0, 0.13, 0.62
    top = y0 + 0.42
    slow = it.delays[0]
    dtau = upper(np.abs(slow[:, None] - slow[None, :]).astype(float))
    mism = upper(tt.band_mismatch(it))
    cm_delay = LinearSegmentedColormap.from_list("d", [tint(DELAY, 0.2), DELAY, shade(DELAY, 0.6)])
    cm_disp = LinearSegmentedColormap.from_list("q", [tint(DISP, 0.2), DISP, shade(DISP, 0.6)])
    rows = [(R, XC, "Pearson $r_{ij}$", r"max. cross-corr."), (XC, P, r"max. cross-corr.", "phase-aligned $P_{ij}$")]
    lo = min(-0.05, min(float(upper(stat(mats, f, R)).min()) for f in FAM) - 0.03)
    handles = {}
    for rr, (kx, ky, lx, ly) in enumerate(rows):
        for cc, fam in enumerate(FAM):
            ax = axes_in(fig, x0 + cc * (size + gap), top + rr * (size + 0.4), size, size)
            vx, vy = upper(stat(mats, fam, kx)), upper(stat(mats, fam, ky))
            lim = (lo if kx == R else -0.02, 0.72)
            ax.plot([-1, 1], [-1, 1], color=HAIR, lw=0.7, zorder=0)
            if fam == "synchronous":
                ax.scatter(vx, vy, s=4, lw=0, color="#B4B7BD", zorder=2)
            elif rr == 0:
                o = np.argsort(dtau)
                handles["d"] = ax.scatter(vx[o], vy[o], s=4, lw=0, c=dtau[o], cmap=cm_delay, vmin=0, vmax=15, zorder=2)
            else:
                cval = mism if fam == "dispersive" else np.zeros_like(mism)
                o = np.argsort(cval)
                handles["q"] = ax.scatter(vx[o], vy[o], s=4, lw=0, c=cval[o], cmap=cm_disp, vmin=0, vmax=float(mism.max()),
                                          zorder=2)
            rugs(ax, vx, vy, shade(HUE[kx], 0.85), shade(HUE[ky], 0.85))
            z_label(ax, z(stat(mats, fam, kx), stat(mats, fam, ky)))
            ax.set(xlim=lim, ylim=(-0.02, 0.72), xticks=[0, 0.3, 0.6], yticks=[0, 0.3, 0.6])
            ax.set_xticklabels(["0", ".3", ".6"]); ax.set_yticklabels(["0", ".3", ".6"])
            if cc:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(ly)
            if cc == 1:
                ax.set_xlabel(lx)
            if rr == 0:
                ax.text(0.5, 1.06, FAM[fam], transform=ax.transAxes, ha="center", va="bottom", fontsize=8)
    for k, (key, lab, ticks) in enumerate((("d", r"$|\tau_i-\tau_j|$, slow band", [0, 15]),
                                           ("q", r"band mismatch $|\Delta^{\mathrm{s}}_{ij}-\Delta^{\mathrm{f}}_{ij}|$",
                                            [0, int(mism.max())]))):
        cax = axes_in(fig, x0 + 3 * size + 2 * gap + 0.1, top + k * (size + 0.4) + 0.25, 0.05, size - 0.5)
        cb = fig.colorbar(handles[key], cax=cax)
        cb.set_ticks(ticks)
        cb.ax.tick_params(labelsize=7, length=2, pad=1)
        cb.outline.set_linewidth(0.4)
        cb.set_label(lab, fontsize=7, labelpad=1)


def panel_z(fig, frame, pop, y0=5.08):
    panel_title(fig, 4.75, y0, "e", "Agreement across recordings", f"{N_REP} recordings per family; open: population")
    ax = axes_in(fig, 5.1, y0 + 0.42, 1.75, 1.0)
    rng = np.random.default_rng(2)
    specs = [(tt.pair_name(R, XC), (R, "xcorr"), -0.16, "o", "-", INK, r"$z(r,\max)$"),
             (tt.pair_name(XC, P), ("xcorr", "P"), 0.0, "s", (0, (3, 1.5)), INK, r"$z(\max,P)$"),
             (tt.pair_name(XC, CM), ("xcorr", "cohmag"), 0.16, "D", (0, (1, 1.2)), MUTED, r"$z(\max,\mathrm{coh})$")]
    for name, (a, b), dx, mk, ls, col, lab in specs:
        sub = frame[frame.pair == name]
        means = []
        for k, fam in enumerate(FAM):
            v = sub[sub.family == fam].z.to_numpy()
            ax.scatter(k + dx + rng.uniform(-0.04, 0.04, len(v)), v, s=2.5, lw=0, color=MUTED, alpha=0.6, zorder=2)
            pa = pop[fam]["r" if a == R else a]
            ax.scatter(k + dx, z(pa, pop[fam][b]), s=14, marker=mk, facecolor="white", edgecolor=col, lw=0.7, zorder=5)
            means.append(v.mean())
        ax.plot(np.arange(3) + dx, means, color=col, lw=0.9, ls=ls, marker=mk, ms=2.8, mew=0, zorder=3)
        ax.text(2 + dx + 0.22, means[-1] + {"o": 0, "s": 0.07, "D": -0.07}[mk], lab, fontsize=7, va="center", ha="left", color=col)
    ax.set(xlim=(-0.4, 2.4), ylim=(0, 1.06), xticks=range(3), yticks=[0, 0.5, 1])
    ax.set_xticklabels(["sync", "delay", "disp."]); ax.set_yticklabels(["0", ".5", "1"])
    ax.set_ylabel("$z$")


def panel_limits(fig, nb, y0=6.82):
    panel_title(fig, 4.75, y0, "f", "Narrow bands", "population; both bands shrink about their centres")
    ax = axes_in(fig, 5.1, y0 + 0.42, 1.75, 0.82)
    for fam, col, ls in (("delayed", DELAY, "-"), ("dispersive", DISP, "-")):
        g = nb[nb.family == fam].sort_values("kappa")
        ax.plot(g.kappa, g.z_xcorr_P, color=col, lw=0.9, ls=ls, marker="s", ms=2.8, mew=0)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set(ylim=(0.4, 1.04), yticks=[0.5, 1], xticks=[1, 0.1, 0.02])
    ax.set_xticklabels(["1", ".1", ".02"]); ax.set_yticklabels([".5", "1"])
    ax.set_xlabel(r"relative bandwidth $\kappa$")
    ax.set_ylabel(r"$z(\max,P)$")
    ax.text(0.5, 0.08, "dispersive", transform=ax.transAxes, color=DISP, fontsize=7, ha="left")
    ax.text(0.5, 0.8, "pure delay", transform=ax.transAxes, color=DELAY, fontsize=7, ha="left")


def build(live_text=False, recompute=False):
    frame, mats, pop, nb = compute(recompute)
    style(live_text)
    it = internals()
    order = np.argsort(it.delays[0], kind="stable")
    chosen = choose_pairs(it, mats)
    fig = plt.figure(figsize=(W, H))
    panel_construction(fig, it, chosen[-1][1])
    panel_matrices(fig, mats, order)
    panel_pairs(fig, mats, chosen)
    panel_planes(fig, mats, it)
    panel_z(fig, frame, pop)
    panel_limits(fig, nb)
    zs = frame.groupby(["family", "pair"]).z.agg(["mean", "std"]).round(4)
    summary = {"params": PRM, "n_rep": N_REP, "rep_shown": REP,
               "pairs": [(fam, i + 1, j + 1) for fam, (i, j) in chosen],
               "z": {f"{a}|{b}": v for (a, b), v in zs["mean"].items()},
               "z_sd": {f"{a}|{b}": v for (a, b), v in zs["std"].items()},
               "z_population": {f"{fam}|{a}-{b}": z(pop[fam][a], pop[fam][b]) for fam in FAM
                                for a, b in (("r", "xcorr"), ("xcorr", "P"), ("xcorr", "cohmag"))},
               "narrowband": nb.round(4).to_dict(orient="records")}
    return fig, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live-text", action="store_true")
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args(argv)
    fig, summary = build(args.live_text, args.recompute)
    print(save(fig, OUT, "timing_composite" + ("_live-text" if args.live_text else "")))
    (OUT / "timing_composite.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
