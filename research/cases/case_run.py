"""Shared runner and system figure for the one-knob case studies (2026-10-28).

A case supplies its knob levels, a generator call, an SPI function and its two rungs. ``run`` measures
every level over replicate recordings; ``z`` always uses all M(M-1) ordered off-diagonal entries
(the ``unified_ordered_v3`` contract). ``system_figure`` is a working figure for discussion, not the
paper figure: per level, a carpet of one recording, its three SPI matrices and its two edge planes,
then every recording placed in the plane of the two rung coordinates.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.cases.composite_style import HAIR, INK, MUTED, axes_in, carpet, save, seq, style, z_label

ROOT = Path(__file__).resolve().parents[2]


def ordered(mat: np.ndarray) -> np.ndarray:
    """All M(M-1) ordered off-diagonal entries."""
    return np.asarray(mat)[~np.eye(len(mat), dtype=bool)]


def z(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(ordered(a), ordered(b))[0, 1])


@dataclass
class Case:
    name: str                                   # output folder stem
    levels: dict                                # label -> generator kwargs
    make: Callable                              # (kwargs, rep) -> (x, internals)
    stats: Callable                             # x -> {short name: M x M matrix}
    spis: tuple                                 # (short, label, hue) for the three SPIs, restrictive first
    rungs: tuple                                # ((a, b), (b, c)) as short names
    level_colours: dict
    pair_colour: Callable                       # (label, internals) -> (values M x M, cmap or None)
    reps: int = 20
    extra_z: Callable | None = None             # mats -> {"x_...": value}, reported but not plotted
    extra: dict = field(default_factory=dict)   # recorded verbatim in summary.json


def _one(job):
    case, label, rep = job
    x, it = case.make(case.levels[label], rep)
    mats = case.stats(x)
    names = [s for s, _, _ in case.spis]
    row = {"level": label, "rep": rep}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            row[f"z_{a}_{b}"] = z(mats[a], mats[b])
    if case.extra_z is not None:
        row.update(case.extra_z(mats))
    return row, ((x, it, mats) if rep == 0 else None)


def run(case: Case, out: Path, workers: int = 8):
    jobs = [(case, label, rep) for label in case.levels for rep in range(case.reps)]
    with Pool(workers) as pool:
        results = pool.map(_one, jobs, chunksize=1)
    frame = pd.DataFrame([r for r, _ in results])
    examples = {job[1]: ex for job, (_, ex) in zip(jobs, results) if ex is not None}
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "z_by_recording.csv", index=False)
    cols = [c for c in frame.columns if c.startswith(("z_", "x_"))]
    summary = {"levels": {k: {kk: (vv if np.isscalar(vv) else list(vv)) for kk, vv in v.items()}
                          for k, v in case.levels.items()},
               "reps": case.reps, "z_mean": frame.groupby("level", sort=False)[cols].mean().round(4).to_dict("index"),
               "z_sd": frame.groupby("level", sort=False)[cols].std().round(4).to_dict("index"), **case.extra}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    return frame, examples


def system_figure(case: Case, frame: pd.DataFrame, examples: dict, out: Path, t_window=(0, 300)):
    style(live_text=True)
    labels = list(case.levels)
    n = len(labels)
    W, row_h = 7.4, 1.05
    H = 0.35 + n * row_h + 2.2
    fig = plt.figure(figsize=(W, H))
    short = [s for s, _, _ in case.spis]
    lims = {s: (min(np.nanmin(ordered(examples[l][2][s])) for l in labels),
                max(np.nanmax(ordered(examples[l][2][s])) for l in labels)) for s in short}
    for k, label in enumerate(labels):
        x, it, mats = examples[label]
        y = 0.35 + k * row_h
        fig.text(0.0, 1 - (y + 0.45) / H, label, fontsize=8, color=case.level_colours[label], rotation=90,
                 va="center", ha="left", transform=fig.transFigure)
        cp = axes_in(fig, 0.22, y, 1.7, 0.85)
        carpet(cp, x[slice(*t_window)].T)
        if k == 0:
            cp.set_title("one recording", fontsize=7.5, pad=2)
        for j, (s, lab, hue) in enumerate(case.spis):
            ax = axes_in(fig, 2.05 + j * 0.92, y, 0.85, 0.85)
            m = np.array(mats[s], float)
            np.fill_diagonal(m, np.nan)
            ax.imshow(m, cmap=seq(hue), vmin=lims[s][0], vmax=lims[s][1], interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if k == 0:
                ax.set_title(lab, fontsize=7.5, pad=2)
        vals, cmap = case.pair_colour(label, it)
        for j, (a, b) in enumerate(case.rungs):
            ax = axes_in(fig, 5.2 + j * 1.25, y, 0.85, 0.85)
            va, vb = ordered(mats[a]), ordered(mats[b])
            c = ordered(vals)
            if cmap is None:
                ax.scatter(va, vb, s=2.5, lw=0, c=list(c))
            else:
                o = np.argsort(c)
                ax.scatter(va[o], vb[o], s=2.5, lw=0, c=c[o], cmap=cmap, vmin=0, vmax=max(np.nanmax(ordered(vals)), 1e-9))
            lo, hi = min(lims[a][0], lims[b][0]), max(lims[a][1], lims[b][1])
            pad = 0.05 * (hi - lo)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=HAIR, lw=0.6, zorder=0)
            ax.set(xlim=(lims[a][0] - pad, lims[a][1] + pad), ylim=(lims[b][0] - pad, lims[b][1] + pad))
            ax.tick_params(labelsize=6, length=2, pad=1)
            z_label(ax, z(mats[a], mats[b]), x=0.04, y=0.96, fontsize=6.5)
            if k == n - 1:
                ax.set_xlabel(dict((s, l) for s, l, _ in case.spis)[a], fontsize=7, labelpad=1)
            ax.set_ylabel(dict((s, l) for s, l, _ in case.spis)[b], fontsize=7, labelpad=1)
    # z-space: every recording at (rung 1, rung 2)
    (a1, b1), (a2, b2) = case.rungs
    k1, k2 = f"z_{a1}_{b1}", f"z_{a2}_{b2}"
    ax = axes_in(fig, 0.55, 0.35 + n * row_h + 0.35, 1.6, 1.6)
    for label in labels:
        f = frame[frame.level == label]
        ax.scatter(f[k1], f[k2], s=7, lw=0, color=case.level_colours[label], alpha=0.8, label=label)
        ax.plot(f[k1].mean(), f[k2].mean(), "o", ms=5, mfc="none", mec=case.level_colours[label], mew=0.9)
    lab = dict((s, l) for s, l, _ in case.spis)
    ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), xticks=[0, 0.5, 1], yticks=[0, 0.5, 1],
           xlabel=f"$z$({lab[a1]}, {lab[b1]})", ylabel=f"$z$({lab[a2]}, {lab[b2]})")
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.02, 1), handletextpad=0.2, labelspacing=0.3)
    ax.set_title(f"{case.reps} recordings per level", fontsize=7.5, pad=2)
    # z by level, all three pairs
    ax = axes_in(fig, 4.2, 0.35 + n * row_h + 0.35, 2.6, 1.6)
    cols = [c for c in frame.columns if c.startswith("z_")]
    mk = ("o", "s", "D")
    for i, c in enumerate(cols):
        g = frame.groupby("level", sort=False)[c]
        a, b = c[2:].split("_", 1)
        ax.errorbar(np.arange(n) + (i - 1) * 0.08, g.mean()[labels], yerr=g.std()[labels], marker=mk[i], ms=3,
                    lw=0.9, capsize=1.5, color=(INK, "#0072B2", MUTED)[i], label=f"$z$({lab[a]}, {lab[b]})")
    ax.set(ylim=(-0.05, 1.05), yticks=[0, 0.5, 1], xticks=range(n))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=6.5, loc="lower left", labelspacing=0.2, handlelength=1.6)
    return save(fig, out, f"{case.name}_system")
