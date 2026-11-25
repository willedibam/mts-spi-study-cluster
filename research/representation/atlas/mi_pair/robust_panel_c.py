"""Panel (c) of the MI figure (plot_character_draft.py) redrawn for every sampling-rate and window arm: per 10-s (or
long) window, Δz = (real − surrogate) minus its session's eyes-closed wake mean, one raincloud per state, all arms on
one shared raw-Δz axis so sizes compare across arms (the main figure prints wake SDs, which differ by arm). Numbers
above each cloud: its window mean; in the title: the window-pooled drug mean (session or animal weighting in
lowrate_summary.txt differs by up to .002). At 100 s one sleep session has no eyes-closed window, so its 35 windows
have no reference and are dropped.

usage: robust_panel_c.py <anchor results dir> <low-rate scores dir> <output stem>
  anchor: results/representation/atlas-mi-pair-261117 (zns/); low-rate: atlas-mi-pair-lowrate-261125/scores/<arm>/
"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from mtsspi.figures import figure_style, raincloud
from plot_character import read
from plot_character_draft import COLOR, DRUGS, GREY, SHORT, STATES

# (arm, rate Hz, window s, band, samples per window); None: a cell still running, drawn empty
GRID = [[("anchor", 100, 10, "0.5–40 Hz", 1000), ("L20", 100, 10, "0.5–8 Hz", 1000), ("L10", 100, 10, "0.5–4 Hz", 1000)],
        [("S20", 20, 10, "subsampled", 200), ("D20", 20, 10, "0.5–8 Hz, decimated", 200), ("W50", 100, 50, "0.5–40 Hz", 5000)],
        [("S20L", 20, 50, "subsampled", 1000), ("D20L", 20, 50, "0.5–8 Hz, decimated", 1000), None],
        [("S10", 10, 10, "subsampled", 100), ("D10", 10, 10, "0.5–4 Hz, decimated", 100), None],
        [("S10L", 10, 100, "subsampled", 1000), ("D10L", 10, 100, "0.5–4 Hz, decimated", 1000), None]]
RUNNING = {"W50"}                                        # not read: its prediction is written before it reports


def windows(directory):
    df, _ = read(directory)
    df["group"] = np.where(df.state == "drug", df.agent, np.where(df.state == "sleep", "SLP", df.state))
    df["d"] = df.residual - df.unit.map(df[df.state == "awake_ec"].groupby("unit").residual.mean())
    return df


def main(anchor, low, stem):
    figure_style()
    plt.rcParams.update({"font.size": 7, "xtick.labelsize": 6, "ytick.labelsize": 6})
    fig, axes = plt.subplots(len(GRID), 3, figsize=(7.2, 9.4), sharey=True, layout="constrained")
    lo, hi = -.075, .045
    for r, row in enumerate(GRID):
        for c, cell in enumerate(row):
            ax = axes[r, c]
            if cell is None:
                ax.axis("off"); continue
            arm, fs, T, band, n = cell
            title = f"{arm}: {fs} Hz $\\times$ {T} s, {band}\n{n} samples per window"
            ax.axhline(0, color=GREY, lw=.5, ls="--", zorder=0)
            ax.set_xticks(range(len(STATES)), [SHORT[s] for s in STATES], fontsize=5.6)
            for tick, s in zip(ax.get_xticklabels(), STATES):
                tick.set_color(COLOR[s])
            ax.tick_params(axis="x", length=0); ax.set_xlim(-.6, len(STATES) - .4)
            if arm in RUNNING:
                ax.set_title(title, fontsize=6.6, loc="left")
                ax.text(.5, .5, "running", transform=ax.transAxes, ha="center", va="center", color=GREY, fontsize=8)
                continue
            df = windows(anchor / "zns" if arm == "anchor" else low / arm)
            df = df[df.d.notna()]; rng = np.random.default_rng(0)
            for i, s in enumerate(STATES):
                v = df[df.group == s].d.values
                raincloud(ax, np.clip(v, lo, hi), i, COLOR[s], rng, width=.64, box=True, point_sizes=1.2,
                          point_kw={"alpha": .3})
                ax.text(i, hi - .001, f"{v.mean():+.3f}".replace("0.", "."), ha="center", va="top", fontsize=5.4,
                        color=COLOR["ink"] if s in DRUGS else GREY, fontweight="bold" if s in DRUGS else None)
            dr = f"{df[df.state == 'drug'].d.mean():+.4f}".replace("0.", ".")
            ax.set_title(title + rf", drug $\mathbf{{{dr}}}$", fontsize=6.6, loc="left")
    axes[0, 0].set_ylim(lo, hi + .004)
    axes[2, 2].text(0, .95, "Rows: rate $\\times$ window.  Columns: subsampled (every\nn-th sample, no filter), "
                    "decimated (anti-alias\nfiltered first), and 100 Hz variants.\n\n"
                    "$\\Delta z$: window's real $-$ surrogate $z$, minus its\nsession's eyes-closed wake mean.  "
                    "Same raw\naxis in every cell; bold: drugs.", transform=axes[2, 2].transAxes, va="top",
                    fontsize=6, color=GREY, linespacing=1.3)
    for a in axes[:, 0]:
        a.set_ylabel("$\\Delta z$ per window")
    fig.suptitle("Panel (c) per sampling arm: $\\Delta z$ per window by state, shared raw axis "
                 f"(clipped to [{lo}, {hi}])", fontsize=7.5)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=300)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
