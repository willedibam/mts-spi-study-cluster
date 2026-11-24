"""Supplementary to fig:mi (draft): is the drug change in Δz sleep depth? Every macaque window, Δz against its delta
fraction, one panel per animal (delta_scatter() of plot_character_draft.py).

    python plot_delta_supp.py <results/representation/atlas-mi-pair-261117> <output stem>
    (as committed: output results/representation/atlas-mi-pair-261117/mi-figure-supp-delta; writes .svg, .pdf, .png)

Shaded: the delta range both the animal's sleep and drug windows reach (only Chibi and George were recorded in sleep);
the medians written there are of the windows inside it. Line: Δz on delta over the eyes-open, eyes-closed and sleep
windows, fitted per animal (review/sleepmatch_ns.py fits one common slope with state terms), dashed where it is
extrapolated beyond those windows' delta range. Delta fraction is animal-specific (George sits higher throughout), so
panels are not pooled.
"""
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from mtsspi.figures import figure_style
from plot_character_draft import load, delta_scatter, COLOR, GREY

ANIMALS = ("Chibi", "George", "Kin2", "Su")
NAME = {"awake_eo": "wake, recovery", "SLP": "sleep", "KT": "ketamine", "KTMD": "ketamine + medetomidine",
        "MD": "medetomidine", "PF": "propofol"}


def draw(root, output):
    df, _ = load(root)
    figure_style()
    plt.rcParams.update({"font.size": 7, "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "axes.linewidth": .7})
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.1), sharey=True, layout="constrained")
    for ax, an in zip(axs, ANIMALS):
        d = df[df.animal == an]; delta_scatter(ax, d, s=4)
        ax.set_title(an, fontsize=7.5, pad=3)
        ax.set_xticks([.2, .4, .6, .8, 1], [".2", ".4", ".6", ".8", "1"])
        ax.tick_params(length=2, pad=1.5)
        sl, dr = d[d.group == "SLP"], d[d.state == "drug"]
        if len(sl):
            lo, hi = max(sl.delta.min(), dr.delta.min()), min(sl.delta.max(), dr.delta.max())
            m = d.delta.between(lo, hi)
            txt = "in the shaded range\n" + "\n".join(
                rf"{lab} {e.sd.median():+.1f} ({len(e)})".replace("-", "--")
                for lab, e in (("sleep", d[m & (d.group == "SLP")]), ("drug", d[m & (d.state == "drug")])))
            ax.text(.03, .04, txt, transform=ax.transAxes, fontsize=5.6, color=GREY, va="bottom", linespacing=1.15)
        else:
            ax.text(.03, .04, "no sleep recording", transform=ax.transAxes, fontsize=5.6, color=GREY, va="bottom")
    axs[0].set_yticks([0, -5, -10], ["0", "--5", "--10"])
    axs[0].set_ylabel(r"$\Delta z$ per 10-s window, wake SDs", fontsize=6.6)
    fig.supxlabel("delta fraction of the window (0.5--4 Hz share of 0.5--40 Hz power)", fontsize=6.6)
    h = [Line2D([], [], marker="o", ls="", ms=3, color=COLOR[k], alpha=.5 if k == "awake_eo" else .85) for k in NAME]
    h += [Line2D([], [], color=COLOR["ink"], lw=.7), Line2D([], [], color=COLOR["ink"], lw=.7, ls=(0, (2, 1.4)))]
    fig.legend(h, list(NAME.values()) + ["wake and sleep fit", "extrapolated"], ncol=8, loc="outside upper center",
               fontsize=5.8, frameon=False, handletextpad=.2, columnspacing=.9, handlelength=1.6)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(output.with_suffix("." + ext), dpi=180 if ext == "png" else None)
    for an in ANIMALS:
        d = df[df.animal == an]; f = d[d.group.isin(("awake_eo", "SLP")) | (d.state == "awake_ec")]
        print(an, "fit slope per +0.1 delta", round(np.polyfit(f.delta, f.sd, 1)[0] / 10, 2),
              "no-drug delta range", round(f.delta.min(), 2), round(f.delta.max(), 2),
              "drug delta range", round(d[d.state == "drug"].delta.min(), 2), round(d[d.state == "drug"].delta.max(), 2))


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path); p.add_argument("output", type=Path)
    a = p.parse_args(); draw(a.root, a.output)
