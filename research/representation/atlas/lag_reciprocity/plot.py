"""Plot 20-ms lag reciprocity on the original common-average montage."""
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def draw(scores, output):
    df = pd.read_csv(scores).query("lag_ms == 20 and montage == 'CAR'")
    groups = ["KT", "KTMD", "MD", "PF", "SLP"]
    labels = ["Ketamine", "Ket. + med.", "Medetomidine", "Propofol", "Sleep"]
    names = ["Chibi", "George", "Kin2", "Su"]
    colors = ["#006A86", "#BA542B", "#674593", "#468253"]
    with plt.rc_context({"font.size": 8, "font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "mathtext.fontset": "cm",
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 3, figsize=(8, 2.65), layout="constrained")
        ax = axes[0]
        ax.plot([-.15,.8], [-.15,.8], "--", color="#889298", lw=1)
        # One illustrative dyad and its transpose. Projection splits symmetry from asymmetry.
        ax.plot([0,.35],[0,.35], color="#536973", lw=2)
        ax.plot([.35,.60],[.35,.10], color="#994370", lw=2)
        ax.scatter([.60,.10],[.10,.60], color=["#994370","#536973"], s=20)
        ax.text(.12,.62,"reversed edge", fontsize=7)
        ax.text(.42,.015,"forward edge", fontsize=7)
        ax.text(.01,.29,"symmetric", rotation=45, fontsize=7, color="#536973")
        ax.text(.41,.30,"asymmetric", rotation=-45, fontsize=7, color="#994370")
        ax.set(xlim=(-.15,.8), ylim=(-.15,.8), aspect="equal", xlabel=r"$C_{ij}(20\,\mathrm{ms})$",
               ylabel=r"$C_{ji}(20\,\mathrm{ms})$", xticks=[0,.4,.8], yticks=[0,.4,.8])
        for axis, key, factor, xlabel in zip(axes[1:], ["stable_share","stable_odd"], [100,1000],
                ["Change in share (percentage points)", r"Change in asymmetric energy ($10^{-3}\,r^2$)"]):
            wide = df.pivot(index=["agent", "unit", "animal"], columns="state", values=key)
            delta = ((wide.drug.combine_first(wide.sleep) - wide.awake_ec)*factor).rename("delta").reset_index()
            for j, group in enumerate(groups):
                part = delta[delta.agent == group]
                for animal, values in part.groupby("animal"):
                    idx = names.index(animal)
                    y = j + (idx - 1.5)*.11
                    axis.scatter(values.delta, np.full(len(values), y), marker="|", color=colors[idx], s=35, alpha=.65)
                    axis.scatter(values.delta.mean(), y, color=colors[idx], s=18)
            axis.axvline(0, ls="--", lw=.8, color="#758189", zorder=0)
            axis.set(xlabel=xlabel, ylim=(4.55,-.65), yticks=range(5))
            axis.set_yticklabels(labels if key == "stable_share" else [])
        for axis, title in zip(axes, ["a   Exchange past and future", "b   Relative to all repeatable variation",
                                      "c   Repeatable asymmetry alone"]):
            axis.set_title(title, loc="left", fontweight="bold", fontsize=8.4, pad=8)
        handles = [plt.Line2D([], [], color=c, marker="o", ls="", ms=4) for c in colors]
        fig.legend(handles, names, loc="outside lower center", ncol=4, frameon=False, handlelength=.8)
        for ext in ("pdf", "png"):
            fig.savefig(output.with_suffix("."+ext), dpi=220, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    draw(args.scores, args.output)
