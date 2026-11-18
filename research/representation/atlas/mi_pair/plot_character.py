"""Render the normal-scored MI result from the checksum-listed zns arrays."""
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GROUPS = ("KT", "KTMD", "MD", "PF", "SLP")
LABELS = ("Ketamine", "Ket. + med.", "Medetomidine", "Propofol", "Sleep")
BLUE, ORANGE, GREY = "#176387", "#BA542B", "#78838A"


def read(directory):
    frames, profiles = [], {}
    for path in sorted(directory.glob("zns_[AB]_*.npz")):
        with np.load(path) as data:
            frame = pd.DataFrame({k: data[k] for k in ("id", "unit", "animal", "state", "agent")})
            for j, key in enumerate(("real", "null")):
                frame[key] = data["V"][:, 1, j]
                # Independent audit of every archived window and matched null from the edge arrays.
                z = [np.corrcoef(g, k)[0, 1] for g, k in data["E"][:, 1, j]]
                np.testing.assert_allclose(frame[key], z, atol=1e-6)
            if path.stem.endswith("_PF"):
                profiles.update(zip(frame.id, data["E"][:, 1, 0]))
            frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    assert df.id.is_unique
    df["residual"] = df.real - df.null
    return df, profiles


def session_effects(df):
    scores = df.groupby(["unit", "animal", "agent", "state"])[["real", "null", "residual"]].mean().unstack()
    target = scores.xs("drug", axis=1, level="state").combine_first(scores.xs("sleep", axis=1, level="state"))
    return (target - scores.xs("awake_ec", axis=1, level="state")).reset_index()


def draw(directory, output):
    df, profiles = read(directory)
    sessions = session_effects(df)
    style = {"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
             "mathtext.fontset": "cm", "font.size": 8, "axes.spines.top": False,
             "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.8), layout="constrained",
                                 gridspec_kw={"width_ratios": [1.04, 1, 1.08]})
        ax, middle, right = axes
        # Deterministic illustration: first propofol date, median-residual window of each state.
        sample = df[df.unit == min(df.loc[df.agent == "PF", "unit"])]
        upper = 0
        for state, name, color in (("awake_ec", "Wake", BLUE), ("drug", "Propofol", ORANGE)):
            rows = sample[sample.state == state]
            representative = rows.loc[(rows.residual - rows.residual.median()).abs().idxmin()]
            g, k = profiles[representative.id]
            upper = max(upper, g.max(), k.max())
            ax.scatter(g, k, s=5, alpha=.52, color=color, linewidth=0,
                       label=f"{name}  $z$ = {representative.real:.3f}")
        ax.plot([0, upper], [0, upper], lw=.7, ls="--", color=GREY, zorder=0)
        ax.set(xlabel="Gaussian MI (nats)", ylabel="Nearest-neighbour MI (nats)", aspect="equal")
        ax.legend(loc="upper left", fontsize=7, handletextpad=.1)
        for j, group in enumerate(GROUPS):
            part = sessions[sessions.agent == group]
            mean = part[["real", "null"]].mean()
            middle.plot(mean, [j, j], color=GREY, lw=1)
            middle.scatter(mean.real, j, color=BLUE, s=18, label="Observed" if j == 0 else None)
            middle.scatter(mean.null, j, color="white", edgecolor=GREY, s=18,
                           label="Surrogate" if j == 0 else None)
            for _, animal in part.groupby("animal"):
                right.scatter(animal.residual.mean(), j, color=BLUE, alpha=.72, s=22, linewidth=0)
            right.scatter(part.residual.mean(), j+.17, marker="|", s=65, color="black")
        middle.set_yticks(range(len(GROUPS)), LABELS)
        right.set_yticks(range(len(GROUPS)), [])
        for axis in (middle, right):
            axis.axvline(0, color=GREY, lw=.7, ls="--", zorder=0)
            axis.set_ylim(4.55, -.65)
        middle.set_xlabel(r"State minus wake, $\Delta z$")
        right.set_xlabel(r"$\Delta(z_{\rm observed}-z_{\rm surrogate})$")
        middle.legend(loc="upper center", bbox_to_anchor=(.48, -.24), ncol=2,
                      handlelength=.8, fontsize=7)
        right.text(.5, -.25, "Dot per animal   |   session mean", transform=right.transAxes,
                   ha="center", va="top", fontsize=7)
        for axis, title in zip(axes, ("a   One dot per edge", "b   State contrast", "c   Beyond the spectral null")):
            axis.set_title(title, loc="left", fontsize=9, weight="bold", pad=9)
        for ext in ("pdf", "png"):
            fig.savefig(output.with_suffix("."+ext), dpi=220, bbox_inches="tight")
        plt.close(fig)
    print(sessions.groupby("agent")[["real","null","residual"]].mean().to_string())


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    draw(args.directory, args.output)
