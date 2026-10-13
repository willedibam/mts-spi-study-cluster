"""Visual summary of completed three-class VAR mechanism results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(root):
    result = json.loads((root / "results.json").read_text())
    summary = result["summary"]["both_M_and_T_changed"]
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    styles = [("VAR", "Direct VAR estimates"), ("z", "Intact SPI–SPI z"),
              ("m_rich", "Richer SPI marginals"), ("z_null_mean", "Disrupted alignment (3 runs)")]
    for key, label in styles:
        s = summary[key]
        line, = axes[0].plot(result["labelled_per_class"], s["balanced_accuracy"], "o-", label=label)
        lo, hi = np.asarray(s["conditional_95_CI"]).T
        axes[0].fill_between(result["labelled_per_class"], lo, hi, color=line.get_color(), alpha=.1)
    axes[0].axhline(1/3, ls=":", color="grey", label="Chance")
    axes[0].set(xlabel="Labelled realizations per class", ylabel="Balanced accuracy",
                xticks=[2, 4, 8], ylim=(.25, 1.035), title="Changed physical size and duration")
    axes[0].legend(frameon=False, fontsize=8, loc="center right", bbox_to_anchor=(1, .77))
    with np.load(root / "features.npz", allow_pickle=False) as bank:
        values = bank["VAR"]
        labels = np.asarray([r.split("|")[0] for r in bank["row_id"]])
    names = {"var-phi-0.2_cpl-0.4": "Low self / moderate coupling",
             "var-phi-0.2_cpl-0.8": "Low self / high coupling",
             "var-phi-0.95_cpl-0.4": "High self / moderate coupling"}
    for name in sorted(names):
        mask = labels == name
        axes[1].scatter(values[mask, 0], values[mask, 1], s=8, alpha=.45, label=names[name], rasterized=True)
    axes[1].set(xlabel="Estimated mean self coefficient", ylabel="Estimated mean cross-channel total",
                title="Two direct descriptors separate the regimes")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    for extension in ["png", "svg"]:
        fig.savefig(root / f"mechanism-summary.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    plot(parser.parse_args().results)
