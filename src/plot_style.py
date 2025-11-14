from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

_STYLE_INITIALISED = False


def apply_plot_style() -> None:
    global _STYLE_INITIALISED
    if _STYLE_INITIALISED:
        return
    plt.close("all")
    plt.rcdefaults()
    sns.reset_orig()
    sns.set_theme(
        style="white",
        rc={
            "text.usetex": True,
            "font.size": 12,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "axes.labelsize": 16,
            "axes.titlesize": 20,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        },
    )
    _STYLE_INITIALISED = True

