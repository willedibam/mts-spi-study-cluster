"""Figure helpers shared by the report and case-study plots: one serif style,
one way to save a figure, and the raincloud mark."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def style():
    plt.rcParams.update({'text.usetex': False, 'font.family': 'serif', 'font.serif': ['CMU Serif', 'DejaVu Serif'],
                        'mathtext.fontset': 'cm', 'font.size': 9, 'axes.titlesize': 10,
                        'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                        'xtick.direction': 'out', 'ytick.direction': 'out', 'figure.dpi': 180,
                        'savefig.bbox': 'tight', 'axes.grid': False})


def save(fig, output: Path, name: str):
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f'{name}.png', dpi=180)
    fig.savefig(output / f'{name}.svg')
    return fig


def raincloud(ax, values, position, color, rng, width=.2):
    values = np.asarray(values); values = values[np.isfinite(values)]
    if len(values) > 2 and np.std(values) > 1e-9:
        parts = ax.violinplot(values, positions=[position], widths=width, showextrema=False)
        for body in parts['bodies']:
            verts = body.get_paths()[0].vertices
            verts[:, 0] = np.maximum(verts[:, 0], position)
            body.set_facecolor(color); body.set_edgecolor('none'); body.set_alpha(.25)
    ax.scatter(position-rng.uniform(.025, width*.55, len(values)), values, color=color, s=7, alpha=.5, linewidths=0)
    ax.plot([position-width*.12, position+width*.12], [np.median(values)]*2, color=color, lw=1.6)
