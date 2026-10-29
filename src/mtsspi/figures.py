"""Figure helpers shared across studies: two styles (`style` without LaTeX, `figure_style` with it), both
Helvetica text with Computer Modern math, one way to save a figure, and the raincloud mark."""
from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np


# Helvetica text, Computer Modern math. Under LaTeX the text font is `helvet` (URW Nimbus Sans, Helvetica
# metrics) and matplotlib's default tick labels ($\mathdefault{...}$) stay in it.
SANS = {'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'], 'mathtext.fontset': 'cm'}
SANS_PREAMBLE = (r'\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}'
                 r'\DeclareMathAlphabet{\mathhv}{OT1}{phv}{m}{n}\renewcommand{\mathdefault}[1]{\mathhv{#1}}')


def style():
    plt.rcParams.update({'text.usetex': False, **SANS, 'font.size': 9, 'axes.titlesize': 10,
                        'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                        'xtick.direction': 'out', 'ytick.direction': 'out', 'figure.dpi': 180,
                        'savefig.bbox': 'tight', 'axes.grid': False})


def save(fig, output: Path, name: str):
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f'{name}.png', dpi=180)
    fig.savefig(output / f'{name}.svg')
    return fig


def raincloud(ax, values, position, color, rng, width=.2, *, box=False, point_sizes=7, point_markers=None, point_kw=None,
              outline=False, line_scale=1.):
    """``outline`` edges the density in the box colour; ``line_scale`` thins or thickens all box/density lines."""
    values = np.asarray(values); finite = np.isfinite(values)
    point_sizes = np.asarray(point_sizes)
    if point_sizes.ndim:
        if point_sizes.shape != values.shape:
            raise ValueError('Point sizes must align with the unfiltered values.')
        point_sizes = point_sizes[finite]
    if point_markers is not None:
        point_markers = np.asarray(point_markers)
        if point_markers.shape != values.shape:
            raise ValueError('Point markers must align with the unfiltered values.')
        point_markers = point_markers[finite]
    values = values[finite]
    cloud_position = position + width*.14 if box else position
    if len(values) > 2 and np.std(values) > 1e-9:
        parts = ax.violinplot(values, positions=[cloud_position], widths=width, showextrema=False)
        for body in parts['bodies']:
            verts = body.get_paths()[0].vertices
            verts[:, 0] = np.maximum(verts[:, 0], cloud_position)
            body.set_facecolor(to_rgba(color, .25))
            body.set_edgecolor(color if outline else 'none'); body.set_linewidth(.9*line_scale)
    jitter = position-rng.uniform(width*.14 if box else .025, width*.55, len(values))
    point_kw = {'alpha': .5, 'linewidths': 0, **(point_kw or {})}
    if point_markers is None:
        ax.scatter(jitter, values, color=color, s=point_sizes, **point_kw)
    else:
        sizes = np.broadcast_to(point_sizes, values.shape)
        for marker in dict.fromkeys(point_markers):
            use = point_markers == marker
            ax.scatter(jitter[use], values[use], color=color, s=sizes[use], marker=marker, **point_kw)
    if box:
        ax.boxplot([values], positions=[position], widths=width*.16, whis=1.5,
                   showfliers=False, manage_ticks=False, patch_artist=True,
                   boxprops={'facecolor':'white','edgecolor':color,'linewidth':.9*line_scale},
                   whiskerprops={'color':color,'linewidth':.8*line_scale},
                   capprops={'color':color,'linewidth':.8*line_scale},
                   medianprops={'color':color,'linewidth':1.3*line_scale})
    else:
        ax.plot([position-width*.12, position+width*.12], [np.median(values)]*2, color=color, lw=1.6)


def figure_style():
    if not (shutil.which('latex') and shutil.which('dvipng')):
        raise RuntimeError('Requested LaTeX rendering requires latex and dvipng.')
    plt.rcParams.update({'text.usetex': True, 'text.latex.preamble': SANS_PREAMBLE, **SANS,
        'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.direction': 'out', 'ytick.direction': 'out', 'legend.frameon': False,
        'lines.linewidth': 1.7, 'lines.markersize': 2.7,
        'figure.constrained_layout.use': True, 'savefig.bbox': 'tight'})
