"""Mock-up for fig:mi panel (b): each window's off-band count by state, against chance and wake (offband-result.md).

    python plot_offband_ref.py <results/representation/atlas-mi-pair-261117> <output stem> [--null <dir with t1_*.npz>]

Count per 10-s window as band() of plot_character_draft.py (recording pairs outside the surrogate's 95% band about the
recording's own G-K line; 496 pairs). References: the surrogate-against-surrogate 95% range of the count (pooled over
windows, from offband_null.py's draws; nominal 24.8 if no draws are given) and the median eyes-open wake window. Ringed:
the five windows (b) draws. A separate script, so plot_character_draft.py's default output is untouched.
"""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mtsspi.figures import figure_style, raincloud
from plot_character_draft import COLOR, SHORT, STATES, GREY, LIGHT
from offband import load, load_null, PANEL_B, Q


def draw(root, out, null=None):
    df, E, er, e0 = load(root / 'zns')
    if null:
        pos, e, _, _ = load_null(df, E, null, 't1'); L = df.L.values[pos]
        c0 = (np.abs(e[:, 1:]) > L[:, None, None]).sum(2); lo, hi = np.percentile(c0, [2.5, 97.5]); ref = r'surrogate vs.\ surrogate, 95\% range'
    else:
        lo = hi = .05 * 496; ref = r'nominal 5\%'
    figure_style(); plt.rcParams.update({'font.size': 7.5, 'axes.labelsize': 7.3, 'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8,
                                         'axes.linewidth': .7, 'svg.hashsalt': 'fig-mi-offband'})
    fig, ax = plt.subplots(figsize=(3.0, 1.9)); rng = np.random.default_rng(0)
    ax.axhspan(lo, hi, color=LIGHT, lw=0, zorder=0) if hi > lo else ax.axhline(lo, color=GREY, lw=.6, ls=(0, (3, 1.6)), zorder=0)
    wm = df[df.group == 'awake_eo'].c.median(); ax.axhline(wm, color=GREY, lw=.7, zorder=1)
    for i, s in enumerate(STATES):
        w = df[df.group == s]
        raincloud(ax, w.c.values, i, COLOR[s], rng, width=.64, box=True, point_sizes=2.3, point_kw={'alpha': .35})
        ax.scatter(i, w[w.id == PANEL_B[s]].c, s=14, fc='none', ec=COLOR['ink'], lw=.6, zorder=5) if s in PANEL_B else None
        ax.text(i, 1.01, f'{w.c.median():.0f}', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=6.4, color=COLOR['ink'])
    ax.set_xticks(range(len(STATES)), [SHORT[s] for s in STATES]); ax.set_xlim(-.6, len(STATES) - .4); ax.set_ylim(0, 5 * np.ceil(df.c.max() / 5) + 2)
    ax.text(-.55, 1.01, 'median', transform=ax.get_xaxis_transform(), ha='right', va='bottom', fontsize=5.6, color=GREY)
    ax.set_ylabel('off-band pairs of 496'); ax.tick_params(axis='x', length=0)
    ax.text(len(STATES) - .45, lo - 2, ref, ha='right', va='top', fontsize=5.6, color=GREY)
    ax.text(len(STATES) - .45, wm + 2, 'wake median', ha='right', va='bottom', fontsize=5.6, color=GREY)
    for ext in ('svg', 'pdf', 'png'):
        fig.savefig(out.with_suffix('.' + ext), dpi=600, metadata={'svg': {'Date': None}, 'pdf': {'CreationDate': None}}.get(ext))
    plt.close(fig)


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__); p.add_argument('root', type=Path); p.add_argument('output', type=Path)
    p.add_argument('--null', type=Path, default=None, help='offband_null.py output dir (t1_*.npz)')
    a = p.parse_args(); draw(a.root, a.output, a.null)
