"""Mock-up for fig:mi panel (b): off-band pairs against chance, wake and sleep (offband-result.md).

    python plot_offband_ref.py <results/representation/atlas-mi-pair-261117> <output stem> [--null <dir with t1_*.npz>] [--both]

Default: each window's off-band count by state (surrogate reference, as band() of plot_character_draft.py: recording pairs
outside the surrogate's 95% band about the recording's own G-K line; 496 pairs), against the surrogate-against-surrogate
95% range of the count (pooled over windows, from offband_null.py's draws; nominal 24.8 without them) and the median
eyes-open wake window. Ringed: the five windows (b) draws.
--both: the two references kept separate. Top, (b)'s five windows with two highlight layers: filled, off the surrogate
band (shaded; "beyond Gaussian"); ringed, off the wake band (dashed; the same session's eyes-open residual spread in
surrogate SDs, offband.wake_counts; "beyond the awake state"). Below, the count by state under each reference with its
null range (surrogate against surrogate; eyes-open windows left out one at a time). Wake is ~24.8 under the wake
reference by construction. A separate script, so plot_character_draft.py's default output is untouched.
"""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mtsspi.figures import figure_style, raincloud
from plot_character_draft import COLOR, SHORT, STATES, NAME, GREY, LIGHT
from offband import load, load_null, wake_counts, PANEL_B


def strip(ax, df, col, lo, hi, ref, top, rng, wake_line=True):
    ax.axhspan(lo, hi, color=LIGHT, lw=0, zorder=0) if hi > lo else ax.axhline(lo, color=GREY, lw=.6, ls=(0, (3, 1.6)), zorder=0)
    wm = df[df.group == 'awake_eo'][col].median()
    if wake_line: ax.axhline(wm, color=GREY, lw=.7, zorder=1)
    for i, s in enumerate(STATES):
        w = df[df.group == s]
        raincloud(ax, w[col].values, i, COLOR[s], rng, width=.64, box=True, point_sizes=2.3, point_kw={'alpha': .35})
        ax.scatter(i, w[w.id == PANEL_B[s]][col], s=14, fc='none', ec=COLOR['ink'], lw=.6, zorder=5) if s in PANEL_B else None
        ax.text(i, 1.01, f'{w[col].median():.0f}', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=6.4, color=COLOR['ink'])
    ax.set_xticks(range(len(STATES)), [SHORT[s] for s in STATES]); ax.set_xlim(-.6, len(STATES) - .4); ax.set_ylim(0, top)
    ax.text(-.55, 1.01, 'median', transform=ax.get_xaxis_transform(), ha='right', va='bottom', fontsize=5.6, color=GREY)
    ax.set_ylabel('off-band pairs of 496'); ax.tick_params(axis='x', length=0)
    ax.text(len(STATES) - .45, lo - 2, ref, ha='right', va='top', fontsize=5.6, color=GREY)
    if wake_line: ax.text(len(STATES) - .45, wm + 2, 'wake median', ha='right', va='bottom', fontsize=5.6, color=GREY)


def draw(root, out, null=None, both=False):
    df, E, er, e0 = load(root / 'zns')
    if null:
        pos, e, _, _ = load_null(df, E, null, 't1'); L = df.L.values[pos]
        c0 = (np.abs(e[:, 1:]) > L[:, None, None]).sum(2); lo, hi = np.percentile(c0, [2.5, 97.5]); ref = r'surrogate vs.\ surrogate, 95\% range'
    else:
        lo = hi = .05 * 496; ref = r'nominal 5\%'
    figure_style(); plt.rcParams.update({'font.size': 7.5, 'axes.labelsize': 7.3, 'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8,
                                         'axes.linewidth': .7, 'svg.hashsalt': 'fig-mi-offband'})
    rng = np.random.default_rng(0); top = 5 * np.ceil(df.c.max() / 5) + 2
    if not both:
        fig, ax = plt.subplots(figsize=(3.0, 1.9)); strip(ax, df, 'c', lo, hi, ref, top, rng)
    else:
        wake_counts(df, er, e0); n0 = df[df.group == 'awake_eo'].cw; wlo, whi = np.percentile(n0, [2.5, 97.5])
        fig = plt.figure(figsize=(7.2, 4.1)); g = fig.add_gridspec(2, 5, height_ratios=[1, 1.15])
        for j, s in enumerate(('awake_eo', 'SLP', 'KT', 'KTMD', 'PF')):
            ax = fig.add_subplot(g[0, j]); i = int(np.where(df.id == PANEL_B[s])[0][0]); (gg, kk) = E[i, 0]
            c = np.polyfit(gg, kk, 1); sd = e0[i].std(); xx = np.linspace(-.05, .5, 50); ink = COLOR['ink'] if s == 'awake_eo' else COLOR[s]
            ax.fill_between(xx, np.polyval(c, xx) - df.L[i], np.polyval(c, xx) + df.L[i], color=LIGHT, lw=0, zorder=0)
            for sg in (-1, 1): ax.plot(xx, np.polyval(c, xx) + sg * df.Lw[i] * sd, color=GREY, lw=.5, ls=(0, (2.5, 1.5)), zorder=1)
            os_, ow = np.abs(er[i]) > df.L[i], np.abs(er[i]) > df.Lw[i] * sd
            ax.scatter(gg[~os_], kk[~os_], s=3, lw=0, color=GREY, alpha=.5, zorder=2)
            ax.scatter(gg[os_], kk[os_], s=7, lw=0, color=ink, zorder=3)
            ax.scatter(gg[ow], kk[ow], s=16, fc='none', ec=COLOR['ink'], lw=.45, zorder=4)
            ax.set(xlim=(-.02, .42), ylim=(-.17, .52), xticks=[0, .2, .4], yticks=[0, .2, .4], aspect='equal')
            ax.set_title(f'{NAME[s]}\n{os_.sum()} / {ow.sum()}', fontsize=6.8, color=ink, pad=2)
            ax.set_xlabel('$G$', labelpad=1); ax.set_ylabel('$K$', labelpad=1) if j == 0 else ax.set_yticklabels([])
        for j, (col, a, b, r, name) in enumerate((('c', lo, hi, ref, r'surrogate reference (beyond Gaussian)'),
                                                 ('cw', wlo, whi, r'eyes-open wake, leave-one-out 95\% range', r'wake reference (beyond the awake state)'))):
            ax = fig.add_subplot(g[1, 0:3] if j == 0 else g[1, 3:5])
            strip(ax, df, col, a, b, r, top, rng, wake_line=col == 'c'); ax.set_title(name, fontsize=7, pad=12)
    for ext in ('svg', 'pdf', 'png'):
        fig.savefig(out.with_suffix('.' + ext), dpi=600, metadata={'svg': {'Date': None}, 'pdf': {'CreationDate': None}}.get(ext))
    plt.close(fig)


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__); p.add_argument('root', type=Path); p.add_argument('output', type=Path)
    p.add_argument('--null', type=Path, default=None, help='offband_null.py output dir (t1_*.npz)')
    p.add_argument('--both', action='store_true', help='surrogate and wake references side by side')
    a = p.parse_args(); draw(a.root, a.output, a.null, a.both)
