"""Panel (c) of fig:mi redrawn with mean difference kurtosis in place of Δz (post hoc, 26 November; not in a prereg).

Panels a-e, panel (c)'s construction for each readout: per 10-s window, the value minus its session's eyes-closed wake mean, in
units of the eyes-open wake windows' SD of that difference; rainclouds per state; the printed number is the window mean (as
(c)); open markers are animal means. Readouts, left to right:
  Δz      panel (c) itself, z = corr(G, K) over the 496 pairs, real minus its IAAFT surrogate (reproduction gate);
  real    mean over the 496 pairs of the excess kurtosis k04 of the normal-scored electrode difference, recording only;
  null    the same on the window's single spectrum-final IAAFT draw (kdiff.py);
  KD      real - null, kdiff's primary readout; real and null are drawn on KD's scale, so real - null = KD exactly;
  KZ      per pair (k04 real - mean of 100 marginal-matched IAAFT draws) / their SD, averaged (kens.py, kens_score.py).
Panel f, absolute levels (no baseline): window-mean k04 of the recording and the two 100-draw null means; dots are animal
means, bars their mean.
  python kd_panel.py <results/representation/atlas-mi-pair-261117> <output stem>
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent)]
from mtsspi.figures import figure_style, raincloud                                    # noqa: E402
from plot_character_draft import COLOR, DRUGS, GREY, MARK, SHORT, STATES, load       # noqa: E402  (read only)


def units(df, col):
    """Panel (c)'s units: window minus its session's eyes-closed mean, over the eyes-open windows' SD of that."""
    d = df[col] - df.unit.map(df[df.state == 'awake_ec'].groupby('unit')[col].mean())
    return d, d[df.state == 'awake_eo'].std(ddof=1)


def main(root, out):
    root = Path(root)
    ke = pd.read_csv(root / 'ensemble/ke-windows.csv.gz')
    kd = pd.read_csv(root / 'cumulants/kd-windows.csv.gz')[['id', 'KD_real', 'KD_surr', 'KD']]
    df = ke.merge(kd, on='id', suffixes=('', '_kd')); assert len(df) == len(ke) == len(kd)
    assert np.allclose(df.KD, df.KD_kd)
    # gate: Δz from the table equals panel (c)'s, window by window (plot_character_draft.load on the zns arrays)
    ref, _ = load(root); ref = ref.set_index('id').loc[df.id]
    gap = np.abs(df.ZGK_sd.values - ref.sd.values).max(); print(f'gate: max |Δz table - panel (c)| = {gap:.1e}')
    assert gap < 1e-9
    dnet, snet = units(df, 'KD')
    cols = {'dz': df.ZGK_sd, 'real': units(df, 'KD_real')[0] / snet, 'null': units(df, 'KD_surr')[0] / snet,
            'KD': dnet / snet, 'KZ': df.KZ_sd}
    assert np.allclose(cols['real'] - cols['null'], cols['KD'])
    for k, v in cols.items():
        df['c_' + k] = v

    # ---- numbers printed beside the figure
    print('\nwindow means per state (panel (c)\'s printed numbers), panel (c) units')
    print(df[df.group.isin(STATES)].groupby('group')[['c_' + k for k in cols]].mean().loc[list(STATES)].round(2).to_string())
    S = df.groupby(['unit', 'animal', 'group'])[['c_' + k for k in cols]].mean().reset_index()
    print('\nanimal means (session means first), drug pooled = mean over agents per animal; t(3) over 4 animals')
    for k in cols:
        A = S[S.group.isin(DRUGS)].groupby('animal')['c_' + k].mean(); t = ttest_1samp(A, 0)
        sl = S[S.group == 'SLP'].groupby('animal')['c_' + k].mean()
        ag = S[S.group.isin(DRUGS)].pivot_table(index='animal', columns='group', values='c_' + k).round(2)
        print(f'{k:5s} drug {A.mean():+.2f} [{" ".join(f"{v:+.2f}" for v in A)}] t(3) {t.statistic:+.2f} p {t.pvalue:.3f}'
              f'  sleep {" ".join(f"{a} {v:+.2f}" for a, v in sl.items())}')
        print('      by agent:', ag.to_dict('index'))

    # ---- figure: 2 x 3; (c') rows are panel (c)'s construction, the last panel absolute levels
    figure_style(); plt.rcParams.update({'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.3,
                                         'ytick.labelsize': 6.3, 'svg.hashsalt': 'kd-panel'})
    fig = plt.figure(figsize=(7.2, 6.4), layout='constrained')
    axs = fig.subplots(2, 3); ax = [axs[0, 0], axs[0, 1], axs[0, 2], axs[1, 0], axs[1, 1]]
    rng = np.random.default_rng(0); win = df[df.group.isin(STATES)]
    titles = {'dz': r'\textbf{a}\enspace $\Delta z$: panel (c) itself', 'real': r'\textbf{b}\enspace real $\bar k_{04}$',
              'null': r'\textbf{c}\enspace null $\bar k_{04}$ (its 1 IAAFT draw)',
              'KD': r'\textbf{d}\enspace KD = real $-$ null', 'KZ': r'\textbf{e}\enspace KZ: z against 100 matched nulls'}
    yl = (-9.5, 8.5)
    for a, k in zip(ax, cols):
        for i, s in enumerate(STATES):
            w = win[win.group == s]; v = w['c_' + k].values
            raincloud(a, np.clip(v, *yl), i, COLOR[s], rng, width=.64, box=True, point_sizes=1.6,
                      point_kw={'alpha': .3}, line_scale=.7)
            m = v.mean(); a.text(i, yl[1] - .1, f'{m:+.1f}'.replace('-', '--'), ha='center', va='top', fontsize=6,
                                 color=COLOR['ink'] if s in DRUGS else GREY,
                                 fontweight='bold' if s in DRUGS else 'normal')
            for an, mm in S[S.group == s].groupby('animal')['c_' + k].mean().items():
                a.scatter(i + .02, np.clip(mm, *yl), s=10, marker=MARK[an], facecolor='white', edgecolor=COLOR['ink'],
                          lw=.55, zorder=6)
        a.axhline(0, color=GREY, lw=.5, ls='--', zorder=0)
        a.set_xticks(range(len(STATES)), [SHORT[s] for s in STATES], fontsize=5.4)
        for tk, s in zip(a.get_xticklabels(), STATES):
            tk.set_color(COLOR[s])
        a.tick_params(axis='x', length=0); a.set_xlim(-.6, len(STATES) - .4); a.set_ylim(*yl)
        a.set_title(titles[k], fontsize=6.8, pad=3, loc='left')
    for a in (ax[0], ax[3]):
        a.set_ylabel('per 10-s window, wake SDs')
    for a in (ax[1], ax[2]):
        a.set_ylabel("per 10-s window, KD's wake SDs", fontsize=6)
    ax[0].legend([Line2D([], [], ls='', marker=MARK[a_], mfc='white', mec=COLOR['ink'], ms=3.5) for a_ in MARK],
                 list(MARK), loc='lower left', ncol=2, fontsize=5.4, handletextpad=.1, columnspacing=.6,
                 title='animal means', title_fontsize=5.4)

    b1 = axs[1, 2]; G = ('awake_eo', 'SLP') + DRUGS
    A2 = df.groupby(['unit', 'animal', 'group'])[['raw_k04_real', 'raw_k04_im', 'raw_k04_ia']].mean().reset_index() \
        .groupby(['animal', 'group']).mean(numeric_only=True).reset_index()
    for j, (c, nm, col) in enumerate((('raw_k04_real', 'recording', COLOR['ink']),
                                      ('raw_k04_ia', 'IAAFT, mean of 100', '#A7AFB4'),
                                      ('raw_k04_im', 'marginal-matched IAAFT', '#2E86AB'))):
        for i, g in enumerate(G):
            w = A2[A2.group == g]; x = i + (j - 1) * .24
            b1.scatter(np.full(len(w), x), w[c], s=9, lw=.5, marker='o', facecolor='white' if j else col,
                       edgecolor=col, zorder=3)
            b1.plot([x - .1, x + .1], [w[c].mean()] * 2, color=col, lw=1.1, zorder=4, label=nm if i == 0 else None)
    b1.axhline(0, color=GREY, lw=.5, ls='--', zorder=0)
    b1.set_xticks(range(len(G)), [SHORT[g] for g in G], fontsize=5.4)
    for tk, g in zip(b1.get_xticklabels(), G):
        tk.set_color(COLOR[g])
    b1.tick_params(axis='x', length=0); b1.set_ylabel(r'window-mean $k_{04}$, absolute')
    b1.legend(fontsize=5.4, loc='upper right', handletextpad=.3, handlelength=1)
    b1.set_title(r'\textbf{f}\enspace Absolute: animal means above every null, every state', fontsize=6.8, pad=3, loc='left')
    fig.suptitle(r"Panel (c)'s construction with mean difference kurtosis over the 496 pairs (a--e): window minus its "
                 r"session's eyes-closed mean, in eyes-open SDs. Numbers: window means; markers: animal means; "
                 r"clipped at the axis.", x=0, ha='left', fontsize=6.6)
    out = Path(out)
    for ext in ('png', 'pdf'):
        fig.savefig(out.with_suffix('.' + ext), dpi=300, metadata={'CreationDate': None} if ext == 'pdf' else None)
    print('wrote', out.with_suffix('.png'))


if __name__ == '__main__':
    main(*sys.argv[1:3])
