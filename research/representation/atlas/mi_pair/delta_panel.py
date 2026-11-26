"""Panel (c) of fig:mi beside the same construction for slow-wave share (post hoc, 26 November; not in a prereg).

Per 10-s window, the value minus its session's eyes-closed wake mean, in units of the eyes-open windows' SD of that difference
(panel (c)'s units), rainclouds per state, printed number the window mean, open markers animal means.
  a  Δz, panel (c) itself (plot_character_draft.load);
  b  slow-wave share: prepare.delta_fraction, the 0.5-4 Hz share of 0.5-40 Hz power, same construction;
  c  the two per window, Chibi and George (the animals with sleep); large markers animal-by-state means.
Printed: window AUCs per animal, drug against sleep and sleep against same-session eyes-closed wake, for both readouts.
  python delta_panel.py <dir holding zns/> <output stem>
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.metrics import roc_auc_score

from mtsspi.figures import figure_style, raincloud
from plot_character_draft import COLOR, DRUGS, GREY, MARK, SHORT, STATES, load       # read only


def units(df, col):
    d = df[col] - df.unit.map(df[df.state == 'awake_ec'].groupby('unit')[col].mean())
    return d / d[df.state == 'awake_eo'].std(ddof=1)


def auc(a, b):
    return roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])


def main(root, out):
    df, _ = load(Path(root)); df['c_dz'], df['c_sw'] = df.sd, units(df, 'delta')
    print('window means per state, panel (c) units:')
    print(df[df.group.isin(STATES)].groupby('group')[['c_dz', 'c_sw']].mean().loc[list(STATES)].round(2).to_string())
    print('\nwindow AUC (|AUC - .5| read; drug vs sleep, sleep vs same-session eyes closed)')
    for an in ('Chibi', 'George'):
        d = df[df.animal == an]; sl = d[d.group == 'SLP']; dr = d[d.state == 'drug']
        ec = d[(d.state == 'awake_ec') & d.unit.isin(sl.unit)]
        for c, nm in (('c_dz', 'Δz'), ('c_sw', 'slow-wave')):
            a1, a2 = auc(dr[c], sl[c]), auc(sl[c], ec[c])
            print(f'  {an:6s} {nm:9s} drug/sleep {max(a1, 1 - a1):.2f}  sleep/EC {max(a2, 1 - a2):.2f}  (n drug {len(dr)}, sleep {len(sl)}, EC {len(ec)})')

    figure_style(); plt.rcParams.update({'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.3,
                                         'ytick.labelsize': 6.3, 'svg.hashsalt': 'delta-panel'})
    fig = plt.figure(figsize=(7.2, 2.7), layout='constrained'); ax = fig.subplots(1, 3, width_ratios=(1, 1, 1.05))
    rng = np.random.default_rng(0); win = df[df.group.isin(STATES)]
    S = df.groupby(['unit', 'animal', 'group'])[['c_dz', 'c_sw']].mean().reset_index()
    lims = {'c_dz': (-9.5, 4.5), 'c_sw': (-4.5, 7.5)}
    titles = {'c_dz': r'\textbf{a}\enspace $\Delta z$: panel (c)', 'c_sw': r'\textbf{b}\enspace Slow-wave share, same construction'}
    for a, k in zip(ax[:2], ('c_dz', 'c_sw')):
        yl = lims[k]
        for i, s in enumerate(STATES):
            v = win[win.group == s][k].values
            raincloud(a, np.clip(v, *yl), i, COLOR[s], rng, width=.64, box=True, point_sizes=1.6, point_kw={'alpha': .3}, line_scale=.7)
            a.text(i, yl[1] - .1, f'{v.mean():+.1f}'.replace('-', '--'), ha='center', va='top', fontsize=6,
                   color=COLOR['ink'] if s in DRUGS else GREY, fontweight='bold' if s in DRUGS else 'normal')
            for an, mm in S[S.group == s].groupby('animal')[k].mean().items():
                a.scatter(i + .02, np.clip(mm, *yl), s=10, marker=MARK[an], facecolor='white', edgecolor=COLOR['ink'], lw=.55, zorder=6)
        a.axhline(0, color=GREY, lw=.5, ls='--', zorder=0)
        a.set_xticks(range(len(STATES)), [SHORT[s] for s in STATES], fontsize=5.4)
        for tk, s in zip(a.get_xticklabels(), STATES):
            tk.set_color(COLOR[s])
        a.tick_params(axis='x', length=0); a.set_xlim(-.6, len(STATES) - .4); a.set_ylim(*yl)
        a.set_title(titles[k], fontsize=6.8, pad=3, loc='left'); a.set_ylabel('per 10-s window, wake SDs')
    ax[0].legend([Line2D([], [], ls='', marker=MARK[a_], mfc='white', mec=COLOR['ink'], ms=3.5) for a_ in MARK], list(MARK),
                 loc='lower left', ncol=2, fontsize=5.4, handletextpad=.1, columnspacing=.6, title='animal means', title_fontsize=5.4)
    c = ax[2]; P = win[win.animal.isin(('Chibi', 'George'))]
    for s in STATES:
        w = P[P.group == s]
        c.scatter(w.c_sw, np.clip(w.c_dz, -14, 4), s=2.5, color=COLOR[s], alpha=.35, lw=0, zorder=2)
        for an, mm in w.groupby('animal')[['c_sw', 'c_dz']].mean().iterrows():
            c.scatter(mm.c_sw, max(mm.c_dz, -14), s=22, marker=MARK[an], facecolor=COLOR[s], edgecolor='white', lw=.6, zorder=5)
    c.axhline(0, color=GREY, lw=.5, ls='--', zorder=0); c.axvline(0, color=GREY, lw=.5, ls='--', zorder=0)
    c.set_xlabel('slow-wave share, wake SDs'); c.set_ylabel(r'$\Delta z$, wake SDs'); c.set_ylim(-14.5, 4.5)
    c.set_title(r'\textbf{c}\enspace Per window, Chibi and George', fontsize=6.8, pad=3, loc='left')
    c.legend([Line2D([], [], ls='', marker='o', mfc=COLOR[s], mec='none', ms=3.5) for s in STATES], [SHORT[s].replace('\n', '') for s in STATES],
             loc='lower left', fontsize=5.4, handletextpad=.1, ncol=2, columnspacing=.6)
    fig.suptitle(r"Window minus its session's eyes-closed mean, in eyes-open SDs. Numbers: window means; open markers: animal means; "
                 r"(c) large markers: animal-by-state means. Clipped at the axes.", x=0, ha='left', fontsize=6.4)
    for ext in ('png', 'pdf'):
        fig.savefig(Path(out).with_suffix('.' + ext), dpi=300, metadata={'CreationDate': None} if ext == 'pdf' else None)
    print('wrote', Path(out).with_suffix('.png'))


if __name__ == '__main__':
    main(*sys.argv[1:3])
