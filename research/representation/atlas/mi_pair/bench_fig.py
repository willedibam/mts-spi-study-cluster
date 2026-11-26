"""Figure for bench.py's score (post hoc display of preregs/bench-prereg.md; reads bench-score.json and bench-windows.csv.gz).
a: per-agent drug effect and sleep, oriented wake SDs (session means), metrics in rows grouped by TIER, with the frozen
criterion (C1 same sign in all 10 drug cells, C2 sleep/drug ratio 90% CI below .25) and the ratio; b: Δz against spectral
entropy per window, Chibi and George.
TIER (from the literature, not from these results): ours; clinical (star): an index of, or a component of, a deployed
depth-of-anaesthesia monitor (GE Entropy: spectral entropy, Viertiö-Oja et al. 2004; SEF95; BIS beta ratio, here a band-truncated
proxy); research: published markers of anaesthetic or conscious state (slow-wave and alpha power, Purdon et al. 2013; Lempel-Ziv,
Schartner et al. 2015; permutation entropy, Olofsen et al. 2008; aperiodic slope, Lendner et al. 2020; wSMI, King et al. 2013;
wPLI, Vinck et al. 2011); generic: connectivity summaries not specific to anaesthesia; ablation: Δz's ingredient without its
surrogate.
  python bench_fig.py <bench results dir> <output stem>
"""
import sys, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mtsspi.figures import figure_style
from plot_character_draft import COLOR, GREY, MARK, SHORT, STATES               # read only

NAMES = {'DZ': r'$\bm{\Delta z}$ (ours)', 'SW': 'slow-wave share', 'ALPHA': 'rel. alpha', 'BR': 'beta ratio (BIS)', 'SEF95': 'SEF95',
         'SE': 'spectral entropy', 'AP': 'aperiodic slope', 'LZC': 'Lempel-Ziv', 'PE': 'perm. entropy', 'ABSR': r'mean $|r|$',
         'EIG': 'top eigenvalue', 'COHD': 'coherence 1-4', 'COHA': 'coherence 8-13', 'WPLID': 'wPLI 1-4', 'WPLIA': 'wPLI 8-13',
         'KURA': 'Kuramoto 8-13', 'WSMI': 'wSMI', 'MG': 'mean Gaussian MI', 'MK': 'mean KSG MI', 'MKG': 'mean KSG - Gaussian (ablation)'}
TIERS = (('ours', ('DZ',)), ('clinical', ('SE', 'SEF95', 'BR')), ('research', ('SW', 'ALPHA', 'LZC', 'PE', 'AP', 'WSMI', 'WPLID', 'WPLIA')),
         ('generic', ('ABSR', 'EIG', 'COHD', 'COHA', 'KURA', 'MG', 'MK')), ('ablation', ('MKG',)))
TIER = {m: t for t, ms in TIERS for m in ms}
ORDER = [m for _, ms in TIERS for m in ms]
TSTYLE = {'ours': ('#2B2B2B', 1.0), 'clinical': ('#1B7F5E', 1.0), 'research': ('#3E6FB0', .7), 'generic': ('#8C969C', .45),
          'ablation': ('#8C969C', .45)}           # outline colour, opacity
TLABEL = {'ours': 'ours', 'clinical': r'$\star$ clinical monitor', 'research': 'published research marker', 'generic': 'generic connectivity',
          'ablation': 'ablation (no surrogate)'}


def label(m, bold=True):
    n = NAMES[m] + (r'$^{\star}$' if TIER[m] == 'clinical' else '')
    return r'\textbf{' + n + '}' if bold else n


def main(d, out):
    d = Path(d); S = json.load(open(d / 'bench-score.json'))['res']; W = pd.read_csv(d / 'bench-windows.csv.gz')
    M = [m for m in ORDER if m in S]                     # BSR omitted: no window below the threshold, constant 0
    figure_style(); plt.rcParams.update({'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.3, 'ytick.labelsize': 6.3, 'svg.hashsalt': 'bench'})
    plt.rcParams['text.latex.preamble'] += r'\usepackage{bm}'
    fig = plt.figure(figsize=(7.2, 4.6), layout='constrained'); ax = fig.subplots(1, 2, width_ratios=(1.55, 1))
    y = np.arange(len(M))[::-1]; cols = ('KT', 'KTMD', 'MD', 'PF')
    H = np.array([[S[m]['by_agent'][g] for g in cols] + [S[m]['sleep']] for m in M])
    a = ax[0]; a.imshow(np.clip(H, -6, 6), cmap='RdBu_r', vmin=-6, vmax=6, aspect='auto', extent=(-.5, 4.5, -.5, len(M) - .5))
    for i, m in enumerate(M):
        for j in range(5):
            a.text(j, y[i], f'{H[i, j]:+.1f}'.replace('-', '--'), ha='center', va='center', fontsize=5, color='k')
        r = S[m]; col, al = TSTYLE[TIER[m]]
        a.text(5.0, y[i], 'yes' if r['C1'] else 'no', ha='center', va='center', fontsize=5.4, color=COLOR['ink'] if r['C1'] else GREY)
        t = f"{r['ratio']:+.2f}".replace('-', '--'); t = r'\textbf{' + t + '}' if r['C1'] and r['C2'] else t
        a.text(5.9, y[i], t, ha='center', va='center', fontsize=5.4, color=COLOR['ink'] if r['C2'] else GREY)
    a.set_xlim(-.5, 6.4)
    edges = np.cumsum([len([m for m in ms if m in S]) for _, ms in TIERS])[:-1]
    for e in edges:
        a.axhline(len(M) - e - .5, color='white', lw=1.6)
    a.set_yticks(y, [label(m) for m in M], fontsize=6)
    for tk, m in zip(a.get_yticklabels(), M):
        col, al = TSTYLE[TIER[m]]; tk.set_color(col); tk.set_alpha(max(al, .6))
    a.set_xticks(list(range(5)) + [5.0, 5.9], [SHORT[g] for g in cols + ('SLP',)] + ['C1', 'sleep/\ndrug'], fontsize=5.4)
    for tk, g in zip(a.get_xticklabels()[:5], cols + ('SLP',)):
        tk.set_color(COLOR[g])
    a.tick_params(length=0); [sp.set_visible(False) for sp in a.spines.values()]
    a.set_title(r'\textbf{a}\enspace Effect in the drug direction (wake SDs)', fontsize=6.8, loc='left', pad=3)
    a.legend([Line2D([], [], ls='', marker='s', mfc=TSTYLE[t][0], mec='none', alpha=max(TSTYLE[t][1], .6), ms=4) for t, _ in TIERS[:4]],
             [TLABEL[t] for t, _ in TIERS[:4]], loc='upper left', bbox_to_anchor=(-.62, -.04), ncol=4, fontsize=5.4, handletextpad=.2, columnspacing=.8)
    e = ax[1]; P = W[W.animal.isin(('Chibi', 'George')) & W.group.isin(STATES)]
    for s in STATES:
        w = P[P.group == s]; e.scatter(w.SE, np.clip(w.DZ, -4, 15), s=2.5, color=COLOR[s], alpha=.4, lw=0)
        for an, mm in w.groupby('animal')[['SE', 'DZ']].mean().iterrows():
            e.scatter(mm.SE, mm.DZ, s=20, marker=MARK[an], facecolor=COLOR[s], edgecolor='white', lw=.6, zorder=5)
    e.axhline(0, color=GREY, lw=.5, ls='--'); e.axvline(0, color=GREY, lw=.5, ls='--'); e.set_ylim(-4.5, 15)
    e.set_xlabel(r'\textbf{spectral entropy}$^{\star}$, drug direction'); e.set_ylabel(r'$\bm{\Delta z}$, drug direction')
    e.legend([Line2D([], [], ls='', marker='o', mfc=COLOR[s], mec='none', ms=3.5) for s in STATES], [SHORT[s].replace('\n', '') for s in STATES],
             loc='upper left', fontsize=5.4, handletextpad=.1, ncol=2, columnspacing=.6)
    e.set_title(r'\textbf{b}\enspace Two axes, per window (Chibi, George)', fontsize=6.8, loc='left', pad=3)
    fig.suptitle("Window minus its session's eyes-closed mean, in eyes-open SDs, signed so that drug is positive; session means. C1: same sign in all "
                 "10 drug cells; sleep/drug in black where its 90\\% CI is below .25 (C2), bold where both pass.\n(b) large markers: animal-by-state means. "
                 "BSR omitted (constant 0).", x=0, ha='left', fontsize=6)
    for ext in ('png', 'pdf'):
        fig.savefig(Path(out).with_suffix('.' + ext), dpi=300, metadata={'CreationDate': None} if ext == 'pdf' else None)
    print('wrote', Path(out).with_suffix('.png'))


if __name__ == '__main__':
    main(*sys.argv[1:3])
