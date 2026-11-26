"""Figure for bench.py's score (post hoc display of preregs/bench-prereg.md; reads bench-score.json and bench-windows.csv.gz).
a: per-agent drug effect and sleep, oriented wake SDs (session means), metrics in rows; b: sleep/drug ratio with its 90% session
bootstrap CI, filled where C1 passes (same sign in all 10 drug cells), dashed line the .25 margin; c: matched slow-wave AUC, drug
against sleep, per animal, with 90% CI; d: Δz against spectral entropy per window, Chibi and George.
  python bench_fig.py <bench results dir> <output stem>
"""
import sys, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mtsspi.figures import figure_style
from plot_character_draft import COLOR, GREY, MARK, SHORT, STATES               # read only

NAMES = {'DZ': r'$\Delta z$', 'SW': 'slow-wave share', 'ALPHA': 'rel. alpha', 'BR': 'beta ratio (BIS)', 'SEF95': 'SEF95',
         'SE': 'spectral entropy', 'AP': 'aperiodic slope', 'LZC': 'Lempel-Ziv', 'PE': 'perm. entropy', 'ABSR': r'mean $|r|$',
         'EIG': 'top eigenvalue', 'COHD': 'coherence 1-4', 'COHA': 'coherence 8-13', 'WPLID': 'wPLI 1-4', 'WPLIA': 'wPLI 8-13',
         'KURA': 'Kuramoto 8-13', 'WSMI': 'wSMI', 'MG': 'mean Gaussian MI', 'MK': 'mean KSG MI', 'MKG': 'mean KSG - Gaussian'}


def main(d, out):
    d = Path(d); S = json.load(open(d / 'bench-score.json'))['res']; W = pd.read_csv(d / 'bench-windows.csv.gz')
    M = [m for m in S if m != 'BSR']                         # BSR: no window below the threshold, constant 0
    figure_style(); plt.rcParams.update({'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.3, 'ytick.labelsize': 6.3, 'svg.hashsalt': 'bench'})
    fig = plt.figure(figsize=(7.2, 4.4), layout='constrained'); ax = fig.subplots(1, 4, width_ratios=(1.35, 1, 1, 1.25))
    y = np.arange(len(M))[::-1]; cols = ('KT', 'KTMD', 'MD', 'PF')
    H = np.array([[S[m]['by_agent'][g] for g in cols] + [S[m]['sleep']] for m in M])
    a = ax[0]; im = a.imshow(np.clip(H, -6, 6), cmap='RdBu_r', vmin=-6, vmax=6, aspect='auto', extent=(-.5, 4.5, -.5, len(M) - .5))
    for i, m in enumerate(M):
        for j in range(5):
            a.text(j, y[i], f'{H[i, j]:+.1f}'.replace('-', '--'), ha='center', va='center', fontsize=4.8, color='k')
    a.set_yticks(y, [NAMES[m] for m in M], fontsize=6); a.set_xticks(range(5), [SHORT[g] for g in cols + ('SLP',)], fontsize=5.4)
    for tk, g in zip(a.get_xticklabels(), cols + ('SLP',)):
        tk.set_color(COLOR[g])
    a.tick_params(length=0); a.set_title(r'\textbf{a}\enspace Effect, drug direction (wake SDs)', fontsize=6.8, loc='left', pad=3)
    b = ax[1]
    for i, m in enumerate(M):
        r = S[m]; lo, hi = r['ratio_ci90']; ok = r['C1']; c = COLOR['ink'] if ok else GREY
        b.plot([max(lo, -1), min(hi, 1.5)], [y[i]] * 2, color=c, lw=.9)
        b.scatter(np.clip(r['ratio'], -1, 1.5), y[i], s=14, facecolor=c if ok else 'white', edgecolor=c, lw=.7, zorder=3)
    b.axvline(0, color=GREY, lw=.5, ls='--'); b.axvline(.25, color=COLOR['KT'], lw=.6, ls=':'); b.set_xlim(-1, 1.5)
    b.set_yticks(y, []); b.set_ylim(-.5, len(M) - .5); b.set_xlabel('sleep / drug')
    b.set_title(r'\textbf{b}\enspace Sleep moved toward drug', fontsize=6.8, loc='left', pad=3)
    c = ax[2]
    for k, (an, off) in enumerate((('Chibi', .15), ('George', -.15))):
        for i, m in enumerate(M):
            r = S[m][an]; lo, hi = r['matched_ci90']
            c.plot([lo, hi], [y[i] + off] * 2, color=COLOR['ink'], lw=.6, alpha=.6)
            c.scatter(r['matched'], y[i] + off, s=10, marker=MARK[an], facecolor='white', edgecolor=COLOR['ink'], lw=.6, zorder=3, label=an if i == 0 else None)
    c.axvline(.5, color=GREY, lw=.5, ls='--'); c.set_xlim(0, 1); c.set_yticks(y, []); c.set_ylim(-.5, len(M) - .5)
    c.set_xlabel('AUC, drug vs sleep'); c.legend(fontsize=5.4, loc='lower left', handletextpad=.1)
    c.set_title(r'\textbf{c}\enspace At matched slow-wave', fontsize=6.8, loc='left', pad=3)
    e = ax[3]; P = W[W.animal.isin(('Chibi', 'George')) & W.group.isin(STATES)]
    for s in STATES:
        w = P[P.group == s]; e.scatter(w.SE, np.clip(w.DZ, -4, 15), s=2.5, color=COLOR[s], alpha=.4, lw=0)
        for an, mm in w.groupby('animal')[['SE', 'DZ']].mean().iterrows():
            e.scatter(mm.SE, mm.DZ, s=20, marker=MARK[an], facecolor=COLOR[s], edgecolor='white', lw=.6, zorder=5)
    e.axhline(0, color=GREY, lw=.5, ls='--'); e.axvline(0, color=GREY, lw=.5, ls='--'); e.set_ylim(-4.5, 15)
    e.set_xlabel('spectral entropy, drug direction'); e.set_ylabel(r'$\Delta z$, drug direction')
    e.set_title(r'\textbf{d}\enspace Two axes, per window', fontsize=6.8, loc='left', pad=3)
    fig.suptitle("Window minus its session's eyes-closed mean, in eyes-open SDs, signed so that drug is positive. (b) filled: same sign in all 10 "
                 "drug cells (C1); dotted: .25 margin (C2). \n(c) oriented to the drug direction, so near 0 also separates (reversed); 90\\% bootstrap CIs; (d) large markers animal-by-state means.", x=0, ha='left', fontsize=6)
    for ext in ('png', 'pdf'):
        fig.savefig(Path(out).with_suffix('.' + ext), dpi=300, metadata={'CreationDate': None} if ext == 'pdf' else None)
    print('wrote', Path(out).with_suffix('.png'))


if __name__ == '__main__':
    main(*sys.argv[1:3])
