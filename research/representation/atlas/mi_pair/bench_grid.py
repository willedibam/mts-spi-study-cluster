"""Panel (c)'s construction for every metric of bench.py (post hoc display; reads bench-score.json and bench-windows.csv.gz).
Per 10-s window, the value minus its session's eyes-closed mean, in eyes-open SD units, in its natural sign (not drug-oriented);
rainclouds per state, printed number the window mean, open markers animal means. Y-limits differ by metric (1st-99th percentile of
the windows shown, at least +-3), and windows outside are drawn at the limit. Title tags: C1 (same sign in all 10 drug cells),
r = sleep/drug ratio (C2 needs its 90% CI below .25). BSR omitted (constant 0).
  python bench_grid.py <bench results dir> <output stem>
"""
import sys, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mtsspi.figures import figure_style, raincloud
from plot_character_draft import COLOR, DRUGS, GREY, MARK, SHORT, STATES       # read only
from bench_fig import NAMES


def main(d, out):
    d = Path(d); J = json.load(open(d / 'bench-score.json')); S, sign = J['res'], J['sign']
    W = pd.read_csv(d / 'bench-windows.csv.gz'); W = W[W.group.isin(STATES)]
    M = [m for m in S if m != 'BSR']
    figure_style(); plt.rcParams.update({'font.size': 6, 'axes.labelsize': 6, 'xtick.labelsize': 5, 'ytick.labelsize': 5.4, 'svg.hashsalt': 'bench-grid'})
    fig = plt.figure(figsize=(7.2, 9.2), layout='constrained'); axs = fig.subplots(5, 4).ravel(); rng = np.random.default_rng(0)
    for a, m in zip(axs, M):
        v = W[m] * sign[m]; lo, hi = np.percentile(v, [1, 99]); lo, hi = min(lo, -3), max(hi, 3); pad = .12 * (hi - lo); yl = (lo, hi + pad)
        for i, s in enumerate(STATES):
            x = v[W.group == s].values
            raincloud(a, np.clip(x, *yl), i, COLOR[s], rng, width=.64, box=True, point_sizes=1.1, point_kw={'alpha': .3}, line_scale=.6)
            a.text(i, yl[1], f'{x.mean():+.1f}'.replace('-', '--'), ha='center', va='top', fontsize=5,
                   color=COLOR['ink'] if s in DRUGS else GREY, fontweight='bold' if s in DRUGS else 'normal')
            g = W[W.group == s].assign(v=v).groupby(['unit', 'animal']).v.mean().groupby('animal').mean()
            for an, mm in g.items():
                a.scatter(i + .02, np.clip(mm, *yl), s=6, marker=MARK[an], facecolor='white', edgecolor=COLOR['ink'], lw=.45, zorder=6)
        a.axhline(0, color=GREY, lw=.45, ls='--', zorder=0); a.set_ylim(*yl); a.set_xlim(-.6, len(STATES) - .4)
        a.set_xticks(range(len(STATES)), [SHORT[s] for s in STATES], fontsize=4.6)
        for tk, s in zip(a.get_xticklabels(), STATES):
            tk.set_color(COLOR[s])
        a.tick_params(axis='x', length=0)
        r = S[m]; tag = ('C1' if r['C1'] else 'no C1') + f", r {r['ratio']:+.2f}".replace('-', '--')
        a.set_title(f'{NAMES[m]}\\enspace\\textit{{{tag}}}' if m != 'DZ' else r'\textbf{$\Delta z$ (panel (c))}\enspace' + f'\\textit{{{tag}}}',
                    fontsize=6.2, loc='left', pad=2)
    for a in axs[len(M):]:
        a.axis('off')
    for a in axs[::4]:
        a.set_ylabel('wake SDs')
    fig.suptitle(r"Panel (c)'s construction per metric: window minus its session's eyes-closed mean, in eyes-open SDs, natural sign. "
                 r"Numbers: window means; markers: animal means; y-scales differ by metric; clipped at the axes.", x=0, ha='left', fontsize=6)
    for ext in ('png', 'pdf'):
        fig.savefig(Path(out).with_suffix('.' + ext), dpi=300, metadata={'CreationDate': None} if ext == 'pdf' else None)
    print('wrote', Path(out).with_suffix('.png'))


if __name__ == '__main__':
    main(*sys.argv[1:3])
