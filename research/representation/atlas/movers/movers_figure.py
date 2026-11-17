"""Minimal figure for the movers result (descriptive). Usage: python movers_figure.py <score.json> <movers.json> <out stem> <agents_all.json>
a: drop test per decomposed feature (fraction of held-out net Δz kept after removing the top-1% carrying edges), against
   leverage-matched random drops; b: each of the confirmed movers' change by agent, in units of its mean drug change."""
import sys, json, shutil, numpy as np, matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

try:
    from mtsspi.figures import SANS, SANS_PREAMBLE
    TEX = dict(**SANS, **{'text.usetex': bool(shutil.which('latex') and shutil.which('dvipng')), 'text.latex.preamble': SANS_PREAMBLE})
except Exception:
    TEX = {}
mpl.rcParams.update({**TEX, 'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'figure.dpi': 180,
                     'savefig.dpi': 600, 'savefig.bbox': 'tight', 'figure.constrained_layout.use': True})


def family(f):
    n = f['name']
    if f['anchor']: return 'Gaussian vs model-free'
    if 'prec' in n.split('|')[0] and any(k in n for k in ('cohmag', 'plv', 'ppc')): return 'precision vs phase coherence'
    if 'pcorr' in n or 'prec_' in n: return 'direct vs marginal dependence'
    if 'chebyshev' in n: return 'extreme-sample distance'
    if ('gaussian' in n and ('kozachenko' in n or 'kraskov' in n)) or 'kernel' in n or 'kozachenko' in n: return 'Gaussian vs model-free'
    return 'other'


COL = {'direct vs marginal dependence': '#1b6ca8', 'precision vs phase coherence': '#c4502e', 'extreme-sample distance': '#7a5195',
       'Gaussian vs model-free': '#2a9d8f', 'other': '#8a8a8a'}

if __name__ == '__main__':
    s = json.load(open(sys.argv[1])); m = json.load(open(sys.argv[2])); F = sorted(s['features'], key=lambda f: f['f_hot'])
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.6, 1]))
    a = ax[0]; x = np.arange(len(F))
    a.fill_between(x, [f['f_rand'][0] for f in F], [f['f_rand'][2] for f in F], color='0.85', lw=0, step='mid', label='random drops (range)')
    for fam, c in COL.items():
        idx = [i for i, f in enumerate(F) if family(f) == fam]
        if idx: a.plot(idx, [F[i]['f_hot'] for i in idx], 'o', ms=3.2, color=c, label=fam)
    for y, ls in ((.8, ':'), (.9, '--'), (1, '-')): a.axhline(y, color='0.4', lw=.6, ls=ls)
    a.text(len(F) - .5, .8, 'localised below', ha='right', va='top', fontsize=6, color='0.3')
    a.text(len(F) - .5, .9, 'broad above', ha='right', va='bottom', fontsize=6, color='0.3')
    a.set_xticks([]); a.set_xlabel(f'decomposed movers (n = {len(F)}), sorted')
    a.set_ylabel('net $\\Delta z$ kept after dropping\nthe top-1\\% carrying edges')
    a.legend(loc='lower right', fontsize=6, handletextpad=.3, borderaxespad=.2)
    a.set_title('a  Most decomposed movers are carried broadly', loc='left', fontsize=8)
    b = ax[1]; o = json.load(open(sys.argv[4])); sg = np.array(o['sign'])  # all-window nets per agent (post hoc)
    dm = np.mean([np.array(o[g]) * sg for g in ('KT', 'KTMD', 'PF')], 0)
    eo = np.mean([np.array(o['EO_' + g]) * sg for g in ('KT', 'KTMD', 'PF')], 0)
    R = [np.array(o[g]) * sg / dm for g in ('KT', 'KTMD', 'PF', 'MD', 'SLP')] + [eo / dm]
    b.axhline(0, color='0.4', lw=.6); b.axhline(1, color='0.4', lw=.6, ls=':')
    b.boxplot(R, widths=.55, showfliers=False, medianprops=dict(color='k', lw=1.2),
              boxprops=dict(lw=.7), whiskerprops=dict(lw=.7), capprops=dict(lw=.7))
    b.set_xticks(range(1, 7), ['ketamine', 'ket.–medet.', 'propofol', 'medet.', 'sleep', 'eyes open'], rotation=30)
    b.set_ylabel('change / mean drug change')
    b.set_title(f'b  {len(sg)} movers by state (all windows)', loc='left', fontsize=8)
    for ext in ('svg', 'pdf', 'png'): fig.savefig(f'{sys.argv[3]}.{ext}')
    print('saved', sys.argv[3])
