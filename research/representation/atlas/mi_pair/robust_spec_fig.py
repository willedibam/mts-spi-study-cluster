"""Specification curve for robustness-result.md: drug - awake EC Δz (normal-scored z real - IAAFT) per configuration,
animal means and their mean, grouped by axis, with the drug-blind stationary offsets where calibrated. Reads only the
text outputs in results/.../robust/ (score10/scoreS2/scoreC2.txt, precise.txt, lowrate_summary.txt,
bands_summary.txt); the 10-s kshift row is ksg-bias-result.md's (animal means rounded there to 3 dp).

  python robust_spec_fig.py <robust results dir> <stem>        writes <stem>.svg/.pdf/.png
"""
import re
import shutil
import sys
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mtsspi.figures import SANS, SANS_PREAMBLE

mpl.rcParams.update({
    "text.usetex": bool(shutil.which("latex") and shutil.which("dvipng")),
    "text.latex.preamble": SANS_PREAMBLE, **SANS,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.top": False, "ytick.right": False,
    "axes.grid": False, "legend.frameon": False,
    "lines.linewidth": 1.7, "lines.markersize": 2.7,
    "figure.constrained_layout.use": True,
    "figure.dpi": 180, "savefig.dpi": 600, "savefig.bbox": "tight",
})
INK = '#2B2B2B'
VCOL = {'SURVIVES': INK, 'PARTIAL': '#D9A21B', 'LOST': '#C8553D'}
NUM = r'([+-]?\d*\.\d+|[+-]?nan)'
RB = ['w0 k4', 'w0 k10', 'auto k4', 'auto k4 pool', 'auto k10', 'kshift', '2te k4', '2te k4 pool', '2te k10',
      '4auto k4', '4auto k4 pool', '4auto k10']
LAB = {'w0 k4': 'w0', 'w0 k10': 'w0 k10', 'auto k4': 'AUTO', 'auto k4 pool': 'AUTO pool', 'auto k10': 'AUTO k10',
       'kshift': 'AUTO kshift', '2te k4': r'2$\tau$', '2te k4 pool': r'2$\tau$ pool', '2te k10': r'2$\tau$ k10',
       '4auto k4': '4AUTO', '4auto k4 pool': '4AUTO pool', '4auto k10': '4AUTO k10'}
LOW = [('L20', '0.5-8 Hz @100'), ('L10', '0.5-4 Hz @100'), ('LP2', '0.5-2 Hz @100'), ('HP2', '2-40 Hz @100'), ('S20', '20 Hz sub'), ('D20', '20 Hz dec'),
       ('S10', '10 Hz sub'), ('D10', '10 Hz dec'), ('S20L', '50 s @20 sub'), ('D20L', '50 s @20 dec'),
       ('S10L', '100 s @10 sub'), ('D10L', '100 s @10 dec')]
YLIM = (-.063, .024)
KSHIFT10 = dict(a=np.array([-.022, -.010, -.012, -.013]), t=-5.73, p=.0053, ag=np.array([-.0154, -.0176, -.0064, -.0148]),
                R=1.11)


def verdict(r):
    """ksgbias rule: SURVIVES 4/4, p < .05, every agent < 0, R >= .5; LOST R < .25 or <= 2/4 animals."""
    n = int((r['a'] < 0).sum())
    if r['R'] < .25 or n <= 2:
        return 'LOST'
    return 'SURVIVES' if n == 4 and r['p'] < .05 and (r['ag'] < 0).all() and r['R'] >= .5 else 'PARTIAL'


def rb(path):
    out, lines = {}, Path(path).read_text().splitlines()
    for i, l in enumerate(lines):
        m = re.match(r'^([\w ]+): animal means \[([^\]]+)\]', l)
        if not m:
            continue
        r = dict(a=np.array(m.group(2).split(), float),
                 t=float(re.search(r't\(3\) = ' + NUM + ', p', l).group(1)), p=float(re.search(r'p = ' + NUM, l).group(1)),
                 R=float(re.search(r'R ' + NUM, l).group(1)),
                 ag=np.array([float(re.search(r'\b' + g + ' ' + NUM, lines[i + 1]).group(1)) for g in ('KT', 'KTMD', 'MD', 'PF')]))
        out[m.group(1)] = r
    return out


def low(path):
    out = {}
    for l in Path(path).read_text().splitlines():
        m = re.match(r'^(\w+)\s+ns\s', l)
        if not m:
            continue
        a = np.array([float(x) for x in re.findall(r'(?:Chibi|George|Kin2|Su) ' + NUM, l)])
        out[m.group(1)] = dict(a=a, t=float(re.search(r't\(3\) ' + NUM, l).group(1)), p=float(re.search(r'p ' + NUM, l).group(1)),
                               ag=np.array([float(re.search(r'\b' + g + ' ' + NUM, l).group(1)) for g in ('KT', 'KTMD', 'MD', 'PF')]))
    ref = out['100Hz']['a'].mean()
    for r in out.values():
        r['R'] = r['a'].mean() / ref
    return out


def calib(path):
    out = {}
    for l in Path(path).read_text().splitlines():
        m = re.match(r'^calib T(\d+) ([\w ]+): offset ' + NUM + ' se ' + NUM, l)
        if m:
            out[(int(m.group(1)), m.group(2))] = (float(m.group(3)), float(m.group(4)))
    return out


def figure(d, stem):
    d = Path(d); C = calib(d / 'precise.txt'); L = low(d / 'lowrate_summary.txt') | low(d / 'bands_summary.txt')
    s10 = rb(d / 'score10.txt'); s10['kshift'] = KSHIFT10
    groups = [('10 s, 100 Hz', [(k, LAB[k], s10[k], C.get((1000, k))) for k in RB]),
              ('2 s, own bank', [(k, LAB[k], r, C.get((200, k))) for k in RB for r in [rb(d / 'scoreS2.txt')[k]]]),
              ('2 s, cut from 10 s', [(k, LAB[k], r, C.get((200, k))) for k in RB for r in [rb(d / 'scoreC2.txt')[k]]]),
              ('rate, band\n10 s', [(k, l, L[k], None) for k, l in LOW[:8]]),
              ('N = 1000\n50-100 s', [(k, l, L[k], None) for k, l in LOW[8:]])]
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    x, ticks, labs, rng = 0, [], [], np.random.default_rng(0)
    ref = s10['auto k4']['a'].mean()
    for gi, (gname, rows) in enumerate(groups):
        x0 = x
        for k, lab, r, cal in rows:
            v = verdict(r); m = r['a'].mean()
            ax.scatter(x + rng.uniform(-.18, .18, 4), r['a'], s=5, color='#B9BFC3', lw=0, zorder=2)
            ax.plot(x, m, 'o', ms=4.2, color=VCOL[v], zorder=3, mfc=VCOL[v] if v == 'SURVIVES' else 'white', mew=1.1)
            assert YLIM[0] < r['a'].min() and r['a'].max() < YLIM[1] - .006, (gname, k)  # no clipped animal, clear of labels
            if cal is not None:  # the -2 SE bar is the frozen 2-s rule; at 10 s no rule was frozen, so diamond only
                ax.errorbar(x + .32, cal[0], yerr=[[2 * cal[1] if gname.startswith('2 s') else 0], [0]], fmt='D', ms=2.6, mfc='white', color='#6B7378', lw=.7,
                            capsize=1.5, mew=.7, zorder=2)
            ticks.append(x); labs.append(lab); x += 1
        ax.text((x0 + x - 1) / 2, YLIM[1] - .005, gname, ha='center', va='bottom', fontsize=7.5, linespacing=1.1)
        if gi < len(groups) - 1:
            ax.axvline(x, color='#D9DDE0', lw=.6, zorder=0)
        x += 1
    ax.axhline(0, color=INK, lw=.5); ax.axhline(ref, color=INK, lw=.5, ls=(0, (2, 2)))
    ax.text(x - 1.2, ref - .0012, 'anchor', fontsize=6.5, ha='right', va='top')
    ax.set_xticks(ticks); ax.set_xticklabels(labs, rotation=70, fontsize=6.5, ha='right', rotation_mode='anchor')
    ax.set(xlim=(-1, x - 1), ylim=YLIM, ylabel=r'drug $-$ awake EC $\Delta z$')
    h = [plt.Line2D([], [], marker='o', ls='', ms=4.2, color=VCOL['SURVIVES'], label='mean of animals: SURVIVES'),
         plt.Line2D([], [], marker='o', ls='', ms=4.2, color=VCOL['PARTIAL'], mfc='white', mew=1.1, label='PARTIAL'),
         plt.Line2D([], [], marker='o', ls='', ms=4.2, color=VCOL['LOST'], mfc='white', mew=1.1, label='LOST'),
         plt.Line2D([], [], marker='o', ls='', ms=2.5, color='#B9BFC3', label='animal means (4)'),
         plt.Line2D([], [], marker='D', ls='', ms=2.6, mfc='white', color='#6B7378',
                    label=r'stationary offset, synthetic (2 s bar: rule threshold)')]
    fig.legend(handles=h, ncol=5, fontsize=7, loc='outside lower center', handletextpad=.3, columnspacing=1.2)
    for ext in ('svg', 'pdf', 'png'):
        fig.savefig(f'{stem}.{ext}', dpi=180 if ext == 'png' else None)


if __name__ == '__main__':
    figure(sys.argv[1], sys.argv[2])
