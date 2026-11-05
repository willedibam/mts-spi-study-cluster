"""Composite figure, one row per axis: the test system, what the estimator computes, calibration and selectivity.

    python -m research.estimators.composite [--out results/estimators/composite]

Calibration and selectivity are read from the scored tables (tables/, tables/transfer/, tables/factorial/); the
first two columns are computed here from fresh recordings at T = 2^14.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

from mtsspi.figures import save, style
from research.estimators import estimators as E
from research.estimators.figure import COLOUR, ENOUGH
from research.estimators.systems import make, make_transfer

HERE = Path(__file__).parent
T_DEMO = 2 ** 14
NAME = dict(nonlinearity='nonlinearity $N$', lag=r'lag share $\ell$', variability='variability $V$',
            direction='directionality $D$ (auxiliary)')
# own systems: (family, system, label)
OWNERS = dict(
    nonlinearity=(('first', 'linearity', 'He$_2$ mixture'), ('transfer', 'rectifier', 'rectifier')),
    lag=(('first', 'lag', 'Hilbert phase'), ('transfer', 'delay', 'delay')),
    variability=(('first', 'lag_variability', 'phase drift'), ('first', 'intermittency', 'on/off'),
                 ('transfer', 'warp', 'delay jitter'), ('transfer', 'telegraph', 'telegraph')),
    direction=(('first', 'direction', 'linear VAR'), ('transfer', 'direction_nl', 'tanh VAR')))
STYLE = [dict(ls='-', marker='o'), dict(ls='-', marker='s'), dict(ls='--', marker='^'), dict(ls='--', marker='v')]
GREY = '.6'


def _tables():
    first = pd.read_csv(HERE / 'tables' / 'summary.csv').assign(family='first')
    transfer = pd.read_csv(HERE / 'tables' / 'transfer' / 'summary.csv').assign(family='transfer')
    return pd.concat([first, transfer], ignore_index=True), pd.read_csv(HERE / 'tables' / 'factorial' / 'summary.csv')


def _prep(x, y):
    g = [E.whiten(E.circular(E.gaussianise(v))) for v in (x, y)]
    return g


# ---- column 1: systems -------------------------------------------------------------------------------

def system_nonlinearity(ax, rng):
    for th, c, lab in ((0., GREY, r'$\theta = 0$'), (.8, COLOUR['nonlinearity'], r'$\theta = 0.8$')):
        x, y = make('linearity', th, 600, rng, strength=.25)
        ax.plot(x, y, 'o', ms=1.6, color=c, alpha=.7, mew=0, label=lab)
    ax.set(xlabel='$x_t$', ylabel='$y_t$', xlim=(-3.2, 3.2))
    ax.legend(loc='upper left', fontsize=6.5, handletextpad=.1, markerscale=2.5)
    ax.set_title(r'$y = \sqrt{1-\theta}\,x + \sqrt{\theta}\,\mathrm{He}_2(x) + \sigma\eta$', fontsize=8)


def system_lag(ax, rng):
    x, y = make('lag', .8, 2000, rng, strength=0., spec='narrow')
    t = np.arange(140)
    ax.plot(t, x[500:640], color=GREY, lw=1.2, label='$x$')
    ax.plot(t, y[500:640], color=COLOUR['lag'], lw=1.2, label=r'$y$ ($\theta = 0.8$, $\sigma = 0$)')
    ax.set(xlabel='sample', yticks=[])
    ax.legend(loc='upper right', fontsize=6.5, ncol=2, handlelength=1.2)
    ax.set_title(r'$y = \cos\varphi\,x + \sin\varphi\,\mathcal{H}[x] + \sigma\eta$, $\sin^2\varphi = \theta$', fontsize=8)


def system_variability(ax, rng):
    k = 20
    s = np.sqrt(-np.log1p(-.6))
    phi = np.pi / 4 + s * rng.standard_normal(k)
    on = np.zeros(k, bool); on[rng.choice(k, 8, replace=False)] = True
    seg = np.arange(k + 1)
    ax.stairs(phi / np.pi, seg, baseline=None, color=COLOUR['variability'], lw=1.4, label=r'phase drift: $\varphi_k/\pi$')
    ax.stairs(np.where(on, 1., 0.) - 1.6, seg, baseline=None, color='#00684A', lw=1.4, label='on/off: gain $a_k$ (offset)')
    ax.axhline(.25, color=GREY, lw=.6, ls=':')
    ax.set(xlabel=f'segment $k$ ($W = {E.W}$ samples)', yticks=[], ylim=(-1.9, 1.3))
    ax.legend(loc='upper right', fontsize=6.5, handlelength=1.2)
    ax.set_title(r'coupling changes per segment, $\theta = 0.6$', fontsize=8)


def system_direction(ax, rng):
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    for xy, lab in (((.2, .5), '$x$'), ((.8, .5), '$y$')):
        ax.add_patch(plt.Circle(xy, .09, fc='white', ec='k', lw=1))
        ax.text(*xy, lab, ha='center', va='center', fontsize=10)
    for (a, b, rad, w, lab) in (((.3, .55), (.7, .55), -.35, 3.2, r'$c\,(1+\theta)/2$'), ((.7, .45), (.3, .45), -.35, 1., r'$c\,(1-\theta)/2$')):
        ax.add_patch(FancyArrowPatch(a, b, connectionstyle=f'arc3,rad={rad}', arrowstyle='-|>', mutation_scale=10,
                                     lw=w, color=COLOUR['direction']))
        ax.text(.5, .74 if a[0] < b[0] else .24, lab, ha='center', fontsize=8)
    ax.set_title('bivariate VAR, own AR(1) 0.5, lag-2 cross coupling', fontsize=8)


# ---- column 2: construction --------------------------------------------------------------------------

def construction_nonlinearity(ax, rng):
    ths = (0., .4, .8)
    vals = []
    for th in ths:
        x, y = make('linearity', th, T_DEMO, rng)
        xg, yg = E.gaussianise(x), E.gaussianise(y)
        f = E._fits(xg, yg)
        r = 1 - (1 - f) / (1 - E._analytic_null(len(x)))
        vals.append((max(r[0], 0), r[1]))
    vals = np.array(vals)
    i = np.arange(len(ths))
    ax.bar(i - .18, vals[:, 0], .34, color=GREY, label=r'linear window $R^2_{\mathrm{lin}}$')
    ax.bar(i + .18, vals[:, 1], .34, color=COLOUR['nonlinearity'], label=r'+ Volterra terms $R^2_{\mathrm{full}}$')
    for j, (l, fu) in enumerate(vals):
        ax.text(j, fu + .01, f'$N = {1 - l / fu:.2f}$', ha='center', fontsize=7)
    ax.set_xticks(i, [rf'$\theta = {t:g}$' for t in ths])
    ax.set(ylabel='out-of-sample $R^2$ (cost-corrected)', ylim=(0, max(vals[:, 1]) * 1.3))
    ax.legend(loc='upper left', fontsize=6.5)
    ax.set_title(r'$N = 1 - R^2_{\mathrm{lin}} / R^2_{\mathrm{full}}$, cross-validated', fontsize=8)


def _unit(ax, r=1.):
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(r * np.cos(t), r * np.sin(t), color='.85', lw=.6, zorder=0)
    ax.axhline(0, color='.85', lw=.6, zorder=0); ax.axvline(0, color='.85', lw=.6, zorder=0)
    ax.set_aspect('equal')


def construction_lag(ax, rng):
    cmap = plt.get_cmap('YlOrBr')
    for th, c in ((0., GREY), (.4, cmap(.35)), (.8, COLOUR['lag'])):
        x, y = make('lag', th, T_DEMO, rng)
        xw, yw = _prep(x, y)
        C = E._coherency(*E._tapered(xw, yw, E.W))[:, 0].mean(0)
        lag, _ = E._lag(E._coherency(*E._tapered(xw, yw, E.W))[:, 0])
        ax.plot(C.real, C.imag, 'o', ms=2.2, color=c, label=rf'$\theta = {th:g}$: $\ell = {lag:.2f}$')
    _unit(ax, .8)
    ax.set(xlim=(-.15, .85), ylim=(-.15, .85), xlabel=r'Re $C(f)$', ylabel=r'Im $C(f)$')
    ax.legend(loc='upper right', fontsize=6.5, handletextpad=.1)
    ax.set_title(r'$\ell = \sum_f \mathrm{Im}\,C(f)^2 / \sum_f |C(f)|^2$, mean coherency', fontsize=8)


def construction_variability(ax, rng):
    # stronger coupling than the scored systems, so each segment's point is visible against its own noise
    for (sys_, th, kw, c, m, lab) in (('lag_variability', 0., dict(strength=.3), GREY, 'o', 'steady'),
                                      ('lag_variability', .6, dict(strength=.3), COLOUR['variability'], 'o', r'phase drift $0.6$'),
                                      ('intermittency', .6, dict(strength=.6), '#00684A', 's', r'on/off $0.6$')):
        x, y = make(sys_, th, T_DEMO, rng, **kw)
        xw, yw = _prep(x, y)
        d = E._delay(xw, yw)
        c4 = E._coherency(*E._tapered(xw, np.roll(yw, -d), E.W, 4), local=True)
        v, _ = E._local(c4)
        ck = c4.mean((1, 2))  # one point per segment: local coherency averaged over quarters and frequencies
        if sys_ == 'lag_variability':
            ck = ck.conj()
        ax.plot(ck.real, ck.imag, m, ms=2, color=c, alpha=.75, mew=0, label=f'{lab}: $V = {v:.2f}$')
    _unit(ax, 1.)
    ax.set(xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), xlabel=r'Re $c_k$', ylabel=r'Im $c_k$')
    ax.legend(loc='upper left', fontsize=6.5, handletextpad=.1, markerscale=2)
    ax.set_title(r'$V = 1 - \sum_f |\mathrm{E}\,c|^2 / \sum_f \mathrm{E}\,|c|^2$, one point per segment', fontsize=8)


def construction_direction(ax, rng):
    from research.estimators.systems import _var
    labels = (r'$G_{x\to y}$', r'$G_{y\to x}$', r'$\tilde G_{x\to y}$', r'$\tilde G_{y\to x}$')
    for j, (th, c) in enumerate(((0., GREY), (.75, COLOUR['direction']))):
        x, y = _var(th, T_DEMO, rng, .6)
        x, y = E.gaussianise(x), E.gaussianise(y)
        g = np.clip([E._granger(x, y), E._granger(y, x), E._granger(x[::-1], y[::-1]), E._granger(y[::-1], x[::-1])], 0, None)
        dval = abs((g[0] - g[1]) - (g[2] - g[3])) / g.sum()
        ax.bar(np.arange(4) + (j - .5) * .38, g, .36, color=c, label=rf'$\theta = {th:g}$: $D = {dval:.2f}$')
    ax.set_xticks(range(4), labels)
    ax.set(ylabel='linear Granger (log RSS ratio)', ylim=(0, ax.get_ylim()[1] * 1.25))
    ax.legend(loc='upper center', fontsize=6.5)
    ax.set_title(r'$D = |(G_{xy} - G_{yx}) - (\tilde G_{xy} - \tilde G_{yx})| / \sum G$, $\sim$: time reversed', fontsize=8)


# ---- columns 3 and 4: calibration and selectivity ------------------------------------------------------

def calibration(ax, summ, axis):
    for (fam, s, lab), st in zip(OWNERS[axis], STYLE):
        m = summ[(summ.family == fam) & (summ.system == s) & (summ.axis == axis) & (summ.condition == 'base')].sort_values('theta')
        ok = m.defined >= ENOUGH
        ax.plot(m.theta, m.truth, color='k', lw=.8, ls=st['ls'], alpha=.55)
        ax.errorbar(m.theta[ok], m['mean'][ok], m.sd[ok], color=COLOUR[axis], ls=st['ls'], marker=st['marker'], ms=3,
                    lw=1.3, capsize=1.5, mfc='white' if fam == 'transfer' else COLOUR[axis], label=f'{lab} ({fam})')
        if (~ok).any():
            ax.plot(m.theta[~ok], m['mean'][~ok], st['marker'], ms=3, mfc='none', color=COLOUR[axis], alpha=.5)
    ax.set(xlim=(-.05, 1.05), ylim=(-.2, 1.15), xlabel=r'knob $\theta$', ylabel='estimate')
    ax.axhline(0, color='.88', lw=.6, zorder=0); ax.axhline(1, color='.88', lw=.6, zorder=0)
    ax.legend(loc='upper left', fontsize=6, handlelength=2.2)
    ax.set_title('mean $\\pm$ SD, 40 recordings; black: truth', fontsize=8)


def selectivity(ax, summ, fact, axis):
    owners = {s for _, s, _ in OWNERS[axis]}
    rows = []
    for fam in ('first', 'transfer'):
        m = summ[(summ.family == fam) & (summ.axis == axis) & (summ.condition == 'base') & ~summ.system.isin(owners | {'independent'})
                 & (summ.defined >= ENOUGH) & summ.truth.notna()]
        for s, g in m.groupby('system', sort=False):
            rows.append((s.replace('_', ' '), fam, float(g.bias.abs().max())))
    f = fact[(fact.axis == axis) & (fact.defined >= ENOUGH)]
    if len(f):
        rows.append(('factorial (16 crossed cells)', 'factorial', float((f['mean'] - f['long']).abs().max())))
    y = np.arange(len(rows))[::-1]
    for yy, (lab, fam, v) in zip(y, rows):
        mk = dict(first=dict(marker='o', mfc=COLOUR[axis]), transfer=dict(marker='o', mfc='white'),
                  factorial=dict(marker='*', mfc=COLOUR[axis], ms=7))[fam]
        ax.plot([0, v], [yy, yy], color='.85', lw=.8, zorder=0)
        ax.plot(v, yy, ls='none', color=COLOUR[axis], **{'ms': 4, **mk})
    ax.axvspan(0, .10, color='.94', zorder=-1)
    ax.axvline(.10, color='.6', lw=.7, ls='--')
    ax.set_yticks(y, [r[0] for r in rows], fontsize=6.5)
    ax.set(xlim=(0, max(.3, max(r[2] for r in rows) * 1.08)), xlabel=r'worst $|$estimate $-$ truth$|$')
    ax.set_title('other characters present (shaded: tolerance)', fontsize=8)


SYSTEM = dict(nonlinearity=system_nonlinearity, lag=system_lag, variability=system_variability, direction=system_direction)
CONSTRUCTION = dict(nonlinearity=construction_nonlinearity, lag=construction_lag, variability=construction_variability,
                    direction=construction_direction)


def main(out):
    style()
    summ, fact = _tables()
    axes = tuple(NAME)
    fig, axs = plt.subplots(len(axes), 4, figsize=(13, 12.5), gridspec_kw=dict(width_ratios=(1.05, 1, 1, .9)),
                            constrained_layout=True)
    rng = np.random.default_rng(20261106)
    for i, a in enumerate(axes):
        SYSTEM[a](axs[i, 0], rng)
        CONSTRUCTION[a](axs[i, 1], rng)
        calibration(axs[i, 2], summ, a)
        selectivity(axs[i, 3], summ, fact, a)
        axs[i, 0].annotate(NAME[a], xy=(-.32, .5), xycoords='axes fraction', rotation=90, ha='center', va='center',
                           fontsize=10, color=COLOUR[a], weight='bold')
    for j, t in enumerate(('test system (knob $\\theta$)', 'what the estimator computes ($T = 2^{14}$)',
                           'calibration at $T = 2000$', 'selectivity: off-axis readings')):
        axs[0, j].annotate(t, xy=(.5, 1.2), xycoords='axes fraction', ha='center', fontsize=9.5)
    save(fig, out, 'composite'); plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=Path('results/estimators/composite'))
    main(ap.parse_args().out)
