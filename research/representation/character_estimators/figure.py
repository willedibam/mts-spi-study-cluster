"""Figures for the construction validation: response grid and nuisance bias maps."""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from mtsspi.figures import save, style
from research.representation.character_estimators.systems import OWN, SYSTEMS

ROWS = ('nonlinearity', 'lag', 'variability', 'direction')
LABEL = dict(nonlinearity='nonlinearity $N$', nonlinearity_raw='$N$ (raw scale)', lag='lag share $\\ell$',
             variability='variability $V$', direction='directionality $D$ (aux.)')
TITLE = dict(linearity='knob: nonlinear share', lag='knob: quadrature share', lag_variability='knob: phase drift',
             intermittency='knob: off-fraction', direction='knob: one-way share')
# Okabe-Ito
COLOUR = dict(nonlinearity='#0072B2', nonlinearity_raw='#56B4E9', lag='#E69F00', variability='#009E73',
              direction='#D55E00')
ENOUGH = .8


def _band(ax, m, axis, ls='-'):
    m = m.sort_values('theta')
    ok = m.defined >= ENOUGH
    ax.fill_between(m.theta, m['mean'] - m.sd, m['mean'] + m.sd, where=ok, color=COLOUR[axis], alpha=.18, lw=0)
    ax.plot(m.theta, np.where(ok, m['mean'], np.nan), color=COLOUR[axis], ls=ls, lw=1.7)
    ax.plot(m.theta, np.where(ok, np.nan, m['mean']), color=COLOUR[axis], ls=':', lw=1)
    t = m.truth.values
    ax.plot(m.theta, t, color='k', ls='--', lw=.9, marker='o', ms=2.2)


def response(summ, out, systems=SYSTEMS, own=None, titles=TITLE, name='response', raw_with='nonlinearity'):
    own = own or dict(OWN, linearity='nonlinearity')
    base = summ[summ.condition == 'base']
    fig, ax = plt.subplots(len(ROWS) + 1, len(systems), figsize=(11, 10.5), sharex='col', squeeze=False, constrained_layout=True)
    for j, s in enumerate(systems):
        b = base[base.system == s]
        a = ax[0, j]
        g = b[b.axis == 'lag'].sort_values('theta')
        a.plot(g.theta, g.pearson.abs(), color='.35', lw=1.5, label='|Pearson r|')
        a.plot(g.theta, g.coherence, color='.65', lw=1.5, ls='--', label='mean coherence$^2$')
        a.set(ylim=(0, 1), title=f'{s.replace("_", " ")}\n{titles[s]}')
        if j == 0:
            a.set_ylabel('strength lenses'); a.legend(loc='upper right', fontsize=6.5)
        for i, axis in enumerate(ROWS, 1):
            a = ax[i, j]
            _band(a, b[b.axis == axis], axis)
            if axis == 'nonlinearity':
                _band(a, b[b.axis == 'nonlinearity_raw'], 'nonlinearity_raw', ls='-.')
            a.set(ylim=(-.15, 1.15)); a.axhline(0, color='.85', lw=.6, zorder=0); a.axhline(1, color='.85', lw=.6, zorder=0)
            if axis == (raw_with if own[s] == 'nonlinearity_raw' else own[s]):
                for sp in a.spines.values():
                    sp.set_visible(True); sp.set_linewidth(1.2)
            if j == 0:
                a.set_ylabel(LABEL[axis])
            if i == len(ROWS):
                a.set_xlabel(r'knob $\theta$')
    n = int(base.n.min())
    fig.suptitle(f'Estimates (mean $\\pm$ SD over {n} recordings, T = 2000; dotted: defined in < 80%) '
                 'against truth (black dashed); boxed: the system\'s own axis', fontsize=9)
    save(fig, out, name); plt.close(fig)


def nuisance(summ, out):
    own = dict(OWN, linearity='nonlinearity')
    rows = [(s, own[s]) for s in SYSTEMS] + [('linearity', 'nonlinearity_raw')]
    nu = summ[(summ.condition != 'base') & (summ.system != 'independent')]
    conds = list(dict.fromkeys(nu.condition))
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.6), gridspec_kw=dict(width_ratios=(1, 1, 1)), constrained_layout=True)
    for k, which in enumerate(('low', 'high')):
        M = np.full((len(rows), len(conds)), np.nan)
        for i, (s, a) in enumerate(rows):
            g = nu[(nu.system == s) & (nu.axis == a)]
            th = sorted(set(g.theta))[0 if which == 'low' else -1]
            for jj, c in enumerate(conds):
                r = g[(g.theta == th) & (g.condition == c)]
                if len(r) and r.defined.iloc[0] >= ENOUGH:
                    M[i, jj] = r.bias.iloc[0]
        ax = axs[k]
        im = ax.imshow(M, cmap='RdBu_r', vmin=-.3, vmax=.3, aspect='auto')
        for i in range(M.shape[0]):
            for jj in range(M.shape[1]):
                ax.text(jj, i, '·' if np.isnan(M[i, jj]) else f'{M[i, jj]:+.2f}'.replace('0.', '.'), ha='center',
                        va='center', fontsize=5.5, color='k' if np.isnan(M[i, jj]) or abs(M[i, jj]) < .2 else 'w')
        ax.set_xticks(range(len(conds)), conds, rotation=70, fontsize=6.5)
        ax.set_yticks(range(len(rows)), [f'{LABEL[a]} ({s.replace("_", " ")})' for s, a in rows], fontsize=7)
        ax.set_title(f'own-axis bias, {which} knob', fontsize=9)
    fig.colorbar(im, ax=axs[1], shrink=.8, label='estimate − truth')
    # worst off-axis |bias|
    axes = list(LABEL)
    M = np.full((len(axes), len(conds)), np.nan)
    for i, a in enumerate(axes):
        owners = [s for s, o in rows if o == a]
        g = nu[(nu.axis == a) & ~nu.system.isin(owners) & (nu.defined >= ENOUGH) & nu.truth.notna()]
        for jj, c in enumerate(conds):
            r = g[g.condition == c]
            if len(r):
                M[i, jj] = r.bias.abs().max()
    ax = axs[2]
    im = ax.imshow(M, cmap='Greys', vmin=0, vmax=.3, aspect='auto')
    for i in range(M.shape[0]):
        for jj in range(M.shape[1]):
            if np.isfinite(M[i, jj]):
                ax.text(jj, i, f'{M[i, jj]:.2f}'.replace('0.', '.'), ha='center', va='center', fontsize=5.5,
                        color='w' if M[i, jj] > .2 else 'k')
    ax.set_xticks(range(len(conds)), conds, rotation=70, fontsize=6.5)
    ax.set_yticks(range(len(axes)), [LABEL[a] for a in axes], fontsize=7)
    ax.set_title('worst off-axis |bias| (other systems, both knobs)', fontsize=9)
    fig.colorbar(im, ax=ax, shrink=.8, label='|estimate − truth|')
    fig.suptitle('Nuisance sweeps, T = 2000 unless stated, 20 recordings per cell; · = defined in < 80%', fontsize=9)
    save(fig, out, 'nuisance'); plt.close(fig)


def figures(summ, out):
    style()
    response(summ, out)
    nuisance(summ, out)
