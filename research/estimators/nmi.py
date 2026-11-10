"""N_MI = 1 - I_Gauss / I_KSG on both families of test systems (README.md, 'N_MI').

    python -m research.estimators.nmi run --out <dir> [--offset 0] [--workers 8]
    python -m research.estimators.nmi analyse --out <dir>

Same systems, nuisances, replicates and null as validate.py and transfer.py; only ``nonlinearity_mi`` is computed,
so it runs locally in minutes. Truth is 0 where the pair is linear-Gaussian, a numerical integral where y given x
is a Gaussian mixture with a known mean (``population``), and otherwise the long run (T = 2^16, 3 seeds, no
shifts).
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
import os
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
import pandas as pd
from scipy import stats

from research.estimators import transfer as TR, validate as V
from research.estimators.estimators import nonlinearity_mi
from research.estimators.systems import make, make_transfer

HERE = Path(__file__).parent
MAKE = dict(main=make, transfer=make_transfer)
LONG_T, LONG_SEEDS = 2 ** 16, 3
LINEAR = {('main', 'lag'), ('main', 'direction'), ('transfer', 'delay')}   # jointly Gaussian at every setting
BEND = dict(linearity=lambda x: (x ** 2 - 1) / np.sqrt(2),
            rectifier=lambda x: (np.abs(x) - np.sqrt(2 / np.pi)) / np.sqrt(1 - 2 / np.pi))
SWITCH = dict(intermittency=.45, telegraph=.4)   # default Pearson r (systems.py)
RESHAPES = ('beta', 'ampmod')    # change the pair's distribution in every system: truth from the long run
CHANGES_X = ('mix', 'xnoise')    # change the regressor: no closed form outside the linear-Gaussian systems
TOL, ENOUGH = .10, .8


def population(parts, lim=6., nx=801, ny=4001):
    """N_MI of (x, y) with x ~ N(0, 1) and y given x a mixture of N(mean_i(x), sd_i^2) with weights w_i
    (``parts``: (w_i, mean_i, sd_i)), by integration on a grid."""
    x = np.linspace(-lim, lim, nx)
    px = stats.norm.pdf(x); px /= px.sum()
    means = [m(x) for _, m, _ in parts]
    y = np.linspace(min(m.min() - lim * p[2] for m, p in zip(means, parts)),
                    max(m.max() + lim * p[2] for m, p in zip(means, parts)), ny)
    c = sum(w * stats.norm.pdf((y[None] - m[:, None]) / sd) / sd for (w, _, sd), m in zip(parts, means))
    c /= c.sum(1, keepdims=True)                      # p(y | x) as weights on the y grid
    py = px @ c
    info = (px[:, None] * c * np.log(np.where(c > 0, c / py[None], 1.))).sum()
    ey = py @ y
    r2 = (px * x * (c @ y)).sum() ** 2 / ((px @ x ** 2) * (py @ y ** 2 - ey ** 2))
    return float(1 + .5 * np.log1p(-r2) / info)


def truth(family, s, th, label, kw):
    """Population N_MI of a condition, or None where it comes from the long run."""
    kind = label.split('=')[0]
    if kind in RESHAPES:
        return None
    if (family, s) in LINEAR:
        return 0.
    if kind in CHANGES_X:
        return None
    strength = kw.get('strength')
    if s in BEND:
        if th == 0:
            return 0.
        return population([(1., lambda x: np.sqrt(1 - th) * x + np.sqrt(th) * BEND[s](x), strength or 1.)])
    if s in SWITCH:
        if th == 0:
            return 0.
        r, q = strength or SWITCH[s], 1 - th
        sd = np.sqrt(1 - r ** 2 / q)
        return population([(q, lambda x: r / q * x, sd), (1 - q, lambda x: 0 * x, sd)])
    return None


def conditions():
    """(family, system, theta, label, make kwargs, T, reps), from the two scored protocols."""
    return [('main',) + c for c in V.conditions()] + [('transfer',) + c for c in TR.conditions()]


def _one(task):
    fam, s, th, label, kw, T, rep, offset = task
    rng = np.random.default_rng([V._seed('nmi', fam, s, th, label, rep), offset])
    x, y = MAKE[fam](s, th, T, rng, **kw)
    return dict(family=fam, system=s, theta=th, condition=label, rep=rep, T=T, pearson=float(np.corrcoef(x, y)[0, 1]),
                **nonlinearity_mi(x, y, rng))


def _long(task):
    fam, s, th, label, kw, offset = task
    v = []
    for seed in range(LONG_SEEDS):
        rng = np.random.default_rng([V._seed('nmi-long', fam, s, th, label, seed), offset])
        v.append(nonlinearity_mi(*MAKE[fam](s, th, LONG_T, rng, **kw), rng, shifts=0)['nonlinearity_mi'])
    return float(np.mean(v))


def run(out, offset, workers, reps=None):
    out.mkdir(parents=True, exist_ok=True)
    conds = conditions()
    recs, long = [], []
    for fam, s, th, label, kw, T, n in conds:
        if s == 'independent':
            continue
        t = truth(fam, s, th, label, kw)
        recs.append(dict(family=fam, system=s, theta=th, condition=label, truth=t,
                         source='long run' if t is None else 'construction' if t == 0 else 'integral'))
        if t is None:
            long.append((fam, s, th, label, kw, offset))
    tasks = [(fam, s, th, lab, kw, T, r, offset) for fam, s, th, lab, kw, T, n in conds for r in range(min(n, reps or n))]
    with Pool(workers) as pool:
        values = iter(pool.map(_long, long, chunksize=1))
        for r in recs:
            if r['truth'] is None:
                r['truth'] = next(values)
        pd.DataFrame(recs).to_csv(out / 'truth.csv', index=False)
        pd.DataFrame(pool.map(_one, tasks, chunksize=4)).to_csv(out / 'rows.csv', index=False)
    (out / 'run.json').write_text(json.dumps(dict(offset=offset, recordings=len(tasks), conditions=len(conds)), indent=1))


def summarise(rows, truth):
    key = ['family', 'system', 'theta', 'condition']
    g = rows.groupby(key)
    summ = pd.DataFrame(dict(n=g.size(), defined=g.nonlinearity_mi_defined.mean(), mean=g.nonlinearity_mi.mean(),
                             sd=g.nonlinearity_mi.std(), mi=g.mi.mean(), pearson=g.pearson.mean())).reset_index()
    summ = summ.merge(truth, on=key, how='left')
    summ['bias'] = summ['mean'] - summ['truth']
    return summ


def verdicts(summ):
    """Pre-registered predictions P1-P6 (README.md). A cell counts where N_MI is defined in at least 80% of recordings."""
    ok, base = summ[summ.defined >= ENOUGH], summ.condition == 'base'
    kind = summ.condition.str.split('=').str[0]
    linear = pd.Series([(f, s) in LINEAR for f, s in zip(summ.family, summ.system)], index=summ.index)
    own = summ.system.isin(list(BEND) + list(SWITCH))
    out = []

    def add(name, cells, passed, value, note=''):
        w = cells.loc[value.abs().idxmax()] if len(value) else None
        out.append(dict(prediction=name, passed=bool(passed), cells=len(cells), counted=int((cells.defined >= ENOUGH).sum()),
                        worst=float(value.abs().max()) if len(value) else np.nan,
                        where=f'{w.system} theta={w.theta:g} {w.condition}' if w is not None else '', note=note))

    c = summ[base & (linear | (own & (summ.theta == 0)))]
    add('P1 linear-Gaussian reads 0', c, (c.defined >= ENOUGH).all() and (c['mean'].abs() <= TOL).all(), c['mean'])
    for s in list(BEND) + list(SWITCH):
        c = summ[base & (summ.system == s)].sort_values('theta')
        rho = stats.spearmanr(c['mean'], c.truth)[0]
        add(f'P2 calibration: {s}', c, (c.defined >= ENOUGH).all() and (c.bias.abs() <= TOL).all() and rho >= .9, c.bias,
            f'Spearman {rho:.2f}')
    c = summ[summ.system == 'independent']
    add('P3 null', c, (c.defined <= .10).all(), c.defined, 'worst = largest defined fraction')
    c = summ[own & (kind == 'strength')]
    k = c[c.defined >= ENOUGH]
    add('P4 follows the population value across strengths', c, (k.bias.abs() <= TOL).all(), k.bias,
        f'population value spans {c.truth.min():.2f} to {c.truth.max():.2f}')
    c = summ[linear & kind.isin(['strength', 'T', 'spec', 'delay', 'mix', 'xnoise'])]
    k = c[c.defined >= ENOUGH]
    add('P5 linear-Gaussian stays 0 under nuisances', c, (k['mean'].abs() <= TOL).all(), k['mean'])
    ref = summ[base].set_index(['family', 'system', 'theta'])['mean']
    for nuisance in RESHAPES:
        c = summ[linear & (kind == nuisance) & (summ.defined >= ENOUGH)]
        rise = c['mean'] - ref.reindex(pd.MultiIndex.from_frame(c[['family', 'system', 'theta']])).values
        add(f'P6 {nuisance} raises N_MI on linear-Gaussian systems', c, len(c) > 0 and (rise > 0).all(), rise,
            f'smallest rise {rise.min():.2f}' if len(c) else '')
    for s in ('lag_variability', 'warp'):
        c = summ[base & (summ.system == s)].sort_values('theta')
        rho = stats.spearmanr(c['mean'], c.theta)[0]
        add(f'P6 {s} rises with theta', c, rho >= .9 and c['mean'].iloc[-1] > TOL, c['mean'], f'Spearman {rho:.2f}')
    return pd.DataFrame(out)


def analyse(out):
    summ = summarise(pd.read_csv(out / 'rows.csv'), pd.read_csv(out / 'truth.csv'))
    v = verdicts(summ)
    tables = HERE / 'tables' / 'nmi'
    tables.mkdir(parents=True, exist_ok=True)
    for d in (out, tables):
        summ.to_csv(d / 'summary.csv', index=False)
        v.to_csv(d / 'verdicts.csv', index=False)
    with pd.option_context('display.width', 220, 'display.max_columns', 20, 'display.max_colwidth', 60):
        print(v.to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=('run', 'analyse'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--reps', type=int, help='cap recordings per condition (dev runs)')
    a = ap.parse_args()
    run(a.out, a.offset, a.workers, a.reps) if a.step == 'run' else analyse(a.out)
