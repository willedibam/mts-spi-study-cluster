"""Factorial test: nonlinearity, lag, phase drift and on/off coupling crossed at two levels (README.md).

    python -m research.estimators.factorial run --out <dir> [--offset 0] [--workers 9]
    python -m research.estimators.factorial analyse --out <dir>

One-knob systems show that each axis ignores the other characters when those are absent; they cannot show that
the axes stay separable when several are present. Two questions per cell and axis: F1, is the long-run value
the pre-registered one (separability of the definitions); F2, does a T = 2000 estimate match the long run
(estimation in crossed cells).
"""
from __future__ import annotations

import argparse
from itertools import product
import json
from multiprocessing import Pool
import os
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
import pandas as pd

from research.estimators import validate as V
from research.estimators.estimators import AXES, estimate
from research.estimators.systems import FACTORIAL, factorial_prediction, make_factorial

HERE = Path(__file__).parent
REPS = 40
CORE = ('nonlinearity', 'nonlinearity_raw', 'lag', 'variability')
KNOBS = tuple(FACTORIAL)


def cells():
    return [dict(zip(KNOBS, v)) for v in product(*FACTORIAL.values())]


def label(c):
    return ' '.join(f'{k}={v:g}' for k, v in c.items())


def _gen(system, theta, T, rng, **kw):
    return make_factorial(T, rng, **kw)


def _one(task):
    c, T, rep, offset = task
    rng = np.random.default_rng([V._seed('factorial', label(c), rep), offset])
    x, y = make_factorial(T, rng, **c)
    return dict(cell=label(c), rep=rep, **c, **estimate(x, y, rng))


def run(out, offset, workers, reps=REPS):
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(c, V.T0, r, offset) for c in cells() for r in range(reps)]
    truth_tasks = [('factorial', 0., label(c), c, offset, (V.LONG_T, V.LONG_SEEDS, V.LONG_SHIFTS), _gen) for c in cells()]
    with Pool(workers) as pool:
        t = pd.DataFrame(pool.map(V._truth, truth_tasks, chunksize=1)).rename(columns=dict(condition='cell'))
        t.drop(columns=['system', 'theta']).to_csv(out / 'truth.csv', index=False)
        rows = pd.DataFrame(pool.map(_one, tasks, chunksize=4))
    rows.to_csv(out / 'rows.csv', index=False)
    (out / 'run.json').write_text(json.dumps(dict(offset=offset, recordings=len(rows), cells=len(cells())), indent=1))


def summarise(rows, truth):
    recs = []
    for c in cells():
        g, t = rows[rows.cell == label(c)], truth[truth.cell == label(c)].iloc[0]
        for a in AXES:
            v = g[a].astype(float)
            p = factorial_prediction(a, **c)
            recs.append(dict(cell=label(c), **c, axis=a, n_rec=len(v), defined=float(v.notna().mean()),
                             mean=float(v.mean()), sd=float(v.std()), long=t[f'{a}_long'],
                             prediction=np.nan if p is None else p, pearson=float(g.pearson.mean()),
                             coherence=float(g.coherence.mean())))
    s = pd.DataFrame(recs)
    s['bias'] = s['mean'] - s['long']
    s['long_minus_prediction'] = s['long'] - s['prediction']
    return s


def verdicts(s):
    """F1: |long run - prediction| <= 0.03 (the C1 tolerance) where a prediction exists. F2: in every cell where
    the axis is defined in >= 80%, |mean - long run| <= 0.10 and SD <= 0.15 (the C2 tolerances)."""
    out = []
    for a in CORE:
        m = s[s.axis == a]
        f1 = m.dropna(subset=['prediction'])
        ok = m[m.defined >= .8]
        bad = ok[(ok.bias.abs() > .10) | (ok.sd > .15)]
        out.append(dict(axis=a, F1_cells=len(f1), F1_pass=bool((f1.long_minus_prediction.abs() <= .03).all()) if len(f1) else None,
                        F1_worst=float(f1.long_minus_prediction.abs().max()) if len(f1) else np.nan,
                        F2_cells_defined=len(ok), F2_pass=bool(len(bad) == 0), F2_worst_bias=float(ok.bias.abs().max()),
                        F2_worst_sd=float(ok.sd.max()), F2_failing='; '.join(bad.cell), min_defined=float(m.defined.min())))
    return pd.DataFrame(out)


def figure(s, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mtsspi.figures import save, style
    from research.estimators.figure import COLOUR, ENOUGH, LABEL
    style()
    rows = ('nonlinearity', 'nonlinearity_raw', 'lag', 'variability', 'direction')
    labels = [label(c) for c in cells()]
    xs = np.arange(len(labels))
    fig, axs = plt.subplots(len(rows), 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    for ax, a in zip(axs, rows):
        m = s[s.axis == a].set_index('cell').loc[labels]
        ok = (m.defined >= ENOUGH).values
        ax.errorbar(xs[ok], m['mean'][ok], m.sd[ok], fmt='o', color=COLOUR[a], ms=4, lw=1.2, capsize=2, label='T = 2000, mean ± SD')
        ax.errorbar(xs[~ok], m['mean'][~ok], m.sd[~ok], fmt='o', mfc='none', color=COLOUR[a], ms=4, lw=.7, capsize=2,
                    alpha=.6, label='defined in < 80%')
        ax.plot(xs, m.long, 'k_', ms=14, mew=1.6, label='long run')
        ax.plot(xs, m.prediction, 'x', color='.45', ms=6, label='pre-registered')
        ax.set(ylim=(-.25, 1.15), ylabel=LABEL[a])
        ax.axhline(0, color='.85', lw=.6, zorder=0); ax.axhline(1, color='.85', lw=.6, zorder=0)
        for i in range(1, len(labels)):
            if i % 4 == 0:
                ax.axvline(i - .5, color='.8', lw=.6)
    axs[0].legend(loc='upper left', fontsize=6.5, ncol=4)
    axs[-1].set_xticks(xs, [l.replace(' ', '\n') for l in labels], fontsize=6)
    fig.suptitle(f'Factorial test: knobs n (nonlinear share), l (quadrature share), d (phase drift), o (off-fraction); '
                 f'{int(s.n_rec.min())} recordings per cell', fontsize=9)
    save(fig, out, 'factorial'); plt.close(fig)


def analyse(out):
    s = summarise(pd.read_csv(out / 'rows.csv'), pd.read_csv(out / 'truth.csv'))
    v = verdicts(s)
    tables = HERE / 'tables' / 'factorial'
    tables.mkdir(parents=True, exist_ok=True)
    for d in (out, tables):
        s.to_csv(d / 'summary.csv', index=False)
        v.to_csv(d / 'verdicts.csv', index=False)
    figure(s, out / 'figures')
    with pd.option_context('display.width', 250, 'display.max_columns', 20, 'display.max_colwidth', 120):
        print(v.to_string(index=False))
        print(s[s.axis.isin(CORE + ('direction',))].pivot(index='cell', columns='axis', values=['mean', 'long', 'prediction']).round(2).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=('run', 'analyse'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--workers', type=int, default=9)
    ap.add_argument('--reps', type=int, default=REPS)
    a = ap.parse_args()
    run(a.out, a.offset, a.workers, a.reps) if a.step == 'run' else analyse(a.out)
