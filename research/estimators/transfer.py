"""Transfer validation: the same estimators and criteria on a second family of test systems (README.md).

    python -m research.estimators.transfer run --out <dir> [--offset 0] [--workers 8]
    python -m research.estimators.transfer analyse --out <dir>

The family (systems.make_transfer) avoids the first family's construction: narrowband signals with white private
noise, a nonlinearity outside the Volterra basis, conventional delays, a smooth time-varying delay, switching
independent of the W-sample grid, and nonlinear directed dynamics. Own-axis truths are integrals over the known
coherence spectrum; off-axis values are 0 by construction where stated, otherwise the long run (T = 2^17,
3 seeds; no shift floor, which a long run does not need).
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

from research.estimators import validate as V
from research.estimators.estimators import estimate
from research.estimators.systems import OWN_TRANSFER, TRANSFER, make_transfer, transfer_truth

HERE = Path(__file__).parent
LEVELS = {s: (0, .25, .5, .75, 1.) if s == 'direction_nl' else (0, .2, .4, .6, .8) for s in TRANSFER}
NUISANCE_THETA = {s: (.25, .75) if s == 'direction_nl' else (.2, .6) for s in TRANSFER}
STRENGTH = dict(rectifier=(.5, 2.), delay=(.5, 2.), warp=(.5, 2.), telegraph=(.3, .5), direction_nl=(.3, .9))
TITLE = dict(rectifier='knob: rectified share', delay='knob: delay (carrier phase)', warp='knob: delay jitter',
             telegraph='knob: off-fraction', direction_nl='knob: one-way share')


def _sigma(condition):
    return float(condition.split('=')[1]) if condition.startswith('strength') else 1.


def _analytic(system):
    return lambda theta, condition: transfer_truth(system, theta, _sigma(condition))


NAN = np.nan
CONSTRUCTION = dict(
    rectifier=dict(nonlinearity_raw='theta', lag=0., variability=0., direction=NAN),
    delay=dict(nonlinearity=0., nonlinearity_raw=0., lag=_analytic('delay'), variability=0.),
    warp=dict(nonlinearity=0., nonlinearity_raw=0., lag=0., variability=_analytic('warp')),
    telegraph=dict(nonlinearity_raw=0., lag=0., variability=_analytic('telegraph'), direction=NAN),
    direction_nl=dict(variability=0.))
OWN = dict(OWN_TRANSFER, rectifier=('nonlinearity', 'nonlinearity_raw'))


def conditions():
    out = []
    for s in TRANSFER:
        for th in LEVELS[s]:
            out.append((s, th, 'base', {}, V.T0, V.REPS_MAIN))
        for th in NUISANCE_THETA[s]:
            for v in STRENGTH[s]:
                out.append((s, th, f'strength={v:g}', dict(strength=v), V.T0, V.REPS_NUISANCE))
    out.append(('independent', 0., 'spec=narrow+white', {}, V.T0, V.REPS_NULL))
    return out


def _one(task):
    s, th, label, kw, T, rep, offset = task
    rng = np.random.default_rng([V._seed('transfer', s, th, label, rep), offset])
    x, y = make_transfer(s, th, T, rng, **kw)
    return dict(system=s, theta=th, condition=label, rep=rep, T=T, **estimate(x, y, rng))


def run(out, offset, workers, reps=None):
    out.mkdir(parents=True, exist_ok=True)
    conds = conditions()
    tasks = [(s, th, lab, kw, T, r, offset) for s, th, lab, kw, T, n in conds for r in range(min(n, reps or n))]
    truth_tasks = [(s, th, lab, kw, offset, (V.LONG_T, V.LONG_SEEDS, 0), make_transfer)
                   for s, th, lab, kw, T, _ in conds if s != 'independent']
    with Pool(workers) as pool:
        pd.DataFrame(pool.map(V._truth, truth_tasks, chunksize=1)).to_csv(out / 'truth.csv', index=False)
        rows = pd.DataFrame(pool.map(_one, tasks, chunksize=4))
    rows.to_csv(out / 'rows.csv', index=False)
    (out / 'run.json').write_text(json.dumps(dict(offset=offset, recordings=len(rows), conditions=len(conds)), indent=1))


def analyse(out):
    rows, truth = pd.read_csv(out / 'rows.csv'), pd.read_csv(out / 'truth.csv')
    summ = V.summarise(rows, truth, CONSTRUCTION)
    v, sel = V.verdicts(summ, TRANSFER, OWN)
    tables = HERE / 'tables' / 'transfer'
    tables.mkdir(parents=True, exist_ok=True)
    for d in (out, tables):
        summ.to_csv(d / 'summary.csv', index=False)
        v.to_csv(d / 'verdicts_own.csv', index=False)
        sel.to_csv(d / 'verdicts_axis.csv', index=False)
    from research.estimators import figure as F
    F.style()
    F.response(summ, out / 'figures', TRANSFER, OWN_TRANSFER, TITLE, name='transfer_response')
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(v.to_string(index=False)); print(sel.to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=('run', 'analyse'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--reps', type=int)
    a = ap.parse_args()
    run(a.out, a.offset, a.workers, a.reps) if a.step == 'run' else analyse(a.out)
