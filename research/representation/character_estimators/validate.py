"""Construction validation of the five character estimators (protocol in README.md).

    python -m research.representation.character_estimators.validate run --out <dir> [--offset 0] [--workers 9]
    python -m research.representation.character_estimators.validate analyse --out <dir>

``run`` writes rows.csv (one row per recording) and truth.csv (one row per condition); ``analyse`` writes
summary.csv and verdicts.csv to <dir> and to tables/ here, and the figures to <dir>/figures.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
import os
from pathlib import Path
import zlib

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')  # one BLAS thread per worker; set before numpy loads
import numpy as np
import pandas as pd

from research.representation.character_estimators.estimators import AXES, estimate
from research.representation.character_estimators.systems import OWN, SYSTEMS, make

HERE = Path(__file__).parent
T0 = 2000
LEVELS = {s: (0, .25, .5, .75, 1.) if s == 'direction' else (0, .2, .4, .6, .8) for s in SYSTEMS}
NUISANCE_THETA = {s: (.25, .75) if s == 'direction' else (.2, .6) for s in SYSTEMS}  # in LEVELS, so T conditions reuse base truth
STRENGTH = dict(linearity=(.5, 2., 3.), lag=(.5, 2., 3.), lag_variability=(.5, 2., 3.), intermittency=(.2, .3, .5),
                direction=(.2, .4, .8))
BASE_STRENGTH = dict(intermittency=.4)  # Pearson r; the others use the systems.py default
REPS_MAIN, REPS_NUISANCE, REPS_NULL = 40, 20, 100
LONG_T, LONG_SEEDS, LONG_SHIFTS = 2 ** 17, 3, 19
# Construction values of every axis at base and strength/T/spec conditions (NaN: no dependence of that kind).
CONSTRUCTION = dict(
    linearity=dict(nonlinearity_raw='theta', lag=0., lag_variability=0., intermittency=0., direction=np.nan),
    lag=dict(nonlinearity=0., nonlinearity_raw=0., lag='theta', lag_variability=0., intermittency=0.),
    lag_variability=dict(nonlinearity=0., nonlinearity_raw=0., lag=.5, lag_variability='theta', intermittency=0.),
    intermittency=dict(nonlinearity=0., nonlinearity_raw=0., lag=0., lag_variability=0., intermittency='theta', direction=np.nan),
    direction=dict(nonlinearity=0., nonlinearity_raw=0., lag_variability=0., intermittency=0.))
KEEPS_CONSTRUCTION = ('base', 'strength', 'T', 'spec')


def conditions():
    """(system, theta, condition label, make kwargs, T, reps)."""
    out = []
    for s in SYSTEMS:
        base = dict(strength=BASE_STRENGTH.get(s))
        for th in LEVELS[s]:
            out.append((s, th, 'base', base, T0, REPS_MAIN))
        for th in NUISANCE_THETA[s]:
            for v in STRENGTH[s]:
                out.append((s, th, f'strength={v:g}', dict(strength=v), T0, REPS_NUISANCE))
            for T in (1000, 4000, 8000):
                out.append((s, th, f'T={T}', base, T, REPS_NUISANCE))
            if s != 'direction':
                for spec in ('broad', 'slow', 'narrow'):
                    out.append((s, th, f'spec={spec}', {**base, 'spec': spec}, T0, REPS_NUISANCE))
            for key, vals in (('beta', (1., 3.)), ('mix', (.2, .5)), ('xnoise', (.5, 1.))):
                for v in vals:
                    out.append((s, th, f'{key}={v:g}', {**base, key: v}, T0, REPS_NUISANCE))
    for spec in ('ar1', 'broad', 'slow', 'narrow'):
        out.append(('independent', 0., f'spec={spec}', dict(spec=spec), T0, REPS_NULL))
    return out


def _seed(*parts):
    return zlib.crc32('|'.join(map(str, parts)).encode())


def _one(task):
    s, th, label, kw, T, rep, offset = task
    rng = np.random.default_rng([_seed(s, th, label, rep), offset])
    x, y = make(s, th, T, rng, **kw)
    e = estimate(x, y, rng)
    return dict(system=s, theta=th, condition=label, rep=rep, T=T, **e)


def _truth(task):
    s, th, label, kw, offset, (long_t, seeds, shifts) = task
    vals = {a: [] for a in AXES}
    for seed in range(seeds):
        rng = np.random.default_rng([_seed('long', s, th, label, seed), offset])
        x, y = make(s, th, long_t, rng, **kw)
        e = estimate(x, y, rng, shifts=shifts)
        for a in AXES:
            vals[a].append(e[a])
    row = dict(system=s, theta=th, condition=label)
    for a in AXES:
        v = np.array(vals[a], float)
        row[f'{a}_long'] = float(np.nanmean(v)) if np.isfinite(v).sum() >= 2 else np.nan
    return row


def run(out, offset, workers, reps=None, long=(LONG_T, LONG_SEEDS, LONG_SHIFTS)):
    """``reps`` caps the recordings per condition (for a dry run)."""
    out.mkdir(parents=True, exist_ok=True)
    conds = conditions()
    tasks = [(s, th, lab, kw, T, r, offset) for s, th, lab, kw, T, n in conds for r in range(min(n, reps or n))]
    # truth does not depend on T: T conditions reuse the base truth at that theta
    truth_tasks = [(s, th, lab, kw, offset, long) for s, th, lab, kw, T, _ in conds if not lab.startswith('T=') and s != 'independent']
    with Pool(workers) as pool:
        truth = pd.DataFrame(pool.map(_truth, truth_tasks, chunksize=1))
        truth.to_csv(out / 'truth.csv', index=False)
        rows = pd.DataFrame(pool.map(_one, tasks, chunksize=4))
    rows.to_csv(out / 'rows.csv', index=False)
    (out / 'run.json').write_text(json.dumps(dict(offset=offset, recordings=len(rows), conditions=len(conds)), indent=1))


def truth_table(truth):
    """Long-run and construction values per (system, theta, condition, axis); construction overrides where it holds."""
    recs = []
    for _, r in truth.iterrows():
        for a in AXES:
            c = CONSTRUCTION.get(r.system, {}).get(a, None)
            kind = r.condition.split('=')[0]
            if kind == 'beta' and a != 'nonlinearity_raw':
                kind = 'base'  # main axes see normal scores, so a monotone distortion of y changes nothing
            cv = (r.theta if c == 'theta' else c) if (c is not None and kind in KEEPS_CONSTRUCTION) else None
            recs.append(dict(system=r.system, theta=r.theta, condition=r.condition, axis=a, long=r[f'{a}_long'],
                             construction=np.nan if cv is None else cv,
                             truth=r[f'{a}_long'] if cv is None else cv, source='long run' if cv is None else 'construction'))
    t = pd.DataFrame(recs)
    base = t[t.condition == 'base']
    tt = []
    for T in (1000, 4000, 8000):  # T conditions share the base truth
        for th in set(t.theta):
            b = base[base.theta == th].assign(condition=f'T={T}')
            tt.append(b)
    return pd.concat([t] + tt, ignore_index=True)


def summarise(rows, truth):
    recs = []
    for (s, th, lab), g in rows.groupby(['system', 'theta', 'condition']):
        for a in AXES:
            v = g[a].astype(float)
            recs.append(dict(system=s, theta=th, condition=lab, axis=a, n=len(v), defined=float(v.notna().mean()),
                             mean=float(v.mean()), sd=float(v.std()), pearson=float(g.pearson.mean()),
                             coherence=float(g.coherence.mean())))
    summ = pd.DataFrame(recs)
    tt = truth_table(truth)
    summ = summ.merge(tt, on=['system', 'theta', 'condition', 'axis'], how='left')
    summ['bias'] = summ['mean'] - summ['truth']
    return summ


def verdicts(summ):
    """Pre-registered criteria C1-C5 (README.md)."""
    tol, sd_tol, enough = .10, .15, .8
    own = dict(OWN, linearity=('nonlinearity', 'nonlinearity_raw'))
    rows = []
    for s in SYSTEMS:
        for a in np.atleast_1d(own[s]):
            m = summ[(summ.system == s) & (summ.axis == a) & (summ.condition == 'base')].sort_values('theta')
            c1 = m.dropna(subset=['construction'])
            c1_ok = bool((np.abs(c1['long'] - c1['construction']) <= .03).all()) if len(c1) else None
            ok = m[m.defined >= enough]
            c2_ok = bool(((ok.bias.abs() <= tol) & (ok.sd <= sd_tol)).all() and len(ok) == len(m)
                         and pd.Series(ok['mean'].values).corr(pd.Series(ok.truth.values), method='spearman') >= .9)
            st = summ[(summ.system == s) & (summ.axis == a) & summ.condition.str.startswith('strength')]
            sto = st[st.defined >= enough]
            c4_ok = bool((sto.bias.abs() <= tol).all())
            rows.append(dict(system=s, axis=a, C1_long_run_matches_construction=c1_ok,
                             C2_accuracy=c2_ok, C2_worst_bias=float(m.bias.abs().max()), C2_worst_sd=float(m.sd.max()),
                             C2_min_defined=float(m.defined.min()), C4_strength=c4_ok,
                             C4_worst_bias=float(sto.bias.abs().max()) if len(sto) else np.nan,
                             C4_min_defined=float(st.defined.min())))
    v = pd.DataFrame(rows)
    sel = []
    for a in AXES:
        owners = [s for s in SYSTEMS if a in np.atleast_1d(own[s])]
        m = summ[(summ.axis == a) & (summ.condition == 'base') & ~summ.system.isin(owners + ['independent'])]
        d = m[(m.defined >= enough) & m.truth.notna()]
        undefined = m[m.truth.isna()]
        sel.append(dict(axis=a, C3_selectivity=bool((d.bias.abs() <= tol).all()),
                        C3_worst_bias=float(d.bias.abs().max()) if len(d) else np.nan,
                        C3_worst_where=(f'{d.loc[d.bias.abs().idxmax(), "system"]} theta={d.loc[d.bias.abs().idxmax(), "theta"]:g}'
                                        if len(d) else ''),
                        C5_null_defined_max=float(summ[(summ.system == 'independent') & (summ.axis == a)].defined.max()),
                        C5_undefined_truth_defined_max=float(undefined.defined.max()) if len(undefined) else np.nan))
    sel = pd.DataFrame(sel)
    sel['C5_null'] = (sel.C5_null_defined_max <= .10) & ~(sel.C5_undefined_truth_defined_max > .10)
    return v, sel


def analyse(out):
    rows, truth = pd.read_csv(out / 'rows.csv'), pd.read_csv(out / 'truth.csv')
    summ = summarise(rows, truth)
    v, sel = verdicts(summ)
    tables = HERE / 'tables'
    tables.mkdir(exist_ok=True)
    for d in (out, tables):
        summ.to_csv(d / 'summary.csv', index=False)
        v.to_csv(d / 'verdicts_own.csv', index=False)
        sel.to_csv(d / 'verdicts_axis.csv', index=False)
    from research.representation.character_estimators.figure import figures
    figures(summ, out / 'figures')
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(v.to_string(index=False)); print(sel.to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=('run', 'analyse'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--workers', type=int, default=9)
    a = ap.parse_args()
    run(a.out, a.offset, a.workers) if a.step == 'run' else analyse(a.out)
