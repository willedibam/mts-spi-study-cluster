"""Run the validation conditions through the pyspi subset (resumable, parallel).

    python -m research.representation.character_axes.validation.run [--workers 9] [--only PREFIX] [--dry]

Each job is one window; outputs go to results/representation/character-axes-validation/raw/<job>.npz,
or to $VALIDATION_OUT. On Gadi, jobs/farm.pbs runs --shard k --shards N tasks under nci-parallel.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import zlib
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = Path(os.environ.get('VALIDATION_OUT', ROOT / 'results' / 'representation' / 'character-axes-validation' / 'raw'))
CONFIG = HERE / 'spis.yaml'
PI = np.pi

BASE = {'B0': dict(cv_s=0.5, r=0.75), 'B1': dict(cv_s=1.0, r=0.9),
        # post hoc: stronger per-edge coupling (2 partners per channel, little private or measurement noise)
        'PHS': dict(cv_s=0.5, r=0.75, degree=2, private=0.05, noise=0.0)}
POSTHOC_KNOBS = ('form_level', 'form_disp', 'warp_level', 'warp_disp')
COMMON_A = dict(M=16, T=2000, degree=3, private=0.2, noise=0.05, mix=0.0, beta=0.0)
KNOBS_A = {
    'lag_level': ('phi', None, [0, PI / 8, PI / 4, 3 * PI / 8, PI / 2]),
    'lag_disp': ('phi', PI / 4, [0, PI / 8, PI / 4, 3 * PI / 8, PI / 2]),
    'band_level': ('f', None, [0.06, 0.15, 0.25, 0.35, 0.44]),
    'band_disp': ('f', 0.25, [0, 0.08, 0.16, 0.24, 0.32]),
    'form_level': ('theta', None, [0, PI / 12, PI / 6, PI / 4, PI / 3]),
    'form_disp': ('theta', PI / 6, [0, PI / 12, PI / 6, PI / 4, PI / 3]),
    'warp_level': ('warp', None, [0, 0.75, 1.5, 2.25, 3.0]),
    'warp_disp': ('warp', 1.5, [0, 0.75, 1.5, 2.25, 3.0]),
}
NUIS_A = {'cv_s': [0.1, 0.3, 0.6, 1.0, 1.5], 'noise': [0, 0.05, 0.25, 0.5, 1.0], 'T': [500, 1000, 2000, 3000, 4000],
          'r': [0.6, 0.75, 0.85, 0.9, 0.95], 'beta': [0, 0.5, 1.0, 1.5, 2.0], 'mix': [0, 0.1, 0.2, 0.35, 0.5],
          'M': [8, 12, 16, 20, 24]}
COMMON_B = dict(M=16, T=2000, degree=3, b=0.6, a=0.3, tau=2, noise=0.05, cv_s=0.3)
KNOBS_B = {'dir_level': (None, [0, 0.25, 0.5, 0.75, 1.0]), 'dir_disp': (0.5, [0, 0.25, 0.5, 0.75, 1.0])}
COMMON_C = dict(M=16, T=2000, a=0.6, noise=0.05, rho=0.3, hubs=2)
LAM = [0, 0.25, 0.5, 0.75, 1.0]
COMMON_K = dict(M=16, T=2000, f0=0.1, K=0.15, cv_s=0.3, sigma_w=0.005, phase_noise=0.02, noise=0.25)
KNOBS_K = {'lag_level': ('alpha', None, [0, PI / 16, PI / 8, 3 * PI / 16, PI / 4]),
           'lag_disp': ('alpha', PI / 8, [0, PI / 16, PI / 8, 3 * PI / 16, PI / 4]),
           'band_level': ('f0', None, [0.06, 0.15, 0.25, 0.35, 0.44])}
SURR_KNOBS = {'form_level', 'form_disp', 'warp_level', 'warp_disp'}
REPS_MAIN, REPS_GRID, WINDOWS = 6, 4, 2


def h(*parts) -> int:
    return zlib.crc32('|'.join(map(str, parts)).encode()) & 0x7FFFFFFF


def conditions(posthoc=False):
    """Yield condition dicts: family, knob, level, value, baseline, grid, reps, surrogate."""
    if posthoc:
        ph = [dict(family='A', knob=k, level=i, value=float(v), baseline='PHS', grid='', reps=REPS_MAIN)
              for k in POSTHOC_KNOBS for i, v in enumerate(KNOBS_A[k][2])]
        return [dict(x, surrogate=s) for s in (False, True) for x in ph]
    c = []
    for bl in ('B0', 'B1'):
        for knob, (_, _, vals) in KNOBS_A.items():
            for i, v in enumerate(vals):
                c.append(dict(family='A', knob=knob, level=i, value=float(v), baseline=bl, grid='', reps=REPS_MAIN))
        for nuis, vals in NUIS_A.items():
            for i, v in enumerate(vals):
                c.append(dict(family='A', knob='nuis_' + nuis, level=i, value=float(v), baseline=bl, grid='', reps=REPS_MAIN))
    for knob, (_, vals) in KNOBS_B.items():
        for i, v in enumerate(vals):
            c.append(dict(family='B', knob=knob, level=i, value=float(v), baseline='B0', grid='', reps=REPS_MAIN))
    for i, v in enumerate(LAM):
        c.append(dict(family='C', knob='med', level=i, value=float(v), baseline='B0', grid='', reps=REPS_MAIN))
    for knob, (_, _, vals) in KNOBS_K.items():
        for i, v in enumerate(vals):
            c.append(dict(family='K', knob=knob, level=i, value=float(v), baseline='B0', grid='', reps=REPS_MAIN))
    surr = [dict(x, surrogate=True) for x in c if x['family'] == 'A' and not x['grid']
            and (x['knob'] in SURR_KNOBS or (x['knob'] == 'lag_disp' and x['baseline'] == 'B0'))]
    grid = []
    for knob, (_, _, vals) in KNOBS_A.items():
        for i in (0, len(vals) - 1):
            for nuis, nv in NUIS_A.items():
                base_val = {**COMMON_A, **BASE['B0']}[nuis]
                for tag, ext in (('lo', nv[0]), ('hi', nv[-1])):
                    if ext == base_val:
                        continue
                    grid.append(dict(family='A', knob=knob, level=i, value=float(vals[i]), baseline='B0',
                                     grid=f'{nuis}:{tag}:{ext}', reps=REPS_GRID))
    out = [dict(x, surrogate=False) for x in c] + surr + [dict(x, surrogate=False) for x in grid]
    return out


def job_id(cond, rep, win):
    g = cond['grid'].replace(':', '_').replace('.', 'p') if cond['grid'] else 'main'
    s = '-surr' if cond['surrogate'] else ''
    return f"{cond['family']}-{cond['knob']}-L{cond['level']}-{cond['baseline']}-{g}-r{rep}-w{win}{s}"


def _signs(seed, n):
    rng = np.random.default_rng(seed)
    return rng.permutation(np.where(np.arange(n) % 2 == 0, 1.0, -1.0))


def build(cond, rep, win):
    """Generate one window; return x (T, M), truth dict."""
    from research.representation.character_axes.validation import generators as G
    fam, bl, grid = cond['family'], cond['baseline'], cond['grid']
    net_seed, noise_seed = h(fam, bl, grid, rep), h(fam, bl, grid, rep, win)
    if fam == 'A':
        kw = {**COMMON_A, **BASE[bl]}
        if cond['knob'].startswith('nuis_'):
            kw[cond['knob'][5:]] = cond['value']
        if grid:
            nuis, _, ext = grid.split(':')
            kw[nuis] = float(ext)
        kw['M'], kw['T'] = int(kw['M']), int(kw['T'])
        E = kw['M'] * kw['degree'] // 2
        sg = _signs(h(net_seed, 'signs'), E)
        per = dict(phi=PI / 8 + sg * PI / 16, f=np.full(E, 0.1), theta=np.zeros(E), warp=np.zeros(E))
        if cond['knob'] in KNOBS_A:
            prop, centre, vals = KNOBS_A[cond['knob']]
            v = vals[cond['level']]
            per[prop] = np.full(E, float(v)) if centre is None else centre + sg * v / 2
        x, truth = G.edge_drivers(net_seed=net_seed, noise_seed=noise_seed, **kw, **per)
    elif fam == 'B':
        E = COMMON_B['M'] * COMMON_B['degree'] // 2
        sg = _signs(h(net_seed, 'signs'), E)
        centre, vals = KNOBS_B[cond['knob']]
        v = vals[cond['level']]
        alpha = np.full(E, float(v)) if centre is None else centre + sg * v / 2
        x, truth = G.var_network(net_seed=net_seed, noise_seed=noise_seed, alpha=alpha, **COMMON_B)
    elif fam == 'C':
        x, truth = G.ggm_network(net_seed=net_seed, noise_seed=noise_seed, lam=cond['value'], **COMMON_C)
    else:
        E = COMMON_K['M'] - 1
        sg = _signs(h(net_seed, 'signs'), E)
        prop, centre, vals = KNOBS_K[cond['knob']]
        v = vals[cond['level']]
        kw = dict(COMMON_K, alpha=PI / 8)
        if prop == 'alpha':
            kw['alpha'] = np.full(E, float(v)) if centre is None else centre + sg * v / 2
        else:
            kw['f0'] = float(v)
        x, truth = G.kuramoto_tree(net_seed=net_seed, noise_seed=noise_seed, **kw)
    if cond['surrogate']:
        x = G.ft_surrogate(x, np.random.default_rng(h(net_seed, noise_seed, 'surr')))
        x = (x - x.mean(0)) / x.std(0)
    return x, truth


def needs_dcorr(cond):
    """dcorr (~45% of a job) is computed only where the r2|dcorr pair is evaluated."""
    if cond['family'] != 'A':
        return False
    form = cond['knob'].startswith('form')
    return form if (cond['grid'] or cond['surrogate']) else True


def _work(args):
    cond, rep, win = args
    path = OUT / (job_id(cond, rep, win) + '.npz')
    if path.exists():
        return path.name, 0.0, None
    import warnings
    warnings.filterwarnings('ignore')
    from mtsspi.compute import run_pyspi
    from research.representation.character_axes.validation.build_subset import WANTED
    full = needs_dcorr(cond)
    names = sorted(WANTED) if full else sorted(WANTED - {'dcorr'})
    t0 = time.time()
    try:
        x, truth = build(cond, rep, win)
        r = run_pyspi(x, config_path=CONFIG if full else HERE / 'spis_nodcorr.yaml', normalise=True, n_jobs=1)
        S = np.stack([r.matrices[n] for n in names]).astype(np.float32)
        ac1 = float(np.mean([np.corrcoef(x[:-1, j], x[1:, j])[0, 1] for j in range(x.shape[1])]))
        tmp = path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, S=S, spi=np.array(names), ac1=ac1,
                            meta=json.dumps(dict(cond, rep=rep, win=win)),
                            errors=json.dumps(r.errors or {}),
                            **{f't_{k}': np.asarray(v) for k, v in truth.items()})
        tmp.rename(path)
        return path.name, time.time() - t0, None
    except Exception as e:  # noqa: BLE001 - log and continue
        return path.name, time.time() - t0, f'{type(e).__name__}: {e}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=9)
    ap.add_argument('--only', default='')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--posthoc', action='store_true')
    ap.add_argument('--shard', type=int, default=-1, help='run jobs[shard::shards] serially in this process (one farm task)')
    ap.add_argument('--shards', type=int, default=1)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(c, r, w) for c in conditions(a.posthoc) for r in range(c['reps']) for w in range(WINDOWS)]
    jobs = [j for j in jobs if job_id(*j).startswith(a.only)]
    if a.shard >= 0:  # slice the fixed job list, not the shrinking to-do list, so concurrent shards never overlap
        jobs = jobs[a.shard::a.shards]
    todo = [j for j in jobs if not (OUT / (job_id(*j) + '.npz')).exists()]
    if a.limit:
        todo = todo[:a.limit]
    print(f'{len(jobs)} jobs, {len(todo)} to run', flush=True)
    if a.dry:
        return
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[k] = '1'
    if a.shard >= 0:
        for job in todo:
            name, dt, err = _work(job)
            print(f"{'ERROR' if err else 'ok'} {name} {dt:.0f}s {err or ''}", flush=True)
        return
    t0, done = time.time(), 0
    with get_context('spawn').Pool(a.workers, maxtasksperchild=200) as pool, open(OUT.parent / 'run.log', 'a') as log:
        for name, dt, err in pool.imap_unordered(_work, todo, chunksize=1):
            done += 1
            if err:
                log.write(f'ERROR {name} {err}\n')
            if done % 25 == 0 or err or done == len(todo):
                el = time.time() - t0
                msg = f'{done}/{len(todo)} done, {el / 60:.1f} min, eta {el / done * (len(todo) - done) / 60:.0f} min'
                print(msg, flush=True); log.write(msg + '\n'); log.flush()


if __name__ == '__main__':
    main()
