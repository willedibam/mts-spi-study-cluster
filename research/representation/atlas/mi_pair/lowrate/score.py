"""edges.py recompute on a low-rate bank, plus Theiler-window diagnostics (preregs/lowrate-prereg.md).

The estimator, subsets, normal-scoring, IAAFT draw and seeds are edges.py's own (imported, not copied): per window
the same calls as edges.one_recompute, and additionally, for each of raw/normal-scored x real/surrogate, the fraction
of the 496 edges whose AUTO Theiler window int(2 <a_i, a_j>) reaches the cap T // 10, the median applied window,
and the cap. Gate B (run first on the original 100 Hz stage-A bank) checks V against the stored zns arrays.

    PYTHONPATH=<repo>/src:<repo>/research/representation/atlas/mi_pair python score.py <bank dir> <out.npz> <agent> <first|row> <n workers>
"""
import json
import sys

import numpy as np
from multiprocessing import Pool

import edges as E
from mtsspi.surrogates import multivariate_iaaft


def theiler(x):
    x = (x - x.mean(0)) / x.std(0); T, M = x.shape; iu = np.triu_indices(M, 1)
    A = np.array([E.acf(x[:, m]) for m in range(M)]); w = np.trunc(2 * (A @ A.T))[iu].astype(int)   # int() as edges.edge_mi
    return [np.mean(w >= T // 10), np.median(np.minimum(w, T // 10)), T // 10]


def one(args):
    x, seed = args; V, Ed, D = [], [], []
    for v in (x, E.ns(x)):                                   # as edges.one_recompute
        s = multivariate_iaaft(v, np.random.default_rng(seed), final='spectrum')[0]
        gr, kr = E.edge_mi(v); gs, ks = E.edge_mi(s)
        V.append([E.zc(gr, kr), E.zc(gs, ks)]); Ed.append([[gr, kr], [gs, ks]]); D.append([theiler(v), theiler(s)])
    return np.array(V), np.array(Ed, np.float32), np.array(D)


if __name__ == '__main__':
    d, out, agent, rowmode, nw = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    man = json.load(open(f'{d}/manifest.json')); W = np.load(f'{d}/windows.npz')
    rows = [r for r in man['rows'] if r['state'] in E.STATES and r['agent'] in agent.split(',')]   # as edges.py
    S0 = E.sub32(rows[0]['M'])
    sub = lambda r: S0 if rowmode == 'first' else E.sub32(r['M'])
    jobs = [(W[r['id']].astype(float)[:, sub(r)], k) for k, r in enumerate(rows)]
    with Pool(nw) as p: res = p.map(one, jobs, chunksize=1)
    m, kept = E.meta(rows, man)
    labels = np.array([np.array(kept[r['unit']][:r['M']])[sub(r)] for r in rows])
    np.savez(out, V=np.array([a for a, _, _ in res]), E=np.array([b for _, b, _ in res]),
             D=np.array([c for _, _, c in res]), chan=labels, rowmode=rowmode, T=man['T'], fs=man['fs'],
             start_s=np.array([r['start_s'] for r in rows]), **m)
    print('done', out, flush=True)
