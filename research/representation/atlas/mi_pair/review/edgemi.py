"""Edge-level Gaussian vs kNN MI on real windows and one IAAFT (spectrum-final) surrogate per window.
Usage: edgemi.py <raw dir> <M> <out.npz> [states,comma]"""
import sys, json, numpy as np, warnings; warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
from surr_pairs import acf, ksg, multivariate_iaaft


def edges(x):
    x = (x - x.mean(0)) / x.std(0); T, M = x.shape; iu = np.triu_indices(M, 1)
    r = np.corrcoef(x.T)[iu]; A = [acf(x[:, m]) for m in range(M)]
    mik = np.array([ksg(x[:, i], x[:, j], min(int(2 * np.dot(A[i], A[j])), T // 10)) for i, j in zip(*iu)])
    return r, mik


def one(x, seed):
    r, k = edges(x); rs, ks = edges(multivariate_iaaft(x, np.random.default_rng(seed), final='spectrum')[0])
    return np.stack([r, k, rs, ks])


if __name__ == '__main__':
    d, M, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    states = sys.argv[4].split(',') if len(sys.argv) > 4 else ['awake_ec', 'awake_eo', 'drug', 'sleep', 'recovery_ec']
    W = np.load(f'{d}/windows.npz'); rows = [r for r in json.load(open(f'{d}/manifest.json'))['rows'] if r['state'] in states]
    sub = np.round(np.arange(M) * (rows[0]['M'] - 1) / (M - 1)).astype(int)
    res = Parallel(n_jobs=8, verbose=0)(delayed(one)(W[r['id']].astype(float)[:, sub], k) for k, r in enumerate(rows))
    np.savez(out, E=np.array(res), sub=sub, **{k: np.array([r.get(k, '') for r in rows]) for k in ('id', 'unit', 'state', 'agent', 'animal')})

