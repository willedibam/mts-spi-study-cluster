"""Window-level z(Gaussian MI, kNN MI), real and joint 32-ch IAAFT, for raw and normal-scored channels.
Out V[w] = [[raw_real, raw_surr], [ns_real, ns_surr]]. PF sessions first (pilot), then the rest."""
import sys, json, numpy as np, warnings; warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
from scipy.stats import norm, rankdata
from surr_pairs import acf, ksg, multivariate_iaaft
def zgk(x):
    x = (x - x.mean(0)) / x.std(0); T, M = x.shape; iu = np.triu_indices(M, 1)
    r = np.corrcoef(x.T)[iu]; A = [acf(x[:, m]) for m in range(M)]
    k = np.array([ksg(x[:, i], x[:, j], min(int(2 * np.dot(A[i], A[j])), T // 10)) for i, j in zip(*iu)])
    return np.corrcoef(-0.5 * np.log(1 - r ** 2), k)[0, 1]
ns = lambda x: norm.ppf((np.apply_along_axis(rankdata, 0, x) - .5) / len(x))
def one(x, seed):
    out = []
    for v in (x, ns(x)):
        out.append([zgk(v), zgk(multivariate_iaaft(v, np.random.default_rng(seed), final='spectrum')[0])])
    return np.array(out)
if __name__ == '__main__':
    d, out, only = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else None)
    rows = [r for r in json.load(open(f'{d}/manifest.json'))['rows'] if r['state'] in ('awake_ec', 'awake_eo', 'drug', 'sleep', 'recovery_ec')]
    if only: rows = [r for r in rows if r['agent'] in only.split(',')]
    W = np.load(f'{d}/windows.npz'); S = lambda m: np.round(np.arange(32) * (m - 1) / 31).astype(int)
    res = Parallel(n_jobs=7)(delayed(one)(W[r['id']].astype(float)[:, S(r['M'])], k) for k, r in enumerate(rows))
    np.savez(out, V=np.array(res), delta_fraction=np.array([r.get('delta_fraction', np.nan) for r in rows], float),
             **{k: np.array([r.get(k, '') for r in rows]) for k in ('id', 'unit', 'state', 'agent', 'animal')})
