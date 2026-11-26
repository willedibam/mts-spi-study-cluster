"""z(Gaussian MI, kNN MI) real and 3 joint-IAAFT draws per accepted monkey B window, with per-pair MI saved.
zgk_pairs reproduces claar/zgk_ns.zgk exactly (asserted on the first window) but also returns the pair vectors.
Variants: ns = CAR 32 ch normal-scored [primary]; raw = CAR not normal-scored; asis = as-recorded, normal-scored.
    python fb_z.py SEL_DIR OUT.npz VARIANT[,VARIANT...]"""
import sys, os, json, numpy as np, warnings; warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'claar'))
from zgk_ns import zgk, ns
from surr_pairs import acf, ksg, multivariate_iaaft


def zgk_pairs(x):
    x = (x - x.mean(0)) / x.std(0); T, M = x.shape; iu = np.triu_indices(M, 1)
    r = np.corrcoef(x.T)[iu]; A = [acf(x[:, m]) for m in range(M)]
    k = np.array([ksg(x[:, i], x[:, j], min(int(2 * np.dot(A[i], A[j])), T // 10)) for i, j in zip(*iu)])
    g = -0.5 * np.log(1 - r ** 2)
    return np.corrcoef(g, k)[0, 1], g, k


def one(w, seed, variant):
    v = ns(w) if variant in ('ns', 'asis') else w
    out = [zgk_pairs(v)] + [zgk_pairs(multivariate_iaaft(v, np.random.default_rng(seed + d), final='spectrum')[0]) for d in range(3)]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out], np.float32), np.array([o[2] for o in out], np.float32)


if __name__ == '__main__':
    d, out, variants = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
    acc = json.load(open(f'{d}/sel.json'))['accepted']
    src = dict(ns=np.load(f'{d}/W32.npy'), raw=np.load(f'{d}/W32.npy'), asis=np.load(f'{d}/W32asis.npy'))
    w0 = ns(src['ns'][0].astype(np.float64)); assert abs(zgk_pairs(w0)[0] - zgk(w0)) < 1e-12
    seeds = [{'W1': 0, 'D': 1, 'W2': 2}[r['state']] * 100000 + r['k'] * 10 for r in acc]
    res = {}
    for var in variants:
        R = Parallel(n_jobs=9, verbose=5)(delayed(one)(src[var][i].astype(np.float64), seeds[i], var) for i in range(len(acc)))
        res[f'Z_{var}'] = np.array([r[0] for r in R])
        if var == 'ns':
            res['G_ns'] = np.array([r[1] for r in R]); res['K_ns'] = np.array([r[2] for r in R])
        np.savez(out, rows=json.dumps(acc), **res); print('done', var, res[f'Z_{var}'].shape, flush=True)
