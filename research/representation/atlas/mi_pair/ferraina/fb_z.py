"""z(Gaussian MI, kNN MI) real and 3 joint-IAAFT draws per accepted monkey B window, with per-pair MI saved.
zgk_pairs reproduces claar/zgk_ns.zgk exactly (asserted on the first window) but also returns the pair vectors.
Variants: ns = CAR 32 ch normal-scored [primary]; raw = CAR not normal-scored; asis = as-recorded, normal-scored;
pa = per-array CAR 32 ch, normal-scored (post hoc, Clarification 9); paall = per-array CAR, all 190 kept channels.
Suffix _kshift: K is replaced by kshift = k4 minus the mean k4 with x_j circularly shifted by T/4, T/2, 3T/4 (as ksgbias.py).
    NJOBS=9 python fb_z.py SEL_DIR OUT.npz VARIANT[,VARIANT...]"""
import sys, os, json, numpy as np, warnings; warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'claar'))
from zgk_ns import zgk, ns
from surr_pairs import acf, ksg, multivariate_iaaft
SRC = dict(ns='W32', raw='W32', asis='W32asis', pa='W32pa', paall='WALLpa')


def zgk_pairs(x, shift=False):
    x = (x - x.mean(0)) / x.std(0); T, M = x.shape; iu = np.triu_indices(M, 1)
    r = np.corrcoef(x.T)[iu]; A = [acf(x[:, m]) for m in range(M)]
    W = [min(int(2 * np.dot(A[i], A[j])), T // 10) for i, j in zip(*iu)]
    k = np.array([ksg(x[:, i], x[:, j], w) for i, j, w in zip(*iu, W)])
    if shift:
        k = k - np.array([np.mean([ksg(x[:, i], np.roll(x[:, j], q * T // 4), w) for q in (1, 2, 3)]) for i, j, w in zip(*iu, W)])
    g = -0.5 * np.log(1 - r ** 2)
    return np.corrcoef(g, k)[0, 1], g, k


def one(w, seed, variant):
    base, shift = variant.split('_')[0], variant.endswith('_kshift')
    v = w if base == 'raw' else ns(w)
    out = [zgk_pairs(v, shift)] + [zgk_pairs(multivariate_iaaft(v, np.random.default_rng(seed + d), final='spectrum')[0], shift) for d in range(3)]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out], np.float32), np.array([o[2] for o in out], np.float32)


if __name__ == '__main__':
    d, out, variants = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
    acc = json.load(open(f'{d}/sel.json'))['accepted']
    w0 = ns(np.load(f'{d}/W32.npy', mmap_mode='r')[0].astype(np.float64)); assert abs(zgk_pairs(w0)[0] - zgk(w0)) < 1e-12
    seeds = [{'W1': 0, 'D': 1, 'W2': 2}[r['state']] * 100000 + r['k'] * 10 for r in acc]
    res = {}
    for var in variants:
        X = np.load(f"{d}/{SRC[var.split('_')[0]]}.npy", mmap_mode='r')
        R = Parallel(n_jobs=int(os.environ.get('NJOBS', 9)), verbose=5)(delayed(one)(X[i].astype(np.float64), seeds[i], var) for i in range(len(acc)))
        res[f'Z_{var}'] = np.array([r[0] for r in R])
        if var not in ('asis', 'raw'):
            res[f'G_{var}'] = np.array([r[1] for r in R]); res[f'K_{var}'] = np.array([r[2] for r in R])
        np.savez(out, rows=json.dumps(acc), **res); print('done', var, res[f'Z_{var}'].shape, flush=True)
