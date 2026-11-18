"""10-s decision test for z(Gaussian MI, kNN MI): static curved coupling vs amplitude-state (gain) coupling.
Per edge (both regression directions): R² of y on [x] (L), [x,x²,x³] (C), [x,g,x·g] (G), all (CG); g = log 1-s RMS of the pair.
Readouts (pair means, real minus IAAFT): curvature unique to C, gain unique to G, and out-of-sample curvature
(fit on one half, score on the other; a static function transfers, a within-window state change does not)."""
import sys, json, numpy as np, warnings; warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
from scipy.ndimage import uniform_filter1d
from surr_pairs import acf, ksg, multivariate_iaaft

ENV = 100   # 1 s at 100 Hz


def r2(X, y, Xt=None, yt=None):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    if Xt is None: Xt, yt = X, y
    return 1 - ((yt - Xt @ b) ** 2).mean() / yt.var()


def design(x, g):
    o = np.ones_like(x)
    return dict(L=np.c_[o, x], C=np.c_[o, x, x ** 2, x ** 3], G=np.c_[o, x, g, x * g], CG=np.c_[o, x, x ** 2, x ** 3, g, x * g])


def pair(x, y):
    g = np.log(uniform_filter1d((x ** 2 + y ** 2) / 2, ENV, mode='reflect') + 1e-12); g = (g - g.mean()) / g.std()
    out = np.zeros(3); T = len(x); h = T // 2
    for a, b in ((x, y), (y, x)):
        D = design(a, g); R = {k: r2(v, b) for k, v in D.items()}
        oos = 0
        for tr, te in ((slice(0, h), slice(h, T)), (slice(h, T), slice(0, h))):
            oos += (r2(D['C'][tr], b[tr], D['C'][te], b[te]) - r2(D['L'][tr], b[tr], D['L'][te], b[te])) / 2
        out += np.array([R['CG'] - R['G'], R['CG'] - R['C'], oos]) / 2
    return out


def metrics(x, strong_frac=.25):
    x = (x - x.mean(0)) / x.std(0); T, M = x.shape; iu = np.triu_indices(M, 1)
    r = np.corrcoef(x.T)[iu]; A = [acf(x[:, m]) for m in range(M)]
    mik = np.array([ksg(x[:, i], x[:, j], min(int(2 * np.dot(A[i], A[j])), T // 10)) for i, j in zip(*iu)])
    P = np.array([pair(x[:, i], x[:, j]) for i, j in zip(*iu)])
    s = np.argsort(np.abs(r))[::-1][:int(strong_frac * len(r))]
    return np.r_[np.corrcoef(-0.5 * np.log(1 - r ** 2), mik)[0, 1], P.mean(0), P[s].mean(0), (mik - (-0.5 * np.log(1 - r ** 2))).mean()]


NAMES = ['zgk', 'curv', 'gain', 'oos_curv', 'curv_strong', 'gain_strong', 'oos_strong', 'knn_minus_gauss']


def one(x, seed):
    return np.stack([metrics(x), metrics(multivariate_iaaft(x, np.random.default_rng(seed), final='spectrum')[0])])


if __name__ == '__main__':
    d, M, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    W = np.load(f'{d}/windows.npz'); rows = json.load(open(f'{d}/manifest.json'))['rows']
    rows = [r for r in rows if r['state'] in ('awake_ec', 'awake_eo', 'drug', 'sleep', 'recovery_ec')]
    sub = np.round(np.arange(M) * (rows[0]['M'] - 1) / (M - 1)).astype(int)
    res = Parallel(n_jobs=8)(delayed(one)(W[r['id']].astype(float)[:, sub], k) for k, r in enumerate(rows))
    np.savez(out, V=np.array(res), names=NAMES, **{k: np.array([r.get(k, np.nan) if k == 'delta_fraction' else r.get(k, '') for r in rows])
                                                   for k in ('id', 'unit', 'state', 'agent', 'animal', 'delta_fraction')})

