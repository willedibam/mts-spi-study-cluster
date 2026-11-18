"""Three mechanism readouts per window (pair means; real minus IAAFT gives the beyond-linear part):
 switch : across 1-s blocks, variance of Fisher-z(block r)                -> coupling strength changes within the window
 comod  : correlation of 1-s log-RMS of the two channels                  -> shared amplitude (gain) modulation
 curv   : on block-standardised, clipped data, out-of-half ΔR² of [x²,x³] -> static curved coupling
plus zgk and the kNN-minus-Gaussian MI level."""
import sys, json, numpy as np, warnings; warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
from surr_pairs import acf, ksg, multivariate_iaaft
B = 100
def r2(X, y, Xt, yt):
    b = np.linalg.lstsq(X, y, rcond=None)[0]; return 1 - ((yt - Xt @ b) ** 2).mean() / yt.var()
def metrics(x, frac=.25, mi=True):
    x = (x - x.mean(0)) / x.std(0); T, M = x.shape; iu = np.triu_indices(M, 1); nb = T // B
    blk = x[:nb * B].reshape(nb, B, M)
    xs = ((blk - blk.mean(1, keepdims=True)) / blk.std(1, keepdims=True)).reshape(nb * B, M).clip(-3, 3)
    r = np.corrcoef(x.T)[iu]
    rb = np.array([np.corrcoef(b.T)[iu] for b in blk]); sw = np.arctanh(rb.clip(-.999, .999)).var(0)
    lr = np.log(blk.std(1)); cm = np.corrcoef(lr.T)[iu]
    h = nb * B // 2; o = np.ones(h)
    cv = []
    for i, j in zip(*iu):
        v = 0
        for a, b in ((xs[:, i], xs[:, j]), (xs[:, j], xs[:, i])):
            for tr, te in ((slice(0, h), slice(h, 2 * h)), (slice(h, 2 * h), slice(0, h))):
                L = lambda z: np.c_[o, z]; C = lambda z: np.c_[o, z, z ** 2, z ** 3]
                v += (r2(C(a[tr]), b[tr], C(a[te]), b[te]) - r2(L(a[tr]), b[tr], L(a[te]), b[te])) / 4
        cv.append(v)
    cv = np.array(cv); s = np.argsort(np.abs(r))[::-1][:int(frac * len(r))]
    out = [sw.mean(), cm.mean(), cv.mean(), sw[s].mean(), cm[s].mean(), cv[s].mean()]
    if mi:
        A = [acf(x[:, m]) for m in range(M)]; g = -0.5 * np.log(1 - r ** 2)
        k = np.array([ksg(x[:, i], x[:, j], min(int(2 * np.dot(A[i], A[j])), T // 10)) for i, j in zip(*iu)])
        out += [np.corrcoef(g, k)[0, 1], (k - g).mean()]
    else: out += [np.nan, np.nan]
    return np.array(out)
NAMES = ['switch', 'comod', 'curv', 'switch_s', 'comod_s', 'curv_s', 'zgk', 'knn_minus_gauss']
def one(x, seed, mi=True):
    return np.stack([metrics(x, mi=mi), metrics(multivariate_iaaft(x, np.random.default_rng(seed), final='spectrum')[0], mi=mi)])
if __name__ == '__main__':
    d, M, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    rows = [r for r in json.load(open(f'{d}/manifest.json'))['rows'] if r['state'] in ('awake_ec', 'awake_eo', 'drug', 'sleep', 'recovery_ec')]
    W = np.load(f'{d}/windows.npz'); sub = np.round(np.arange(M) * (rows[0]['M'] - 1) / (M - 1)).astype(int)
    res = Parallel(n_jobs=8)(delayed(one)(W[r['id']].astype(float)[:, sub], k) for k, r in enumerate(rows))
    np.savez(out, V=np.array(res), names=NAMES, delta_fraction=np.array([r.get('delta_fraction', np.nan) for r in rows], float),
             **{k: np.array([r.get(k, '') for r in rows]) for k in ('id', 'unit', 'state', 'agent', 'animal')})
