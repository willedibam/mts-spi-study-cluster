import numpy as np, warnings; warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
from scipy.signal import lfilter
exec(open('decide.py').read().split("if __name__")[0])
T, M, NW = 200, 16, 40
def ar(rng, n, phi=.9): return lfilter([1], [1, -phi], rng.standard_normal(n))
def gen(kind, strength, seed):
    rng = np.random.default_rng(seed)
    L = np.array([ar(rng, T) for _ in range(4)]).T; L = (L - L.mean(0)) / L.std(0)
    W = rng.uniform(0, 1, (4, M)) * (rng.random((4, M)) < .5)            # heterogeneous linear mixing
    x = L @ W + np.array([ar(rng, T) for _ in range(M)]).T * 1.0
    half = np.arange(M) < M // 2
    if kind == 'nonlinear':      # static nonlinear coupling: a shared latent enters half the channels squared
        q = L[:, 0] ** 2 - 1; x[:, half] += strength * np.outer(q, rng.uniform(.5, 1.5, half.sum()))
    if kind == 'gain':           # linear coupling unchanged; brief shared amplitude bursts on half the channels
        u = lfilter([1], [1, -.8], rng.standard_normal(T)); env = np.exp(strength * (u - u.mean()) / u.std())
        x[:, half] *= env[:, None]
    return x
def one(kind, s, k):
    x = gen(kind, s, k); return run(x, k)
out = {}
for kind, s in (('linear', 0), ('nonlinear', .6), ('nonlinear', 1.2), ('gain', .5), ('gain', 1.0)):
    r = Parallel(n_jobs=8)(delayed(one)(kind, s, 1000 * hash(kind) % 997 + k) for k in range(NW))
    g = lambda key: np.array([x[key] for x in r])
    out[(kind, s)] = dict(real=g('real').mean(), resid=(g('real') - g('glob')).mean(), resid_q=(g('real') - g('quarter')).mean(),
                          gain_resid=(g('gain_real') - g('gain_glob')).mean(), se=(g('real') - g('glob')).std() / np.sqrt(NW))
base = out[('linear', 0)]
print(f"{'generator':18s} {'z real':>7s} {'real−IAAFT':>11s} {'real−quarterIAAFT':>18s} {'gain-norm real−IAAFT':>21s}   (minus linear baseline)")
for k, v in out.items():
    print(f"{k[0]+' '+str(k[1]):18s} {v['real']:7.3f} {v['resid']-base['resid']:+11.3f} {v['resid_q']-base['resid_q']:+18.3f} {v['gain_resid']-base['gain_resid']:+21.3f}   se≈{v['se']:.3f}")
