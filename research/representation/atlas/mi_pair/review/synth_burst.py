"""Sign check: linear Gaussian network (16 ch, AR(2) sources mixed by random loadings) vs the same network under a shared
on/off gain (bursts over an independent noise floor), average-referenced and normal-scored; z residual against joint IAAFT (3 draws)."""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
from zgk_ns import zgk, ns
from surr_pairs import multivariate_iaaft
from mech import ar, episodes
T, M = 1000, 16
def make(kind, rng):
    S = np.column_stack([ar(T, (1.6, -.7), rng) for _ in range(6)]); x = S @ rng.standard_normal((6, M))
    x = x + 0.5 * x.std() * np.column_stack([ar(T, (1.6, -.7), rng) for _ in range(M)])
    if kind == 'burst':
        g = np.where(episodes(T, .4, 80, rng), 1.0, 0.15)[:, None]; x = g * x + 0.15 * x.std() * rng.standard_normal((T, M))
    x = x - x.mean(1, keepdims=True); return ns(x[:, :-1])
def one(kind, seed):
    rng = np.random.default_rng(seed); v = make(kind, rng); bi = np.median(1.4826 * np.median(np.abs(v - np.median(v, 0)), 0) / v.std(0))
    return zgk(v) - np.mean([zgk(multivariate_iaaft(v, np.random.default_rng(seed + 100 + d), final='spectrum')[0]) for d in range(3)])
for kind in ('linear', 'burst'):
    r = np.array(Parallel(n_jobs=4)(delayed(one)(kind, s) for s in range(12)))
    print(f'{kind:7s} residual mean {r.mean():+.4f} ± {r.std(ddof=1) / np.sqrt(len(r)):.4f} (n={len(r)})')
