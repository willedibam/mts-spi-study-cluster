"""32-channel known-answer calibration of gate.block_feats beyond joint IAAFT (spectrum-final), normal-scored, 2 draws.
Kinds: linear (stationary mixing), local (independent per-channel bursts), gain (shared slow envelope on the shared part,
fixed noise floor), mix (gain + local). Edges: top quartile |r| of the real window, held fixed for its surrogates."""
import time, numpy as np, warnings; warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
from surr_pairs import multivariate_iaaft
from zgk_ns import ns
from mech import ar, episodes
from gate import block_feats, top_edges
M, T, K, PH = 32, 1000, 4, (1.6, -.7)
def synth(kind, rng):
    s = np.column_stack([ar(T, PH, rng) for _ in range(K)]); A = rng.normal(size=(M, K)) * rng.uniform(.3, 1.5, (M, 1))
    sh = s @ A.T; sh /= sh.std(0); n = np.column_stack([ar(T, PH, rng) for _ in range(M)]) * .6
    if kind in ('gain', 'mix'): env = np.exp(ar(T, (.995, 0), rng)); sh = sh * (env / env.std())[:, None]
    x = sh + n
    if kind in ('local', 'mix'):
        x = x + np.column_stack([episodes(T, .2, 25, rng)[:T] * 2.5 * ar(T, PH, rng) for _ in range(M)])
    return x
def one(kind, seed):
    rng = np.random.default_rng(seed); x = ns(synth(kind, rng)); E = top_edges([x]); f = block_feats(x, E)
    S = [block_feats(multivariate_iaaft(x, np.random.default_rng(seed * 10 + d), final='spectrum')[0], E) for d in range(2)]
    return f - np.mean(S, 0)
if __name__ == "__main__":
    t = time.time(); one('linear', 999); print(f'one window (2 IAAFT draws): {time.time() - t:.1f} s', flush=True)
    for kind in ('linear', 'local', 'gain', 'mix'):
        R = np.array(Parallel(n_jobs=7)(delayed(one)(kind, s) for s in range(14)))
        print(f'{kind:7s} ' + '  '.join(f'{n} {R[:, i].mean():+.3f} (SD {R[:, i].std(ddof=1):.3f}, t {R[:, i].mean() / R[:, i].std(ddof=1) * np.sqrt(len(R)):+.1f})' for i, n in enumerate(('gate', 'imbal', 'switch'))), flush=True)
