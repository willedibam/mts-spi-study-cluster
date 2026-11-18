"""Post hoc: shared gain on the coupled part plus a spatially correlated (common-mode-heavy) floor, with and without CAR."""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
from gate_calib import ns, top_edges, block_feats, multivariate_iaaft, M, T, K, PH
from mech import ar
import sys; sys.path.insert(0, '/Users/wedi0306/Code/mts-spi-study-cluster/.claude/worktrees/real-world-apps-9b53a1')
from research.representation.atlas.prepare import reference
def synth(rng, floor_common):
    s = np.column_stack([ar(T, PH, rng) for _ in range(K)]); A = rng.normal(size=(M, K)) * rng.uniform(.3, 1.5, (M, 1))
    sh = s @ A.T; sh /= sh.std(0); env = np.exp(ar(T, (.995, 0), rng)); sh = sh * (env / env.std())[:, None]
    c = ar(T, PH, rng)[:, None] * rng.uniform(.8, 1.2, (1, M)); n = np.column_stack([ar(T, PH, rng) for _ in range(M)])
    return sh + .6 * (np.sqrt(floor_common) * c + np.sqrt(1 - floor_common) * n)
def one(fc, car, seed):
    rng = np.random.default_rng(seed); x = synth(rng, fc); x = ns(reference(x, 'car') if car else x); E = top_edges([x])
    return block_feats(x, E)[0] - np.mean([block_feats(multivariate_iaaft(x, np.random.default_rng(seed * 10 + d), final='spectrum')[0], E)[0] for d in range(2)])
for fc in (0, .5, .9):
    for car in (False, True):
        R = np.array(Parallel(n_jobs=7)(delayed(one)(fc, car, s) for s in range(14)))
        print(f'gain, floor common share {fc:.1f}, {"CAR " if car else "asis"}: gate {R.mean():+.3f} (SD {R.std(ddof=1):.3f})', flush=True)
