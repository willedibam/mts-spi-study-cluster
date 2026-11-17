"""Post hoc: the same calibration with an average reference (last channel dropped, as atlas reference('car')) before normal-scoring."""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
from gate_calib import synth, ns, top_edges, block_feats, multivariate_iaaft
import sys; sys.path.insert(0, '/Users/wedi0306/Code/mts-spi-study-cluster/.claude/worktrees/real-world-apps-9b53a1')
from research.representation.atlas.prepare import reference
def one(kind, seed):
    rng = np.random.default_rng(seed); x = ns(reference(synth(kind, rng), 'car')); E = top_edges([x]); f = block_feats(x, E)
    S = [block_feats(multivariate_iaaft(x, np.random.default_rng(seed * 10 + d), final='spectrum')[0], E) for d in range(2)]
    return f - np.mean(S, 0)
for kind in ('linear', 'local', 'gain', 'mix'):
    R = np.array(Parallel(n_jobs=7)(delayed(one)(kind, s) for s in range(14)))
    print(f'CAR {kind:7s} gate {R[:, 0].mean():+.3f} (SD {R[:, 0].std(ddof=1):.3f})', flush=True)
