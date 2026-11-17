"""z(Gaussian MI, kNN MI) real and 3 joint-IAAFT draws per selected Claar window (zgk, ns imported unchanged from zgk_ns.py).
Variants: ns  = average reference (last channel dropped), normal-scored  [primary]
          raw = average reference, not normal-scored                      [secondary]
          asis = as-recorded reference, normal-scored                     [secondary]
    python claar_z.py SEL_DIR OUT.npz VARIANT[,VARIANT...]
Out per window: Z[variant] = [real, surr1, surr2, surr3]."""
import sys, json, glob, os, zlib
import numpy as np, warnings; warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zgk_ns import zgk, ns
from surr_pairs import multivariate_iaaft
sys.path.insert(0, '/Users/wedi0306/Code/mts-spi-study-cluster/.claude/worktrees/real-world-apps-9b53a1')
from research.representation.atlas.prepare import reference
PREP = dict(ns=lambda w: ns(reference(w, 'car')), raw=lambda w: reference(w, 'car'), asis=lambda w: ns(w))


def one(w, seed, variant):
    v = PREP[variant](w.astype(np.float64))
    return [zgk(v)] + [zgk(multivariate_iaaft(v, np.random.default_rng(seed + d), final='spectrum')[0]) for d in range(3)]


if __name__ == '__main__':
    src, out, variants = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
    jobs, rows = [], []
    for fn in sorted(glob.glob(f'{src}/sub-*.npz')):
        Z = np.load(fn); R = json.loads(str(Z['rows'])); s = os.path.basename(fn)[:-4]
        for i, r in enumerate(R):
            rows.append(dict(r, session=s)); jobs.append((fn, i, zlib.crc32(f'{s}:{i}'.encode()) * 10))
    res = {}
    for var in variants:
        res[var] = np.array(Parallel(n_jobs=7, verbose=5)(delayed(one)(np.load(fn)['W'][i], seed, var) for fn, i, seed in jobs))
        np.savez(out, rows=json.dumps(rows), **res)
        print('done', var, res[var].shape, flush=True)
