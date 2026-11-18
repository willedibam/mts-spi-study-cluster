"""Montage check: z(Gaussian MI, kNN MI) real and one joint IAAFT draw, normal-scored, on 32 index-neighbour bipolar
derivations (ch s+1 − ch s at the same 32 sites as zgk_ns.py). Bipolar of the CAR windows equals bipolar of the
as-recorded data (the common average cancels). Out V[w] = [ns_real, ns_surr].
    python zbip.py raw/nt10 zbip_A.npz"""
import sys, json, numpy as np, warnings; warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
from zgk_ns import zgk, ns
from surr_pairs import multivariate_iaaft
def one(x, seed):
    v = ns(x); return np.array([zgk(v), zgk(multivariate_iaaft(v, np.random.default_rng(seed), final='spectrum')[0])])
if __name__ == '__main__':
    d, out = sys.argv[1], sys.argv[2]
    rows = [r for r in json.load(open(f'{d}/manifest.json'))['rows'] if r['state'] in ('awake_ec', 'awake_eo', 'drug', 'sleep', 'recovery_ec')]
    W = np.load(f'{d}/windows.npz'); S = lambda m: np.round(np.arange(32) * (m - 1) / 31).astype(int)
    def bip(w, m):
        s = S(m); a = np.where(s + 1 < m, s, s - 1); return w[:, a + 1] - w[:, a]
    res = Parallel(n_jobs=7)(delayed(one)(bip(W[r['id']].astype(float), r['M']), k) for k, r in enumerate(rows))
    np.savez(out, V=np.array(res), **{k: np.array([r.get(k, '') for r in rows]) for k in ('id', 'unit', 'state', 'agent', 'animal')})
    print('done', out, len(rows), flush=True)
