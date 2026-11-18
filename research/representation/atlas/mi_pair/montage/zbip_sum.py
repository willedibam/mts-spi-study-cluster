"""Bipolar montage vs CAR (zns_*): per-session Δ = state − awake_ec of the normal-scored residual; same units, side by side."""
import glob, numpy as np
from scipy.stats import binom
def load(pat, vi):
    L = [np.load(f) for f in sorted(glob.glob(pat))]; cat = lambda k: np.concatenate([d[k] for d in L]); V = np.concatenate([d['V'] for d in L])
    res = V[vi] if False else (V[:, 1, 0] - V[:, 1, 1] if V.ndim == 3 else V[:, 0] - V[:, 1])
    out = {}
    for u in np.unique(cat('unit')):
        un, st, ag, an = cat('unit') == u, cat('state'), cat('agent'), cat('animal')
        m = lambda s: un & (st == s); b = res[m('awake_ec')].mean(); a = ag[un][0]
        out[u] = dict(ag=a, an=an[un][0], **{s: res[m(s)].mean() - b for s in ('drug', 'sleep', 'awake_eo', 'recovery_ec') if m(s).any()})
    return out
C, B = load('zns_[AB]_*.npz', 1), load('zbip_[AB].npz', 0)
for name, R in (('CAR', C), ('bipolar', B)):
    eo = np.array([R[u]['awake_eo'] for u in R]); sd = eo.std(ddof=1); print(f'{name}: eyes-open null SD {sd:.4f} (n={len(eo)})')
    for a in ('KT', 'KTMD', 'MD', 'PF', 'SLP'):
        s = 'sleep' if a == 'SLP' else 'drug'; us = [u for u in R if R[u]['ag'] == a]; d = np.array([R[u][s] for u in us]); n = len(d); neg = int((d < 0).sum())
        rc = np.nanmean([R[u].get('recovery_ec', np.nan) for u in us]) if a != 'SLP' else np.nan
        print(f'  {a:5s} n={n:2d} Δ {d.mean():+.4f} ({d.mean() / sd:+.1f} SD) {neg}/{n} negative (p={binom.sf(neg - 1, n, .5):.3f}); recovery {rc:+.4f}')
common = [u for u in C if u in B]; x = np.array([C[u].get('drug', np.nan) for u in common]); y = np.array([B[u].get('drug', np.nan) for u in common]); k = np.isfinite(x)
print(f'per-session drug Δ, CAR vs bipolar: r = {np.corrcoef(x[k], y[k])[0, 1]:+.2f} over {k.sum()} sessions')
