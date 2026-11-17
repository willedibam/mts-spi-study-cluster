"""Validation of lagnull: (1) null-of-null construction bias, (2) pooled-spectrum z (low |Im C| floor),
(3) zero-phase null that keeps |S| (PSD-projected) instead of Re S."""
import numpy as np, json, sys
from lagnull import xspec, feats, simulate, fullspec, BAND


def psd_abs(S):                     # zero phase, same magnitude, projected to PSD per frequency
    A = np.abs(S).astype(complex)
    for f in range(len(A)):
        w, V = np.linalg.eigh(A[f].real); A[f] = (V * np.clip(w, 0, None)) @ V.T
    return A


def pooled(xs):
    return fullspec(xs)[BAND]


if __name__ == '__main__':
    M = int(sys.argv[1])
    W = np.load('raw/nt/windows.npz'); rows = json.load(open('raw/nt/manifest.json'))['rows']
    sub = np.round(np.arange(M) * 125 / (M - 1)).astype(int); rng = np.random.default_rng(1); res = []
    keys = sorted({(r['unit'], r['state'], r['agent']) for r in rows if r['state'] in ('awake_ec', 'awake_eo', 'drug', 'sleep', 'recovery_ec')})
    for u, s, a in keys:
        xs = [W[r['id']][:, sub] for r in rows if r['unit'] == u and r['state'] == s]
        xs = [(x - x.mean(0)) / x.std(0) for x in xs]
        S = fullspec(xs)
        nS = simulate(S, len(xs), rng); nSS = simulate(fullspec(nS), len(xs), rng)          # null and null-of-null
        n0 = simulate(S.real.astype(complex), len(xs), rng); nA = simulate(psd_abs(S), len(xs), rng)
        mz = lambda L: float(np.mean([feats(xspec(x))['z'] for x in L]))
        mp = lambda L: float(np.mean([feats(xspec(x))['zphase'] for x in L]))
        d = dict(unit=u, state=s, agent=a)
        for n, L in (('real', xs), ('nS', nS), ('nSS', nSS), ('n0', n0), ('nA', nA)):
            d[n + '_z'] = mz(L); d[n + '_zp'] = mp(L); P = feats(pooled(L)); d[n + '_Pz'] = P['z']; d[n + '_Pzp'] = P['zphase']
        res.append(d); print(a, s, u[:8], ' '.join(f"{k}={v:+.3f}" for k, v in d.items() if k.endswith('_z') or k.endswith('_Pz')), flush=True)
    json.dump(res, open(f'lagnull2_{M}.json', 'w'))

