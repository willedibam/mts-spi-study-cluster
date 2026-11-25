"""Unrounded pooled drug Δz, T = 200 calibration thresholds (offset − 2 SE) and sleep Δz in own eyes-open SD for the rb arms (robustness-result.md). Run in the mi_pair folder of the robust snapshot on Gadi; paths are the gdata outputs."""
import sys, glob, io, contextlib, numpy as np
sys.argv = ['x']; import robust as R
d = '/g/data/ql44/we2614/mts-spi-study/representation/atlas-mi-pair-261117/robust'
# calibration offsets, unrounded
D = np.load(f'{d}/rcalib.npz'); X, kind, sset = D['X'].astype(float), D['kind'], D['set']
off = {}
for T in (1000, 200):
    C = {s: R.configs(X[sset == s], T, True) for s in np.unique(sset) if s.endswith(f'T{T}')}
    for n in ('w0 k4', 'auto k4', 'kshift', 'auto k10'):
        o = {}
        for tag in ('AW', 'KTMD', 'PF'):
            s = f'{tag}-T{T}'; Xs, ks = X[sset == s], kind[sset == s]; K = C[s][n][0]
            b = np.array([R.zc(Xs[w, 0, 0], K[w, 0]) - R.zc(Xs[w, 1, 0], K[w, 1]) for w in np.nonzero(ks == 'block')[0]])
            o[tag] = (b.mean(), b.std(ddof=1) / np.sqrt(len(b)))
        dd = np.mean([o['KTMD'][0], o['PF'][0]]) - o['AW'][0]; se = np.sqrt((o['KTMD'][1] ** 2 + o['PF'][1] ** 2) / 4 + o['AW'][1] ** 2)
        off[(T, n)] = (dd, se); print(f'calib T{T} {n}: offset {dd:+.5f} se {se:.5f} threshold {dd - 2 * se:+.5f}')
for prefix in ('rb10', 'rbS2', 'rbC2'):
    L = [np.load(f) for f in sorted(glob.glob(f'{d}/{prefix}_[AB]_*.npz')) + sorted(glob.glob(f'{d}/{prefix}_S_*.npz'))]
    cat = lambda k: np.concatenate([q[k] for q in L]); X = np.concatenate([q['X'] for q in L]).astype(float); T = int(L[0]['T'])
    u, st, ag, an = cat('unit'), cat('state'), cat('agent'), cat('animal'); sh = bool(np.isfinite(X[:, :, R.CSH]).all())
    C = R.configs(X, T, sh)
    for n in C:
        if n.endswith('pool') and n.startswith('w0'): continue
        K = C[n][0]; Z = np.array([[R.zc(X[w, rs, 0], K[w, rs]) for rs in (0, 1)] for w in range(len(X))]); res = Z[:, 0] - Z[:, 1]
        s = R.stats(res, u, st, ag, an); g, a2, dz, eo = R.session_deltas(res, u, st, ag, an)
        slp = dz[g == 'SLP']; thr = off.get((T, n))
        extra = f' | calib thr {thr[0] - 2 * thr[1]:+.5f} -> {"PASS" if s["pooled"] < thr[0] - 2 * thr[1] else "fail"} ({(s["pooled"] - thr[0]) / thr[1]:+.2f} SE)' if T == 200 and thr else ''
        print(f'{prefix} {n:13s} pooled(sessions) {s["pooled"]:+.5f} animal-mean {s["A"].mean():+.5f} sd {s["sd"]:.5f} | sleep {slp.mean():+.5f} ({slp.mean() / s["sd"]:+.2f} SD, {np.sum(slp < 0)}/{len(slp)}){extra}')
