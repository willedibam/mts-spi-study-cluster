"""Axis screen compute (prereg.md): edge profiles of the stats.py statistics for every window of a bank, on the
normal-scored window and on six nulls, reduced to per-window z matrices and per-group reliabilities.

Versions per window (all from the normal-scored window v):
  real; iaaft0..2: spectrum-final joint multivariate IAAFT of v, seed [extract.seed(id), 10, b];
  stationary / zero_phase / bias: discovery.null_bank's three Gaussian nulls of the window's group (unit x state, sleep
  split into within-archive delta tertiles), built from the group's normal-scored windows.
Per window and version: z over the statistics (all channels), z of double-centred profiles (c + a_i + a_j removed),
node-additivity R^2, and on a 32-channel subset (as the MI anchor) z including KSG MI (k 4, DCE AUTO; mi_pair/edges.py).
Per group: between-window reliability of each statistic (mean pairwise profile correlation) and z of window-mean
profiles, per version. Profiles are kept (float32) for post hoc work.

  python -m research.representation.atlas.axis_screen.run group-count <bank>
  python -m research.representation.atlas.axis_screen.run compute <bank> <out dir> <workers> [--prios] [--sub32]
  python -m research.representation.atlas.axis_screen.run collect <out dir> <out.npz>
"""
import sys, json, zlib, glob, time
import numpy as np
from pathlib import Path
from multiprocessing import Pool

VERSIONS = ('real', 'iaaft0', 'iaaft1', 'iaaft2', 'stationary', 'zero_phase', 'bias')
B_IAAFT = 3


def _ns(x):
    from scipy.stats import norm, rankdata
    return norm.ppf((np.apply_along_axis(rankdata, 0, x) - .5) / len(x))


def _groups(rows):
    from research.representation.character_axes.episodic import sleep_tertiles
    sub = sleep_tertiles(rows) if any(r['state'] == 'sleep' for r in rows) else [r['state'] for r in rows]
    g = {}
    for r, s in zip(rows, sub):
        g.setdefault((r['unit'], r['state']), []).append((r['id'], s))
    return sorted(g.items())


_SYMPINV = {}


def _sym_projector(M):
    if M not in _SYMPINV:
        a, b = np.triu_indices(M, 1)
        X = np.zeros((len(a), M + 1))
        X[:, 0] = 1
        X[np.arange(len(a)), 1 + a] = 1
        X[np.arange(len(a)), 1 + b] = 1
        _SYMPINV[M] = (X, np.linalg.pinv(X))
    return _SYMPINV[M]


def _zmat(P):
    """Correlation between rows of P (S, E) over edges finite in every row."""
    ok = np.isfinite(P).all(0)
    X = P[:, ok] - P[:, ok].mean(1, keepdims=True)
    n = np.linalg.norm(X, axis=1)
    with np.errstate(all='ignore'):
        Y = X / n[:, None]
    return (Y @ Y.T).astype(np.float32), float(ok.mean())


def _reduce(P, M):
    """z, double-centred z, node R^2 for one window-version (P: S x E upper-triangle profiles)."""
    z, fin = _zmat(P)
    X, pinv = _sym_projector(M)
    ok = np.isfinite(P).all(0)
    Q = np.where(ok, P, 0.)
    fit = (pinv @ Q.T).T @ X.T
    R = P - fit
    ss = ((P - P.mean(1, keepdims=True)) ** 2).sum(1)
    r2 = 1 - (R ** 2).sum(1) / ss
    zdc, _ = _zmat(R)
    return z, zdc, r2.astype(np.float32), fin


def _covariates(x_raw, v, fs, row):
    from research.representation.atlas.discovery import spectral
    M = v.shape[1]
    C = np.corrcoef(v.T)
    r = np.abs(C[np.triu_indices(M, 1)])
    lam = np.linalg.eigvalsh(C)
    ent, peak = spectral(x_raw, fs) if x_raw is not None else (np.nan, np.nan)
    burst = (float(np.median(1.4826 * np.median(np.abs(x_raw - np.median(x_raw, 0)), 0) / x_raw.std(0)))
             if x_raw is not None else row.get('burst', np.nan))
    return [r.mean(), r.std(), lam.sum() ** 2 / (lam ** 2).sum(), ent, peak, row.get('delta_fraction', np.nan), burst]


COVARIATE_NAMES = ('mean_abs_r', 'sd_abs_r', 'cov_pr', 'spectral_entropy', 'peak_hz', 'delta_fraction', 'burst_index')


def group(args):
    bank, out, unit, state, members, rows, fs, prios, sub32 = args
    from research.representation.atlas.extract import seed
    from research.representation.atlas.discovery import _mean_periodogram, _zero_phase, gaussian_draws
    from research.representation.atlas.axis_screen.stats import profiles, KEYS
    from mtsspi.surrogates import multivariate_iaaft
    target = Path(out) / f'{unit}__{state}.npz'
    if target.exists():
        return str(target), 0.
    t0 = time.time()
    ids = [i for i, _ in members]
    with np.load(Path(bank) / 'windows.npz') as W:
        raw = [np.asarray(W[i], np.float64) for i in ids]
    if prios:  # PRIOS windows are already normal-scored and may differ in channels: nulls on the group's common set
        V = raw
        raw = [None] * len(V)
    else:
        V = [_ns(x) for x in raw]
    M = {v.shape[1] for v in V}
    assert len(M) == 1, (unit, state, M)
    M = M.pop()
    n, T = len(V), V[0].shape[0]
    ver = {k: [None] * n for k in VERSIONS}
    ver['real'] = V
    for w, (wid, v) in enumerate(zip(ids, V)):
        for b in range(B_IAAFT):
            ver[f'iaaft{b}'][w] = multivariate_iaaft(v, np.random.default_rng([seed(wid), 10, b]), final='spectrum')[0]
    for sub in sorted({s for _, s in members}):
        ix = [w for w, (_, s) in enumerate(members) if s == sub]
        X = np.stack([V[w] for w in ix])
        S = _mean_periodogram(X)
        for kind in ('stationary', 'zero_phase', 'bias'):
            rng = np.random.default_rng([0, zlib.crc32(f'{unit}|{sub}|{kind}|axis'.encode())])
            Sk = _zero_phase(S) if kind == 'zero_phase' else _mean_periodogram(gaussian_draws(S, len(ix), T, rng)) if kind == 'bias' else S
            for w, y in zip(ix, gaussian_draws(Sk, len(ix), T, rng)):
                ver[kind][w] = y
    a, b_ = np.triu_indices(M, 1)
    s32 = np.round(np.arange(32) * (M - 1) / 31).astype(int) if sub32 else np.arange(M)
    e32 = np.flatnonzero(np.isin(a, s32) & np.isin(b_, s32))
    S_ = len(KEYS)
    Pall = np.full((n, len(VERSIONS), S_, len(a)), np.nan, np.float32)
    Z = np.full((n, len(VERSIONS), S_, S_), np.nan, np.float32); Zdc = Z.copy()
    Z32 = np.full((n, len(VERSIONS), S_ + 1, S_ + 1), np.nan, np.float32)
    R2 = np.full((n, len(VERSIONS), S_), np.nan, np.float32)
    K32 = np.full((n, len(VERSIONS), len(e32)), np.nan, np.float32)
    fin = np.full((n, len(VERSIONS)), np.nan)
    cov = np.array([_covariates(raw[w], V[w], fs, rows[ids[w]]) for w in range(n)])
    from research.representation.atlas.mi_pair.edges import edge_mi
    for w in range(n):
        for k, name in enumerate(VERSIONS):
            x = ver[name][w]
            P = profiles(x)
            P = np.array([P[key] for key in KEYS])
            Pall[w, k] = P
            Z[w, k], Zdc[w, k], R2[w, k], fin[w, k] = _reduce(P, M)
            _, kk = edge_mi(x[:, s32])
            K32[w, k] = kk
            Z32[w, k] = _zmat(np.vstack([P[:, e32], kk[None]]))[0]
    rel = np.full((len(VERSIONS), S_ + 1), np.nan, np.float32)
    zmean = np.full((len(VERSIONS), S_, S_), np.nan, np.float32)
    if n >= 3:
        for k in range(len(VERSIONS)):
            for s in range(S_):
                c = np.corrcoef(Pall[:, k, s][:, np.isfinite(Pall[:, k, s]).all(0)])
                rel[k, s] = c[np.triu_indices(n, 1)].mean()
            c = np.corrcoef(K32[:, k])
            rel[k, S_] = c[np.triu_indices(n, 1)].mean()
            zmean[k] = _zmat(np.nanmean(Pall[:, k], 0))[0]
    np.savez(target, ids=np.array(ids), sub=np.array([s for _, s in members]), Z=Z, Zdc=Zdc, Z32=Z32, R2=R2, fin=fin,
             cov=cov, rel=rel, zmean=zmean, M=M, keys=np.array(KEYS), versions=np.array(VERSIONS), e32=e32)
    np.save(Path(out) / 'profiles' / f'{unit}__{state}.npy', Pall)
    np.save(Path(out) / 'profiles' / f'{unit}__{state}__ksg32.npy', K32)
    return str(target), time.time() - t0


def jobs(bank, out, prios=False, sub32=True):
    man = json.loads((Path(bank) / 'manifest.json').read_text())
    rows = {r['id']: r for r in man['rows']}
    fs = man.get('fs', 100)
    J = []
    for (unit, state), members in _groups(man['rows']):
        J.append((str(bank), str(out), unit, state, members, {i: rows[i] for i, _ in members}, fs, prios, sub32))
    return sorted(J, key=lambda j: -len(j[4]))  # largest groups first


def compute(bank, out, workers, prios=False, sub32=True):
    (Path(out) / 'profiles').mkdir(parents=True, exist_ok=True)
    J = jobs(bank, out, prios, sub32)
    print('groups', len(J), 'windows', sum(len(j[4]) for j in J), flush=True)
    with Pool(workers, maxtasksperchild=1) as p:
        for path, sec in p.imap_unordered(group, J):
            print(f'{sec:7.0f}s {path}', flush=True)


def collect(out, target):
    fs = sorted(glob.glob(f'{out}/*__*.npz'))
    D = [np.load(f) for f in fs]
    keys = [str(k) for k in D[0]['keys']]
    per_w = {k: np.concatenate([d[k] for d in D]) for k in ('ids', 'sub', 'Z', 'Zdc', 'Z32', 'R2', 'fin', 'cov')}
    grp = np.concatenate([[Path(f).stem] * len(d['ids']) for f, d in zip(fs, D)])
    np.savez(target, **per_w, group=grp, groups=np.array([Path(f).stem for f in fs]), rel=np.stack([d['rel'] for d in D]),
             zmean=np.stack([d['zmean'] for d in D]), M=np.array([int(d['M']) for d in D]), keys=np.array(keys),
             versions=D[0]['versions'])
    print('collected', len(fs), 'groups', len(per_w['ids']), 'windows ->', target)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'group-count':
        J = jobs(sys.argv[2], '/tmp')
        print(len(J), sum(len(j[4]) for j in J))
    elif cmd == 'compute':
        compute(sys.argv[2], sys.argv[3], int(sys.argv[4]), prios='--prios' in sys.argv, sub32='--sub32' in sys.argv)
    elif cmd == 'collect':
        collect(sys.argv[2], sys.argv[3])
