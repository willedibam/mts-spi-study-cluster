"""Chatterjee's xi on the axis screen's windows (post hoc; criteria in xi_screen.md, committed before the run).

xi is added to the screen's 33 statistics and KSG without recomputing them: per window and version the screen's stored
profiles (axis_screen/<arm>/profiles/<group>.npy, rows stats.KEYS; <group>__ksg32.npy) are stacked with four xi rows
computed here on the same versions, regenerated with the screen's code and seeds (run.py: normal-scored window, 3
spectrum-final joint IAAFT draws, stationary / zero-phase / bias Gaussian nulls). Reproduction is checked per window
and version: the r profile and the old block of z must match the stored ones.

  xi       (xi(i -> j) + xi(j -> i)) / 2 at lag 0 (pyspi xi_tau-0, symmetrised)
  xi_g     its Gaussian value from r: (3/pi) arcsin((1 + r^2)/2) - 1/2 (identities.json)
  xi_asym  |xi(i -> j) - xi(j -> i)| at lag 0 (0 in expectation under any Gaussian process)
  xi1      symmetrised xi at lag 1 (pyspi xi_tau-1)

Outputs per group follow run.py's format with keys = stats.KEYS + XI_KEYS, so the frozen scorer (score.py) runs on the
collected npz, with the xi rows declared higher-order (F1, real - IAAFT) except xi_g, which is a function of r.

  python -m research.representation.atlas.pair_catalogue.xi_screen compute <bank> <screen dir> <out dir> <workers>
  python -m research.representation.atlas.pair_catalogue.xi_screen collect <out dir> <out.npz>
  python -m research.representation.atlas.pair_catalogue.xi_screen score <xi-screen.npz> <manifest.json> <out.json>
"""
import sys, json, glob, time, zlib
import numpy as np
from pathlib import Path
from multiprocessing import Pool

XI_KEYS = ('xi', 'xi_g', 'xi_asym', 'xi1')
XI_NAMES = {'xi': 'xi_tau-0 (symmetrised)', 'xi_g': 'xi Gaussian value of r', 'xi_asym': '|xi_tau-0 asymmetry|', 'xi1': 'xi_tau-1 (symmetrised)'}
XI_HIGHER = {'xi', 'xi_asym', 'xi1'}
PRIMARY = 'xi_g|xi'
SECONDARY = ('r2|xi', 'rho|xi', 'dcorr|xi', 'mcd2|xi', 'xi|ksg', 'xi_g|xi_asym', 'xi|xi1')


def xi_gauss(r):
    return 3 / np.pi * np.arcsin((1 + r ** 2) / 2) - .5


def xi_matrix(x, tau=0):
    """A[i, j] = xi(source x_i[t], target x_j[t + tau]) for every ordered pair, without ties (continuous data):
    sort by the source, then 1 - 3 sum |r_{k+1} - r_k| / (n^2 - 1) over the target's ranks (pyspi's general form
    reduces to this when there are no ties)."""
    T, M = x.shape
    n = T - tau
    R = np.argsort(np.argsort(x[tau:], 0, kind='stable'), 0, kind='stable').astype(np.int32)
    A = np.empty((M, M))
    for i in range(M):
        o = np.argsort(x[:n, i], kind='stable')
        A[i] = 1 - 3 * np.abs(np.diff(R[o], axis=0)).sum(0) / (n * n - 1)
    return A


def xi_profiles(x):
    """(4, E) xi rows on the upper triangle (run.py order), and the r profile for the reproduction check."""
    M = x.shape[1]
    a, b = np.triu_indices(M, 1)
    z = (x - x.mean(0)) / x.std(0)
    r = (z.T @ z / len(z))[a, b]
    A0, A1 = xi_matrix(x), xi_matrix(x, 1)
    return np.stack([(A0[a, b] + A0[b, a]) / 2, xi_gauss(r), np.abs(A0[a, b] - A0[b, a]), (A1[a, b] + A1[b, a]) / 2]), r


def versions(bank, ids, subs, unit):
    """The screen's seven versions of each window (run.group, non-PRIOS branch, same seeds and order of draws)."""
    from research.representation.atlas.extract import seed
    from research.representation.atlas.discovery import _mean_periodogram, _zero_phase, gaussian_draws
    from research.representation.atlas.axis_screen.run import VERSIONS, B_IAAFT, _ns
    from mtsspi.surrogates import multivariate_iaaft
    with np.load(Path(bank) / 'windows.npz') as W:
        V = [_ns(np.asarray(W[i], np.float64)) for i in ids]
    n, T = len(V), V[0].shape[0]
    ver = {k: [None] * n for k in VERSIONS}
    ver['real'] = V
    for w, (wid, v) in enumerate(zip(ids, V)):
        for b in range(B_IAAFT):
            ver[f'iaaft{b}'][w] = multivariate_iaaft(v, np.random.default_rng([seed(wid), 10, b]), final='spectrum')[0]
    for sub in sorted(set(subs)):
        ix = [w for w, s in enumerate(subs) if s == sub]
        X = np.stack([V[w] for w in ix])
        S = _mean_periodogram(X)
        for kind in ('stationary', 'zero_phase', 'bias'):
            rng = np.random.default_rng([0, zlib.crc32(f'{unit}|{sub}|{kind}|axis'.encode())])
            Sk = _zero_phase(S) if kind == 'zero_phase' else _mean_periodogram(gaussian_draws(S, len(ix), T, rng)) if kind == 'bias' else S
            for w, y in zip(ix, gaussian_draws(Sk, len(ix), T, rng)):
                ver[kind][w] = y
    return ver


def group(args):
    bank, screen, out, g = args
    from research.representation.atlas.axis_screen.run import VERSIONS, _reduce, _zmat
    from research.representation.atlas.axis_screen.stats import KEYS
    target = Path(out) / f'{g}.npz'
    if target.exists():
        return str(target), 0., None
    t0 = time.time()
    G = np.load(Path(screen) / f'{g}.npz')
    ids, subs = [str(i) for i in G['ids']], [str(s) for s in G['sub']]
    unit = g.split('__')[0]
    P0 = np.load(Path(screen) / 'profiles' / f'{g}.npy')
    K = np.load(Path(screen) / 'profiles' / f'{g}__ksg32.npy')
    assert P0.shape[:3] == (len(ids), len(VERSIONS), len(KEYS)), (g, P0.shape)
    ver = versions(bank, ids, subs, unit)
    n, S_ = len(ids), len(KEYS) + len(XI_KEYS)
    M = ver['real'][0].shape[1]
    a, b_ = np.triu_indices(M, 1)
    s32 = np.round(np.arange(32) * (M - 1) / 31).astype(int)
    e32 = np.flatnonzero(np.isin(a, s32) & np.isin(b_, s32))
    X = np.full((n, len(VERSIONS), len(XI_KEYS), len(a)), np.nan, np.float32)
    Z = np.full((n, len(VERSIONS), S_, S_), np.nan, np.float32); Zdc = Z.copy()
    Z32 = np.full((n, len(VERSIONS), S_ + 1, S_ + 1), np.nan, np.float32)
    R2 = np.full((n, len(VERSIONS), S_), np.nan, np.float32)
    fin = np.full((n, len(VERSIONS)), np.nan)
    r_err = np.full((n, len(VERSIONS)), np.nan); z_err = r_err.copy()
    ri = KEYS.index('r')
    for w in range(n):
        for k, name in enumerate(VERSIONS):
            x = ver[name][w]
            if x is None:
                continue
            X[w, k], r = xi_profiles(x)
            r_err[w, k] = np.nanmax(np.abs(r - P0[w, k, ri]))
            P = np.vstack([P0[w, k].astype(float), X[w, k]])
            Z[w, k], Zdc[w, k], R2[w, k], fin[w, k] = _reduce(P, M)
            z_err[w, k] = np.nanmax(np.abs(Z[w, k, :len(KEYS), :len(KEYS)] - G['Z'][w, k]))
            Z32[w, k] = _zmat(np.vstack([P[:, e32], K[w, k][None]]))[0]
    rel = np.full((len(VERSIONS), S_ + 1), np.nan, np.float32)
    rel[:, :len(KEYS)], rel[:, S_] = G['rel'][:, :len(KEYS)], G['rel'][:, len(KEYS)]
    zmean = np.full((len(VERSIONS), S_, S_), np.nan, np.float32)
    if n >= 3:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        for k in range(len(VERSIONS)):
            for s in range(len(XI_KEYS)):
                c = [np.corrcoef(X[i, k, s], X[j, k, s])[0, 1] for i, j in pairs]
                rel[k, len(KEYS) + s] = np.nanmean(c)
            zmean[k] = _zmat(np.vstack([np.nanmean(P0[:, k].astype(float), 0), np.nanmean(X[:, k].astype(float), 0)]))[0]
    lev = np.stack([np.nanmean(X[:, :, 0] - X[:, :, 1], -1), np.nanmean(X[:, :, 2], -1), np.nanmean(X[:, :, 0], -1)], -1)
    np.savez(target, ids=np.array(ids), sub=np.array(subs), Z=Z, Zdc=Zdc, Z32=Z32, R2=R2, fin=fin, cov=G['cov'], rel=rel,
             zmean=zmean, M=M, keys=np.array(KEYS + XI_KEYS), versions=np.array(VERSIONS), r_err=r_err, z_err=z_err, lev=lev)
    np.save(Path(out) / 'profiles' / f'{g}.npy', X)
    return str(target), time.time() - t0, (float(np.nanmax(r_err)), float(np.nanmax(z_err)))


def compute(bank, screen, out, workers):
    (Path(out) / 'profiles').mkdir(parents=True, exist_ok=True)
    groups = sorted(Path(f).stem for f in glob.glob(f'{screen}/*__*.npz'))
    sizes = {g: len(np.load(Path(screen) / f'{g}.npz')['ids']) for g in groups}
    J = [(str(bank), str(screen), str(out), g) for g in sorted(groups, key=lambda g: -sizes[g])]
    print('groups', len(J), 'windows', sum(sizes.values()), flush=True)
    with Pool(workers, maxtasksperchild=1) as p:
        for path, sec, err in p.imap_unordered(group, J):
            print(f'{sec:7.0f}s {path} r_err/z_err {err}', flush=True)


def collect(out, target):
    from research.representation.atlas.axis_screen.run import collect as run_collect
    run_collect(out, target)
    fs = sorted(glob.glob(f'{out}/*__*.npz'))
    D = [np.load(f) for f in fs]
    extra = {k: np.concatenate([d[k] for d in D]) for k in ('r_err', 'z_err', 'lev')}
    base = dict(np.load(target))
    np.savez(target, **base, **extra)
    print('reproduction: max |r - stored r|', float(np.nanmax(extra['r_err'])), 'max |z - stored z|', float(np.nanmax(extra['z_err'])))


def _scorer():
    """The frozen scorer (axis_screen/score.py) with the xi rows declared and the pre-stated xi coordinates added to its
    named list; nothing else in it changes."""
    import types
    from research.representation.atlas.axis_screen import score as S
    src = Path(S.__file__).read_text()
    old = "named = ['linf|l2',"
    assert src.count(old) == 1
    src = src.replace(old, "named = NAMED_XI + ['linf|l2',")
    mod = types.ModuleType('xi_score')
    mod.__dict__['NAMED_XI'] = [PRIMARY, *SECONDARY]
    exec(compile(src, S.__file__, 'exec'), mod.__dict__)
    mod.PYSPI.update(XI_NAMES)
    mod.HIGHER |= XI_HIGHER
    return mod


def score(path, manifest, output):
    mod = _scorer()
    mod.score(path, manifest, output)
    res = json.load(open(output))
    D, keys, gi, I = mod.load(path, manifest)
    unit, state, animal, agent = I['unit'], I['state'], I['animal'], I['agent']
    sessions, animals = res['sessions'], res['animals']
    from scipy.stats import ttest_1samp
    lev = D['lev']  # (n, 7, 3): mean (xi - xi_g), mean xi_asym, mean xi over edges
    levels = {}
    for c, nm in enumerate(('xi_minus_gaussian', 'xi_asym', 'xi')):
        net = lev[:, 0, c] - np.nanmean(lev[:, 1:4, c], 1)
        out = {}
        for lab, x in (('real', lev[:, 0, c]), ('iaaft', np.nanmean(lev[:, 1:4, c], 1)), ('net', net)):
            d = np.array([np.nanmean(x[(unit == u) & (state == 'drug')]) - np.nanmean(x[(unit == u) & (state == mod.REF)]) for u in sessions])
            sa = np.array([animal[unit == u][0] for u in sessions]); sg = np.array([agent[unit == u][0] for u in sessions])
            am = np.array([d[sa == k].mean() for k in animals])
            awake = np.nanmean([np.nanmean(x[(unit == u) & (state == mod.REF)]) for u in sessions])
            out[lab] = dict(awake=float(awake), delta=float(d.mean()), animals=am.round(5).tolist(), t3=float(ttest_1samp(am, 0).statistic),
                            agents={g: float(d[sg == g].mean()) for g in mod.DRUGS})
        levels[nm] = out
    res['xi_levels'] = levels
    res['reproduction'] = dict(r_err_max=float(np.nanmax(D['r_err'])), z_err_max=float(np.nanmax(D['z_err'])))
    res['xi_rows'] = {fam: [r for r in F['rows'] if any(k in r['pair'].split('|') for k in XI_KEYS)] for fam, F in res['families'].items()}
    json.dump(res, open(output, 'w'), indent=1, default=float)
    for r in [r for r in res['named'] if any(k in r['pair'].split('|') for k in XI_KEYS)]:
        print(f"{r['pair']:14s} {r['family']} d {r['mean']:+.4f} t3 {r['t3']:+.1f} sd {r['sd_multiple']:+.1f} z0 {r['z0']:.3f} "
              f"rel {r['reliability'][0]:.2f},{r['reliability'][1]:.2f} rMI {r['r_with_mi_net']:+.2f} axis {r['axis']} {r['gates']}")
    print('levels', json.dumps({k: {l: round(v[l]['delta'], 4) for l in v} for k, v in levels.items()}))
    print('xi rows screened', {f: [r['pair'] for r in v] for f, v in res['xi_rows'].items()}, 'reproduction', res['reproduction'])


NETCORR = (('mi_g', 'xi'), ('xi_g', 'xi'), ('r2', 'mcd2'), ('mi_g', 'ksg'), ('icoh_bb', 'xi'), ('icoh_bb', 'r2'))


def netcorr(path, output):
    """Post hoc, exploratory: window-level correlation (drug windows, centred per session) of real - mean IAAFT nets."""
    D = np.load(path)
    keys = [str(k) for k in D['keys']]; vers = [str(v) for v in D['versions']]
    g = [str(x).split('__') for x in D['group']]
    unit = np.array([a for a, _ in g]); state = np.array([b for _, b in g])
    real, ia = vers.index('real'), [i for i, v in enumerate(vers) if v.startswith('iaaft')]

    def net(a, b):
        Z, j = (D['Z32'], len(keys)) if b == 'ksg' else (D['Z'], keys.index(b))
        return Z[:, real, keys.index(a), j] - Z[:, ia, keys.index(a), j].mean(1)
    C = {f'{a}|{b}': net(a, b) for a, b in NETCORR}
    C['xi-xi_g level'] = D['lev'][:, real, 0] - D['lev'][:, ia, 0].mean(1)
    dm = state == 'drug'
    X = []
    for x in C.values():
        x = x.astype(float).copy()
        for u in set(unit[dm]):
            m = dm & (unit == u); x[m] -= np.nanmean(x[m])
        X.append(x[dm])
    X = np.array(X); ok = np.isfinite(X).all(0)
    out = dict(names=list(C), n_windows=int(ok.sum()), r=np.corrcoef(X[:, ok]).round(3).tolist())
    json.dump(out, open(output, 'w'), indent=1)
    print(json.dumps(out))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'compute':
        compute(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    elif cmd == 'collect':
        collect(sys.argv[2], sys.argv[3])
    elif cmd == 'score':
        score(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == 'netcorr':
        netcorr(sys.argv[2], sys.argv[3])
