"""Tail-dependence asymmetry on the axis screen's windows (pre-registered; criteria in tailasym_screen.md, committed
before any run on data).

Per window and version (the screen's seven, regenerated with xi_screen.versions: same code, seeds and order of draws),
four rows on the upper triangle, from average ranks of each channel (pyspi's definition, k = floor((1 - q) T + 1e-9)):
  ta      lambda_U - lambda_L at q .9 (pyspi tailasym_q-0-9)
  ta95    lambda_U - lambda_L at q .95 (pyspi tailasym_q-0-95)
  ta_up   lambda_U at q .9 (pyspi tailasym_upper_q-0-9)
  ta_lo   lambda_L at q .9 (pyspi tailasym_lower_q-0-9)
Ranks are invariant to the per-channel normal-scoring, so real values equal those of the z-scored window.
Levels per window (mean over edges): ta, ta95, |ta|, and the window's median over channels of |skewness| (the added
covariate; skewness is affine-invariant, so it is taken on the bank window before normal-scoring).

The z coordinates go through the frozen scorer (axis_screen/score.py) as in the xi screen, rows ta* declared higher-order.
The primary is a level, scored by `levels` below with the screen's gates 1-5 (same functions' logic, re-stated here
because score.py builds them inside score()).

  python -m research.representation.atlas.pair_catalogue.tailasym_screen check <bank> <n per archive>
  python -m research.representation.atlas.pair_catalogue.tailasym_screen compute <bank> <screen dir> <out dir> <workers>
  python -m research.representation.atlas.pair_catalogue.tailasym_screen collect <out dir> <out.npz>
  python -m research.representation.atlas.pair_catalogue.tailasym_screen score <ta-screen.npz> <manifest.json> <out.json>
"""
import sys, json, glob, time
import numpy as np
from pathlib import Path
from multiprocessing import Pool

TA_KEYS = ('ta', 'ta95', 'ta_up', 'ta_lo')
TA_NAMES = {'ta': 'tailasym_q-0-9', 'ta95': 'tailasym_q-0-95', 'ta_up': 'tailasym_upper_q-0-9', 'ta_lo': 'tailasym_lower_q-0-9'}
TA_HIGHER = set(TA_KEYS)
NAMED_TA = ['ta_lo|ta_up', 'r|ta']
LEVELS = ('ta', 'ta95', 'abs_ta')


def tail_lambdas(x, q):
    """(lambda_U, lambda_L) as M x M matrices, exactly pyspi's TailDependenceAsymmetry._coefficients on finite,
    non-constant channels: average ranks, k = floor((1 - q) T + 1e-9), denominator the geometric mean of tail counts."""
    from scipy.stats import rankdata
    T = len(x)
    k = int(np.floor((1 - q) * T + 1e-9))
    R = rankdata(x, axis=0)
    out = []
    for tail in (R > T - k, R <= k):
        t = tail.astype(np.float64)
        both = t.T @ t
        n = np.diag(both).copy()
        out.append(both / np.sqrt(np.outer(n, n)))
    return out


def ta_profiles(x):
    """(4, E) rows on the upper triangle (run.py order) and the r profile for the reproduction check."""
    M = x.shape[1]
    a, b = np.triu_indices(M, 1)
    z = (x - x.mean(0)) / x.std(0)
    r = (z.T @ z / len(z))[a, b]
    U, L = tail_lambdas(x, .9)
    U95, L95 = tail_lambdas(x, .95)
    up, lo = U[a, b], L[a, b]
    return np.stack([up - lo, U95[a, b] - L95[a, b], up, lo]), r


def skew_abs(w):
    w = np.asarray(w, np.float64)
    c = w - w.mean(0)
    return float(np.median(np.abs((c ** 3).mean(0) / (c ** 2).mean(0) ** 1.5)))


def check(bank, n_per):
    """pyspi TailDependenceAsymmetry (repo pin) against ta_profiles on the first n windows of each archive."""
    from pyspi.statistics.events import TailDependenceAsymmetry
    from research.representation.atlas.axis_screen.run import _ns
    rows = json.load(open(Path(bank) / 'manifest.json'))['rows']
    by = {}
    for r in rows:
        by.setdefault(r['unit'].split('_')[0], []).append(r['id'])
    worst = 0.
    with np.load(Path(bank) / 'windows.npz') as W:
        for arch, ids in sorted(by.items()):
            for i in ids[:n_per]:
                x = _ns(np.asarray(W[i], np.float64))
                P, _ = ta_profiles(x)
                M = x.shape[1]; a, b = np.triu_indices(M, 1)
                for s, (q, stat) in enumerate(((.9, 'asymmetry'), (.95, 'asymmetry'), (.9, 'upper'), (.9, 'lower'))):
                    U, L = TailDependenceAsymmetry(quantile=q, statistic=stat)._coefficients(x.T)
                    A = U - L if stat == 'asymmetry' else U if stat == 'upper' else L
                    worst = max(worst, float(np.nanmax(np.abs(A[a, b] - P[s]))))
            print(arch, 'max |pyspi - ours|', worst, flush=True)
    assert worst < 1e-12, worst
    print('check passed', worst)


def group(args):
    bank, screen, out, g = args
    from research.representation.atlas.axis_screen.run import VERSIONS, _reduce, _zmat
    from research.representation.atlas.axis_screen.stats import KEYS
    from research.representation.atlas.pair_catalogue.xi_screen import versions
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
    with np.load(Path(bank) / 'windows.npz') as W:
        skew = np.array([skew_abs(W[i]) for i in ids])
    n, S_ = len(ids), len(KEYS) + len(TA_KEYS)
    M = ver['real'][0].shape[1]
    a, b_ = np.triu_indices(M, 1)
    s32 = np.round(np.arange(32) * (M - 1) / 31).astype(int)
    e32 = np.flatnonzero(np.isin(a, s32) & np.isin(b_, s32))
    X = np.full((n, len(VERSIONS), len(TA_KEYS), len(a)), np.nan, np.float32)
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
            X[w, k], r = ta_profiles(x)
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
            for s in range(len(TA_KEYS)):
                c = [np.corrcoef(X[i, k, s], X[j, k, s])[0, 1] for i, j in pairs]
                rel[k, len(KEYS) + s] = np.nanmean(c)
            zmean[k] = _zmat(np.vstack([np.nanmean(P0[:, k].astype(float), 0), np.nanmean(X[:, k].astype(float), 0)]))[0]
    lev = np.stack([np.nanmean(X[:, :, 0], -1), np.nanmean(X[:, :, 1], -1), np.nanmean(np.abs(X[:, :, 0]), -1)], -1)
    np.savez(target, ids=np.array(ids), sub=np.array(subs), Z=Z, Zdc=Zdc, Z32=Z32, R2=R2, fin=fin, cov=G['cov'], rel=rel,
             zmean=zmean, M=M, keys=np.array(KEYS + TA_KEYS), versions=np.array(VERSIONS), r_err=r_err, z_err=z_err, lev=lev,
             skew=skew)
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
    extra = {k: np.concatenate([d[k] for d in D]) for k in ('r_err', 'z_err', 'lev', 'skew')}
    base = dict(np.load(target))
    np.savez(target, **base, **extra)
    print('reproduction: max |r - stored r|', float(np.nanmax(extra['r_err'])), 'max |z - stored z|', float(np.nanmax(extra['z_err'])))


def _scorer():
    """The frozen scorer with the ta rows declared higher-order and the pre-stated coordinates named; nothing else changes."""
    import types
    from research.representation.atlas.axis_screen import score as S
    src = Path(S.__file__).read_text()
    old = "named = ['linf|l2',"
    assert src.count(old) == 1
    src = src.replace(old, "named = NAMED_TA + ['linf|l2',")
    mod = types.ModuleType('ta_score')
    mod.__dict__['NAMED_TA'] = NAMED_TA
    exec(compile(src, S.__file__, 'exec'), mod.__dict__)
    mod.PYSPI.update(TA_NAMES)
    mod.HIGHER |= TA_HIGHER
    return mod


def levels(mod, D, I):
    """Gates 1-5 of the screen (prereg.md) on a per-window level net = real - mean of the 3 IAAFT draws. Gate 5 uses the
    F1 covariates (cov columns 0, 1, 2, 6) plus the window's median |skewness|. Logic follows score.py's row()."""
    from scipy.stats import ttest_1samp
    unit, state, agent, animal, start = I['unit'], I['state'], I['agent'], I['animal'], I['start']
    REF, DRUGS = mod.REF, mod.DRUGS
    ag = {u: a for u, a in zip(unit, agent)}; an = {u: a for u, a in zip(unit, animal)}
    sessions = sorted(u for u in set(unit) if ag[u] in DRUGS and ((unit == u) & (state == 'drug')).any() and ((unit == u) & (state == REF)).any())
    animals = sorted({an[u] for u in sessions})
    sa = np.array([an[u] for u in sessions])
    Gi = lambda u, s: np.flatnonzero((unit == u) & (state == s))
    cov = np.column_stack([D['cov'][:, list(mod.COVS['F1'])].astype(float), D['skew']])

    def sdelta(x, s):
        return np.array([np.nanmean(x[Gi(u, s)]) - np.nanmean(x[Gi(u, REF)]) if len(Gi(u, s)) else np.nan for u in sessions])

    def amean(d):
        return np.array([np.nanmean(d[sa == k]) for k in animals])

    def adjusted(x, C_all):
        out = []
        for u in sessions:
            ix = np.concatenate([Gi(u, REF), Gi(u, 'drug')])
            y = x[ix]; d = (state[ix] == 'drug').astype(float)
            C = C_all[ix]
            C = C[:, np.isfinite(C).all(0) & (np.nanstd(C, 0) > 1e-12)]
            C = (C - C.mean(0)) / C.std(0)
            X = np.column_stack([np.ones(len(ix)), d, C]); ok = np.isfinite(y)
            out.append(np.linalg.lstsq(X[ok], y[ok], rcond=None)[0][1])
        return np.array(out)

    res = {}
    for c, nm in enumerate(LEVELS):
        real, ia = D['lev'][:, 0, c].astype(float), D['lev'][:, 1:4, c].astype(float).mean(1)
        x = real - ia
        d = sdelta(x, 'drug'); am = amean(d); m = float(np.nanmean(d))
        sh = []
        for u in sessions:
            r = Gi(u, REF); r = r[np.argsort(start[r])]; h = len(r) // 2
            sh.append(np.nanmean(x[r[-h:]]) - np.nanmean(x[r[:h]]))
        sh = np.array(sh); null_sd = float(np.nanstd(sh, ddof=1) / np.sqrt(2))
        eo = float(np.nanmean(sdelta(x, 'awake_eo')))
        adj = adjusted(x, cov); adj_f1 = adjusted(x, cov[:, :-1])
        t = ttest_1samp(am, 0)
        gates = {'animal_t3': bool(t.pvalue < .05 and len(set(np.sign(am))) == 1), 'size_3sd': bool(abs(m) >= 3 * null_sd),
                 'split_half': bool(abs(np.nanmean(sh)) <= .25 * abs(m)), 'eyes_open': bool(abs(eo) <= .33 * abs(m)),
                 'covariates': bool(np.sign(np.nanmean(adj)) == np.sign(m) and abs(np.nanmean(adj)) >= .5 * abs(m)
                                    and all(np.sign(amean(adj)) == np.sign(m)))}
        res[nm] = dict(
            net=m, animals=dict(zip(animals, am.round(5).tolist())), t3=float(t.statistic), p3=float(t.pvalue),
            sessions_same_sign=f"{int(np.sum(np.sign(d) == np.sign(m)))}/{len(sessions)}", null_sd=null_sd, sd_multiple=m / null_sd,
            split_half=float(np.nanmean(sh)), eyes_open=eo, adjusted=float(np.nanmean(adj)), adjusted_animals=amean(adj).round(5).tolist(),
            adjusted_f1_only=float(np.nanmean(adj_f1)),
            real=float(np.nanmean(sdelta(real, 'drug'))), iaaft=float(np.nanmean(sdelta(ia, 'drug'))),
            awake_real=float(np.nanmean([np.nanmean(real[Gi(u, REF)]) for u in sessions])),
            awake_iaaft=float(np.nanmean([np.nanmean(ia[Gi(u, REF)]) for u in sessions])),
            per_agent={k: float(np.nanmean(d[np.array([ag[u] == k for u in sessions])])) for k in DRUGS},
            recovery=float(np.nanmean(sdelta(x, 'recovery_ec'))), gates=gates, passes=bool(all(gates.values())))
    sk = sdelta(D['skew'].astype(float), 'drug')
    res['skew_covariate'] = dict(awake=float(np.nanmean([np.nanmean(D['skew'][Gi(u, REF)]) for u in sessions])), delta=float(np.nanmean(sk)),
                                 animals=amean(sk).round(4).tolist())
    return res, sessions


def score(path, manifest, output):
    mod = _scorer()
    mod.score(path, manifest, output)
    res = json.load(open(output))
    D, keys, gi, I = mod.load(path, manifest)
    lev, sessions = levels(mod, D, I)
    res['ta_levels'] = lev
    # awake eyes-closed split-half reliability of the ta edge profile (the level's replacement for gate 6), real version
    S = len(keys); ita = keys.index('ta')
    groups = [str(g) for g in D['groups']]
    res['ta_profile_reliability_awake'] = float(np.nanmean([D['rel'][groups.index(f'{u}__{mod.REF}'), 0, ita] for u in sessions]))
    res['reproduction'] = dict(r_err_max=float(np.nanmax(D['r_err'])), z_err_max=float(np.nanmax(D['z_err'])))
    res['ta_rows'] = {fam: [r for r in F['rows'] if any(k in r['pair'].split('|') for k in TA_KEYS)] for fam, F in res['families'].items()}
    json.dump(res, open(output, 'w'), indent=1, default=float)
    for nm, v in lev.items():
        if nm in LEVELS:
            print(f"{nm:7s} net {v['net']:+.5f} t3 {v['t3']:+.2f} sd {v['sd_multiple']:+.1f} {v['sessions_same_sign']} "
                  f"adj {v['adjusted']:+.5f} agents {json.dumps({k: round(x, 5) for k, x in v['per_agent'].items()})} pass {v['passes']} {v['gates']}")
    print('skew covariate', lev['skew_covariate'])
    print('ta profile reliability (awake ec)', round(res['ta_profile_reliability_awake'], 3))
    for r in [r for r in res['named'] if any(k in r['pair'].split('|') for k in TA_KEYS)]:
        print(f"{r['pair']:14s} {r['family']} d {r['mean']:+.4f} t3 {r['t3']:+.1f} sd {r['sd_multiple']:+.1f} z0 {r['z0']:.3f} "
              f"rel {r['reliability'][0]:.2f},{r['reliability'][1]:.2f} rMI {r['r_with_mi_net']:+.2f} axis {r['axis']} {r['gates']}")
    print('ta rows screened', {f: [r['pair'] for r in v] for f, v in res['ta_rows'].items()}, 'reproduction', res['reproduction'])


def netcorr(path, output):
    """Pre-stated link (descriptive): window-level correlation within drug sessions, centred per session, of the ta level
    net with the MI net (mi_g|ksg) and the MinCovDet net (r2|mcd2)."""
    D = np.load(path)
    keys = [str(k) for k in D['keys']]; vers = [str(v) for v in D['versions']]
    g = [str(x).split('__') for x in D['group']]
    unit = np.array([a for a, _ in g]); state = np.array([b for _, b in g])
    real, ia = vers.index('real'), [i for i, v in enumerate(vers) if v.startswith('iaaft')]
    zn = lambda Z, i, j: Z[:, real, i, j] - Z[:, ia, i, j].mean(1)
    C = {'ta level': D['lev'][:, real, 0] - D['lev'][:, ia, 0].mean(1),
         '|ta| level': D['lev'][:, real, 2] - D['lev'][:, ia, 2].mean(1),
         'mi_g|ksg': zn(D['Z32'], keys.index('mi_g'), len(keys)),
         'r2|mcd2': zn(D['Z'], keys.index('r2'), keys.index('mcd2'))}
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
    if cmd == 'check':
        check(sys.argv[2], int(sys.argv[3]))
    elif cmd == 'compute':
        compute(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    elif cmd == 'collect':
        collect(sys.argv[2], sys.argv[3])
    elif cmd == 'score':
        score(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == 'netcorr':
        netcorr(sys.argv[2], sys.argv[3])
