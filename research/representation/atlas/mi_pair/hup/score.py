"""SOZ against matched non-SOZ edge sets on z(Gaussian MI, kNN MI), HUP ds004100 (rules: ../preregs/hup-prereg.md).

    python score.py edges <windows dir> <edges dir> <n workers> [subject ...]   per-edge MI, real and 3 joint IAAFT draws
    python score.py sets <windows dir> <edges dir> <out.json>                    matched sets and per-set statistics
    python score.py summary <out.json>                                           decision rule and descriptive checks

edges writes <edges dir>/<subject>.npz with X (windows, 8, E) float32: Gaussian, KSG (real), then Gaussian, KSG per draw;
E in np.triu_indices(M, 1) order over the subject's analysed channels.
"""
import sys, os, glob, json, zlib, numpy as np, warnings
from multiprocessing import Pool
from scipy.stats import ttest_1samp, binomtest, spearmanr, t as tdist
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))
from edges import edge_mi, ns
from mtsspi.surrogates import multivariate_iaaft
warnings.filterwarnings('ignore')
DRAWS, NEAR, TRIES, KEEP, MIN_SETS, MIN_WIN = 3, (1, 2, 3, 5, 8, 10**6), 20000, 200, 20, 10
TOLS = ((.25, .05), (.40, .08))  # (relative distance, absolute |r|) tolerances; the second only if the first gives < MIN_SETS


# ---------------------------------------------------------------- per-edge MI
def one(args):
    w, seed = args; v = ns(w.astype(float)); out = list(edge_mi(v))
    for d in range(DRAWS):
        out += list(edge_mi(multivariate_iaaft(v, np.random.default_rng(seed + d), final='spectrum')[0]))
    return np.array(out, np.float32)


def edges(src, dst, nw, subs):
    os.makedirs(dst, exist_ok=True); jobs, own = [], []
    for s in subs:
        W = np.load(f'{src}/{s}.npz')['W']
        jobs += [(W[i], zlib.crc32(f'{s}:{i}'.encode()) * 10) for i in range(len(W))]; own += [s] * len(W)
    with Pool(nw) as p: res = p.map(one, jobs, chunksize=1)
    own = np.array(own)
    for s in subs:
        np.savez(f'{dst}/{s}.npz', X=np.array([r for r, o in zip(res, own) if o == s])); print('edges done', s, flush=True)


# ---------------------------------------------------------------- sets
def rowcorr(a, b):
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


def describe(A, eidx, D, R):
    e = eidx(A); iu = np.triu_indices(len(A), 1); r = R[e]
    return np.median(D[np.ix_(A, A)][iu]), np.median(r), np.subtract(*np.percentile(r, [75, 25]))


def grow(pool, n, D, rng):
    """Seed contact, then add one chosen uniformly among the k remaining contacts nearest to the set; k drawn from NEAR
    per candidate, so proposals range from compact clusters (k = 1) to uniform random sets (k = all)."""
    k = NEAR[rng.integers(len(NEAR))]; cur = [int(rng.choice(pool))]
    while len(cur) < n:
        rem = np.array([p for p in pool if p not in cur]); d = D[np.ix_(rem, cur)].min(1)
        cur.append(int(rng.choice(rem[np.argsort(d, kind='stable')[:k]])))
    return sorted(cur)


def matched(S, pool, D, R, eidx, seed):
    """Returns S's descriptors, the accepted sets and the tolerance level used (0 primary, 1 widened)."""
    tgt = describe(S, eidx, D, R); out = []
    if len(pool) < len(S): return tgt, out, -1
    for level, (dt, rt) in enumerate(TOLS):
        rng = np.random.default_rng(seed); out, seen = [], set()
        for _ in range(TRIES):
            A = grow(pool, len(S), D, rng); key = tuple(A)
            if key in seen: continue
            seen.add(key); d = describe(A, eidx, D, R)
            if abs(d[0] - tgt[0]) <= dt * tgt[0] and abs(d[1] - tgt[1]) <= rt and abs(d[2] - tgt[2]) <= rt: out.append(A)
            if len(out) == KEEP: break
        if len(out) >= MIN_SETS: break
    return tgt, out, level


def setstat(A, X, eidx, wmask):
    """Per-set window means of res (z real − mean draw z), z real, and lev; NaN if fewer than MIN_WIN windows."""
    if wmask.sum() < MIN_WIN: return [np.nan] * 3
    e = eidx(A); x = X[wmask][:, :, e].astype(float)
    zr = rowcorr(x[:, 0], x[:, 1]); zs = np.mean([rowcorr(x[:, 2 + 2 * d], x[:, 3 + 2 * d]) for d in range(DRAWS)], 0)
    lr = (x[:, 1] - x[:, 0]).mean(1); ls = np.mean([(x[:, 3 + 2 * d] - x[:, 2 + 2 * d]).mean(1) for d in range(DRAWS)], 0)
    return [float(np.nanmean(zr - zs)), float(np.nanmean(zr)), float(np.nanmean(lr - ls))]


def sets(src, edir, out):
    res = {}
    for f in sorted(glob.glob(f'{edir}/*.npz')):
        s = os.path.basename(f)[:-4]; X = np.load(f)['X']; Z = np.load(f'{src}/{s}.npz'); m = json.loads(str(Z['meta']))
        rows = json.loads(str(Z['rows'])); spike = Z['spike']; M = len(m['names']); run = np.array([r['run'] for r in rows])
        iu = np.triu_indices(M, 1); E = -np.ones((M, M), int); E[iu] = np.arange(len(iu[0]))
        eidx = lambda A: E[np.ix_(A, A)][np.triu_indices(len(A), 1)]  # A sorted, so every pair is upper-triangular
        xyz = np.array([p if p else [np.nan] * 3 for p in m['xyz']], float); D = np.linalg.norm(xyz[:, None] - xyz[None], axis=2)
        R = np.sqrt(1 - np.exp(-2 * X[:, 0].astype(float))).mean(0)
        has = np.isfinite(xyz).all(1); soz = np.array(m['soz']); rsc = np.array(m['resect'])
        S = [i for i in range(M) if soz[i] and has[i]]
        rec = dict(n_S=len(S), windows=int(len(X)), implant=m['implant'], engel=m['engel'], M=M)
        allw = np.ones(len(X), bool)
        for tag, pool in (('primary', [i for i in range(M) if not soz[i] and not rsc[i] and has[i]]),
                          ('pool_resect', [i for i in range(M) if not soz[i] and has[i]])):
            tgt, N, lvl = matched(S, pool, D, R, eidx, zlib.crc32(s.encode()) + (0 if tag == 'primary' else 1))
            v = dict(target=[float(q) for q in tgt], n_sets=len(N), tol_level=lvl, S={}, N={})
            masks = dict(all=lambda A: allw, run1=lambda A: run == 1, run2=lambda A: run == 2,
                         nospike=lambda A: ~spike[:, A].any(1)) if tag == 'primary' else dict(all=lambda A: allw)
            for k, mk in masks.items():
                v['S'][k] = setstat(S, X, eidx, mk(S)); v['N'][k] = [setstat(A, X, eidx, mk(A)) for A in N]
            rec[tag] = v
        res[s] = rec; print(s, rec['n_S'], rec['primary']['n_sets'], rec['pool_resect']['n_sets'], flush=True)
    json.dump(res, open(out, 'w'))


# ---------------------------------------------------------------- summary
def per_subject(res, subs, tag, mask, col):
    """{subject: (D, pseudo-D)}: S against the mean of its matched sets; the first matched set against the rest."""
    out = {}
    for s in subs:
        v = res[s][tag]; a = v['S'][mask][col]; b = np.array([q[col] for q in v['N'][mask]], float); b = b[np.isfinite(b)]
        if np.isfinite(a) and len(b) >= MIN_SETS: out[s] = (a - b.mean(), b[0] - b[1:].mean())
    return out


def contrast(*a):
    d = per_subject(*a); return np.array([q[0] for q in d.values()]), np.array([q[1] for q in d.values()])


def verdict(D, P, sign):
    n = len(D)
    if n < 3: return f'n = {n}: too few'
    sd = P.std(ddof=1); se = D.std(ddof=1) / np.sqrt(n); q = tdist.ppf(.975, n - 1)
    pl, pg = ttest_1samp(D, 0, alternative='less').pvalue, ttest_1samp(D, 0, alternative='greater').pvalue
    pos = int((D > 0).sum()); sl = binomtest(pos, n, .5, alternative='less').pvalue; sg = binomtest(pos, n, .5, alternative='greater').pvalue
    if sign < 0: v = 'SUPPORTED' if D.mean() < 0 and pl < .05 and sl < .05 else 'OPPOSITE' if D.mean() > 0 and pg < .05 else 'NOT DETECTED'
    else: v = 'SUPPORTED' if D.mean() > 0 and pg < .05 and sg < .05 else 'OPPOSITE' if D.mean() < 0 and pl < .05 else 'NOT DETECTED'
    return (f'n {n}; mean {D.mean():+.4f} [{D.mean() - q * se:+.4f}, {D.mean() + q * se:+.4f}]; positive {pos}/{n}; '
            f't p(less) {pl:.4f} p(greater) {pg:.4f}; sign p(less) {sl:.4f} p(greater) {sg:.4f}; null SD {sd:.4f} -> '
            f'{D.mean() / sd:+.2f} SD; d_z {D.mean() / D.std(ddof=1):+.2f}; {v}')


def summary(fn):
    res = json.load(open(fn)); ok = lambda s, fl: res[s]['n_S'] >= fl and res[s]['primary']['n_sets'] >= MIN_SETS
    prim = [s for s in res if ok(s, 8)]
    print(f'included (n_S >= 8, >= {MIN_SETS} matched sets): {len(prim)}/{len(res)}')
    for s in sorted(res): print(f"  {s}: n_S {res[s]['n_S']}, sets {res[s]['primary']['n_sets']}, windows {res[s]['windows']}, {res[s]['implant']} {res[s]['engel']}")
    for name, col, sign in (('PRIMARY D_z (res = z real - z surrogate)', 0, -1), ('SECONDARY D_lev (kNN - Gaussian, beyond surrogate)', 2, +1)):
        print(f'\n{name}:', verdict(*contrast(res, prim, 'primary', 'all', col), sign))
    print('\nDescriptive (none decided):')
    for col, lab in ((0, 'D_z'), (2, 'D_lev'), (1, 'z real only')):
        for desc, subs, tag, mk in (('spike-free windows', prim, 'primary', 'nospike'), ('run 1', prim, 'primary', 'run1'),
                                    ('run 2', prim, 'primary', 'run2'), ('floor n_S >= 6', [s for s in res if ok(s, 6)], 'primary', 'all'),
                                    ('pool with resected', [s for s in prim if res[s]['pool_resect']['n_sets'] >= MIN_SETS], 'pool_resect', 'all'),
                                    ('ECoG', [s for s in prim if res[s]['implant'] == 'ECOG'], 'primary', 'all'),
                                    ('SEEG', [s for s in prim if res[s]['implant'] == 'SEEG'], 'primary', 'all'),
                                    ('Engel 1', [s for s in prim if str(res[s]['engel']).startswith('1')], 'primary', 'all'),
                                    ('Engel 2-4', [s for s in prim if str(res[s]['engel'])[:1] in '234'], 'primary', 'all')):
            print(f'  {lab}, {desc}:', verdict(*contrast(res, subs, tag, mk, col), -1 if col != 2 else +1))
    for col, lab in ((0, 'D_z'), (2, 'D_lev')):
        a, b = (per_subject(res, prim, 'primary', r, col) for r in ('run1', 'run2')); k = [s for s in a if s in b]
        if len(k) > 2:
            d1, d2 = np.array([a[s][0] for s in k]), np.array([b[s][0] for s in k])
            print(f'  run agreement {lab} (n {len(k)}): Spearman {spearmanr(d1, d2)[0]:+.2f}, same sign {(np.sign(d1) == np.sign(d2)).mean():.2f}')


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'edges': edges(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5:] or sorted(os.path.basename(f)[:-4] for f in glob.glob(f'{sys.argv[2]}/sub-*.npz')))
    elif mode == 'sets': sets(*sys.argv[2:5])
    elif mode == 'summary': summary(sys.argv[2])
