"""Robustness of the macaque MI-pair drug Δz: Theiler window, k, window length and the shift-corrected KSG at 2 s.

Same windows, 32-channel subsets, normal-scoring and single spectrum-final joint IAAFT draw as `edges.py recompute` and
`ksgbias.py`. Per edge, on the normal-scored real window and on its surrogate, the KSG-1 count term
c = psi(k) - <psi(nx+1) + psi(ny+1)> at k = 4 and k = 10 under four exclusion windows:
  w0     self only (|t - s| <= 0)
  auto   the edges.py AUTO window min(int(2 <a_i, a_j>), T//10)
  2te    min(2 max(tau_e,i, tau_e,j), T//10), tau_e the first lag with acf < 1/e (the window of ksgbias k2te)
  4auto  min(4 int(2 <a_i, a_j>), T//4)
Primary K = c + psi(T) (the edges.py form: psi is constant over edges, so only the exclusion changes z); the pool-psi form
c + <psi(pool + 1)> is secondary. With --shift, also the mean k = 4 AUTO count term of (x_i, x_j rolled by T/4, T/2, 3T/4),
so kshift = K(auto, 4) - K(shifted) as ksgbias.py. Columns per edge: g, c[w][k] (w-major), csh, w_auto, w_2te, w_4auto.

  python robust.py run <bank dir> <out.npz> <agent> <first|row> <n workers> [--shift] [--cut P]
  python robust.py calib <stage-a bank dir> <out.npz> <n per cell> <n workers>
  python robust.py score <dir> <prefix: rb10|rbS2|rbC2> [gate zns dir]
  python robust.py cscore <calib.npz>

--cut P splits each window into P contiguous pieces (T/P samples), each with its own surrogate (seed P*row + piece).
Decision rules: preregs/robust-prereg.md (frozen before any new real-data value was read).
"""
import sys, glob, json, numpy as np, warnings
from multiprocessing import Pool
from scipy.special import digamma
from scipy.stats import ttest_1samp, ttest_rel
from mtsspi.surrogates import multivariate_iaaft
from edges import acf, ns, zc, sub32, meta, STATES
from ksgbias import pool_psi, tau_e, session_deltas
warnings.filterwarnings('ignore')
KS, WN, SHIFTS = (4, 10), ('w0', 'auto', '2te', '4auto'), (1, 2, 3)
NC = 1 + len(WN) * len(KS) + 1 + 3  # g, c[w][k], csh, w_auto, w_2te, w_4auto
col = lambda w, k: 1 + WN.index(w) * len(KS) + KS.index(k)
CSH = 1 + len(WN) * len(KS)


def counts(dx, dy, ex):
    """KSG-1 count terms for every k in KS under mask ex; for k = 4 identical in value to ksgbias.kcore."""
    d = np.where(ex, np.inf, np.maximum(dx, dy)); P = np.partition(d, [k - 1 for k in KS], 1); out = []
    for k in KS:
        eps = P[:, k - 1]
        nx = np.sum(np.where(ex, False, dx < eps[:, None]), 1); ny = np.sum(np.where(ex, False, dy < eps[:, None]), 1)
        out.append(digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)))
    return out


def edge_sweep(x, shift):
    x = (x - x.mean(0)) / x.std(0); T, M = x.shape; iu = np.triu_indices(M, 1)
    r = np.corrcoef(x.T)[iu]; A = np.array([acf(x[:, m]) for m in range(M)]); te = np.array([tau_e(a) for a in A])
    Wu = (2 * A @ A.T).astype(int)  # as ksgbias.edge_all (reproduces edges.py's int(2 * dot))
    t = np.arange(T); td = np.abs(t[:, None] - t[None, :]); out = []
    for i, j in zip(*iu):
        ws = (0, min(Wu[i, j], T // 10), min(2 * max(te[i], te[j]), T // 10), min(4 * Wu[i, j], T // 4))
        dx = np.abs(x[:, i][:, None] - x[:, i][None, :]); dy = np.abs(x[:, j][:, None] - x[:, j][None, :]); row = []
        for w in ws: row += counts(dx, dy, td <= w)
        sh = np.nan
        if shift:
            ex = td <= ws[1]; s = []
            for q in SHIFTS:
                y = np.roll(x[:, j], q * T // 4); s.append(counts(dx, np.abs(y[:, None] - y[None, :]), ex)[0])
            sh = np.mean(s)
        out.append(row + [sh, ws[1], ws[2], ws[3]])
    return np.column_stack([-0.5 * np.log(1 - r ** 2), np.array(out)]).T  # (NC, E)


def one(args):
    x, seed, shift = args; v = ns(x)
    s = multivariate_iaaft(v, np.random.default_rng(seed), final='spectrum')[0]
    return np.array([edge_sweep(v, shift), edge_sweep(s, shift)], np.float32)  # (2 real/surr, NC, E)


def run(d, out, agent, rowmode, nw, shift, cut):
    man = json.load(open(f'{d}/manifest.json')); Wn = np.load(f'{d}/windows.npz')
    rows = [r for r in man['rows'] if r['state'] in STATES and r['agent'] in agent.split(',')]  # as edges.py recompute
    S0 = sub32(rows[0]['M']); sub = lambda r: S0 if rowmode == 'first' else sub32(r['M'])
    if cut == 1: jobs = [(Wn[r['id']].astype(float)[:, sub(r)], k, shift) for k, r in enumerate(rows)]
    else:
        L = man['T'] // cut
        jobs = [(Wn[r['id']].astype(float)[p * L:(p + 1) * L, sub(r)], cut * k + p, shift) for k, r in enumerate(rows) for p in range(cut)]
    with Pool(nw) as p: res = p.map(one, jobs, chunksize=1)
    m, _ = meta(rows, man); m = {k: np.repeat(v, cut) for k, v in m.items()}
    np.savez(out, X=np.array(res), T=man['T'] // cut, cut=cut, **m)
    print('done', out, flush=True)


def configs(X, T, shift):
    """Per-edge K arrays (W, 2, E) for every configuration; name -> (K, w used)."""
    psiT = digamma(T); P = np.array([pool_psi(T, w) for w in range(T // 4 + 1)]); C = {}
    wcol = {'w0': None, 'auto': -3, '2te': -2, '4auto': -1}
    for w in WN:
        wv = np.zeros(X[:, :, 0].shape, int) if wcol[w] is None else X[:, :, wcol[w]].astype(int)
        for k in KS:
            c = X[:, :, col(w, k)]; C[f'{w} k{k}'] = (c + psiT, wv)
            if k == 4: C[f'{w} k4 pool'] = (c + P[wv], wv)
    if shift: C['kshift'] = (X[:, :, col('auto', 4)] - X[:, :, CSH], X[:, :, -3].astype(int))  # psi terms cancel
    return C


def stats(res, unit, state, agent, animal):
    ag, an, dz, eo = session_deltas(res, unit, state, agent, animal); dr = ag != 'SLP'
    A = np.array([dz[dr & (an == x)].mean() for x in np.unique(an[dr])]); t = ttest_1samp(A, 0, alternative='less')
    sd = eo.std(ddof=1); agm = {a: (dz[ag == a].mean(), int(np.sum(dz[ag == a] < 0)), int(np.sum(ag == a))) for a in ('KT', 'KTMD', 'MD', 'PF', 'SLP')}
    return dict(A=A, t=t.statistic, p=t.pvalue, pooled=dz[dr].mean(), sd=sd, agm=agm, n_neg=int((A < 0).sum()))


def score(d, prefix, gate=None):
    L = [np.load(f) for f in sorted(glob.glob(f'{d}/{prefix}_[AB]_*.npz')) + sorted(glob.glob(f'{d}/{prefix}_S_*.npz'))]
    cat = lambda k: np.concatenate([q[k] for q in L]); X = np.concatenate([q['X'] for q in L]).astype(float); T = int(L[0]['T'])
    unit, state, agent, animal = cat('unit'), cat('state'), cat('agent'), cat('animal'); shift = bool(np.isfinite(X[:, :, CSH]).all())
    C = configs(X, T, shift); Z = {n: np.array([[zc(X[w, rs, 0], K[w, rs]) for rs in (0, 1)] for w in range(len(X))]) for n, (K, _) in C.items()}
    print(f'{prefix}: {len(X)} windows, T = {T}, shift {shift}')
    if gate:
        G = [np.load(f) for f in sorted(glob.glob(f'{gate}/zns_[AB]_*.npz'))]; Vz = np.concatenate([q['V'] for q in G])[:, 1]
        gap = np.abs(Z['auto k4'] - Vz).max(); print(f'reproduction gate: max |auto k4 z − stored zns| = {gap:.2e} ->', 'PASS' if gap < 1e-4 else 'FAIL')
        if gap >= 1e-4: return
    ref = stats(Z['auto k4'][:, 0] - Z['auto k4'][:, 1], unit, state, agent, animal)
    for n, (K, wv) in C.items():
        s = stats(Z[n][:, 0] - Z[n][:, 1], unit, state, agent, animal); pr = ttest_rel(s['A'], ref['A'])
        drug = np.isin(state, ['drug']); aw = np.isin(state, ['awake_ec', 'awake_eo'])
        capv = {'w0': 0, 'auto': T // 10, '2te': T // 10, '4auto': T // 4}[n.split()[0]] if n != 'kshift' else T // 10
        cap = '' if n.startswith('w0') else f' | w median awake/drug {np.median(wv[aw, 0]):.0f}/{np.median(wv[drug, 0]):.0f}, at cap {(wv[aw, 0] >= capv).mean():.3f}/{(wv[drug, 0] >= capv).mean():.3f}'
        print(f'\n{n}: animal means {np.round(s["A"], 4)}; {s["n_neg"]}/4 < 0; t(3) = {s["t"]:.2f}, p = {s["p"]:.4f}; pooled drug Δz {s["pooled"]:+.4f} '
              f'({s["pooled"] / s["sd"]:+.1f} own EO SD; EO SD {s["sd"]:.4f}); R {s["pooled"] / ref["pooled"]:.2f}; paired vs auto k4 t(3) = {pr.statistic:+.2f}{cap}')
        print('  ' + ' | '.join(f'{a} {v[0]:+.4f} ({v[1]}/{v[2]} down)' for a, v in s['agm'].items())
              + f' | awake z real {Z[n][state == "awake_ec", 0].mean():.3f}, K<0 drug real/surr {(K[drug, 0] < 0).mean():.2f}/{(K[drug, 1] < 0).mean():.2f}')


def calib(d, out, n, nw):
    from ksgbias_calib import gen, sources
    src = sources(d); jobs, lab = [], []
    for si, (tag, T) in enumerate([(a, T) for T in (1000, 200) for a in ('AW', 'KTMD', 'PF')]):
        for ki, k in enumerate(('block', 'gain_shared')):
            for i in range(n): jobs.append((k, 7 * 10 ** 6 + 10 ** 5 * si + 10 ** 3 * ki + 2 * i, src[tag][i % len(src[tag])][:T])); lab.append((k, f'{tag}-T{T}'))
    with Pool(nw) as p: R = p.map(calib_one, jobs, chunksize=1)
    np.savez(out, X=np.array(R), kind=np.array([a for a, _ in lab]), set=np.array([b for _, b in lab])); print('done', out, flush=True)


def calib_one(a):
    from ksgbias_calib import gen
    kind, seed, src = a; rng = np.random.default_rng(seed)
    x = ns(gen(kind, src, rng)); s = multivariate_iaaft(x, np.random.default_rng(seed + 1), final='spectrum')[0]
    return np.array([edge_sweep(x, True), edge_sweep(s, True)], np.float32)


def cscore(f):
    D = np.load(f); X, kind, sset = D['X'].astype(float), D['kind'], D['set']
    for T in (1000, 200):
        print(f'\n=== T = {T}: z real − surr, mean ± SE (n per cell); detectability = (block − gain_shared) / SD(block)')
        C = {s: configs(X[sset == s], T, True) for s in np.unique(sset) if s.endswith(f'T{T}')}
        for n in C[f'AW-T{T}']:
            line, off = [], {}
            for tag in ('AW', 'KTMD', 'PF'):
                s = f'{tag}-T{T}'; Xs, ks = X[sset == s], kind[sset == s]; K = C[s][n][0]
                dz = {k: np.array([zc(Xs[w, 0, 0], K[w, 0]) - zc(Xs[w, 1, 0], K[w, 1]) for w in np.nonzero(ks == k)[0]]) for k in ('block', 'gain_shared')}
                b, g = dz['block'], dz['gain_shared']; off[tag] = (b.mean(), b.std(ddof=1) / np.sqrt(len(b)))
                line.append(f'{tag} block {b.mean():+.4f}±{off[tag][1]:.4f} gs {g.mean():+.4f} det {(b.mean() - g.mean()) / b.std(ddof=1):.2f}')
            dd = np.mean([off['KTMD'][0], off['PF'][0]]) - off['AW'][0]; se = np.sqrt((off['KTMD'][1] ** 2 + off['PF'][1] ** 2) / 4 + off['AW'][1] ** 2)
            print(f'{n:13s} ' + ' | '.join(line) + f' || stationary drug − awake offset {dd:+.4f} ± {se:.4f}')


if __name__ == '__main__':
    m = sys.argv[1]
    if m == 'score': score(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    elif m == 'cscore': cscore(sys.argv[2])
    elif m == 'calib': calib(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    elif m == 'run':
        a = sys.argv[2:]; cut = int(a[a.index('--cut') + 1]) if '--cut' in a else 1
        run(a[0], a[1], a[2], a[3], int(a[4]), '--shift' in a, cut)
