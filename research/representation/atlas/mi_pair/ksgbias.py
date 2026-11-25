"""KSG bias check for the macaque MI pair: does the drug Δz of z(G, K) survive bias-corrected KSG estimators?

Same windows, 32-channel subsets, normal-scoring, seeds and single spectrum-final joint IAAFT draw as `edges.py recompute`
and `estimators.py`. Per edge, on the normal-scored real window and on its surrogate:
  g      Gaussian MI from Pearson r (as edges.py)
  k4     KSG k=4, AUTO Theiler window w, psi(N) (as edges.py; reproduction gate)
  kpool  as k4 with psi(N) replaced by the mean of psi(pool + 1) over the Theiler-reduced pools (pyspi _ksg_pool_digamma)
  k2te   as kpool with w = min(2 max(tau_e,i, tau_e,j), T//10), tau_e the first lag with acf < 1/e
  kshift k4 minus the mean k4 of (x_i, x_j circularly shifted by T/4, T/2, 3T/4): each edge's independence bias under the
         window's own serial structure, nonstationarity included (psi terms cancel, so psi(N) vs pool is immaterial)
  w      the AUTO window

  python ksgbias.py <bank dir> <out.npz> <agent> <rowmode: first|row> <n workers>
  python ksgbias.py score <dir with kb_[AB]_*.npz>

Decision rule: preregs/ksgbias-prereg.md (frozen before any corrected real-data value was computed).
"""
import sys, glob, json, numpy as np, warnings
from multiprocessing import Pool
from scipy.special import digamma
from scipy.stats import ttest_1samp
from mtsspi.surrogates import multivariate_iaaft
from edges import acf, ns, zc, sub32, meta, STATES, K_NN
warnings.filterwarnings('ignore')
EST = ('k4', 'kpool', 'k2te', 'kshift')
SHIFTS = (1, 2, 3)  # quarters of T


def pool_psi(N, w):
    i = np.arange(N); return np.mean(digamma(N - np.minimum(i, w) - np.minimum(N - 1 - i, w)))


def kcore(dx, dy, ex):
    """KSG-1 counts term digamma(k) - <psi(nx+1) + psi(ny+1)>, Theiler mask ex; same arithmetic as edges.ksg_rows."""
    d = np.where(ex, np.inf, np.maximum(dx, dy)); eps = np.partition(d, K_NN - 1, 1)[:, K_NN - 1]
    nx = np.sum(np.where(ex, False, dx < eps[:, None]), 1); ny = np.sum(np.where(ex, False, dy < eps[:, None]), 1)
    return digamma(K_NN) - np.mean(digamma(nx + 1) + digamma(ny + 1))


def tau_e(a):
    b = np.nonzero(a < 1 / np.e)[0]; return int(b[0]) if len(b) else len(a)


def edge_all(x):
    x = (x - x.mean(0)) / x.std(0); T, M = x.shape; iu = np.triu_indices(M, 1)
    r = np.corrcoef(x.T)[iu]; A = np.array([acf(x[:, m]) for m in range(M)]); te = np.array([tau_e(a) for a in A])
    W = np.minimum((2 * A @ A.T).astype(int), T // 10)  # identical to edges.py's int(2 * dot) per pair
    t = np.arange(T); tdist = np.abs(t[:, None] - t[None, :]); out = []
    for i, j in zip(*iu):
        w, w2 = W[i, j], min(2 * max(te[i], te[j]), T // 10)
        dx = np.abs(x[:, i][:, None] - x[:, i][None, :]); dy = np.abs(x[:, j][:, None] - x[:, j][None, :]); ex = tdist <= w
        c = kcore(dx, dy, ex); k4 = c + digamma(T); kp = c + pool_psi(T, w)
        k2 = kcore(dx, dy, tdist <= w2) + pool_psi(T, w2)
        sh = []
        for q in SHIFTS:
            y = np.roll(x[:, j], q * T // 4); sh.append(kcore(dx, np.abs(y[:, None] - y[None, :]), ex) + digamma(T))
        out.append([k4, kp, k2, k4 - np.mean(sh), w])
    return np.column_stack([-0.5 * np.log(1 - r ** 2), np.array(out)]).T  # (6, E): g, k4, kpool, k2te, kshift, w


def one(args):
    x, seed = args; v = ns(x)
    s = multivariate_iaaft(v, np.random.default_rng(seed), final='spectrum')[0]
    return np.array([edge_all(v), edge_all(s)], np.float32)  # (2 real/surr, 6, E)


def session_deltas(res, unit, state, agent, animal):
    S = []
    for u in np.unique(unit):
        m = lambda s: (unit == u) & (state == s); ag = agent[unit == u][0]; s = 'sleep' if ag == 'SLP' else 'drug'
        if m(s).any(): S.append((ag, animal[unit == u][0], res[m(s)].mean() - res[m('awake_ec')].mean(),
                                 res[m('awake_eo')].mean() - res[m('awake_ec')].mean()))
    return np.array([q[0] for q in S]), np.array([q[1] for q in S]), np.array([q[2] for q in S]), np.array([q[3] for q in S])


def score(d):
    L = [np.load(f) for f in sorted(glob.glob(f'{d}/kb_[AB]_*.npz'))]; cat = lambda k: np.concatenate([q[k] for q in L])
    X = np.concatenate([q['X'] for q in L]).astype(float); Vz = cat('Vzns')
    unit, state, agent, animal = cat('unit'), cat('state'), cat('agent'), cat('animal')
    z = np.array([[[zc(X[w, rs, 0], X[w, rs, e]) for e in (1, 2, 3, 4)] for rs in (0, 1)] for w in range(len(X))])  # (W, 2, 4)
    gap = np.abs(z[:, :, 0] - Vz).max(); print(f'reproduction gate: max |k4 z − stored| = {gap:.2e} ->', 'PASS' if gap < 1e-4 else 'FAIL')
    if gap >= 1e-4: return
    weak = X[:, 1, 0] < .01  # G_real == G_surr
    for c, e in enumerate(EST):
        res = z[:, 0, c] - z[:, 1, c]; ag, an, dz, eo = session_deltas(res, unit, state, agent, animal)
        dr = ag != 'SLP'; A = np.array([dz[dr & (an == x)].mean() for x in np.unique(an[dr])]); pooled = dz[dr].mean()
        t = ttest_1samp(A, 0, alternative='less'); agm = {a: dz[ag == a].mean() for a in ('KT', 'KTMD', 'MD', 'PF', 'SLP')}
        print(f'\n{e}: animal means {np.round(A, 4)}; {(A < 0).sum()}/4 < 0; t(3) = {t.statistic:.2f}, one-sided p = {t.pvalue:.4f}; '
              f'pooled drug Δz {pooled:+.4f}; eyes-open SD {eo.std(ddof=1):.4f}')
        for a in agm:
            m = (agent == a) & (state == ('sleep' if a == 'SLP' else 'drug')); kr, ks = X[m, 0, 1 + c], X[m, 1, 1 + c]; wk = weak[m]
            print(f'  {a:5s} Δz {agm[a]:+.4f} ({np.sum(dz[ag == a] < 0)}/{np.sum(ag == a)} down) | weak edges (G<.01): '
                  f'K real {kr[wk].mean():+.4f}, surr {ks[wk].mean():+.4f}, frac K<0 real {(kr[wk] < 0).mean():.2f}')
        if e != 'k4':
            print(f'  retention vs k4: pooled {pooled / RET0:.2f}; ' + ' '.join(f'{a} {agm[a] / AGM0[a]:.2f}' for a in ('KT', 'KTMD', 'MD', 'PF')))
        else:
            globals()['RET0'], globals()['AGM0'] = pooled, agm


if __name__ == '__main__':
    if sys.argv[1] == 'score': score(sys.argv[2]); sys.exit()
    d, out, agent, rowmode, nw = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    man = json.load(open(f'{d}/manifest.json')); Wn = np.load(f'{d}/windows.npz')
    rows = [r for r in man['rows'] if r['state'] in STATES and r['agent'] in agent.split(',')]  # as edges.py recompute
    S0 = sub32(rows[0]['M']); sub = lambda r: S0 if rowmode == 'first' else sub32(r['M'])
    with Pool(nw) as p: res = p.map(one, [(Wn[r['id']].astype(float)[:, sub(r)], k) for k, r in enumerate(rows)], chunksize=1)
    m, _ = meta(rows, man); zns = out.replace('/ksgbias/kb_', '/zns/zns_')
    np.savez(out, X=np.array(res), Vzns=np.load(zns)['V'][:, 1], **m)
    print('done', out, flush=True)
