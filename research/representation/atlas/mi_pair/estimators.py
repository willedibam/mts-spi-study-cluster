"""Estimator check for the macaque MI pair: is the real − IAAFT fall of z(Gaussian MI, model-free MI) specific to KSG k=4?

Same windows, 32-channel subsets, normal-scoring, seeds and single spectrum-final joint IAAFT draw as `edges.py recompute`
(so the surrogate is the identical draw). Per edge, on the normal-scored real window and its surrogate:
  g     Gaussian MI from Pearson r (as edges.py)
  k4    KSG k=4, AUTO Theiler window (as edges.py; reproduction check)
  k10   KSG k=10, same Theiler window (same family, different bias/variance trade-off)
  kde   Gaussian-kernel plug-in MI, leave-out of the same Theiler neighbourhood, one bandwidth h = n^(-1/6)
        (Scott, d = 2, unit-variance normal scores) for joint and marginals (different estimator family: fixed bandwidth)

  python estimators.py <bank dir> <out.npz> <agent> <rowmode: first|row> <n workers>
  python estimators.py score <dir with est_[AB]_*.npz>

Decision rule, fixed 19 November 2026 before any output was computed:
- Reproduction gate: k4 z (real, surrogate) must match the stored zns V[:, 1] within 1e-4 in every window; otherwise stop.
- Statistic per estimator e: res = z(g, e)_real − z(g, e)_surr per window; session Δ = mean(drug) − mean(awake_ec); animal mean
  pools the drug sessions (KT, KTMD, MD, PF) of each animal, as montage/animal_level.py.
- e SUPPORTS the MI-pair fall if: 4/4 animal means < 0, one-sided t on animal means p < .05, every agent's session mean < 0,
  and |sleep Δ| < .3 |pooled drug Δ|.
- e is ESTIMATOR-SPECIFIC (the k4 result does not generalise to e) if animal means are negative in ≤ 2/4 or the agent-pooled
  drug Δ is ≥ 0. Anything else is PARTIAL and is reported as such.
- Descriptive, not decided: the level (mean over edges of e − g, real − surrogate, drug − awake_ec), and the eyes-open control.
"""
import sys, glob, json, numpy as np, warnings
from multiprocessing import Pool
from scipy.special import digamma
from scipy.stats import ttest_1samp
from mtsspi.surrogates import multivariate_iaaft
from edges import acf, ns, zc, sub32, meta, STATES
warnings.filterwarnings('ignore')
EST = ('k4', 'k10', 'kde')


def rows_mi(x, i, js, w, tdist, h):
    """KSG (k = 4, 10) and kernel MI for pairs (i, j), j in js; KSG arithmetic as edges.ksg_rows."""
    n = len(x); dx = np.abs(x[:, i][:, None] - x[:, i][None, :]); Kx = np.exp(-.5 * (dx / h) ** 2); out = np.empty((len(js), 3))
    for a, j in enumerate(js):
        dy = np.abs(x[:, j][:, None] - x[:, j][None, :]); ex = tdist <= w[a]; keep = ~ex
        d = np.where(ex, np.inf, np.maximum(dx, dy)); ds = np.sort(d, 1)
        for c, k in enumerate((4, 10)):
            eps = ds[:, k - 1]
            nx = np.sum(keep & (dx < eps[:, None]), 1); ny = np.sum(keep & (dy < eps[:, None]), 1)
            out[a, c] = digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))
        Ky = np.exp(-.5 * (dy / h) ** 2) * keep; Kxk = Kx * keep; cnt = keep.sum(1)
        out[a, 2] = np.mean(np.log(cnt * (Kxk * Ky).sum(1) / (Kxk.sum(1) * Ky.sum(1))))
    return out


def edge_all(x):
    x = (x - x.mean(0)) / x.std(0); T, M = x.shape; iu = np.triu_indices(M, 1)
    r = np.corrcoef(x.T)[iu]; A = np.array([acf(x[:, m]) for m in range(M)])
    W = np.array([[min(int(2 * np.dot(A[i], A[j])), T // 10) for j in range(M)] for i in range(M)])
    t = np.arange(T); tdist = np.abs(t[:, None] - t[None, :]); h = T ** (-1 / 6)
    e = np.concatenate([rows_mi(x, i, np.arange(i + 1, M), W[i, i + 1:], tdist, h) for i in range(M - 1)])
    return np.column_stack([-0.5 * np.log(1 - r ** 2), e]).T  # (4, E): g, k4, k10, kde


def one(args):
    x, seed = args; v = ns(x)  # edges.one_recompute's normal-scored branch: draw seeded by the row index
    s = multivariate_iaaft(v, np.random.default_rng(seed), final='spectrum')[0]
    return np.array([edge_all(v), edge_all(s)], np.float32)  # (2 real/surr, 4, E)


def score(d):
    L = [np.load(f) for f in sorted(glob.glob(f'{d}/est_[AB]_*.npz'))]; cat = lambda k: np.concatenate([q[k] for q in L])
    X = np.concatenate([q['X'] for q in L]).astype(float); Vz = cat('Vzns')
    unit, state, agent, animal = cat('unit'), cat('state'), cat('agent'), cat('animal')
    z = np.array([[[zc(X[w, rs, 0], X[w, rs, e]) for e in (1, 2, 3)] for rs in (0, 1)] for w in range(len(X))])  # (W, 2, 3)
    gap = np.abs(z[:, :, 0] - Vz).max(); print(f'reproduction gate: max |k4 z − stored| = {gap:.2e} ->', 'PASS' if gap < 1e-4 else 'FAIL')
    if gap >= 1e-4: return
    lev = (X[:, :, 1:, :] - X[:, :, :1, :]).mean(-1)  # (W, 2, 3) mean excess over Gaussian
    for c, e in enumerate(EST):
        res, lres = z[:, 0, c] - z[:, 1, c], lev[:, 0, c] - lev[:, 1, c]; S = []
        for u in np.unique(unit):
            m = lambda s: (unit == u) & (state == s); ag = agent[unit == u][0]; s = 'sleep' if ag == 'SLP' else 'drug'
            if m(s).any(): S.append((ag, animal[unit == u][0], *(q[m(s)].mean() - q[m('awake_ec')].mean() for q in (res, lres)),
                                     res[m('awake_eo')].mean() - res[m('awake_ec')].mean()))
        ag, an = np.array([q[0] for q in S]), np.array([q[1] for q in S]); dz, dl, eo = (np.array([q[k] for q in S]) for k in (2, 3, 4))
        dr = ag != 'SLP'; A = np.array([dz[dr & (an == x)].mean() for x in np.unique(an[dr])])
        agm = {a: dz[ag == a].mean() for a in ('KT', 'KTMD', 'MD', 'PF', 'SLP')}; pooled = dz[dr].mean(); p = ttest_1samp(A, 0, alternative='less').pvalue
        sup = (A < 0).all() and p < .05 and all(agm[a] < 0 for a in ('KT', 'KTMD', 'MD', 'PF')) and abs(agm['SLP']) < .3 * abs(pooled)
        spec = (A < 0).sum() <= 2 or pooled >= 0
        print(f'\n{e}: verdict', 'SUPPORTS' if sup else 'ESTIMATOR-SPECIFIC' if spec else 'PARTIAL')
        print(f'  animal means {np.round(A, 4)}; {(A < 0).sum()}/4 < 0; one-sided t p = {p:.4f}; pooled drug {pooled:+.4f}; eyes-open SD {eo.std(ddof=1):.4f}')
        for a in agm: print(f'  {a:5s} Δz {agm[a]:+.4f} ({np.sum(dz[ag == a] < 0)}/{np.sum(ag == a)} down) | level Δ(e − g) real − surr {dl[ag == a].mean():+.4f} nats')


if __name__ == '__main__':
    if sys.argv[1] == 'score': score(sys.argv[2]); sys.exit()
    d, out, agent, rowmode, nw = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    man = json.load(open(f'{d}/manifest.json')); Wn = np.load(f'{d}/windows.npz')
    rows = [r for r in man['rows'] if r['state'] in STATES and r['agent'] in agent.split(',')]  # as edges.py recompute
    S0 = sub32(rows[0]['M']); sub = lambda r: S0 if rowmode == 'first' else sub32(r['M'])
    with Pool(nw) as p: res = p.map(one, [(Wn[r['id']].astype(float)[:, sub(r)], k) for k, r in enumerate(rows)], chunksize=1)
    m, _ = meta(rows, man); zns = sys.argv[2].replace('/est/est_', '/zns/zns_')
    np.savez(out, X=np.array(res), Vzns=np.load(zns)['V'][:, 1], **m)
    print('done', out, flush=True)
