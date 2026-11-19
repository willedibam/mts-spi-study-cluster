"""Reordering test for the macaque MI pair: is the real − IAAFT fall of z(Gaussian MI, kNN MI) a reordering of edges,
and if so, is it channel-level or pair-level?

Why. With one shared scalar gain, x = a(t) u with Gaussian u, every edge is a Gaussian scale mixture with the same mixing
law, so its MI is one monotone function of its correlation: kNN MI bends against Gaussian MI but keeps the edge order.
A channel-specific noise floor, C(a) = a^2 S + N, can reorder edges, but through channel properties. So:
  1 − zP^2  share of the kNN-MI profile variance not explained linearly by Gaussian MI (the existing statistic, Pearson)
  1 − zS^2  the same with Spearman (rank) correlation, descriptive
  R_iso     share not explained by the best monotone increasing function of Gaussian MI (isotonic fit): reordering
  R_pair    share left after the isotonic fit AND an additive node fit (c + a_i + a_j) to its residual: pair-level reordering
Each is computed on the real window and on its surrogate (the stored zns draws), and read as real − surrogate.

  python reorder.py calib <n workers>          synthetic calibration (prints, fixed seeds)
  python reorder.py score <dir with zns_[AB]_*.npz>

Decision rule, fixed 19 November 2026 before any calibration or real output was computed:
- Calibration gate (synthetic, 32 channels, T = 1000, the zns pipeline: normal scores, one spectrum-final joint IAAFT draw,
  edges.edge_mi). Scenarios: linear Gaussian; shared gain with a homogeneous floor; shared gain with a heterogeneous floor;
  pair-specific non-monotone coupling. The R_pair verdict below is DIAGNOSTIC only if R_pair excess is not significantly
  > 0 (one-sided t over windows, p > .05) under both shared-gain scenarios and is significantly > 0 (p < .05) under
  pair-specific coupling. Otherwise R_pair is reported as non-diagnostic and only R_iso is read.
- Real data: per window res = Q_real − Q_surr; session Δ = mean(drug) − mean(awake_ec); animal means pool the drug
  sessions (KT, KTMD, MD, PF) per animal, as montage/animal_level.py. "Passes" for Q means: 4/4 animal means > 0, one-sided
  t on animal means p < .05, and every agent's session mean > 0.
  PAIR-LEVEL REORDERING     R_pair passes (and the gate holds).
  CHANNEL-LEVEL REORDERING  R_iso passes, R_pair does not.
  MONOTONE                  R_iso animal means > 0 in ≤ 2/4, or the pooled drug Δ R_iso ≤ 0: consistent with a shared gain
                            bending the map without reordering.
  PARTIAL                   anything else.
- Descriptive: the share ΔR_iso / Δ(1 − zP^2) of the Pearson-scale excess that is reordering; sleep and eyes-open contrasts.

Outcome of the rule above (19 November): the calibration gate FAILED. A shared gain with a homogeneous floor also raises
R_iso (+.099) and R_pair (+.053) at T = 1000, most plausibly because slow gain inflates kNN estimator noise, which reads as
reordering. Real data: R_iso and R_pair pass (4/4 animals; R_iso is .84 of the Pearson-scale excess), but neither separates
gain from coupling.

Version 2, cross-window reproducibility, fixed 19 November 2026 after the version-1 outputs above and before any version-2
calibration or real output. Estimator noise differs between windows; pair-specific structure repeats.
  e_w       pair-level residual of window w (after the isotonic fit and the additive node fit), real and surrogate
  rho       mean over distinct window pairs, within one session and state, of corr(e_w, e_w'); excess = rho_real − rho_surr
  python reorder.py calib2 <n workers>   synthetic sessions: fixed loadings and pair structure, independent windows
  python reorder.py score2 <dir with zns_[AB]_*.npz>
- Calibration gate: excess not significantly > 0 (one-sided t over synthetic sessions, p > .05) under both shared-gain
  scenarios, and significantly > 0 under pair-specific coupling. If it fails, version 2 is reported as non-diagnostic.
- Real data: session Δ = excess(drug or sleep) − excess(awake_ec); animal means pool drug sessions as above.
  STABLE PAIR-LEVEL REORDERING  4/4 animal means > 0, one-sided t p < .05, every agent's session mean > 0 (gate holds).
  NOT REPRODUCIBLE              animal means > 0 in ≤ 2/4, or pooled drug Δ ≤ 0: consistent with window-specific noise.
  PARTIAL                       anything else. Descriptive: awake_ec excess itself, eyes-open contrast, sleep.

Outcome of version 2 (19 November): the gate FAILED narrowly. Shared gain with a homogeneous floor gives no reproducible
residual (−.002), but shared gain with a heterogeneous, fixed channel floor does (+.037, p = .033; pair-specific coupling
+.90), because SNR-ordered gain leaves a product structure f(SNR_i, SNR_j) that the additive node fit does not remove.
Real data, descriptive only: drug − awake excess, animal means +.082, +.019, +.031, +.045 (one-sided p = .024); KT +.149
(4/4), KTMD +.028 (11/11), MD +.027 (4/4), PF +.008 (2/4), sleep +.006 (5/7); awake_ec excess +.007; eyes-open SD .015.
Only KT clearly exceeds the heterogeneous-gain calibration; propofol's reordering does not repeat across windows.
"""
import sys, glob, numpy as np, warnings
from multiprocessing import Pool
from scipy.optimize import isotonic_regression
from scipy.stats import rankdata, ttest_1samp
from mtsspi.surrogates import multivariate_iaaft
from edges import edge_mi, ns
warnings.filterwarnings('ignore')
NAMES = ('1-zP2', '1-zS2', 'R_iso', 'R_pair')


def stats(g, k, M):
    g, k = np.asarray(g, float), np.asarray(k, float); iu = np.triu_indices(M, 1); ss = np.sum((k - k.mean()) ** 2)
    zP = np.corrcoef(g, k)[0, 1]; zS = np.corrcoef(rankdata(g), rankdata(k))[0, 1]
    o = np.argsort(g, kind='stable'); fit = np.empty_like(k); fit[o] = isotonic_regression(k[o]).x; e = k - fit
    B = np.zeros((len(k), M + 1)); B[:, 0] = 1; B[np.arange(len(k)), 1 + iu[0]] = 1; B[np.arange(len(k)), 1 + iu[1]] = 1
    e2 = e - B @ np.linalg.lstsq(B, e, rcond=None)[0]
    return np.array([1 - zP ** 2, 1 - zS ** 2, np.sum(e ** 2) / ss, np.sum(e2 ** 2) / ss])


# ---------- synthetic calibration ----------
M, T = 32, 1000


def ar(rng, n, phi):
    e = rng.standard_normal((T + 200, n)); x = np.empty_like(e); x[0] = e[0]
    for t in range(1, len(e)): x[t] = phi * x[t - 1] + e[t]
    x = x[200:]; return x / x.std(0)


def latent(rng):  # smooth Gaussian channels with a spread of correlations: 4 factors plus private parts
    L = rng.standard_normal((M, 4)) * rng.uniform(.2, 1.2, (M, 1))
    u = ar(rng, 4, .9) @ L.T + ar(rng, M, .9); return u / u.std(0)


def gain(rng):  # smooth log-normal gain (slow-wave-like), shared
    return np.exp(.8 * ar(rng, 1, .99)[:, 0])


def scenario(args):
    name, seed = args; rng = np.random.default_rng(seed); u = latent(rng)
    if name == 'linear':
        x = u + .3 * rng.standard_normal((T, M))
    elif name == 'gain_homog':
        x = gain(rng)[:, None] * u + .3 * rng.standard_normal((T, M))
    elif name == 'gain_hetero':
        x = gain(rng)[:, None] * u + rng.uniform(.05, 2., M) * rng.standard_normal((T, M))
    elif name == 'pair_nonmono':  # 8 channels gain a non-monotone (quadratic) dependence on one partner each
        x = u + .3 * rng.standard_normal((T, M)); src = rng.choice(M, 8, replace=False); dst = rng.choice(np.setdiff1d(np.arange(M), src), 8, replace=False)
        for s, d in zip(src, dst): x[:, d] = x[:, d] + 1.2 * (x[:, s] ** 2 - 1)
    v = ns(x); s = multivariate_iaaft(v, np.random.default_rng(seed + 1), final='spectrum')[0]
    gr, kr = edge_mi(v); gs, ks = edge_mi(s)
    return stats(gr, kr, M) - stats(gs, ks, M)


def calib(nw, n=10):
    sc = ('linear', 'gain_homog', 'gain_hetero', 'pair_nonmono')
    with Pool(nw) as p: R = p.map(scenario, [(s, 1000 * i + j) for i, s in enumerate(sc) for j in range(n)], chunksize=1)
    R = np.array(R).reshape(len(sc), n, 4); P = {}
    for i, s in enumerate(sc):
        P[s] = ttest_1samp(R[i, :, 3], 0, alternative='greater').pvalue
        print(f'{s:13s} real − surr ' + '  '.join(f'{nm} {R[i, :, c].mean():+.4f}±{R[i, :, c].std(ddof=1) / np.sqrt(n):.4f}' for c, nm in enumerate(NAMES))
              + f'  | R_pair > 0 p = {P[s]:.3f}')
    ok = P['gain_homog'] > .05 and P['gain_hetero'] > .05 and P['pair_nonmono'] < .05
    print('calibration gate:', 'HOLDS (R_pair diagnostic)' if ok else 'FAILS (R_pair non-diagnostic; read R_iso only)')


def resid(g, k, M):
    g, k = np.asarray(g, float), np.asarray(k, float); iu = np.triu_indices(M, 1)
    o = np.argsort(g, kind='stable'); fit = np.empty_like(k); fit[o] = isotonic_regression(k[o]).x; e = k - fit
    B = np.zeros((len(k), M + 1)); B[:, 0] = 1; B[np.arange(len(k)), 1 + iu[0]] = 1; B[np.arange(len(k)), 1 + iu[1]] = 1
    return e - B @ np.linalg.lstsq(B, e, rcond=None)[0]


def xwin(R):  # mean correlation over distinct window pairs
    C = np.corrcoef(np.asarray(R)); return C[np.triu_indices(len(C), 1)].mean()


def window2(args):
    name, seed, w = args; rng = np.random.default_rng(seed)  # session-level structure from seed, window noise from w
    L = rng.standard_normal((M, 4)) * rng.uniform(.2, 1.2, (M, 1)); fl = rng.uniform(.05, 2., M)
    src = rng.choice(M, 8, replace=False); dst = rng.choice(np.setdiff1d(np.arange(M), src), 8, replace=False)
    r2 = np.random.default_rng(seed * 100 + w + 7); u = ar(r2, 4, .9) @ L.T + ar(r2, M, .9); u = u / u.std(0)
    if name == 'linear': x = u + .3 * r2.standard_normal((T, M))
    elif name == 'gain_homog': x = gain(r2)[:, None] * u + .3 * r2.standard_normal((T, M))
    elif name == 'gain_hetero': x = gain(r2)[:, None] * u + fl * r2.standard_normal((T, M))
    elif name == 'pair_nonmono':
        x = u + .3 * r2.standard_normal((T, M))
        for s_, d_ in zip(src, dst): x[:, d_] = x[:, d_] + 1.2 * (x[:, s_] ** 2 - 1)
    v = ns(x); s = multivariate_iaaft(v, np.random.default_rng(seed * 100 + w + 9), final='spectrum')[0]
    gr, kr = edge_mi(v); gs, ks = edge_mi(s); return resid(gr, kr, M), resid(gs, ks, M)


def calib2(nw, ns_=6, nwin=5):
    sc = ('linear', 'gain_homog', 'gain_hetero', 'pair_nonmono')
    jobs = [(s, 10 * i + j + 1, w) for i, s in enumerate(sc) for j in range(ns_) for w in range(nwin)]
    with Pool(nw) as p: out = p.map(window2, jobs, chunksize=1)
    X = np.array([xwin([out[q][0] for q in range(b, b + nwin)]) - xwin([out[q][1] for q in range(b, b + nwin)])
                  for b in range(0, len(jobs), nwin)]).reshape(len(sc), ns_); P = {}
    for i, s in enumerate(sc):
        P[s] = ttest_1samp(X[i], 0, alternative='greater').pvalue
        print(f'{s:13s} cross-window excess {X[i].mean():+.4f} ± {X[i].std(ddof=1) / np.sqrt(ns_):.4f}  (> 0 p = {P[s]:.3f})')
    ok = P['gain_homog'] > .05 and P['gain_hetero'] > .05 and P['pair_nonmono'] < .05
    print('calibration gate (v2):', 'HOLDS' if ok else 'FAILS (v2 non-diagnostic)')


def score2(d):
    L = [np.load(f) for f in sorted(glob.glob(f'{d}/zns_[AB]_*.npz'))]; cat = lambda k: np.concatenate([q[k] for q in L])
    E = np.concatenate([q['E'] for q in L])[:, 1].astype(float); unit, state, agent, animal = cat('unit'), cat('state'), cat('agent'), cat('animal')
    Rr = [resid(E[w, 0, 0], E[w, 0, 1], 32) for w in range(len(E))]; Rs = [resid(E[w, 1, 0], E[w, 1, 1], 32) for w in range(len(E))]
    ex = lambda m: xwin([Rr[w] for w in np.where(m)[0]]) - xwin([Rs[w] for w in np.where(m)[0]]); S = []
    for u in np.unique(unit):
        m = lambda s: (unit == u) & (state == s); ag = agent[unit == u][0]; s = 'sleep' if ag == 'SLP' else 'drug'
        if m(s).sum() > 1: S.append((ag, animal[unit == u][0], ex(m(s)) - ex(m('awake_ec')), ex(m('awake_eo')) - ex(m('awake_ec')), ex(m('awake_ec'))))
    ag, an = np.array([q[0] for q in S]), np.array([q[1] for q in S]); D, EO, B0 = (np.array([q[k] for q in S]) for k in (2, 3, 4)); dr = ag != 'SLP'
    A = np.array([D[dr & (an == x)].mean() for x in np.unique(an[dr])]); p = ttest_1samp(A, 0, alternative='greater').pvalue
    agm = {a: D[ag == a].mean() for a in ('KT', 'KTMD', 'MD', 'PF', 'SLP')}
    ok = (A > 0).all() and p < .05 and all(agm[a] > 0 for a in ('KT', 'KTMD', 'MD', 'PF'))
    v = 'STABLE PAIR-LEVEL REORDERING' if ok else 'NOT REPRODUCIBLE' if (A > 0).sum() <= 2 or D[dr].mean() <= 0 else 'PARTIAL'
    print(f'awake_ec excess (all sessions) {B0.mean():+.4f}; eyes-open contrast mean {EO.mean():+.4f} SD {EO.std(ddof=1):.4f}')
    print(f'animal means {np.round(A, 4)}, one-sided p = {p:.4f}; ' + '  '.join(f'{a} {agm[a]:+.4f} ({np.sum(D[ag == a] > 0)}/{np.sum(ag == a)})' for a in agm))
    print(f'verdict (v2, subject to its gate): {v}')


# ---------- real data ----------
def score(d):
    L = [np.load(f) for f in sorted(glob.glob(f'{d}/zns_[AB]_*.npz'))]; cat = lambda k: np.concatenate([q[k] for q in L])
    E = np.concatenate([q['E'] for q in L])[:, 1].astype(float)  # normal-scored: (W, real/surr, g/k, 496)
    unit, state, agent, animal = cat('unit'), cat('state'), cat('agent'), cat('animal')
    Q = np.array([stats(E[w, 0, 0], E[w, 0, 1], 32) - stats(E[w, 1, 0], E[w, 1, 1], 32) for w in range(len(E))])
    S = []
    for u in np.unique(unit):
        m = lambda s: (unit == u) & (state == s); ag = agent[unit == u][0]; s = 'sleep' if ag == 'SLP' else 'drug'
        if m(s).any(): S.append((ag, animal[unit == u][0], Q[m(s)].mean(0) - Q[m('awake_ec')].mean(0), Q[m('awake_eo')].mean(0) - Q[m('awake_ec')].mean(0)))
    ag, an = np.array([q[0] for q in S]), np.array([q[1] for q in S]); D, EO = np.array([q[2] for q in S]), np.array([q[3] for q in S]); dr = ag != 'SLP'

    def passes(c):
        A = np.array([D[dr & (an == x), c].mean() for x in np.unique(an[dr])]); p = ttest_1samp(A, 0, alternative='greater').pvalue
        return (A > 0).all() and p < .05 and all(D[ag == a, c].mean() > 0 for a in ('KT', 'KTMD', 'MD', 'PF')), A, p
    for c, nm in enumerate(NAMES):
        ok, A, p = passes(c)
        print(f'{nm:6s} animal means {np.round(A, 4)} (one-sided p = {p:.4f}, passes {ok}); eyes-open SD {EO[:, c].std(ddof=1):.4f}; '
              + '  '.join(f'{a} {D[ag == a, c].mean():+.4f} ({np.sum(D[ag == a, c] > 0)}/{np.sum(ag == a)})' for a in ('KT', 'KTMD', 'MD', 'PF', 'SLP')))
    iso, pair = passes(2), passes(3); pooled_iso = D[dr, 2].mean()
    v = ('PAIR-LEVEL REORDERING' if pair[0] else 'CHANNEL-LEVEL REORDERING' if iso[0]
         else 'MONOTONE' if (iso[1] > 0).sum() <= 2 or pooled_iso <= 0 else 'PARTIAL')
    print(f'verdict (R_pair subject to the calibration gate): {v}')
    print(f'share of the Pearson-scale excess that is reordering, ΔR_iso / Δ(1 − zP²), pooled drugs: {pooled_iso / D[dr, 0].mean():.2f}; '
          f'pair-level ΔR_pair / Δ(1 − zP²): {D[dr, 3].mean() / D[dr, 0].mean():.2f}')


if __name__ == '__main__':
    {'calib': lambda a: calib(int(a)), 'score': score, 'calib2': lambda a: calib2(int(a)), 'score2': score2}[sys.argv[1]](sys.argv[2])
