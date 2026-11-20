"""Axis screen MinCovDet mechanism check (post hoc; criteria in result.md, committed before this ran).

The guide (docs/interpreting-spi-pairs.md, robust against classical) reads a robust-classical divergence as
time-heterogeneous dependence, and "episodic coupling" only if the down-weighted samples are temporally clustered,
involve several channels and carry a distinct network. sklearn's MinCovDet covariance is the classical covariance of its
reweighted support, so the classical-MCD difference is carried entirely by the samples outside that support.

Per window and version (real and the screen's 3 IAAFT draws, same seeds), on the z-scored normal-scored window and the
screen's MinCovDet fit (random_state 42):
  eps      fraction of samples outside the reweighted support
  on the top K = 5% of samples by robust Mahalanobis distance (a fixed count, so real and IAAFT compare at equal n):
  fano     variance / mean of top-K counts in 0.5-s bins (about 1 - K/T for independent samples)
  run      mean run length of top-K samples, in samples
  n_ext    mean over top-K samples of the number of channels with |x| > 2 (normal scores: about 5.7 of 126 by chance)
  pr_node  participation ratio over channels of node energy sum_j d_ij^2 of d = corr(top K) - corr(rest): the effective
           number of channels the outlying network involves (about M for unstructured noise)
  amp      mean over channels of x^2 at top-K samples over that at the rest (amplitude, against pattern atypicality)
  r_net    correlation of the top-K samples' edge correlation profile with the rest's
  z_pc, z_c2   the screen's z(pcorr, pcorr_mcd) and z(r2, mcd2), recomputed from the same fit
Per unit x state and version: recurrence = mean pairwise correlation across windows of corr(top K) - corr(rest).

prep (preprocessing sensitivity, result.md): 'ns' (default) normal-scores each channel per window, as the screen; 'z' only
z-scores it (the pyspi treatment). A '-m' suffix ends the IAAFT on the marginal step (exact marginals, approximate
spectra) instead of the spectrum step. On z-scored heavy-tailed data the spectrum-final draws lose the marginal tails,
and MinCovDet responds to marginal kurtosis alone, so the z-scored test needs '-m'.
  kurt     mean excess kurtosis over channels of the prepared window (0 after normal-scoring)

  python -m research.representation.atlas.axis_screen.mechanism compute <bank> <out dir> <workers> [ns|z|ns-m|z-m]
  python -m research.representation.atlas.axis_screen.mechanism collect <out dir> <manifest.json> <out.npz>
  python -m research.representation.atlas.axis_screen.mechanism score <mechanism.npz> <manifest.json> <out.json>
"""
import sys, json, glob, time, warnings
import numpy as np
from pathlib import Path
from multiprocessing import Pool

KF, BIN = .05, 50
VERS = ('real', 'iaaft0', 'iaaft1', 'iaaft2')
METRICS = ('eps', 'fano', 'run', 'n_ext', 'pr_node', 'amp', 'r_net', 'z_pc', 'z_c2', 'kurt')


def _corr(u, v):
    return float(np.corrcoef(u, v)[0, 1])


def metrics(v):
    """Metrics and the top-K minus rest correlation profile for one normal-scored window (T x M)."""
    from sklearn.covariance import MinCovDet
    from scipy.stats import kurtosis
    from research.representation.atlas.axis_screen.stats import zscore
    x = zscore(np.asarray(v, np.float64))
    T, M = x.shape
    a, b = np.triu_indices(M, 1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mcd = MinCovDet(random_state=42).fit(x)
    K = int(round(KF * T))
    top = np.zeros(T, bool); top[np.argsort(mcd.dist_)[-K:]] = True
    n = top[: T // BIN * BIN].reshape(-1, BIN).sum(1)
    e = np.diff(np.r_[0, top.astype(int), 0])
    g = (x ** 2).mean(1)
    co, ci = np.corrcoef(x[top].T)[a, b], np.corrcoef(x[~top].T)[a, b]
    Dm = np.zeros((M, M)); Dm[a, b] = co - ci; Dm += Dm.T
    s_ = (Dm ** 2).sum(1)
    C = x.T @ x / T
    pc = lambda Q: -Q / np.sqrt(np.outer(np.diag(Q), np.diag(Q)))
    out = dict(eps=1 - mcd.support_.mean(), fano=n.var() / n.mean(), run=K / (e == 1).sum(), n_ext=(np.abs(x[top]) > 2).sum(1).mean(),
               pr_node=s_.sum() ** 2 / (s_ ** 2).sum(), amp=g[top].mean() / g[~top].mean(), r_net=_corr(co, ci),
               z_pc=_corr(pc(np.linalg.inv(C))[a, b], pc(mcd.precision_)[a, b]),
               z_c2=_corr(C[a, b] ** 2, mcd.covariance_[a, b] ** 2), kurt=float(kurtosis(x).mean()))
    return np.array([out[m] for m in METRICS]), (co - ci).astype(np.float32)


def window(args):
    bank, out, wid, prep = args
    from research.representation.atlas.extract import seed
    from research.representation.atlas.axis_screen.run import _ns
    from research.representation.atlas.axis_screen.stats import zscore
    from mtsspi.surrogates import multivariate_iaaft
    target = Path(out) / f'{wid}.npz'
    if target.exists():
        return wid, 0.
    t0 = time.time()
    with np.load(Path(bank) / 'windows.npz') as W:
        v = np.asarray(W[wid], np.float64)
    v = _ns(v) if prep.split('-')[0] == 'ns' else zscore(v)
    final = 'marginal' if prep.endswith('-m') else 'spectrum'
    X = [v] + [multivariate_iaaft(v, np.random.default_rng([seed(wid), 10, k]), final=final)[0] for k in range(3)]
    R = [metrics(x) for x in X]
    np.savez(target, m=np.stack([r[0] for r in R]), d=np.stack([r[1] for r in R]))
    return wid, time.time() - t0


def compute(bank, out, workers, prep='ns'):
    from research.representation.atlas.axis_screen.run import jobs
    Path(out).mkdir(parents=True, exist_ok=True)
    J = [(str(bank), str(out), i, prep) for j in jobs(bank, out) for i, _ in j[4]]
    print('windows', len(J), flush=True)
    with Pool(workers, maxtasksperchild=8) as p:
        for k, (wid, sec) in enumerate(p.imap_unordered(window, J)):
            print(f'{k:5d} {sec:6.0f}s {wid}', flush=True)


def collect(out, manifest, target):
    rows = {r['id']: r for r in json.load(open(manifest))['rows']}
    fs = sorted(glob.glob(f'{out}/*.npz'))
    ids = [Path(f).stem for f in fs]
    m, grp, rec = [], {}, {}
    for i, f in zip(ids, fs):
        with np.load(f) as D:
            m.append(D['m'])
            grp.setdefault(f"{rows[i]['unit']}__{rows[i]['state']}", []).append(D['d'])
    for g, L in grp.items():
        D = np.stack(L)  # (n, version, E)
        r = np.full(len(VERS), np.nan)
        if len(D) >= 3:
            for k in range(len(VERS)):
                Y = D[:, k] - D[:, k].mean(1, keepdims=True); Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                G = Y @ Y.T
                r[k] = G[np.triu_indices(len(D), 1)].mean()
        rec[g] = r
    np.savez(target, ids=np.array(ids), m=np.stack(m), metrics=np.array(METRICS), versions=np.array(VERS),
             groups=np.array(list(rec)), recurrence=np.stack(list(rec.values())))
    print('collected', len(ids), 'windows', len(rec), 'groups ->', target)


def score(path, manifest, output):
    from scipy.stats import ttest_1samp
    from research.representation.atlas.axis_screen.score import DRUGS, REF
    D = np.load(path)
    rows = {r['id']: r for r in json.load(open(manifest))['rows']}
    ids = [str(i) for i in D['ids']]
    unit = np.array([rows[i]['unit'] for i in ids]); state = np.array([rows[i]['state'] for i in ids])
    start = np.array([rows[i].get('start_s', rows[i].get('onset_s', np.nan)) for i in ids], float)
    ag = {rows[i]['unit']: rows[i].get('agent', '') for i in ids}; an = {rows[i]['unit']: rows[i].get('animal', '') for i in ids}
    M = D['m'].astype(float)  # (window, version, metric)
    net = M[:, 0] - M[:, 1:].mean(1)
    has = lambda u, s: ((unit == u) & (state == s)).any()
    ses = sorted(u for u in set(unit) if ag[u] in DRUGS and has(u, 'drug') and has(u, REF))
    animals = sorted({an[u] for u in ses}); A = np.array([an[u] for u in ses])
    mean = lambda x, u, s: np.nanmean(x[(unit == u) & (state == s)], 0)
    delta = lambda x, units, s: np.array([mean(x, u, s) - mean(x, u, REF) for u in units if has(u, s)])

    def animal(d, units):
        a = np.array([an[u] for u in units])
        am = np.array([d[a == k].mean() for k in sorted(set(a))])
        return dict(mean=float(d.mean()), animals=am.round(4).tolist(), t3=float(ttest_1samp(am, 0).statistic) if len(am) > 1 else None,
                    positive=int((am > 0).sum()), negative=int((am < 0).sum()))

    def split_half(x):  # within-awake second half minus first, as score.py
        o = []
        for u in ses:
            r = np.flatnonzero((unit == u) & (state == REF)); r = r[np.argsort(start[r])]; h = len(r) // 2
            o.append(np.nanmean(x[r[-h:]]) - np.nanmean(x[r[:h]]))
        return np.array(o)

    out = dict(sessions=len(ses), animals=animals, metrics={})
    MN = [str(n) for n in D['metrics']]  # as stored (older outputs lack kurt)
    for k, name in enumerate(MN):
        x, xr, xi = net[:, k], M[:, 0, k], M[:, 1:, k].mean(1)
        lev = lambda s, y: float(np.mean([mean(y, u, s) for u in ses]))
        dd = delta(x, ses, 'drug'); sh = split_half(x); nsd = float(np.nanstd(sh, ddof=1) / np.sqrt(2))
        out['metrics'][name] = dict(
            awake=dict(real=lev(REF, xr), iaaft=lev(REF, xi)), drug=dict(real=lev('drug', xr), iaaft=lev('drug', xi)),
            drug_level_net=animal(np.array([mean(x, u, 'drug') for u in ses]), ses),
            awake_level_net=animal(np.array([mean(x, u, REF) for u in ses]), ses),
            delta_net=animal(dd, ses), null_sd=nsd, sd_multiple=float(dd.mean() / nsd), split_half=float(np.nanmean(sh)),
            delta_real=float(delta(xr, ses, 'drug').mean()), delta_iaaft=float(delta(xi, ses, 'drug').mean()),
            agents={a: float(dd[np.array([ag[u] for u in ses]) == a].mean()) for a in DRUGS},
            eyes_open=float(delta(x, ses, 'awake_eo').mean()),
            sleep=float(np.mean(delta(x, [u for u in set(unit) if has(u, 'sleep') and has(u, REF)], 'sleep'))),
            recovery=float(np.mean(delta(x, [u for u in ses if has(u, 'recovery_ec')], 'recovery_ec'))))
    # within drug sessions, window-level correlation of the axis net with each metric's net (session-demeaned)
    dw = np.isin(unit, ses) & (state == 'drug')
    dm = lambda y: y[dw] - np.array([np.nanmean(y[dw & (unit == u)]) for u in unit[dw]])
    out['window_r_with_axis'] = {n: float(np.corrcoef(dm(net[:, MN.index('z_pc')]), dm(net[:, k]))[0, 1])
                                 for k, n in enumerate(MN) if n != 'z_pc'}
    G = [str(g) for g in D['groups']]; R = D['recurrence']
    rec = lambda s: np.array([R[G.index(f'{u}__{s}')] for u in ses])
    rd, ra = rec('drug'), rec(REF)
    out['recurrence'] = dict(drug_real=float(rd[:, 0].mean()), drug_iaaft=float(rd[:, 1:].mean()),
                             awake_real=float(ra[:, 0].mean()), awake_iaaft=float(ra[:, 1:].mean()),
                             drug_level_net=animal(rd[:, 0] - rd[:, 1:].mean(1), ses),
                             delta_net=animal((rd[:, 0] - rd[:, 1:].mean(1)) - (ra[:, 0] - ra[:, 1:].mean(1)), ses))
    json.dump(out, open(output, 'w'), indent=1)
    for n, r in out['metrics'].items():
        print(f"{n:7s} awake real {r['awake']['real']:+.3f} iaaft {r['awake']['iaaft']:+.3f} | drug real {r['drug']['real']:+.3f} "
              f"iaaft {r['drug']['iaaft']:+.3f} | drug net {r['drug_level_net']['mean']:+.4f} ({r['drug_level_net']['positive']}+) "
              f"Δnet {r['delta_net']['mean']:+.4f} ({r['sd_multiple']:+.1f} SD) t3 {r['delta_net']['t3']:+.2f} ({r['delta_net']['positive']}+/{r['delta_net']['negative']}-) "
              f"eo {r['eyes_open']:+.4f} sleep {r['sleep']:+.4f} rec {r['recovery']:+.4f}")
    print('recurrence', out['recurrence'])
    print('window r with axis', {k: round(v, 3) for k, v in out['window_r_with_axis'].items()})


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'compute':
        compute(sys.argv[2], sys.argv[3], int(sys.argv[4]), *sys.argv[5:6])
    elif cmd == 'collect':
        collect(*sys.argv[2:5])
    elif cmd == 'score':
        score(*sys.argv[2:5])
