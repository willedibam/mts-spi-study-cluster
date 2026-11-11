"""Atlas step-1 discovery analysis on z (design.md, "Analysis"; reading rules in docs/interpreting-spi-pairs.md).

reduce: per window, from the extracted MPIs (extract.py shards) and the bank's raw windows:
  - z over ordered off-diagonal entries (upper triangle), and z of double-centred profiles (source and target
    means removed: pair-level correspondence only, an exploratory estimand);
  - per-SPI edge mean and SD (mean-SPI), and node-additivity R^2 (share of the profile explained by a source
    plus a target channel effect: high = the SPI reads channel properties, not pair coupling);
  - validated core readouts (mtsspi.character_axes.core_scores);
  - covariates: mean and SD of |r| over pairs, covariance participation ratio, spectral entropy and peak
    frequency of the channel-mean spectrum, plus the bank's delta fraction.
analyse: per arm, the coordinate set is the SPIs finite and non-constant in every window. Per window: effective
  rank of z, mean between-family z (pyspi modules), the precision-rank block, core readouts. Within-unit
  contrasts of each state against the reference state, raw and adjusted for the covariates, with the
  eyes-open null for scale; and the shared drug direction in z and in mean-SPI, its consistency across units,
  how much of it lies in correlation-family x correlation-family pairs or on node-additive SPIs, and whether
  its size tracks the change in coupling strength.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.signal import welch

from mtsspi.character_axes import core_scores
from research.representation.character_axes.episodic import sleep_tertiles

PRECISION = ('prec_', 'prec-sq_', 'pcorr_')
PRECISION_SKIP = ('EmpiricalCovariance', 'GraphicalLasso')
RANK = ('spearmanr', 'spearmanr-sq', 'kendalltau', 'kendalltau-sq', 'ids')
CORR_FAMILY = ('cov_', 'cov-sq_', 'corr_', 'spearmanr', 'kendalltau', 'xcorr', 'mi_gaussian', 'tlmi_gaussian',
               'je_gaussian', 'ce_gaussian', 'lmfit', 'pdist_', 'gc_gaussian')
COVARIATES = ('mean_abs_r', 'sd_abs_r', 'cov_pr', 'delta_fraction')


_PINV = {}


def additive_projector(M):
    """Least-squares source + target channel-effect fit over the ordered off-diagonal entries:
    (pseudo-inverse of the design, source index, target index); the fit is coef[i] + coef[M + j]."""
    if M not in _PINV:
        i, j = np.nonzero(~np.eye(M, dtype=bool))
        X = np.zeros((len(i), 2 * M))
        X[np.arange(len(i)), i] = 1
        X[np.arange(len(i)), M + j] = 1
        _PINV[M] = (np.linalg.pinv(X), i, M + j)
    return _PINV[M]


def offdiag_stats(A):
    """Ordered off-diagonal profile, its double-centred version (source and target effects removed by least
    squares) and node-additivity R^2."""
    M = A.shape[0]
    p = A[~np.eye(M, dtype=bool)]
    if not np.isfinite(p).all():
        return p, np.full_like(p, np.nan), np.nan
    pinv, i, j = additive_projector(M)
    coef = pinv @ p
    fit = coef[i] + coef[j]
    resid = p - fit
    ss = np.sum((p - p.mean()) ** 2)
    return p, resid, (1 - np.sum(resid ** 2) / ss) if ss > 0 else np.nan


def corr_upper(P):
    """Pearson correlation between rows of P (K, E), upper triangle; NaN rows for nonfinite/constant profiles."""
    X = P - P.mean(1, keepdims=True)
    n = np.linalg.norm(X, axis=1)
    ok = np.isfinite(n) & (n > 1e-12) & np.isfinite(X).all(1)
    Y = np.zeros_like(X)
    Y[ok] = X[ok] / n[ok, None]
    Z = Y @ Y.T
    Z[~ok] = np.nan
    Z[:, ~ok] = np.nan
    return Z[np.triu_indices(len(P), 1)].astype(np.float32), ok


def spectral(x, fs):
    f, p = welch(x, fs=fs, nperseg=min(len(x), int(2 * fs)), axis=0)
    p = p.mean(1)
    band = (f >= 0.5) & (f <= 40)
    q = p[band] / p[band].sum()
    return float(-(q * np.log(q)).sum() / np.log(len(q))), float(f[band][np.argmax(p[band])])


def reduce_window(mpis, spis, x, fs):
    K = len(spis)
    profiles, centred, r2 = [], [], np.full(K, np.nan)
    for k in range(K):
        p, c, r2[k] = offdiag_stats(np.asarray(mpis[k], dtype=np.float64))
        profiles.append(p); centred.append(c)
    P, Q = np.array(profiles), np.array(centred)
    z, valid = corr_upper(P)
    zdc, _ = corr_upper(Q)
    # pyspi leaves MPI diagonals NaN; channels are z-scored (normalise=True), so the covariance is the correlation.
    cov = np.array(mpis[spis.index('cov_EmpiricalCovariance')], dtype=np.float64)
    np.fill_diagonal(cov, 1.0)
    r = cov[np.triu_indices(len(cov), 1)]
    lam = np.linalg.eigvalsh(cov)
    entropy, peak = spectral(x, fs)
    with np.errstate(all='ignore'):
        core, names = core_scores(P[None], spis)
    return dict(z=z, zdc=zdc, valid=valid, mean=np.nanmean(P, 1), sd=np.nanstd(P, 1), node_r2=r2,
                core=dict(zip(names, np.asarray(core).ravel().tolist())),
                mean_abs_r=float(np.mean(np.abs(r))), sd_abs_r=float(np.std(np.abs(r))),
                cov_pr=float(lam.sum() ** 2 / np.sum(lam ** 2)), spectral_entropy=entropy, peak_hz=peak)


def reduce_shard(path, bank, fs):
    d = np.load(path, allow_pickle=True)
    spis = [str(s) for s in d['__spis__']]
    out = {}
    with np.load(bank / 'windows.npz') as archive:
        for wid in [k for k in d.files if not k.startswith('__')]:
            out[wid] = reduce_window(d[wid], spis, np.asarray(archive[wid], dtype=np.float64), fs)
    meta = json.loads(str(d['__meta__']))
    return spis, [s['module'].split('.')[-1] for s in meta['pyspi']['spis']], out


def reduce(bank, pyspi_dir, output, jobs):
    manifest = json.loads((bank / 'manifest.json').read_text())
    shards = sorted(pyspi_dir.glob('shard-*.npz'))
    results = Parallel(n_jobs=jobs)(delayed(reduce_shard)(s, bank, manifest['fs']) for s in shards)
    spis, modules = results[0][0], results[0][1]
    merged = {}
    for s, _, out in results:
        if s != spis:
            raise ValueError('SPI order differs between shards')
        merged.update(out)
    rows = [r for r in manifest['rows'] if r['id'] in merged]
    w = [merged[r['id']] for r in rows]
    core_names = sorted({k for v in w for k in v['core']})
    np.savez_compressed(
        output, ids=np.array([r['id'] for r in rows]), spis=np.array(spis), modules=np.array(modules),
        z=np.stack([v['z'] for v in w]), zdc=np.stack([v['zdc'] for v in w]), valid=np.stack([v['valid'] for v in w]),
        mean=np.stack([v['mean'] for v in w]), sd=np.stack([v['sd'] for v in w]), node_r2=np.stack([v['node_r2'] for v in w]),
        core=np.array([[v['core'].get(n, np.nan) for n in core_names] for v in w]), core_names=np.array(core_names),
        covariates=np.array([[v[k] if k in v else r.get(k, np.nan) for k in COVARIATES + ('spectral_entropy', 'peak_hz')]
                             for v, r in zip(w, rows)]),
        covariate_names=np.array(COVARIATES + ('spectral_entropy', 'peak_hz')),
        rows=np.array(json.dumps(rows)))


def summaries(Z, spis, modules, keep):
    """Per-window effective rank, mean between-family z and the precision-rank block on the kept SPIs."""
    K = len(spis)
    iu = np.triu_indices(K, 1)
    names = [spis[k] for k in keep]
    mods = np.array([modules[k] for k in keep])
    prec = np.array([n.startswith(PRECISION) and not n.endswith(PRECISION_SKIP) for n in names])
    rank = np.array([n in RANK for n in names])
    out = np.full((len(Z), 3), np.nan)
    for w, z in enumerate(Z):
        full = np.eye(K)
        full[iu] = z
        full = full + full.T - np.eye(K)
        S = full[np.ix_(keep, keep)]
        lam = np.clip(np.linalg.eigvalsh(np.nan_to_num(S)), 0, None)
        between = mods[:, None] != mods[None, :]
        out[w] = [lam.sum() ** 2 / np.sum(lam ** 2), np.nanmean(S[between]), np.nanmean(S[np.ix_(prec, rank)])]
    return out, ['effective_rank', 'between_family', 'precision_rank']


def unit_contrast(values, unit, state, s, ref, covariates=None):
    """Per-unit difference of state s against ref; with covariates, the state coefficient of a within-unit OLS."""
    diffs = []
    for u in np.unique(unit):
        a, b = (unit == u) & (state == s), (unit == u) & (state == ref)
        if not a.any() or not b.any():
            continue
        if covariates is None:
            diffs.append(np.nanmean(values[a]) - np.nanmean(values[b]))
            continue
        m = a | b
        X = np.column_stack([np.ones(m.sum()), a[m].astype(float), covariates[m]])
        ok = np.isfinite(values[m]) & np.isfinite(X).all(1)
        if ok.sum() <= X.shape[1] + 2:
            continue
        Xs = X[ok].copy()
        Xs[:, 2:] = (Xs[:, 2:] - Xs[:, 2:].mean(0)) / (Xs[:, 2:].std(0) + 1e-12)
        diffs.append(np.linalg.lstsq(Xs, values[m][ok], rcond=None)[0][1])
    diffs = np.array(diffs)
    if not len(diffs):
        return None
    return dict(mean_difference=float(np.mean(diffs)), units=len(diffs), units_positive=int(np.sum(diffs > 0)),
                d=float(np.mean(diffs) / np.std(diffs, ddof=1)) if len(diffs) > 2 and np.std(diffs) > 0 else None)


def unit_means(X, unit, state, s):
    us = [u for u in np.unique(unit) if ((unit == u) & (state == s)).any()]
    return us, np.array([np.nanmean(X[(unit == u) & (state == s)], 0) for u in us])


def directions(Zk, mean_k, cov, unit, state, agent, ref, corr_pair, node_pair):
    """Per-agent drug direction in z (standardised by reference-state SD within unit) and in mean-SPI."""
    out = {}
    for s in sorted(set(state) - {ref}):
        for a in sorted(set(agent[state == s])):
            sel = agent == a
            deltas, mdeltas, dstrength = [], [], []
            for u in np.unique(unit[sel & (state == s)]):
                A, R = (unit == u) & (state == s), (unit == u) & (state == ref)
                if not R.any():
                    continue
                sd = np.nanstd(Zk[R], 0) + 1e-3
                deltas.append((np.nanmean(Zk[A], 0) - np.nanmean(Zk[R], 0)) / sd)
                msd = np.nanstd(mean_k[R], 0) + 1e-9
                mdeltas.append((np.nanmean(mean_k[A], 0) - np.nanmean(mean_k[R], 0)) / msd)
                dstrength.append(np.nanmean(cov[A, 0]) - np.nanmean(cov[R, 0]))
            if len(deltas) < 2:
                continue
            D, MD = np.nan_to_num(np.array(deltas)), np.nan_to_num(np.array(mdeltas))
            cos = lambda X: float(np.mean([np.dot(X[i], X[j]) / (np.linalg.norm(X[i]) * np.linalg.norm(X[j]) + 1e-12)
                                           for i in range(len(X)) for j in range(i + 1, len(X))]))
            mean_dir = D.mean(0)
            proj = [float(np.dot(D[i], np.delete(D, i, 0).mean(0)) / (np.linalg.norm(np.delete(D, i, 0).mean(0)) + 1e-12))
                    for i in range(len(D))] if len(D) > 2 else []
            mass = np.abs(mean_dir) / (np.abs(mean_dir).sum() + 1e-12)
            out[f'{a}:{s}'] = dict(
                units=len(D), z_consistency=cos(D), mean_spi_consistency=cos(MD),
                share_corr_x_corr=float(mass[corr_pair].sum()), share_node_additive=float(mass[node_pair].sum()),
                coordinate_share_corr_x_corr=float(corr_pair.mean()), coordinate_share_node_additive=float(node_pair.mean()),
                projection_vs_strength_change=float(np.corrcoef(proj, dstrength)[0, 1]) if len(proj) > 2 else None,
                mean_direction=mean_dir.astype(np.float32).tolist())
    return out


def analyse(reduced, output, reference):
    r = np.load(reduced, allow_pickle=False)
    rows = json.loads(str(r['rows']))
    spis, modules = [str(s) for s in r['spis']], [str(m) for m in r['modules']]
    keep = np.flatnonzero(r['valid'].all(0))
    K = len(spis)
    iu = np.triu_indices(K, 1)
    kept = np.isin(iu[0], keep) & np.isin(iu[1], keep)
    unit = np.array([x['unit'] for x in rows])
    agent = np.array([x.get('agent', 'all') for x in rows])
    state = np.array(sleep_tertiles(rows) if any('delta_fraction' in x for x in rows) else [x['state'] for x in rows])
    cov = r['covariates']
    names = [str(n) for n in r['covariate_names']]
    adjust = cov[:, [names.index(c) for c in COVARIATES]]
    summ, summ_names = summaries(r['z'], spis, modules, keep)
    summ_dc, _ = summaries(r['zdc'], spis, modules, keep)
    metrics = {**{n: summ[:, k] for k, n in enumerate(summ_names)},
               **{f'dc_{n}': summ_dc[:, k] for k, n in enumerate(summ_names)},
               **{f'core_{n}': r['core'][:, k] for k, n in enumerate(r['core_names'].astype(str))},
               **{f'cov_{n}': cov[:, k] for k, n in enumerate(names)}}
    contrasts = {}
    for s in sorted(set(state) - {reference}):
        for a in sorted(set(agent[state == s])):
            sel = agent == a
            contrasts[f'{a}:{s}'] = {m: dict(raw=unit_contrast(v[sel], unit[sel], state[sel], s, reference),
                                             adjusted=None if m.startswith('cov_') else
                                             unit_contrast(v[sel], unit[sel], state[sel], s, reference, adjust[sel]))
                                     for m, v in metrics.items()}
    node = np.nanmedian(r['node_r2'][state == reference], 0) > .5
    corr_spi = np.array([n.startswith(CORR_FAMILY) for n in spis])
    a_idx, b_idx = iu[0][kept], iu[1][kept]
    report = dict(
        reduced=str(reduced), reference=reference, windows=len(rows), spis_kept=len(keep), spis_dropped=[spis[k] for k in range(K) if k not in set(keep)],
        core_readouts=r['core_names'].astype(str).tolist(), covariates_adjusted=list(COVARIATES),
        node_additive_spis={spis[k]: float(np.nanmedian(r['node_r2'][state == reference][:, k])) for k in np.flatnonzero(node)},
        contrasts=contrasts,
        directions=directions(r['z'][:, kept], r['mean'][:, keep], cov, unit, state, agent, reference,
                              corr_spi[a_idx] & corr_spi[b_idx], node[a_idx] | node[b_idx]),
        top_direction_pairs={})
    for k, v in report['directions'].items():
        md = np.array(v.pop('mean_direction'))
        report['top_direction_pairs'][k] = [dict(a=spis[a_idx[i]], b=spis[b_idx[i]], delta=float(md[i]))
                                           for i in np.argsort(-np.abs(md))[:15]]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('reduce')
    a.add_argument('bank', type=Path)
    a.add_argument('pyspi', type=Path)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--jobs', type=int, default=8)
    b = sub.add_parser('analyse')
    b.add_argument('reduced', type=Path)
    b.add_argument('--output', type=Path, required=True)
    b.add_argument('--reference', default='awake_ec')
    args = p.parse_args()
    if args.command == 'reduce':
        reduce(args.bank, args.pyspi, args.output, args.jobs)
    else:
        analyse(args.reduced, args.output, args.reference)
