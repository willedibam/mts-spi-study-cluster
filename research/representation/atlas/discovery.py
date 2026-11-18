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
STORED_COVARIATES = ('mean_abs_r', 'sd_abs_r', 'cov_pr', 'delta_fraction', 'spectral_entropy', 'peak_hz')
COVARIATES = ('mean_abs_r', 'sd_abs_r', 'cov_pr')  # strength; delta_fraction exists only where sleep is staged
SPECTRAL = COVARIATES + ('spectral_entropy', 'peak_hz')


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
        covariates=np.array([[v[k] if k in v else r.get(k, np.nan) for k in STORED_COVARIATES]
                             for v, r in zip(w, rows)]),
        covariate_names=np.array(STORED_COVARIATES),
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


def unit_effects(Y, unit, state, s, ref, covariates=None):
    """Per-unit effect of state s against ref on every column of Y (n, P): raw difference, or with covariates
    the state coefficient of a within-unit OLS (all columns solved at once). Returns (units, P)."""
    out = []
    for u in np.unique(unit):
        a, b = (unit == u) & (state == s), (unit == u) & (state == ref)
        if not a.any() or not b.any():
            continue
        if covariates is None:
            out.append(np.nanmean(Y[a], 0) - np.nanmean(Y[b], 0))
            continue
        m = a | b
        C = covariates[m]
        C = (C - C.mean(0)) / (C.std(0) + 1e-12)
        X = np.column_stack([np.ones(m.sum()), a[m].astype(float), C])
        if m.sum() <= X.shape[1] + 2:
            continue
        out.append(np.linalg.lstsq(X, np.nan_to_num(Y[m]), rcond=None)[0][1])
    return np.array(out)


def screen(reduced, output, reference, drug_state='drug', top=60, double_centred=False):
    """Every kept z coordinate: per-agent drug effects (raw and strength-adjusted), the eyes-open null, sleep and
    recovery. Keeps coordinates with the same sign under every agent, in both versions, in at least 75% of each
    agent's units, and a mean beyond three eyes-open-null SDs; drops pairs touching a node-additive SPI."""
    r = np.load(reduced, allow_pickle=False)
    rows = json.loads(str(r['rows']))
    spis = [str(s) for s in r['spis']]
    keep = np.flatnonzero(r['valid'].all(0))
    K = len(spis)
    iu = np.triu_indices(K, 1)
    kept = np.isin(iu[0], keep) & np.isin(iu[1], keep)
    a_idx, b_idx = iu[0][kept], iu[1][kept]
    Z = (r['zdc'] if double_centred else r['z'])[:, kept].astype(np.float64)
    unit = np.array([x['unit'] for x in rows])
    agent = np.array([x.get('agent', 'all') for x in rows])
    state = np.array(sleep_tertiles(rows) if any('delta_fraction' in x for x in rows) else [x['state'] for x in rows])
    names = [str(n) for n in r['covariate_names']]
    cov = r['covariates'][:, [names.index(c) for c in COVARIATES]]
    node = np.nanmedian(r['node_r2'][state == reference], 0) > .5
    usable = ~(node[a_idx] | node[b_idx])
    agents = sorted(set(agent[state == drug_state]))
    eff = {}
    for a in agents:
        sel = agent == a
        eff[a] = dict(raw=unit_effects(Z[sel], unit[sel], state[sel], drug_state, reference),
                      adjusted=unit_effects(Z[sel], unit[sel], state[sel], drug_state, reference, cov[sel]))
    null = np.concatenate([unit_effects(Z[agent == a], unit[agent == a], state[agent == a], 'awake_eo', reference)
                           for a in sorted(set(agent)) if ((agent == a) & (state == 'awake_eo')).any()]) \
        if (state == 'awake_eo').any() else np.zeros((2, Z.shape[1]))
    null_sd = null.std(0, ddof=1) + 1e-6
    sign = np.sign(np.mean([eff[a]['raw'].mean(0) for a in agents], 0))
    ok = usable.copy()
    for a in agents:
        for v in ('raw', 'adjusted'):
            e = eff[a][v]
            ok &= np.mean(np.sign(e) == sign, 0) >= .75
        ok &= np.abs(eff[a]['raw'].mean(0)) > 3 * null_sd
    strength = np.min([np.abs(eff[a]['adjusted'].mean(0)) / null_sd for a in agents], 0)
    order = np.argsort(-np.where(ok, strength, -np.inf))[:top]
    other = {}
    for s_name in sorted(set(state) - {reference, drug_state, 'awake_eo'}):
        sel = np.isin(agent, sorted(set(agent[state == s_name])))
        e = unit_effects(Z[sel], unit[sel], state[sel], s_name, reference)
        if len(e):
            other[s_name] = e
    hits = []
    for i in order[np.isfinite(strength[order]) & ok[order]]:
        hits.append(dict(
            a=spis[a_idx[i]], b=spis[b_idx[i]], reference_mean=float(np.nanmean(Z[state == reference, i])),
            per_agent={a: dict(raw=float(eff[a]['raw'][:, i].mean()), adjusted=float(eff[a]['adjusted'][:, i].mean()),
                               units=len(eff[a]['raw']), same_sign=int(np.sum(np.sign(eff[a]['raw'][:, i]) == sign[i])))
                       for a in agents},
            eyes_open_null_sd=float(null_sd[i]), min_adjusted_over_null=float(strength[i]),
            other_states={k: float(v[:, i].mean()) for k, v in other.items()}))
    report = dict(reduced=str(reduced), double_centred=double_centred, reference=reference, drug_state=drug_state,
                  agents=agents, coordinates=int(Z.shape[1]), usable=int(usable.sum()), passing=int(ok.sum()),
                  excluded_node_additive=[spis[k] for k in np.flatnonzero(node)], hits=hits)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + '\n')


_MT = lambda f, band='0_fmax-0-5', stat='mean': f'{f}_multitaper_{stat}_fs-1_fmin-{band}'
_LOW, _HIGH = '0_fmax-0-25', '0-25_fmax-0-5'
_SGC = lambda order, band='1e-05_fmax-0-5': f'sgc_parametric_mean_fs-1_fmin-{band}_order-{order}'
# Named minimal (or near-minimal) pairs: (character, A, B). Readings and confounds: docs/interpreting-spi-pairs.md.
PAIRS = [
    ('non-Gaussian dependence', 'mi_gaussian', 'mi_kraskov_NN-4_DCE-AUTO'),
    ('non-Gaussian dependence', 'mi_gaussian', 'mi_kernel_W-0.25'),
    ('non-Gaussian dependence', 'cov-sq_EmpiricalCovariance', 'dcorr'),
    ('non-Gaussian dependence', 'je_gaussian', 'je_kozachenko'),
    ('non-Gaussian dependence', 'xme_gaussian_k1', 'xme_kozachenko_k1'),
    ('monotone (rank) nonlinearity', 'cov_EmpiricalCovariance', 'spearmanr'),
    ('monotone (rank) nonlinearity', 'cov-sq_EmpiricalCovariance', 'spearmanr-sq'),
    ('monotone (rank) nonlinearity', 'corr_pearson_tau-1', 'corr_spearman_tau-1'),
    ('lagged vs zero-lag phase coupling', _MT('cohmag', _LOW), _MT('icoh', _LOW)),
    ('lagged vs zero-lag phase coupling', _MT('cohmag'), _MT('icoh')),
    ('lagged vs zero-lag phase coupling', _MT('cohmag', _HIGH), _MT('icoh', _HIGH)),
    ('lagged vs zero-lag phase coupling', _MT('plv'), _MT('pli')),
    ('lagged vs zero-lag phase coupling', _MT('plv'), _MT('wpli')),
    ('lagged vs zero-lag phase coupling', _MT('ppc'), _MT('dswpli')),
    ('lagged vs zero-lag amplitude coupling', 'pec', 'pec_orth'),
    ('instantaneous vs lagged dependence', 'mi_gaussian', 'tlmi_gaussian'),
    ('instantaneous vs lagged dependence', 'mi_kraskov_NN-4_DCE-AUTO', 'tlmi_kraskov_NN-4_DCE-AUTO'),
    ('instantaneous vs lagged dependence', 'cov-sq_EmpiricalCovariance', 'xcorr-sq_max_sig-True'),
    ('rigid vs elastic alignment', 'xpdist_euclidean_tau-1_min_rmse', 'dtw'),
    ('rigid vs elastic alignment', 'dtw', 'dtw_constraint-sakoe-chiba_radius-auto'),
    ('rigid vs elastic alignment', 'pdist_euclidean', 'dtw'),
    ('rigid vs elastic alignment', 'lcss', 'lcss_constraint-sakoe-chiba'),
    ('predictive vs contemporaneous', 'mi_gaussian', 'gc_gaussian_k-1_kt-1_l-1_lt-1'),
    ('predictive vs contemporaneous', 'tlmi_gaussian', 'gc_gaussian_k-1_kt-1_l-1_lt-1'),  # link: target's past only
    ('predictive vs contemporaneous', _MT('cohmag'), _SGC('None')),
    ('nonlinear vs linear prediction', 'gc_gaussian_k-1_kt-1_l-1_lt-1', 'te_kraskov_NN-4_DCE-AUTO_k-1_kt-1_l-1_lt-1'),
    ('nonlinear vs linear prediction', 'te_symbolic_k-3', 'te_kraskov_NN-4_DCE-AUTO_k-1_kt-1_l-1_lt-1'),
    ('interaction memory', 'te_kraskov_NN-4_DCE-AUTO_k-1_kt-1_l-1_lt-1', 'te_kraskov_NN-4_DCE-AUTO_k-2_kt-1_l-1_lt-1'),
    ('interaction memory', _SGC('1'), _SGC('20')),
    ('interaction memory', 'gc_gaussian_k-1_kt-1_l-1_lt-1', 'gc_gaussian_MAX-CORR-AIS_k-max-10_tau-max-2'),
    ('direct vs indirect (mediation)', 'cov-sq_EmpiricalCovariance', 'prec-sq_EmpiricalCovariance'),
    ('direct vs indirect (mediation)', 'cov-sq_LedoitWolf', 'prec-sq_LedoitWolf'),
    ('direct vs indirect (mediation)', _MT('dcoh'), _MT('pdcoh')),
    ('bulk vs atypical samples', 'cov-sq_EmpiricalCovariance', 'cov-sq_MinCovDet'),
    ('bulk vs atypical samples', 'prec-sq_LedoitWolf', 'prec-sq_MinCovDet'),
    ('bulk vs atypical samples', 'pdist_euclidean', 'pdist_chebyshev'),
    ('bulk vs atypical samples', 'pdist_euclidean', 'pdist_cityblock'),
    ('band locus', _MT('cohmag', _LOW), _MT('cohmag', _HIGH)),
    ('band locus', _MT('plv', _LOW), _MT('plv', _HIGH)),
    ('broadband vs peaked', _MT('cohmag'), _MT('cohmag', stat='max')),
    ('broadband vs peaked', _MT('plv'), _MT('plv', stat='max')),
    ('phase vs amplitude coupling', _MT('plv'), 'pec'),
    ('phase vs amplitude coupling', _MT('icoh'), 'pec_orth'),
    # added 12 November (findings.md, "Further candidate pairs")
    ('amplitude-weighted vs amplitude-free synchrony', _MT('cohmag'), _MT('plv')),
    ('amplitude-weighted vs amplitude-free synchrony', _MT('cohmag', _LOW), _MT('plv', _LOW)),
    ('amplitude-weighted vs amplitude-free synchrony', _MT('cohmag', _HIGH), _MT('plv', _HIGH)),
    ('burst vs scale-free amplitude coupling', 'pec', 'pec_log'),
    ('burst vs scale-free amplitude coupling', 'pec_orth', 'pec_orth_log'),
    ('sign (anticorrelation) structure', 'cov_EmpiricalCovariance', 'cov-sq_EmpiricalCovariance'),
    ('sign (anticorrelation) structure', 'spearmanr', 'spearmanr-sq'),
    ('direct vs indirect (mediation)', 'cov_EmpiricalCovariance', 'pcorr_LedoitWolf'),
    ('direct vs indirect (mediation)', 'cov_EmpiricalCovariance', 'pcorr_EmpiricalCovariance'),
    ('phase lead vs predictive direction', _MT('psi'), 'gc_gaussian_k-1_kt-1_l-1_lt-1'),
    ('phase lead vs predictive direction', _MT('psi', _LOW), 'gc_gaussian_k-1_kt-1_l-1_lt-1'),
    ('lag-concentrated vs lag-spread', 'xcorr_max_sig-True', 'xcorr_mean_sig-True'),
    ('control: saturation only', 'cov-sq_EmpiricalCovariance', 'mi_gaussian'),
    ('control: identity', 'cov_EmpiricalCovariance', 'pdist_euclidean'),
]


def pairs(reduced, output, reference, named=PAIRS):
    """Named SPI pairs: z in the reference state and every state's within-unit effect per agent, raw, adjusted for
    strength, and adjusted for strength and the channel-mean spectrum (peak frequency, spectral entropy)."""
    r = np.load(reduced, allow_pickle=False)
    rows = json.loads(str(r['rows']))
    spis = [str(s) for s in r['spis']]
    K = len(spis)
    index = np.full((K, K), -1)
    index[np.triu_indices(K, 1)] = np.arange(K * (K - 1) // 2)
    index = np.maximum(index, index.T)
    keep = set(np.flatnonzero(r['valid'].all(0)).tolist())
    unit = np.array([x['unit'] for x in rows])
    agent = np.array([x.get('agent', 'all') for x in rows])
    state = np.array(sleep_tertiles(rows) if any('delta_fraction' in x for x in rows) else [x['state'] for x in rows])
    names = [str(n) for n in r['covariate_names']]
    adjust = {'adjusted': r['covariates'][:, [names.index(c) for c in COVARIATES]],
              'spectral': r['covariates'][:, [names.index(c) for c in SPECTRAL]]}
    node = np.nanmedian(r['node_r2'][state == reference], 0)
    present = [(c, a, b) for c, a, b in named if a in spis and b in spis and spis.index(a) in keep and spis.index(b) in keep]
    cols = [index[spis.index(a), spis.index(b)] for _, a, b in present]
    Z = r['z'][:, cols].astype(np.float64)
    report = dict(reduced=str(reduced), reference=reference, adjusted_for=list(COVARIATES), spectral_for=list(SPECTRAL),
                  missing=[[c, a, b] for c, a, b in named if (c, a, b) not in present], pairs=[])
    effects = {}
    for s in sorted(set(state) - {reference}):
        for g in sorted(set(agent[state == s])):
            sel = agent == g
            effects[f'{g}:{s}'] = {v: unit_effects(Z[sel], unit[sel], state[sel], s, reference,
                                                   None if v == 'raw' else adjust[v][sel])
                                   for v in ('raw', 'adjusted', 'spectral')}
    null = [e['raw'] for k, e in effects.items() if k.endswith(':awake_eo') and len(e['raw'])]
    null_sd = np.concatenate(null).std(0, ddof=1) if null and sum(len(n) for n in null) > 2 else np.full(len(cols), np.nan)
    for i, (c, a, b) in enumerate(present):
        report['pairs'].append(dict(
            character=c, a=a, b=b, node_r2=[float(node[spis.index(a)]), float(node[spis.index(b)])],
            reference_z=float(np.nanmean(Z[state == reference, i])), eyes_open_null_sd=float(null_sd[i]),
            effects={k: {v: None if not len(e) else dict(mean=float(e[:, i].mean()), units=len(e),
                                                         positive=int(np.sum(e[:, i] > 0)))
                         for v, e in d.items()} for k, d in effects.items()}))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + '\n')



def _profiles(path, wanted):
    d = np.load(path, allow_pickle=True)
    spis = [str(s) for s in d['__spis__']]
    out = {}
    for wid in [k for k in d.files if not k.startswith('__')]:
        m = d[wid]
        out[wid] = np.stack([np.asarray(m[spis.index(w)], dtype=np.float64)[~np.eye(m.shape[1], dtype=bool)]
                             for w in wanted]).astype(np.float32)
    return out


def reliability(bank, pyspi_dir, output, reference, named=PAIRS, jobs=8):
    """Edge-profile reliability of the named pairs' SPIs: the mean correlation between the profiles of different
    windows of one unit and state. Per unit and state: each SPI's reliability, mean level and share of negative
    edges, and each pair's z raw and disattenuated (z / sqrt(rel_A rel_B)); then per agent the state-minus-reference
    difference of both, with unit and animal sign counts. A z that falls only because a profile gets noisier falls
    raw but not disattenuated."""
    manifest = json.loads((bank / 'manifest.json').read_text())
    shards = sorted(pyspi_dir.glob('shard-*.npz'))
    with np.load(shards[0], allow_pickle=True) as d:
        have = {str(x) for x in d['__spis__']}
    named = [(c, a, b) for c, a, b in named if a in have and b in have]
    wanted = sorted({x for _, a, b in named for x in (a, b)})
    merged = {}
    for part in Parallel(n_jobs=jobs)(delayed(_profiles)(s, wanted) for s in shards):
        merged.update(part)
    rows = [r for r in manifest['rows'] if r['id'] in merged]
    state = np.array(sleep_tertiles(rows) if any('delta_fraction' in x for x in rows) else [x['state'] for x in rows])
    unit = np.array([r['unit'] for r in rows])
    groups = {}
    for u, s in sorted(set(zip(unit, state))):
        P = np.stack([merged[rows[i]['id']] for i in np.flatnonzero((unit == u) & (state == s))])  # (n, SPI, edges)
        ok = np.isfinite(P).all(axis=(0, 2))
        rel = np.full(len(wanted), np.nan)
        if len(P) > 1:
            for k in np.flatnonzero(ok):
                c = np.corrcoef(P[:, k])
                rel[k] = c[~np.eye(len(P), dtype=bool)].mean()
        z = {}
        for c_, a, b in named:
            ia, ib = wanted.index(a), wanted.index(b)
            zw = [np.corrcoef(P[w, ia], P[w, ib])[0, 1] for w in range(len(P))] if ok[ia] and ok[ib] else [np.nan]
            raw = float(np.nanmean(zw))
            z[f'{a}|{b}'] = dict(raw=raw, disattenuated=raw / np.sqrt(rel[ia] * rel[ib]) if rel[ia] > 0 and rel[ib] > 0 else np.nan)
        groups[(u, s)] = dict(windows=len(P), reliability=dict(zip(wanted, rel.tolist())),
                              level=dict(zip(wanted, np.nanmean(P, axis=(0, 2)).tolist())),
                              negative=dict(zip(wanted, np.nanmean(P < 0, axis=(0, 2)).tolist())), z=z)
    agent = {r['unit']: r.get('agent', 'all') for r in rows}
    animal = {r['unit']: r.get('animal') or r['unit'] for r in rows}
    contrasts = {}
    for u, s in groups:
        if s == reference or (u, reference) not in groups:
            continue
        g, ref = groups[(u, s)], groups[(u, reference)]
        e = contrasts.setdefault(f'{agent[u]}:{s}', {})
        for key, v in g['z'].items():
            for kind in ('raw', 'disattenuated'):
                e.setdefault(f'{key}:{kind}', []).append((animal[u], v[kind] - ref['z'][key][kind]))
        for w in wanted:
            e.setdefault(f'{w}:reliability', []).append((animal[u], g['reliability'][w] - ref['reliability'][w]))
    def summarise(pairs_):
        d = np.array([x for _, x in pairs_], dtype=float)
        by = {}
        for a, x in pairs_:
            by.setdefault(a, []).append(x)
        means = [np.nanmean(v) for v in by.values()]
        return dict(mean=float(np.nanmean(d)), units=int(np.isfinite(d).sum()), positive=int(np.sum(d > 0)),
                    animals=len(by), animals_positive=int(np.sum(np.array(means) > 0)))
    report = dict(bank=str(bank), reference=reference, spis=wanted,
                  reference_state={w: dict(reliability=float(np.nanmean([g['reliability'][w] for (u, s), g in groups.items() if s == reference])),
                                           level=float(np.nanmean([g['level'][w] for (u, s), g in groups.items() if s == reference])))
                                   for w in wanted},
                  states={f'{agent[u]}:{s}': {} for u, s in groups},
                  contrasts={k: {m: summarise(v) for m, v in e.items()} for k, e in contrasts.items()})
    for (u, s), g in groups.items():
        st = report['states'][f'{agent[u]}:{s}']
        for w in wanted:
            st.setdefault(w, []).append([g['reliability'][w], g['level'][w], g['negative'][w]])
    report['states'] = {k: {w: np.nanmean(v, 0).tolist() for w, v in d.items()} for k, d in report['states'].items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + '\n')


def _family(name):
    """Collapse estimator, band and statistic variants so one representative per family pair is reported."""
    return name.split('_')[0]


def explore(reduced, output, reference, conditions=None, top=20, ceiling=.98):
    """Every kept z coordinate, beyond the all-agents screen. Per condition (agent:state), per-unit effects raw and
    strength-adjusted, scaled by the eyes-open null SD (by the condition's own between-unit SD where no eyes-open
    state exists, i.e. Cohen's d). A condition is consistent on a coordinate when at least 75% of units agree in
    sign, raw and adjusted, and every animal's mean agrees. Rankings:
      single:    each condition's strongest consistent coordinates;
      specific:  consistent in one condition at >= 4 scale units while every other condition stays within 1.5;
      opposite:  two conditions consistent with opposite signs, ranked by the weaker of the two;
      majority:  at least all-but-one drug conditions consistent with one sign at >= 3, ranked by the weakest of
                 the agreeing ones.
    Excludes node-additive SPIs, ceiling coordinates (|z| in the reference >= ceiling) and structural zeros, and
    reports one coordinate per SPI-family pair."""
    r = np.load(reduced, allow_pickle=False)
    rows = json.loads(str(r['rows']))
    spis = [str(s) for s in r['spis']]
    keep = np.flatnonzero(r['valid'].all(0))
    K = len(spis)
    iu = np.triu_indices(K, 1)
    kept = np.isin(iu[0], keep) & np.isin(iu[1], keep)
    a_idx, b_idx = iu[0][kept], iu[1][kept]
    Z = r['z'][:, kept].astype(np.float64)
    unit = np.array([x['unit'] for x in rows])
    animal_of = {x['unit']: x.get('animal') or x['unit'] for x in rows}
    agent = np.array([x.get('agent', 'all') for x in rows])
    state = np.array(sleep_tertiles(rows) if any('delta_fraction' in x for x in rows) else [x['state'] for x in rows])
    names = [str(n) for n in r['covariate_names']]
    cov = r['covariates'][:, [names.index(c) for c in COVARIATES]]
    node = np.nanmedian(r['node_r2'][state == reference], 0) > .5
    z0 = np.nanmean(Z[state == reference], 0)
    if conditions is None:
        conditions = sorted({f'{g}:{s}' for g, s in zip(agent, state)
                             if s not in (reference, 'awake_eo') and not s.startswith('recovery')})
    eff, units = {}, {}
    for c in conditions:
        g, s = c.split(':')
        sel = agent == g
        us = [u for u in np.unique(unit[sel]) if ((unit == u) & (state == s)).any() and ((unit == u) & (state == reference)).any()]
        eff[c] = dict(raw=unit_effects(Z[sel], unit[sel], state[sel], s, reference),
                      adjusted=unit_effects(Z[sel], unit[sel], state[sel], s, reference, cov[sel]))
        units[c] = np.array([animal_of[u] for u in us])
    recovery = {}
    for g in sorted(set(agent)):
        sel = agent == g
        if ((state == 'recovery_ec') & sel).any():
            recovery[g] = unit_effects(Z[sel], unit[sel], state[sel], 'recovery_ec', reference).mean(0)
    if (state == 'awake_eo').any():
        null = np.concatenate([unit_effects(Z[agent == g], unit[agent == g], state[agent == g], 'awake_eo', reference)
                               for g in sorted(set(agent)) if ((agent == g) & (state == 'awake_eo')).any()])
        scale = {c: null.std(0, ddof=1) for c in conditions}
        scale_kind = 'eyes-open null SD'
    else:
        scale = {c: eff[c]['adjusted'].std(0, ddof=1) for c in conditions}
        scale_kind = "condition's own between-unit SD (Cohen's d)"
    strength, consistent = {}, {}
    for c in conditions:
        m = eff[c]['adjusted'].mean(0)
        sign = np.sign(m)
        ok = (np.mean(np.sign(eff[c]['adjusted']) == sign, 0) >= .75) & (np.mean(np.sign(eff[c]['raw']) == sign, 0) >= .75)
        for a in np.unique(units[c]):
            ok &= np.sign(eff[c]['adjusted'][units[c] == a].mean(0)) == sign
        strength[c] = np.where(scale[c] > 1e-4, m / np.maximum(scale[c], 1e-4), 0.0)
        consistent[c] = ok
    usable = ~(node[a_idx] | node[b_idx]) & (np.abs(z0) < ceiling) & np.all([scale[c] > 1e-4 for c in conditions], 0)

    def describe(i):
        return dict(a=spis[a_idx[i]], b=spis[b_idx[i]], reference_z=float(z0[i]),
                    conditions={c: dict(adjusted=float(eff[c]['adjusted'][:, i].mean()), raw=float(eff[c]['raw'][:, i].mean()),
                                        scaled=float(strength[c][i]), consistent=bool(consistent[c][i]),
                                        units=len(eff[c]['raw']), animals=int(len(np.unique(units[c]))))
                                for c in conditions},
                    recovery={g: float(v[i]) for g, v in recovery.items()})

    def ranked(score, mask):
        out, seen = [], set()
        for i in np.argsort(-np.where(mask & usable, score, -np.inf)):
            if not (mask[i] and usable[i]) or len(out) >= top:
                break
            fam = tuple(sorted((_family(spis[a_idx[i]]), _family(spis[b_idx[i]]))))
            if fam in seen:
                continue
            seen.add(fam)
            out.append(describe(i))
        return out

    S = {c: np.abs(strength[c]) for c in conditions}
    drugs = [c for c in conditions if c.endswith(':drug') or c.endswith(':isoflurane')]
    report = dict(reduced=str(reduced), reference=reference, scale=scale_kind, conditions=conditions,
                  coordinates=int(Z.shape[1]), usable=int(usable.sum()),
                  single={c: ranked(S[c], consistent[c]) for c in conditions},
                  specific={c: ranked(S[c], consistent[c] & (S[c] >= 4) &
                                      np.all([S[o] < 1.5 for o in conditions if o != c], 0)) for c in conditions},
                  opposite={}, majority=[])
    for k, c1 in enumerate(conditions):
        for c2 in conditions[k + 1:]:
            m = consistent[c1] & consistent[c2] & (np.sign(strength[c1]) == -np.sign(strength[c2]))
            report['opposite'][f'{c1} vs {c2}'] = ranked(np.minimum(S[c1], S[c2]), m & (np.minimum(S[c1], S[c2]) >= 3))
    if len(drugs) >= 3:
        best = []
        for sgn in (1, -1):
            agree = np.array([consistent[c] & (np.sign(strength[c]) == sgn) & (S[c] >= 3) for c in drugs])
            n = agree.sum(0)
            weakest = np.min(np.where(agree, np.array([S[c] for c in drugs]), np.inf), 0)
            best.append((n >= len(drugs) - 1, np.where(n >= len(drugs) - 1, weakest, 0)))
        mask = best[0][0] | best[1][0]
        report['majority'] = ranked(np.maximum(best[0][1], best[1][1]), mask)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + '\n')


STABILITY_SPIS = ('mi_gaussian', 'mi_kraskov_NN-4_DCE-AUTO', 'dcorr', 'xcorr-sq_max_sig-True', 'cov-sq_EmpiricalCovariance',
                  'cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-25')
SEPARATIONS = (0, 150, 400, np.inf)  # seconds between window starts


def stability(bank, pyspi_dir, output, reference, wanted=STABILITY_SPIS, jobs=8):
    """Edge-profile agreement between windows of one unit and state as a function of their separation in time.
    States are sampled at different spacings (drug windows about twice as far apart as wake), so each state is
    compared with the reference within the same separation bin. Drift predicts agreement that falls with
    separation; fast fluctuation predicts low agreement already in the nearest bin."""
    manifest = json.loads((bank / 'manifest.json').read_text())
    shards = sorted(pyspi_dir.glob('shard-*.npz'))
    merged = {}
    for part in Parallel(n_jobs=jobs)(delayed(_profiles)(sh, list(wanted)) for sh in shards):
        merged.update(part)
    rows = [r for r in manifest['rows'] if r['id'] in merged]
    state = np.array(sleep_tertiles(rows) if any('delta_fraction' in x for x in rows) else [x['state'] for x in rows])
    unit = np.array([r['unit'] for r in rows])
    start = np.array([r.get('start_s', np.nan) for r in rows], dtype=float)
    agent = {r['unit']: r.get('agent', 'all') for r in rows}
    animal = {r['unit']: r.get('animal') or r['unit'] for r in rows}
    nb = len(SEPARATIONS) - 1
    table = {}
    for u, s in sorted(set(zip(unit, state))):
        ix = np.flatnonzero((unit == u) & (state == s))
        P = np.stack([merged[rows[i]['id']] for i in ix])
        a, b = np.triu_indices(len(ix), 1)
        sep = np.abs(start[ix][a] - start[ix][b])
        bins = np.digitize(sep, SEPARATIONS[1:-1])
        agree = np.full((len(wanted), nb), np.nan)
        for k in range(len(wanted)):
            if not np.isfinite(P[:, k]).all():
                continue
            c = np.corrcoef(P[:, k])[a, b]
            for j in range(nb):
                if (bins == j).any():
                    agree[k, j] = c[bins == j].mean()
        table[(u, s)] = dict(agree=agree, pairs=np.bincount(bins, minlength=nb).tolist(),
                             median_separation=float(np.median(sep)) if len(sep) else np.nan)
    report = dict(bank=str(bank), reference=reference, spis=list(wanted),
                  separation_bins_s=[f'{SEPARATIONS[j]}-{SEPARATIONS[j + 1]}' for j in range(nb)],
                  states={}, contrasts={})
    for (u, s), t in table.items():
        e = report['states'].setdefault(f'{agent[u]}:{s}', dict(agree=[], median_separation=[]))
        e['agree'].append(t['agree']); e['median_separation'].append(t['median_separation'])
    for k, e in report['states'].items():
        report['states'][k] = dict(agree={w: np.nanmean(np.array(e['agree'])[:, i], 0).tolist() for i, w in enumerate(wanted)},
                                   median_separation_s=float(np.nanmedian(e['median_separation'])))
    for (u, s), t in table.items():
        if s == reference or (u, reference) not in table:
            continue
        d = t['agree'] - table[(u, reference)]['agree']
        e = report['contrasts'].setdefault(f'{agent[u]}:{s}', {})
        for i, w in enumerate(wanted):
            for j in range(nb):
                if np.isfinite(d[i, j]):
                    e.setdefault(f'{w}:{report["separation_bins_s"][j]}', []).append((animal[u], float(d[i, j])))
    for k, e in report['contrasts'].items():
        for m, v in e.items():
            x = np.array([y for _, y in v])
            by = {}
            for a_, y in v:
                by.setdefault(a_, []).append(y)
            e[m] = dict(mean=float(x.mean()), units=len(x), positive=int((x > 0).sum()), animals=len(by),
                        animals_positive=int(sum(np.mean(y) > 0 for y in by.values())))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + '\n')


def _r_profiles(X):
    """Upper-triangle Pearson r of each window (n, T, M) -> (n, E)."""
    Z = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    C = np.einsum('ntm,ntk->nmk', Z, Z) / X.shape[1]
    a, b = np.triu_indices(X.shape[2], 1)
    return C[:, a, b]


def stationary_draws(X, n, rng):
    """Stationary Gaussian windows whose expected cross-periodogram is the mean of the windows' (n_w, T, M):
    each frequency bin is an independent complex-Gaussian mixture of the windows' Fourier coefficients."""
    F = np.fft.rfft((X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True), axis=1)
    w = (rng.normal(size=(n, len(X), F.shape[1])) + 1j * rng.normal(size=(n, len(X), F.shape[1]))) / np.sqrt(2)
    w[:, :, 0] = rng.normal(size=(n, len(X)))  # real bins carry real weights
    if X.shape[1] % 2 == 0:
        w[:, :, -1] = rng.normal(size=(n, len(X)))
    return np.fft.irfft(np.einsum('swf,wfm->sfm', w, F) / np.sqrt(len(X)), n=X.shape[1], axis=1)


def _agreement(P):
    c = np.corrcoef(P)
    return c[np.triu_indices(len(P), 1)]


def spectral_null(bank, output, reference, draws=10, seed=0):
    """Is the drop in between-window agreement of coupling profiles more than the spectrum predicts? For each
    unit and state, the agreement of r and r^2 profiles between its real windows is set against the same
    agreement among stationary Gaussian windows sharing the state's mean cross-spectrum (same T, same count):
    agreement that sampling noise alone gives at that spectral concentration. Real agreement below it means
    the profile changes between windows (non-stationary or non-Gaussian dependence); a drug-minus-reference drop
    reproduced by the null is the effective-sample-size explanation. Pyspi-free."""
    manifest = json.loads((bank / 'manifest.json').read_text())
    rows = manifest['rows']
    state = np.array(sleep_tertiles(rows) if any('delta_fraction' in x for x in rows) else [x['state'] for x in rows])
    unit = np.array([r['unit'] for r in rows])
    start = np.array([r.get('start_s', np.nan) for r in rows], dtype=float)
    agent = {r['unit']: r.get('agent', 'all') for r in rows}
    animal = {r['unit']: r.get('animal') or r['unit'] for r in rows}
    rng = np.random.default_rng(seed)
    measures = ('real', 'real_near', 'null', 'excess', 'excess_near')
    table = {}
    with np.load(bank / 'windows.npz') as archive:
        for u, s in sorted(set(zip(unit, state))):
            ix = np.flatnonzero((unit == u) & (state == s))
            if len(ix) < 3:
                continue
            X = np.stack([np.asarray(archive[rows[i]['id']], dtype=np.float64) for i in ix])
            a, b = np.triu_indices(len(ix), 1)
            near = np.abs(start[ix][a] - start[ix][b]) < SEPARATIONS[1]
            R = _r_profiles(X)
            entry = {}
            for name, f in (('r', lambda p: p), ('r2', np.square)):
                real = _agreement(f(R))
                null = np.mean([_agreement(f(_r_profiles(stationary_draws(X, len(ix), rng)))).mean()
                                for _ in range(draws)])
                rn = real[near].mean() if near.any() else np.nan
                entry[name] = dict(real=real.mean(), real_near=rn, null=null, excess=real.mean() - null,
                                   excess_near=rn - null)
            table[(u, s)] = entry
    report = dict(bank=str(bank), reference=reference, draws=draws, near_s=SEPARATIONS[1], states={}, contrasts={})
    for (u, s), t in table.items():
        e = report['states'].setdefault(f'{agent[u]}:{s}', {})
        for name, v in t.items():
            for m in measures:
                e.setdefault(f'{name}:{m}', []).append(v[m])
        if s == reference or (u, reference) not in table:
            continue
        c = report['contrasts'].setdefault(f'{agent[u]}:{s}', {})
        for name, v in t.items():
            for m in measures:
                d = v[m] - table[(u, reference)][name][m]
                if np.isfinite(d):
                    c.setdefault(f'{name}:{m}', []).append((animal[u], float(d)))
    for e in report['states'].values():
        for m, v in e.items():
            e[m] = float(np.nanmean(v))
    for e in report['contrasts'].values():
        for m, v in e.items():
            x = np.array([y for _, y in v])
            by = {}
            for a_, y in v:
                by.setdefault(a_, []).append(y)
            e[m] = dict(mean=float(x.mean()), units=len(x), positive=int((x > 0).sum()), animals=len(by),
                        animals_positive=int(sum(np.mean(y) > 0 for y in by.values())))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + '\n')



def gaussian_draws(S, n, T, rng):
    """n stationary Gaussian windows of length T whose expected rfft cross-periodogram is S (F x M x M, PSD)."""
    w, V = np.linalg.eigh(S)
    L = V * np.sqrt(np.clip(w, 0, None))[:, None, :]
    Z = (rng.normal(size=(n, *S.shape[:2])) + 1j * rng.normal(size=(n, *S.shape[:2]))) / np.sqrt(2)
    Z[:, 0] = rng.normal(size=(n, S.shape[1]))
    if T % 2 == 0:
        Z[:, -1] = rng.normal(size=(n, S.shape[1]))
    return np.fft.irfft(np.einsum('fij,nfj->nfi', L, Z), n=T, axis=1)


def _mean_periodogram(X):  # (n,T,M) -> (F,M,M): mean rfft cross-periodogram of the standardised windows
    F = np.fft.rfft((X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True), axis=1)
    return np.einsum('nfi,nfj->fij', F, F.conj()) / len(X)


def _zero_phase(S):  # same |S|, every phase zero, projected to PSD per frequency
    w, V = np.linalg.eigh(np.abs(S))
    return (V * np.clip(w, 0, None)[:, None, :]) @ V.transpose(0, 2, 1)


LAG_PAIRS = (('cohmag_low', 'icoh_low'), ('ppc', 'dswpli'), ('pec', 'pec_orth'), ('cov2', 'xcorr2_max'))


def lag_profiles(x, nw=3, k=5):
    """Pyspi-free edge profiles of one window (T x M) for the lag pairs, upper triangle, and its low-band
    coherency (F x E). Multitaper (NW 3, 5 tapers, as pyspi's spectral_connectivity default) with fs = 1:
    cohmag = |C|^2 and icoh = |Im C| over (0, .25]; ppc and dswpli debiased over tapers, over (0, .5);
    pec and pec_orth (pairwise-orthogonalised, symmetrised) on the broadband analytic signal; cov2 = r^2 and
    xcorr2_max = the largest r(l)^2 over |l| <= T/4 among lags with |r(l)| > 1.96/sqrt(T), else 0 (pyspi
    xcorr-sq_max_sig-True)."""
    from scipy.signal import hilbert
    from scipy.signal.windows import dpss
    T, M = x.shape
    x = (x - x.mean(0)) / x.std(0)
    a, b = np.triu_indices(M, 1)
    f = np.fft.rfftfreq(T)
    Xk = np.fft.rfft(dpss(T, nw, k)[:, :, None] * x[None], axis=1)        # k x F x M
    low, full = (f > 0) & (f <= .25), (f > 0) & (f < .5)
    S = np.einsum('kfi,kfj->fij', Xk[:, low], Xk[:, low].conj()) / k
    d = np.real(np.einsum('fii->fi', S))
    C = (S / np.sqrt(d[:, :, None] * d[:, None, :]))[:, a, b]
    out = dict(cohmag_low=(np.abs(C) ** 2).mean(0), icoh_low=np.abs(C.imag).mean(0))
    P = Xk[:, full][:, :, a] * Xk[:, full][:, :, b].conj()                    # k x F x E per-taper cross-spectra
    U = P / np.abs(P)
    out['ppc'] = ((np.abs(U.sum(0)) ** 2 - k) / (k * (k - 1))).mean(0)
    im = P.imag
    sq = (im ** 2).sum(0)
    out['dswpli'] = np.nanmean((im.sum(0) ** 2 - sq) / (np.abs(im).sum(0) ** 2 - sq), 0)
    h = hilbert(x, axis=0)
    env = np.abs(h)
    out['pec'] = np.corrcoef(env.T)[a, b]
    orth = lambda i, j: np.abs((h[:, j] * h[:, i].conj()).imag / env[:, i])  # |j orthogonalised to i|
    corr = lambda u, v: np.corrcoef(u, v)[0, 1]
    out['pec_orth'] = np.array([(corr(env[:, i], orth(i, j)) + corr(env[:, j], orth(j, i))) / 2 for i, j in zip(a, b)])
    r = np.fft.irfft(np.fft.rfft(x, 2 * T, axis=0)[:, a] * np.fft.rfft(x, 2 * T, axis=0)[:, b].conj(), axis=0) / T
    r = np.concatenate([r[-(T // 4):], r[:T // 4 + 1]])                     # lags -T/4..T/4
    out['cov2'] = r[T // 4] ** 2
    out['xcorr2_max'] = np.where(np.abs(r) > 1.96 / np.sqrt(T), r ** 2, 0).max(0)
    return out, C


def _lag_state(X, k=5):
    """Pair z per window (mean over windows) and per-edge across-window dispersion of the low-band coherency:
    phase dispersion 1 - |sum_w C_w| / sum_w |C_w| (zero when only magnitudes vary) and magnitude dispersion
    SD_w |C_w| / mean_w |C_w|, each averaged over frequency and edges."""
    zs, Cs = [], []
    for x in X:
        prof, C = lag_profiles(x, k=k)
        zs.append([np.corrcoef(prof[p], prof[q])[0, 1] for p, q in LAG_PAIRS])
        Cs.append(C)
    Cs = np.stack(Cs)
    A = np.abs(Cs)
    out = dict(zip((f'{p}|{q}' for p, q in LAG_PAIRS), np.nanmean(zs, 0)))
    out['phase_dispersion'] = float(np.mean(1 - np.abs(Cs.sum(0)) / A.sum(0)))
    out['magnitude_dispersion'] = float(np.mean(A.std(0) / A.mean(0)))
    return out


def lag_null(bank, output, reference, draws=3, channels=32, seed=0):
    """Is a state change in the lag pairs more than estimation noise at the state's spectrum? For each unit
    and state (a subset of evenly spaced channels), the lag-pair z and the coherency dispersions of the real
    windows are set against two stationary Gaussian nulls with the same T and window count:
      zero_phase: the state's mean cross-spectrum with every phase set to zero (no lag at all), giving the
        estimation floor;
      stationary: the state's mean cross-spectrum itself (its mean lags, fixed across windows).
    floor = zero_phase, mean_lag = stationary - zero_phase, residual = real - stationary (any window-to-window
    non-stationarity or non-Gaussianity), bias = a stationary null re-drawn from a stationary draw's own
    spectrum minus the stationary null (construction bias, should be ~0). Pyspi-free."""
    manifest = json.loads((bank / 'manifest.json').read_text())
    rows = manifest['rows']
    state = np.array(sleep_tertiles(rows) if any('delta_fraction' in x for x in rows) else [x['state'] for x in rows])
    unit = np.array([r['unit'] for r in rows])
    agent = {r['unit']: r.get('agent', 'all') for r in rows}
    animal = {r['unit']: r.get('animal') or r['unit'] for r in rows}
    rng = np.random.default_rng(seed)
    table = {}
    with np.load(bank / 'windows.npz') as archive:
        M = np.asarray(archive[rows[0]['id']]).shape[1]
        sub = np.unique(np.round(np.linspace(0, M - 1, min(channels, M))).astype(int))
        for u, s in sorted(set(zip(unit, state))):
            ix = np.flatnonzero((unit == u) & (state == s))
            if len(ix) < 3:
                continue
            X = np.stack([np.asarray(archive[rows[i]['id']], dtype=np.float64)[:, sub] for i in ix])
            n, T = len(X), X.shape[1]
            S = _mean_periodogram(X)
            mean = lambda L: {key: float(np.mean([d[key] for d in L])) for key in L[0]}
            real = _lag_state(X)
            zero = mean([_lag_state(gaussian_draws(_zero_phase(S), n, T, rng)) for _ in range(draws)])
            null = [gaussian_draws(S, n, T, rng) for _ in range(draws)]
            stat = mean([_lag_state(Y) for Y in null])
            again = _lag_state(gaussian_draws(_mean_periodogram(null[0]), n, T, rng))
            table[(u, s)] = {key: dict(real=real[key], floor=zero[key], mean_lag=stat[key] - zero[key],
                                       residual=real[key] - stat[key], bias=again[key] - stat[key])
                             for key in real}
    report = dict(bank=str(bank), reference=reference, draws=draws, channels=sub.tolist(), states={}, contrasts={})
    for (u, s), t in table.items():
        e = report['states'].setdefault(f'{agent[u]}:{s}', {})
        for key, v in t.items():
            for m, y in v.items():
                e.setdefault(f'{key}:{m}', []).append(y)
        if s == reference or (u, reference) not in table:
            continue
        c = report['contrasts'].setdefault(f'{agent[u]}:{s}', {})
        for key, v in t.items():
            for m, y in v.items():
                d = y - table[(u, reference)][key][m]
                if np.isfinite(d):
                    c.setdefault(f'{key}:{m}', []).append((animal[u], float(d)))
    for e in report['states'].values():
        for m, v in e.items():
            e[m] = float(np.nanmean(v))
    for e in report['contrasts'].values():
        for m, v in e.items():
            x = np.array([y for _, y in v])
            by = {}
            for a_, y in v:
                by.setdefault(a_, []).append(y)
            e[m] = dict(mean=float(x.mean()), units=len(x), positive=int((x > 0).sum()), animals=len(by),
                        animals_positive=int(sum(np.mean(y) > 0 for y in by.values())))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + '\n')


NULL_KINDS = ('zero_phase', 'stationary', 'bias')


def null_bank(bank, output, kind, units=None, seed=0):
    """A Gaussian null of a window bank, for pyspi: real = floor + mean lag + residual for every coordinate, as
    lag_null does for its pyspi-free pairs. Each unit and state (sleep tertiles where staged, as the analysis groups
    them) is replaced by stationary Gaussian windows with the same count, T and M:
      stationary: expected cross-periodogram = the state's mean standardised cross-periodogram (its mean lags);
      zero_phase: the same with every phase set to zero (no lag), projected to PSD per frequency (the floor);
      bias: drawn from the mean cross-periodogram of one stationary draw; bias - stationary is the construction
        bias (the mean periodogram has rank <= n per frequency bin, so the null's coherence is inflated).
    Window ids and manifest rows are unchanged, so extract's per-window seeds and reduce pair with the real bank.
    The manifest's 'null' record holds the kind, seed, group sizes and, per group, the share of eigenvalue mass
    clipped when projecting to PSD (0 for stationary and bias). Pyspi-free."""
    import zlib
    manifest = json.loads((bank / 'manifest.json').read_text())
    rows = [r for r in manifest['rows'] if units is None or r['unit'] in units]
    state = np.array(sleep_tertiles(rows) if any('delta_fraction' in x for x in rows) else [x['state'] for x in rows])
    unit = np.array([r['unit'] for r in rows])
    windows, groups = {}, {}
    with np.load(bank / 'windows.npz') as archive:
        for u, s in sorted(set(zip(unit, state))):
            ix = np.flatnonzero((unit == u) & (state == s))
            X = np.stack([np.asarray(archive[rows[i]['id']], dtype=np.float64) for i in ix])
            n, T = len(X), X.shape[1]
            rng = np.random.default_rng([seed, zlib.crc32(f'{u}|{s}|{kind}'.encode())])
            S = _mean_periodogram(X)
            clipped = 0.
            if kind == 'zero_phase':
                w = np.linalg.eigvalsh(np.abs(S))
                clipped = float(-w[w < 0].sum() / np.abs(w).sum())
                S = _zero_phase(S)
            elif kind == 'bias':
                S = _mean_periodogram(gaussian_draws(S, n, T, rng))
            for i, y in zip(ix, gaussian_draws(S, n, T, rng)):
                windows[rows[i]['id']] = y
            groups[f'{u}:{s}'] = dict(n=int(n), clipped=clipped)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / 'windows.npz', **windows)
    manifest = {**manifest, 'rows': rows, 'null': dict(kind=kind, seed=seed, source=str(bank), groups=groups)}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')


NULL_PAIRS = (('mi_gaussian', 'mi_kraskov_NN-4'), ('mi_gaussian', 'mi_kraskov_NN-4_DCE-AUTO'),
              ('ppc_multitaper_mean_fs-1_fmin-0_fmax-0-5', 'dswpli_multitaper_mean_fs-1_fmin-0_fmax-0-5'),
              ('cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-25', 'icoh_multitaper_mean_fs-1_fmin-0_fmax-0-25'),
              ('pec', 'pec_orth'), ('cov-sq_EmpiricalCovariance', 'xcorr-sq_max_sig-True'),
              ('cov-sq_EmpiricalCovariance', 'cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-25'),
              ('cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-25', 'pec_log'),
              ('cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-25', 'cohmag_multitaper_mean_fs-1_fmin-0-25_fmax-0-5'))
BEYOND = (.04, .06)  # one |net| threshold for every condition, so drug, eyes-open and recovery counts compare


def null_decompose(discovery, output, reference, units=None):
    """All-unit decomposition of z against the pyspi null banks (null_bank; reduced as null-<kind>-reduced.npz next
    to neurotycho-short-reduced.npz). Per unit and state, the contrast against the unit's reference state in the real
    bank and in each null splits as lag_null does: floor = zero_phase, mean_lag = stationary - zero_phase, residual =
    real - stationary, bias = bias - stationary (the construction bias of a null drawn from an estimated spectrum),
    net = residual + bias (the stationary null is itself such a draw, so real minus a true-spectrum null is about
    residual + bias). Floor and mean lag are readable for coherence-type pairs only: the zero-phase PSD projection
    also moves time-domain strength. Per condition (agent:state): RMS of each part over coordinates, the share of the
    real contrast's sum of squares left in the residual, unit-level t RMS, coordinates whose |net| exceeds each
    BEYOND threshold with every unit's residual of one sign, net RMS per SPI and per module pair, the named pairs
    (NULL_PAIRS) with session and animal counts, and covariate contrasts per bank."""
    banks = {'real': 'neurotycho-short', **{k: f'null-{k}' for k in NULL_KINDS}}
    data = {}
    for key, name in banks.items():
        with np.load(discovery / f'{name}-reduced.npz') as d:
            data[key] = dict(ids=[str(x) for x in d['ids']], spis=[str(x) for x in d['spis']], z=d['z'],
                             cov=d['covariates'], rows=json.loads(str(d['rows'])))
            if key == 'real':
                modules, cov_names = [str(x) for x in d['modules']], [str(x) for x in d['covariate_names']]
    for key, v in data.items():
        assert v['ids'] == data['real']['ids'] and v['spis'] == data['real']['spis'], key
    rows, spis = data['real']['rows'], data['real']['spis']
    unit, state = np.array([r['unit'] for r in rows]), np.array([r['state'] for r in rows])
    agent = {r['unit']: r['agent'] for r in rows}
    animal = {r['unit']: r['animal'] for r in rows}
    K = len(spis)
    a, b = np.triu_indices(K, 1)
    index = {(spis[i], spis[j]): n for n, (i, j) in enumerate(zip(a, b))}
    named = {f'{p}|{q}': index.get((p, q), index.get((q, p))) for p, q in NULL_PAIRS}
    named = {k: n for k, n in named.items() if n is not None}
    mean = {(key, u, s): (np.nanmean(v['z'][(unit == u) & (state == s)], 0),
                          v['cov'][(unit == u) & (state == s)].mean(0))
            for key, v in data.items() for u, s in set(zip(unit, state)) if units is None or u in units}
    contrasts = {}
    for (key, u, s), _ in mean.items():
        if key != 'real' or s == reference or ('real', u, reference) not in mean:
            continue
        c = {k: mean[(k, u, s)][0] - mean[(k, u, reference)][0] for k in banks}
        parts = dict(real=c['real'], floor=c['zero_phase'], mean_lag=c['stationary'] - c['zero_phase'],
                     residual=c['real'] - c['stationary'], bias=c['bias'] - c['stationary'])
        parts['net'] = parts['residual'] + parts['bias']
        cov = {k: (mean[(k, u, s)][1] - mean[(k, u, reference)][1]).round(4).tolist() for k in banks}
        contrasts.setdefault(f'{agent[u]}:{s}', []).append((u, parts, cov))
    rms = lambda x: float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)))
    report = dict(discovery=str(discovery), reference=reference, units=units, covariate_names=cov_names,
                  coordinates=len(a), conditions={})
    for cond, L in sorted(contrasts.items()):
        n = len(L)
        P = {k: np.stack([p[k] for _, p, _ in L]) for k in L[0][1]}
        M = {k: np.nanmean(v, 0) for k, v in P.items()}
        ok = np.all([np.isfinite(v) for v in M.values()], 0)
        lost = int((np.isfinite(M['real']) & ~ok).sum())
        e = dict(units=n, animals=len({animal[u] for u, _, _ in L}), coordinates=int(ok.sum()), lost=lost)
        e['rms'] = {k: rms(v[ok]) for k, v in M.items()}
        ss = float(np.sum(M['real'][ok] ** 2))
        e['residual_share'] = float(np.sum(M['residual'][ok] ** 2) / ss)
        e['net_share'] = float(np.sum(M['net'][ok] ** 2) / ss)
        e['bias_residual_r'] = float(np.corrcoef(M['bias'][ok], M['residual'][ok])[0, 1])
        if n > 2:
            t = lambda X: np.nanmean(X, 0)[ok] / (np.nanstd(X, 0, ddof=1)[ok] / np.sqrt(n))
            e['t_rms'] = {k: rms(np.nan_to_num(t(P[k]))) for k in ('residual', 'bias', 'net')}
        same = np.all(np.sign(P['residual']) == np.sign(M['residual']), 0)
        e['beyond'] = {}
        for floor in BEYOND:
            hit = ok & same & (np.abs(M['net']) > floor)
            e['beyond'][str(floor)] = dict(coordinates=int(hit.sum()), positive=int((hit & (M['net'] > 0)).sum()))
        per_spi = np.zeros(K)
        for i in range(K):
            sel = ok & ((a == i) | (b == i))
            per_spi[i] = rms(M['net'][sel]) if sel.any() else np.nan
        top = np.argsort(-np.nan_to_num(per_spi))[:15]
        e['net_rms_by_spi'] = {spis[i]: round(float(per_spi[i]), 4) for i in top}
        mods = {}
        for n_ in np.flatnonzero(ok):
            mods.setdefault('|'.join(sorted((modules[a[n_]], modules[b[n_]]))), []).append(M['net'][n_])
        e['net_rms_by_module_pair'] = dict(sorted(((k, dict(rms=round(rms(np.array(v)), 4), mean=round(float(np.mean(v)), 4),
                                                            coordinates=len(v))) for k, v in mods.items()),
                                                  key=lambda kv: -kv[1]['rms']))
        e['named'] = {}
        for key, n_ in named.items():
            row = {}
            for k, X in P.items():
                x = X[:, n_]
                by = {}
                for (u, _, _), y in zip(L, x):
                    by.setdefault(animal[u], []).append(y)
                row[k] = dict(mean=round(float(np.nanmean(x)), 4), positive=int((x > 0).sum()),
                              animals_positive=int(sum(np.nanmean(y) > 0 for y in by.values())))
            e['named'][key] = row
        e['covariates'] = {k: np.mean([c[k] for _, _, c in L], 0).round(4).tolist() for k in banks}
        report['conditions'][cond] = e
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + '\n')


def sign_null(bank, output, reference, draws=10, seed=0):
    """Is the fall of z(r, r^2) (sign structure; findings.md, "Further candidate pairs") anticorrelation structure or
    sampling noise? Per unit and state: the per-window z(r, r^2) (real), z of the window-mean r profile (noise
    averaged out: the state's sign structure), and both on stationary Gaussian windows with the state's mean
    cross-spectrum (null: what sampling noise adds at that spectral concentration). Read before the run: a fall
    carried by the window-mean profile is sign structure (reference-induced or not); a per-window fall that the
    null reproduces while the window-mean profile holds is fewer effective samples. Pyspi-free."""
    manifest = json.loads((bank / 'manifest.json').read_text())
    rows = manifest['rows']
    state = np.array(sleep_tertiles(rows) if any('delta_fraction' in x for x in rows) else [x['state'] for x in rows])
    unit = np.array([r['unit'] for r in rows])
    agent = {r['unit']: r.get('agent', 'all') for r in rows}
    animal = {r['unit']: r.get('animal') or r['unit'] for r in rows}
    rng = np.random.default_rng(seed)
    zsq = lambda R: float(np.mean([np.corrcoef(x, x ** 2)[0, 1] for x in np.atleast_2d(R)]))
    table = {}
    with np.load(bank / 'windows.npz') as archive:
        for u, s in sorted(set(zip(unit, state))):
            ix = np.flatnonzero((unit == u) & (state == s))
            if len(ix) < 3:
                continue
            X = np.stack([np.asarray(archive[rows[i]['id']], dtype=np.float64) for i in ix])
            R = _r_profiles(X)
            nulls = [_r_profiles(stationary_draws(X, len(ix), rng)) for _ in range(draws)]
            table[(u, s)] = dict(real=zsq(R), real_mean_profile=zsq(R.mean(0)), null=np.mean([zsq(n) for n in nulls]),
                                 null_mean_profile=np.mean([zsq(n.mean(0)) for n in nulls]), mean_r=float(R.mean()),
                                 negative_share=float((R < 0).mean()))
    report = dict(bank=str(bank), reference=reference, draws=draws, states={}, contrasts={})
    for (u, s), t in table.items():
        e = report['states'].setdefault(f'{agent[u]}:{s}', {})
        for m, v in t.items():
            e.setdefault(m, []).append(v)
        if s == reference or (u, reference) not in table:
            continue
        c = report['contrasts'].setdefault(f'{agent[u]}:{s}', {})
        for m, v in t.items():
            c.setdefault(m, []).append((animal[u], v - table[(u, reference)][m]))
    report['states'] = {k: {m: float(np.mean(v)) for m, v in e.items()} for k, e in report['states'].items()}
    for e in report['contrasts'].values():
        for m, v in e.items():
            x = np.array([y for _, y in v])
            e[m] = dict(mean=float(x.mean()), units=len(x), positive=int((x > 0).sum()))
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
    c = sub.add_parser('screen')
    c.add_argument('reduced', type=Path)
    c.add_argument('--output', type=Path, required=True)
    c.add_argument('--reference', default='awake_ec')
    c.add_argument('--drug-state', default='drug')
    c.add_argument('--double-centred', action='store_true')
    d = sub.add_parser('pairs')
    d.add_argument('reduced', type=Path)
    d.add_argument('--output', type=Path, required=True)
    d.add_argument('--reference', default='awake_ec')
    e = sub.add_parser('reliability')
    e.add_argument('bank', type=Path)
    e.add_argument('pyspi', type=Path)
    e.add_argument('--output', type=Path, required=True)
    e.add_argument('--reference', default='awake_ec')
    e.add_argument('--jobs', type=int, default=8)
    f = sub.add_parser('explore')
    f.add_argument('reduced', type=Path)
    f.add_argument('--output', type=Path, required=True)
    f.add_argument('--reference', default='awake_ec')
    g = sub.add_parser('stability')
    g.add_argument('bank', type=Path)
    g.add_argument('pyspi', type=Path)
    g.add_argument('--output', type=Path, required=True)
    g.add_argument('--reference', default='awake_ec')
    g.add_argument('--jobs', type=int, default=8)
    h = sub.add_parser('spectral-null')
    h.add_argument('bank', type=Path)
    h.add_argument('--output', type=Path, required=True)
    h.add_argument('--reference', default='awake_ec')
    h.add_argument('--draws', type=int, default=10)
    i = sub.add_parser('lag-null')
    i.add_argument('bank', type=Path)
    i.add_argument('--output', type=Path, required=True)
    i.add_argument('--reference', default='awake_ec')
    i.add_argument('--draws', type=int, default=3)
    i.add_argument('--channels', type=int, default=32)
    j = sub.add_parser('sign-null')
    j.add_argument('bank', type=Path)
    j.add_argument('--output', type=Path, required=True)
    j.add_argument('--reference', default='awake_ec')
    k = sub.add_parser('null-bank')
    k.add_argument('bank', type=Path)
    k.add_argument('--output', type=Path, required=True)
    k.add_argument('--kind', choices=NULL_KINDS, required=True)
    k.add_argument('--units', nargs='+')
    k.add_argument('--seed', type=int, default=0)
    m = sub.add_parser('null-decompose')
    m.add_argument('discovery', type=Path)
    m.add_argument('--output', type=Path, required=True)
    m.add_argument('--reference', default='awake_ec')
    m.add_argument('--units', nargs='+')
    args = p.parse_args()
    if args.command == 'null-decompose':
        null_decompose(args.discovery, args.output, args.reference, args.units)
    elif args.command == 'null-bank':
        null_bank(args.bank, args.output, args.kind, args.units, args.seed)
    elif args.command == 'lag-null':
        lag_null(args.bank, args.output, args.reference, args.draws, args.channels)
    elif args.command == 'sign-null':
        sign_null(args.bank, args.output, args.reference)
    elif args.command == 'spectral-null':
        spectral_null(args.bank, args.output, args.reference, args.draws)
    elif args.command == 'reduce':
        reduce(args.bank, args.pyspi, args.output, args.jobs)
    elif args.command == 'reliability':
        reliability(args.bank, args.pyspi, args.output, args.reference, jobs=args.jobs)
    elif args.command == 'stability':
        stability(args.bank, args.pyspi, args.output, args.reference, jobs=args.jobs)
    elif args.command == 'explore':
        explore(args.reduced, args.output, args.reference)
    elif args.command == 'pairs':
        pairs(args.reduced, args.output, args.reference)
    elif args.command == 'screen':
        screen(args.reduced, args.output, args.reference, args.drug_state, double_centred=args.double_centred)
    else:
        analyse(args.reduced, args.output, args.reference)
