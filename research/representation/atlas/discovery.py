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
    args = p.parse_args()
    if args.command == 'reduce':
        reduce(args.bank, args.pyspi, args.output, args.jobs)
    elif args.command == 'reliability':
        reliability(args.bank, args.pyspi, args.output, args.reference, jobs=args.jobs)
    elif args.command == 'pairs':
        pairs(args.reduced, args.output, args.reference)
    elif args.command == 'screen':
        screen(args.reduced, args.output, args.reference, args.drug_state, double_centred=args.double_centred)
    else:
        analyse(args.reduced, args.output, args.reference)
