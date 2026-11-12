"""Assumption classes over the whole z atlas (reading rules: docs/interpreting-spi-pairs.md).

Every valid SPI is parsed from its pyspi identifier into attributes (statistic, estimator, sign handling, lag, band,
aggregation over frequency or lag, and family-specific options). Two SPIs whose attributes differ in exactly one
place form a minimal pair, and minimal pairs that differ in the same way form a class: one assumption, many
estimator families. The class, not the coordinate, is the unit of inference: a character reading is supported when
the member pairs of a class move together across statistic families, which a single estimator quirk cannot do.

catalogue: the class table for a list of SPIs, with each class's character reading, literal reading and known
  confound. Written and committed before any effect is computed.
score: per class and condition (agent:state against eyes-closed wake, within unit), the member pairs' effects raw,
  adjusted for strength, and adjusted for strength and the channel-mean spectrum, in eyes-open-null SD and in raw z
  units; the share of members agreeing in sign; the same on the eyes-open contrast as the class null; partner
  reliability where a reliability file covers both SPIs. Then the reverse direction: the coordinates most
  consistent across the drug conditions, annotated with how their SPIs differ and, for compound pairs, the
  one-difference links between them and each link's own effect.

Reading tiers, fixed before the first run (13 November 2026); tiers, not gates. Per class and condition:
  convergent: at least 3 usable members from at least 2 statistics; class median beyond 3 eyes-open SD raw and
    after the spectral adjustment, same sign; at least 75% of members share that sign; eyes-open class median
    within 1 SD; median |dz| larger than the saturation control's |dz| in that condition when the class sits near
    the ceiling (median |z0| > .9); a majority of members with both partners at reference reliability >= .3;
  partial: the size and sign-share criteria hold raw but not after the spectral adjustment, or the class has
    fewer members or statistics;
  estimator-local: one member carries it.
Classes are read across conditions afterwards (all drugs, a subset, drugs against sleep); a subset pattern is a
result, not a failure. Reverse-ranked coordinates are post-selection over about 24,000 and are read only through
their links.
"""
import argparse
import json
import re
from collections import deque
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np

BANDS = {('0', '0-5'): 'broad', ('1e-05', '0-5'): 'broad', ('0', '0-25'): 'low', ('1e-05', '0-25'): 'low',
         ('0-25', '0-5'): 'high'}
ANTISYMMETRIC = ('phase_multitaper', 'psi_multitaper', 'psi_wavelet', 'pli_multitaper', 'wpli_multitaper', 'gd_')
_INFO = ('mi', 'tlmi', 'je', 'ce', 'cce', 'xme', 'si', 'di', 'te', 'gc')


@lru_cache(maxsize=None)
def parse(name):
    """pyspi identifier -> attribute dict. Linear correlation variants share stat 'corr' so that covariance
    estimators, rank transforms and lags are attributes of one statistic: on z-scored channels the empirical
    covariance is Pearson r (pyspi normalises by default)."""
    a = dict(stat=None, est=None, sign='signed', lag=0, band=None, agg=None, opt=None)
    m = re.fullmatch(r'(cov|prec)(-sq)?_(\w+)', name)
    if m:
        a.update(stat='corr' if m[1] == 'cov' else 'prec', est=m[3].replace('Covariance', ''), sign='sq' if m[2] else 'signed')
        return a
    m = re.fullmatch(r'pcorr_(\w+)', name)
    if m:
        return {**a, 'stat': 'pcorr', 'est': m[1].replace('Covariance', '')}
    m = re.fullmatch(r'(spearmanr|kendalltau)(-sq)?', name)
    if m:
        return {**a, 'stat': 'corr', 'est': 'spearman' if m[1] == 'spearmanr' else 'kendall', 'sign': 'sq' if m[2] else 'signed'}
    m = re.fullmatch(r'corr_(pearson|spearman|kendall)_tau-(\d+)(-sq)?', name)
    if m:
        return {**a, 'stat': 'corr', 'est': 'Empirical' if m[1] == 'pearson' else m[1], 'lag': int(m[2]),
                'sign': 'sq' if m[3] else 'signed'}
    m = re.fullmatch(r'xcorr(-sq)?_(max|mean)_sig-(True|False)', name)
    if m:
        return {**a, 'stat': 'xcorr', 'sign': 'sq' if m[1] else 'signed', 'agg': m[2], 'opt': f'threshold-{m[3]}'}
    m = re.fullmatch(r'(\w+?)_(multitaper|parametric|wavelet)_(mean|max)_fs-1_fmin-([\d\-e]+)_fmax-([\d\-]+?)(_order-(\w+)|_mean)?', name)
    if m:
        stat = m[1] if m[2] != 'wavelet' else f'{m[1]}_wavelet'
        return {**a, 'stat': stat, 'agg': m[3], 'band': BANDS.get((m[4], m[5]), f'{m[4]}-{m[5]}'),
                'opt': f'order-{m[7]}' if m[7] and m[7] != 'None' else None, 'sign': 'antisymmetric' if name.startswith(ANTISYMMETRIC) else 'signed'}
    m = re.fullmatch(r'(mi|tlmi|je|ce|cce|xme|si|di)_(gaussian|kraskov|kozachenko|kernel)(_NN-4)?(_W-[\d.]+)?(_DCE-AUTO)?(_n-\d+|_k-?\d+)?', name)
    if m:
        return {**a, 'stat': m[1], 'est': m[2] + ('_DCE' if m[5] else ''), 'opt': m[6].lstrip('_') if m[6] else None}
    m = re.fullmatch(r'(te|gc)_(gaussian|kraskov|kernel|symbolic)(_NN-4)?(_W-[\d.]+)?(_DCE-AUTO)?(_MAX-CORR-AIS_k-max-10_tau-max-2)?_k-(\d)(_kt-1_l-1_lt-1)?', name) \
        or re.fullmatch(r'(gc)_(gaussian)()()()(_MAX-CORR-AIS)_k-max-10_tau-max-2()()', name)
    if m:
        return {**a, 'stat': 'te', 'est': m[2] + ('_DCE' if m[5] else ''),
                'opt': 'history-auto' if m[6] else f'history-{m[7]}'}
    m = re.fullmatch(r'pec(_orth)?(_log)?(_abs)?', name)
    if m:
        return {**a, 'stat': 'pec', 'est': 'log' if m[2] else 'linear', 'opt': 'orth' if m[1] else 'plain',
                'sign': 'abs' if m[3] else 'signed'}
    m = re.fullmatch(r'pdist_(\w+)', name)
    if m:
        return {**a, 'stat': 'pdist', 'est': m[1]}
    m = re.fullmatch(r'(dtw|lcss)(_constraint-(itakura|sakoe-chiba)(_radius-auto)?)?', name)
    if m:
        return {**a, 'stat': m[1], 'opt': m[3] or 'unconstrained'}
    m = re.fullmatch(r'bary(-sq)?_(\w+?)_(mean|max|corr)', name)
    if m:
        return {**a, 'stat': 'bary', 'est': m[2], 'agg': m[3], 'sign': 'sq' if m[1] else 'signed'}
    m = re.fullmatch(r'dcorr(_biased)?', name)
    if m:
        return {**a, 'stat': 'dcorr', 'est': 'biased' if m[1] else 'unbiased'}
    m = re.fullmatch(r'phi_(star|Geo)_t-1_norm-(\d)', name)
    if m:
        return {**a, 'stat': 'phi', 'est': m[1], 'opt': f'norm-{m[2]}'}
    m = re.fullmatch(r'coint_johansen_(max_eig|trace)_stat_order-(\d)_ardiff-(\d+)', name)
    if m:
        return {**a, 'stat': 'coint_johansen', 'est': m[1], 'opt': f'order-{m[2]}', 'lag': int(m[3])}
    m = re.fullmatch(r'coint_aeg_tstat_trend-(c|ct)_autolag-(aic|bic)_maxlag-10', name)
    if m:
        return {**a, 'stat': 'coint_aeg', 'est': m[2], 'opt': f'trend-{m[1]}'}
    m = re.fullmatch(r'xpdist_euclidean_tau-1_(min|mean)_rmse', name)
    if m:
        return {**a, 'stat': 'xpdist', 'agg': m[1]}
    m = re.fullmatch(r'lmfit_(\w+)', name)
    if m:
        return {**a, 'stat': 'lmfit', 'est': m[1]}
    return {**a, 'stat': name}


# One-assumption links between different statistics. Key: frozenset of the two stats (attributes otherwise equal).
LINKS = {
    frozenset(('corr', 'prec')): 'mediation',
    frozenset(('corr', 'pcorr')): 'mediation (normalised)',
    frozenset(('prec', 'pcorr')): 'precision normalisation',
    frozenset(('cohmag', 'icoh')): 'zero-lag removal (coherence)',
    frozenset(('ppc', 'dswpli')): 'zero-lag removal (debiased phase)',
    frozenset(('cohmag', 'plv')): 'amplitude weighting of synchrony',
    frozenset(('plv', 'ppc')): 'phase-locking debiasing',
    frozenset(('pli', 'wpli')): 'imaginary-part weighting',
    frozenset(('dspli', 'dswpli')): 'imaginary-part weighting',
    frozenset(('cohmag', 'sgc')): 'precedence (spectral)',
    frozenset(('mi', 'tlmi')): 'lag one sample (information)',
    frozenset(('tlmi', 'te')): 'conditioning on the target past',
    frozenset(('mi', 'je')): 'marginal entropies added',
    frozenset(('je', 'ce')): 'one marginal entropy removed',
    frozenset(('dtw', 'lcss')): 'elastic distance against elastic similarity',
}
# Cross-stat links whose attributes are not otherwise equal by construction of parse(): (a-pattern, b-pattern).
EXTRA = [
    ('saturation control', r'cov-sq_EmpiricalCovariance', r'mi_gaussian'),
    ('identity control (distance)', r'cov_EmpiricalCovariance', r'pdist_euclidean'),
    ('best lag against zero lag', r'cov_EmpiricalCovariance', r'xcorr_max_sig-True'),
    ('best lag against zero lag', r'cov-sq_EmpiricalCovariance', r'xcorr-sq_max_sig-True'),
    ('any dependence against linear', r'cov-sq_EmpiricalCovariance', r'dcorr'),
    ('rigid against elastic alignment', r'pdist_euclidean', r'dtw'),
    ('conditioning on the target past', r'tlmi_gaussian', r'gc_gaussian_k-1_kt-1_l-1_lt-1'),
    ('conditioning on the target past', r'tlmi_kraskov_NN-4_DCE-AUTO', r'te_kraskov_NN-4_DCE-AUTO_k-1_kt-1_l-1_lt-1'),
    ('conditioning on the target past', r'tlmi_kernel_W-0.25', r'te_kernel_W-0.25_k-1'),
    ('estimator: gaussian vs symbolic', r'gc_gaussian_k-1_kt-1_l-1_lt-1', r'te_symbolic_k-3'),
    ('zero-lag removal (envelope)', r'pec', r'pec_orth'),
    ('zero-lag removal (envelope)', r'pec_log', r'pec_orth_log'),
]

# Curated readings. Key: class name. (character reading, literal reading, known confound or status)
READINGS = {
    'sign handling: signed vs sq': (
        'sign structure: whether the strongest couplings include anticorrelations',
        'organisation of signed against squared dependence', 'sampling noise at high spectral concentration (stationary null reproduces the per-window fall except under ketamine); common-average reference induces anticorrelations'),
    'estimator: Empirical vs spearman': (
        'monotone-nonlinear dependence (rank invariance); channel marginal shape plus copula departure',
        'organisation of linear against rank dependence', 'marginal shape (raw); copula part needs normal scores or IAAFT'),
    'estimator: kendall vs spearman': ('weighting of large rank discordances (fixed relation under a Gaussian copula)', 'two rank correlations', 'near-identity ceiling'),
    'estimator: Empirical vs kendall': ('monotone-nonlinear dependence (as Spearman)', 'linear against Kendall rank dependence', 'marginal shape'),
    'estimator: Empirical vs MinCovDet': (
        'time-heterogeneous dependence: a minority of samples carries a pattern different from the bulk',
        'classical against minimum-covariance-determinant estimate', 'spatially focal transients; spectral flag clustering'),
    'estimator: LedoitWolf vs MinCovDet': ('time-heterogeneous dependence (shrunk classical against robust)', 'shrunk against robust estimate', 'shrinkage as well as robustness differ'),
    'estimator: Empirical vs LedoitWolf': (None, 'shrinkage', 'identity for covariance (uniform scaling); conditioning for precision at T=200, M=125'),
    'estimator: Empirical vs OAS': (None, 'shrinkage', 'as LedoitWolf'),
    'estimator: Empirical vs Shrunk': (None, 'shrinkage', 'as LedoitWolf'),
    'estimator: gaussian vs kraskov_DCE': (
        'non-Gaussian dependence: how far each pair departs from the dependence a Gaussian process with its covariance would carry',
        'organisation of Gaussian against kNN (bias-corrected) information (mi, tlmi) or prediction (te: GC against TE)', 'spectral concentration, kNN estimator bias at T=200; MI part survives IAAFT; TE part tracks spectral entropy'),
    'estimator: gaussian vs kernel': ('non-Gaussian dependence (kernel estimator)', 'Gaussian against kernel information', 'kernel bandwidth; autocorrelation bias'),
    'estimator: gaussian vs kozachenko': ('non-Gaussian dependence (entropy estimator)', 'Gaussian against Kozachenko-Leonenko entropy', 'marginal entropies are node-level'),
    'estimator: gaussian vs kraskov': (None, 'Gaussian against uncorrected kNN information', 'autocorrelation bias of uncorrected KSG dominates'),
    'estimator: kraskov vs kraskov_DCE': (None, 'kNN information with and without the dynamic correlation exclusion', 'estimator diagnostic: autocorrelation bias'),
    'estimator: gaussian vs symbolic': ('ordinal-pattern (amplitude-free) predictive structure', 'Gaussian against symbolic TE', 'pattern length 3; T'),
    'band: low vs high': ('frequency locus of coupling: which pairs couple fast and which slow', 'organisation of low- against high-band coupling', 'within-band location; spectral'),
    'band: broad vs low': (None, 'broadband against low band', 'dominated by the low band'),
    'band: broad vs high': (None, 'broadband against high band', 'mostly band power'),
    'aggregation: mean vs max': ('spectral peakedness of coupling: narrowband (one frequency) against broadband coupling', 'peak against mean over frequencies or lags', 'spectral concentration of the channels'),
    'lag: 0 vs 1': ('one-sample persistence of coupling', 'zero-lag against lag-1 correlation', 'autocorrelation, dominant frequency (10 ms per sample)'),
    'mediation': ('mediation: direct against indirect coupling (marginal against conditional on all other channels)', 'organisation of marginal against partial dependence', 'montage, reference, M and T (precision near-singular at T=200)'),
    'mediation (normalised)': ('mediation (partial correlation)', 'marginal against partial correlation', 'as mediation'),
    'precision normalisation': (None, 'precision against partial correlation', 'node normalisation'),
    'zero-lag removal (coherence)': ('non-zero-lag share of coupling', 'coherence magnitude against imaginary coherence', 'estimation floor of icoh (zero-phase null reproduces 50-75% of the drug rise)'),
    'zero-lag removal (debiased phase)': ('non-zero-lag share of phase coupling', 'PPC against debiased wPLI', 'dswpli unreliable (reliability .03)'),
    'zero-lag removal (envelope)': ('lagged share of amplitude coupling (volume-conduction-free)', 'envelope correlation with and without orthogonalisation', 'pec_orth unreliable'),
    'amplitude weighting of synchrony': ('amplitude-weighted against pure phase synchrony', 'coherence against phase-locking value', 'medetomidine exception at high band'),
    'phase-locking debiasing': (None, 'PLV against its unbiased square', 'estimator diagnostic'),
    'precedence (spectral)': ('predictive (directed) share of spectral coupling', 'coherence against spectral Granger', 'AR model order'),
    'lag one sample (information)': ('one-sample persistence of dependence', 'MI against lagged MI', 'autocorrelation'),
    'conditioning on the target past': ('information the source adds beyond the target own past', 'lagged MI against TE/GC', 'autocorrelation'),
    'saturation control': (None, 'r^2 against -1/2 log(1-r^2): a fixed monotone function', 'pure saturation and range restriction: the floor any ceiling class must beat'),
    'identity control (distance)': (None, 'Euclidean distance is a fixed function of r on z-scored data', 'identity, z = -1'),
    'best lag against zero lag': ('coupling carried at a non-zero best lag', 'zero-lag against best-lag correlation', 'dominant frequency; xcorr reliability'),
    'any dependence against linear': ('non-linear (any-form) dependence share', 'r^2 against distance correlation', 'saturation; beyond-IAAFT ladder reported by independent review'),
    'rigid against elastic alignment': ('variable-lag (warped) alignment', 'Euclidean against DTW distance', 'warp failed construction validation'),
    # Added after the first run (labels only; tiers unchanged):
    'estimator: cityblock vs euclidean': ('tail weight of the pair disagreement signal x - y (mean absolute against RMS difference; constant for Gaussian x - y)', 'L1 against L2 distance', 'ceiling; marginal shape of each channel enters x - y'),
    'estimator: chebyshev vs euclidean': ('worst-case disagreement: intermittent large divergence of a pair', 'L-infinity against L2 distance', 'one-sample maximum is noisy; transients'),
    'estimator: linear vs log': ('burst-driven against proportional amplitude coupling', 'envelope correlation on raw against log envelopes', 'heavy-tailed envelopes'),
}


PREFERRED = ['signed', 'gaussian', 'Empirical', '0', 'broad', 'low', 'mean', 'linear', 'unconstrained', 'order-None',
             'history-1', 'kraskov']


def classify(a, b):
    """Class name of a minimal pair, or None when the SPIs differ in more than one attribute."""
    A, B = parse(a), parse(b)
    for name, pa, pb in EXTRA:
        if {a, b} == {pa, pb}:
            return name
    diff = [k for k in A if A[k] != B[k]]
    if len(diff) != 1:
        return None
    k = diff[0]
    if k == 'stat':
        return LINKS.get(frozenset((A['stat'], B['stat'])))
    if 'antisymmetric' in (A['sign'], B['sign']):
        return None
    va, vb = sorted([str(A[k]), str(B[k])], key=lambda v: (PREFERRED.index(v) if v in PREFERRED else len(PREFERRED), v))
    if k == 'lag' and va != '0':
        return 'lag: adjacent or longer'
    if k == 'lag':
        vb = '1' if vb == '1' else 'L>1'
    return {'sign': 'sign handling', 'est': 'estimator', 'lag': 'lag', 'band': 'band', 'agg': 'aggregation',
            'opt': 'option'}[k] + f': {va} vs {vb}'.replace('None', 'default')


def edges(spis):
    """All minimal pairs among spis: [(class, a, b)] with the class orientation a -> b as listed."""
    return list(_edges(tuple(spis)))


@lru_cache(maxsize=4)
def _edges(spis):
    out = []
    for a, b in combinations(spis, 2):
        c = classify(a, b)
        if c is not None:
            out.append((c, a, b))
    return out


def catalogue(spis, output):
    groups = {}
    for c, a, b in edges(spis):
        groups.setdefault(c, []).append([a, b])
    rows = []
    for c, members in sorted(groups.items(), key=lambda x: -len(x[1])):
        char, lit, conf = READINGS.get(c, (None, None, None))
        stats = sorted({parse(x)['stat'] for m in members for x in m})
        rows.append(dict(cls=c, members=len(members), statistics=stats, character=char, literal=lit, confound=conf,
                         pairs=members))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(spis=len(spis), classes=rows), indent=1) + '\n')
    return rows


def links(spis, a, b, depth=3):
    """Shortest chain of minimal pairs from a to b (breadth-first), as [(class, x, y)], or None."""
    graph = {}
    for c, x, y in edges(spis):
        graph.setdefault(x, []).append((c, y))
        graph.setdefault(y, []).append((c, x))
    prev, seen, q = {}, {a}, deque([(a, 0)])
    while q:
        x, d = q.popleft()
        if x == b:
            chain = []
            while x != a:
                c, p = prev[x]
                chain.append((c, p, x))
                x = p
            return chain[::-1]
        if d == depth:
            continue
        for c, y in graph.get(x, []):
            if y not in seen:
                seen.add(y)
                prev[y] = (c, x)
                q.append((y, d + 1))
    return None


def score(reduced, output, reference, reliability=None, top=40, ceiling=.98):
    from research.representation.atlas.discovery import COVARIATES, SPECTRAL, unit_effects
    from research.representation.character_axes.episodic import sleep_tertiles
    r = np.load(reduced, allow_pickle=False)
    rows = json.loads(str(r['rows']))
    spis = [str(s) for s in r['spis']]
    K = len(spis)
    index = np.full((K, K), -1)
    index[np.triu_indices(K, 1)] = np.arange(K * (K - 1) // 2)
    index = np.maximum(index, index.T)
    keep = np.flatnonzero(r['valid'].all(0))
    valid = [spis[i] for i in keep]
    unit = np.array([x['unit'] for x in rows])
    agent = np.array([x.get('agent', 'all') for x in rows])
    animal_of = {x['unit']: x.get('animal') or x['unit'] for x in rows}
    state = np.array(sleep_tertiles(rows) if any('delta_fraction' in x for x in rows) else [x['state'] for x in rows])
    names = [str(n) for n in r['covariate_names']]
    adjust = {'raw': None, 'adjusted': r['covariates'][:, [names.index(c) for c in COVARIATES]],
              'spectral': r['covariates'][:, [names.index(c) for c in SPECTRAL]]}
    node = dict(zip(spis, np.nanmedian(r['node_r2'][state == reference], 0).tolist()))
    rel, drel = {}, {}
    if reliability is not None:
        d = json.loads(Path(reliability).read_text())
        rel = {k: v['reliability'] for k, v in d['reference_state'].items()}
        drel = {c: {k.rsplit(':', 1)[0]: v['mean'] for k, v in e.items()} for c, e in d['contrasts'].items()}

    pairs = [(a, b) for a, b in combinations(valid, 2)]
    cols = np.array([index[spis.index(a), spis.index(b)] for a, b in pairs])
    Z = r['z'][:, cols].astype(np.float64)
    z0 = np.nanmean(Z[state == reference], 0)
    structural = np.nanmax(np.abs(Z), 0) < 1e-9
    conditions = sorted({f'{g}:{s}' for g, s in zip(agent, state)
                         if s not in (reference, 'awake_eo') and not s.startswith('recovery')})
    eff, animals = {}, {}
    for c in conditions + [f'{g}:awake_eo' for g in sorted(set(agent[state == 'awake_eo']))]:
        g, s = c.split(':')
        sel = agent == g
        us = [u for u in np.unique(unit[sel]) if ((unit == u) & (state == s)).any() and ((unit == u) & (state == reference)).any()]
        eff[c] = {v: unit_effects(Z[sel], unit[sel], state[sel], s, reference, None if C is None else C[sel])
                  for v, C in adjust.items()}
        animals[c] = np.array([animal_of[u] for u in us])
    null = np.concatenate([e['raw'] for c, e in eff.items() if c.endswith(':awake_eo')])
    null_sd = null.std(0, ddof=1) + 1e-6

    def condition_summary(E, A):
        """E (units, P) -> per-column mean, sign-consistent unit share and all-animals-agree flag."""
        mean = E.mean(0)
        share = np.mean(np.sign(E) == np.sign(mean), 0)
        by = np.array([E[A == x].mean(0) for x in np.unique(A)]) if len(A) == len(E) else E
        return mean, share, np.all(np.sign(by) == np.sign(mean), 0)

    summ = {c: {v: condition_summary(E, animals[c]) for v, E in d.items() if len(E)} for c, d in eff.items()}
    drugs = [c for c in conditions if not c.startswith('SLP')]
    report = dict(reduced=str(reduced), reference=reference, conditions=conditions, scale='eyes-open null SD',
                  reliability=str(reliability) if reliability else None, classes=[], top=[])

    position = {p: i for i, p in enumerate(pairs)}
    member_edges = edges(valid)
    groups = {}
    for c, a, b in member_edges:
        groups.setdefault(c, []).append((a, b))
    for c, members in sorted(groups.items(), key=lambda x: -len(x[1])):
        ix = np.array([position[(a, b)] if (a, b) in position else position[(b, a)] for a, b in members])
        usable = ~structural[ix] & np.array([node[a] <= .5 and node[b] <= .5 for a, b in members])
        char, lit, conf = READINGS.get(c, (None, None, None))
        entry = dict(cls=c, members=len(members), usable=int(usable.sum()), character=char, literal=lit, confound=conf,
                     statistics=sorted({parse(x)['stat'] for m in members for x in m}), conditions={}, pairs=[])
        if not usable.any():
            report['classes'].append(entry)
            continue
        iu = ix[usable]
        for cond in conditions + ['eyes_open']:
            src = {v: (np.concatenate([eff[k][v] for k in eff if k.endswith(':awake_eo')])[:, iu] if cond == 'eyes_open'
                       else eff[cond][v][:, iu]) for v in adjust if cond == 'eyes_open' or len(eff[cond][v])}
            row = {}
            for v, E in src.items():
                m = E.mean(0)
                s = m / null_sd[iu]
                sign = np.sign(np.median(s))
                row[v] = dict(median_sd=float(np.median(s)), median_dz=float(np.median(m)),
                              same_sign=float(np.mean(np.sign(s) == sign)),
                              beyond_3sd=float(np.mean((np.sign(s) == sign) & (np.abs(s) > 3))))
            entry['conditions'][cond] = row
        for k, (a, b) in enumerate(members):
            if not usable[k]:
                continue
            j = ix[k]
            entry['pairs'].append(dict(
                a=a, b=b, z0=float(z0[j]), null_sd=float(null_sd[j]), reliability=[rel.get(a), rel.get(b)],
                reliability_change={cond: [drel.get(cond, {}).get(a), drel.get(cond, {}).get(b)] for cond in conditions},
                effects={cond: dict(dz=float(summ[cond]['raw'][0][j]), sd=float(summ[cond]['raw'][0][j] / null_sd[j]),
                                    spectral_sd=float(summ[cond]['spectral'][0][j] / null_sd[j]) if 'spectral' in summ[cond] else None,
                                    share=float(summ[cond]['raw'][1][j]), animals_agree=bool(summ[cond]['raw'][2][j]))
                         for cond in conditions}))
        report['classes'].append(entry)

    # Reverse: coordinates consistent under every drug condition (raw and spectral-adjusted, >= 75% of units and
    # every animal agreeing), ranked by the weakest drug effect in null SD; ceiling, node-additive, structural out.
    ok = ~structural & (np.abs(z0) < ceiling) & np.array([node[a] <= .5 and node[b] <= .5 for a, b in pairs])
    signs = np.sign(np.mean([summ[c]['raw'][0] for c in drugs], 0))
    for c in drugs:
        for v in ('raw', 'spectral'):
            if v in summ[c]:
                m, share, agree = summ[c][v]
                ok &= (np.sign(m) == signs) & (share >= .75) & agree
    weakest = np.min([np.abs(summ[c]['raw'][0]) / null_sd for c in drugs], 0)
    order = [j for j in np.argsort(-np.where(ok, weakest, -np.inf)) if ok[j]]
    report['consistent'] = int(ok.sum())
    report['usable'] = int((~structural & (np.abs(z0) < ceiling)).sum())
    for j in order[:top]:
        a, b = pairs[j]
        A, B = parse(a), parse(b)
        chain = links(valid, a, b)
        item = dict(a=a, b=b, z0=float(z0[j]), weakest_sd=float(weakest[j]),
                    differs={k: [A[k], B[k]] for k in A if A[k] != B[k]}, cls=classify(a, b),
                    reliability=[rel.get(a), rel.get(b)],
                    effects={c: dict(dz=float(summ[c]['raw'][0][j]), sd=float(summ[c]['raw'][0][j] / null_sd[j]),
                                     spectral_sd=float(summ[c]['spectral'][0][j] / null_sd[j]) if 'spectral' in summ[c] else None)
                             for c in conditions}, links=[])
        for c, x, y in chain or []:
            jj = position[(x, y)] if (x, y) in position else position[(y, x)]
            item['links'].append(dict(cls=c, a=x, b=y, z0=float(z0[jj]),
                                      sd={k: float(summ[k]['raw'][0][jj] / null_sd[jj]) for k in conditions},
                                      dz={k: float(summ[k]['raw'][0][jj]) for k in conditions}))
        report['top'].append(item)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + '\n')


def reliability_all(bank, pyspi_dir, reduced, output, reference, chunk=40, jobs=8):
    """discovery.reliability for every valid SPI of the reduced arm, in chunks (one pass needs about 200 GB):
    reference-state reliability per SPI and its per-condition change."""
    from research.representation.atlas.discovery import reliability
    r = np.load(reduced, allow_pickle=False)
    spis = [str(s) for i, s in enumerate(r['spis']) if r['valid'][:, i].all()]
    out = dict(bank=str(bank), reference=reference, reference_state={}, contrasts={})
    for i in range(0, len(spis), chunk):
        part = output.with_name(f'{output.stem}.part{i}.json')
        reliability(bank, pyspi_dir, part, reference, named=[('self', s, s) for s in spis[i:i + chunk]], jobs=jobs)
        d = json.loads(part.read_text())
        out['reference_state'].update(d['reference_state'])
        for c, e in d['contrasts'].items():
            out['contrasts'].setdefault(c, {}).update({k: v for k, v in e.items() if k.endswith(':reliability')})
        part.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('catalogue')
    a.add_argument('spis', type=Path, help='text file, one SPI per line')
    a.add_argument('--output', type=Path, required=True)
    b = sub.add_parser('score')
    b.add_argument('reduced', type=Path)
    b.add_argument('--output', type=Path, required=True)
    b.add_argument('--reference', default='awake_ec')
    b.add_argument('--reliability', type=Path)
    c = sub.add_parser('reliability')
    c.add_argument('bank', type=Path)
    c.add_argument('pyspi', type=Path)
    c.add_argument('reduced', type=Path)
    c.add_argument('--output', type=Path, required=True)
    c.add_argument('--reference', default='awake_ec')
    c.add_argument('--jobs', type=int, default=8)
    args = p.parse_args()
    if args.command == 'reliability':
        reliability_all(args.bank, args.pyspi, args.reduced, args.output, args.reference, jobs=args.jobs)
    elif args.command == 'catalogue':
        catalogue([x for x in args.spis.read_text().split() if x], args.output)
    else:
        score(args.reduced, args.output, args.reference, args.reliability)
