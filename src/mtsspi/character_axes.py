"""Named axes of dependence character, read from minimal pairs of statistics.

A pairwise statistic reports how strongly each channel pair is coupled under the
assumptions it fixes, so on its own it cannot say where a coupling lies along
those assumptions. Two statistics that differ in one assumption can. Across the
channel pairs of a recording, the correlation of their edge profiles is high
when the property that separates them is constant or tracks shared strength,
and falls when that property varies between pairs independently of strength.

Each axis lists such minimal pairs from the benchmarked p90 catalogue. Its
organisation score is the mean profile correlation over its pairs, with the
same ordered off-diagonal entries and validity rule as the unified z contract.
Organisation is a statement about heterogeneity across pairs, not the level of
the property: equal slowing of every connection barely moves it. Where both
statistics share units, a level score reads the typical magnitude instead.

Antisymmetric statistics (A_ji = -A_ij) correlate with any symmetric profile at
exactly zero over ordered entries, so they enter axes as magnitudes |A|.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import warnings

import numpy as np

BANDS = {'broad': ('0', '0-5'), 'low': ('0', '0-25'), 'high': ('0-25', '0-5')}
LAGS = (1, 2, 3, 4, 5, 10, 20)
ANTISYMMETRIC_PREFIXES = ('phase_multitaper_mean', 'psi_multitaper_mean', 'psi_wavelet_mean',
                          'pli_multitaper_mean', 'wpli_multitaper_mean', 'gd_multitaper_delay')


def mt(family, agg, band):
    lo, hi = BANDS[band]
    return f'{family}_multitaper_{agg}_fs-1_fmin-{lo}_fmax-{hi}'


def sgc(agg, band, order=None):
    lo, hi = BANDS[band]
    lo = '1e-05' if lo == '0' else lo
    name = f'sgc_{"nonparametric" if order is None else "parametric"}_{agg}_fs-1_fmin-{lo}_fmax-{hi}'
    return name if order is None else f'{name}_order-{order}'


@dataclass(frozen=True)
class Axis:
    name: str
    low: str
    high: str
    pairs: tuple[tuple[str, str], ...]
    caveat: str = ''
    role: str = 'axis'          # 'axis', 'identity' or 'nuisance'


def _axes():
    sq = lambda n: n.replace('cov_', 'cov-sq_').replace('prec_', 'prec-sq_')
    A = []
    A.append(Axis('form_monotone', 'linear', 'monotone nonlinear',
        tuple([('cov_EmpiricalCovariance', 'spearmanr'), ('cov_EmpiricalCovariance', 'kendalltau'),
               ('cov-sq_EmpiricalCovariance', 'spearmanr-sq'), ('cov-sq_EmpiricalCovariance', 'kendalltau-sq')]
              + [(f'corr_pearson_tau-{k}{s}', f'corr_{r}_tau-{k}{s}') for k in LAGS for r in ('spearman', 'kendall') for s in ('', '-sq')]),
        'Rank statistics also differ from Pearson under heavy-tailed marginals; amplitude-preserving surrogates separate the two.'))
    A.append(Axis('form_information', 'Gaussian', 'non-Gaussian dependence',
        (('mi_gaussian', 'mi_kraskov_NN-4'), ('mi_gaussian', 'mi_kernel_W-0.25'),
         ('tlmi_gaussian', 'tlmi_kraskov_NN-4'), ('tlmi_gaussian', 'tlmi_kernel_W-0.25'),
         ('gc_gaussian_k-1_kt-1_l-1_lt-1', 'te_kraskov_NN-4_k-1_kt-1_l-1_lt-1'),
         ('gc_gaussian_k-1_kt-1_l-1_lt-1', 'te_kernel_W-0.25_k-1'),
         ('cov-sq_EmpiricalCovariance', 'dcorr')),
        'Nonparametric estimators carry sample-size-dependent bias; compare at matched T only.'))
    A.append(Axis('timing_fixed_lag', 'instantaneous', 'lagged at fixed delays',
        tuple([('cov_EmpiricalCovariance', f'corr_pearson_tau-{k}') for k in LAGS]
              + [('cov-sq_EmpiricalCovariance', f'corr_pearson_tau-{k}-sq') for k in LAGS]
              + [('mi_gaussian', 'tlmi_gaussian'), ('mi_kraskov_NN-4', 'tlmi_kraskov_NN-4')]),
        'Fixed sample lags weight the cross-spectrum by cos(2 pi f tau): this axis also moves with dominant frequency.'))
    A.append(Axis('timing_best_shift', 'zero shift', 'best rigid shift',
        (('cov_EmpiricalCovariance', 'xcorr_max_sig-True'), ('cov-sq_EmpiricalCovariance', 'xcorr-sq_max_sig-True'),
         ('pdist_euclidean', 'xpdist_euclidean_tau-1_min_rmse'))))
    A.append(Axis('timing_warping', 'rigid shift', 'elastic (variable-lag) alignment',
        (('xpdist_euclidean_tau-1_min_rmse', 'dtw'), ('xpdist_euclidean_tau-1_min_rmse', 'dtw_constraint-itakura'),
         ('xpdist_euclidean_tau-1_min_rmse', 'dtw_constraint-sakoe-chiba_radius-auto'),
         ('bary_euclidean_max', 'bary_dtw_max')),
        'Barycentre means are structurally ~0 on standardized inputs, so only barycentre maxima enter.'))
    phase_lag = [(mt('cohmag', g, b), mt('icoh', g, b)) for g in ('mean', 'max') for b in BANDS]
    phase_lag += [(mt('plv', 'mean', b), mt(x, 'mean', b)) for b in BANDS for x in ('pli', 'wpli')]
    phase_lag += [(mt('ppc', 'mean', b), mt(x, 'mean', b)) for b in BANDS for x in ('dspli', 'dswpli')]
    phase_lag += [('pec', 'pec_orth'), ('pec_log', 'pec_orth_log')]
    A.append(Axis('phase_lag', 'zero-phase', 'phase-lagged', tuple(phase_lag),
        'In scalp EEG zero-phase coupling includes volume conduction; this axis is its main confound and its main use.'))
    predictive = [('mi_gaussian', 'gc_gaussian_k-1_kt-1_l-1_lt-1'),
                  ('mi_kraskov_NN-4', 'te_kraskov_NN-4_k-1_kt-1_l-1_lt-1'), ('mi_kernel_W-0.25', 'te_kernel_W-0.25_k-1')]
    predictive += [(mt('cohmag', g, b), sgc(g, b)) for g in ('mean', 'max') for b in BANDS]
    A.append(Axis('predictive', 'contemporaneous', 'predictive (Granger-type)', tuple(predictive)))
    history = [('gc_gaussian_k-1_kt-1_l-1_lt-1', 'gc_gaussian_MAX-CORR-AIS_k-max-10_tau-max-2'),
               ('te_kraskov_NN-4_DCE-AUTO_k-1_kt-1_l-1_lt-1', 'te_kraskov_NN-4_DCE-AUTO_k-2_kt-1_l-1_lt-1')]
    history += [(sgc(g, b, 1), sgc(g, b, 20)) for g in ('mean', 'max') for b in BANDS]
    A.append(Axis('history', 'short history', 'long history', tuple(history)))
    mediation = [(f'{c}_{e}', f'{p}_{e}') for e in ('EmpiricalCovariance', 'LedoitWolf') for c, p in (('cov', 'prec'), ('cov-sq', 'prec-sq'))]
    mediation += [(mt(t, g, b), mt(d, g, b)) for t, d in (('dtf', 'ddtf'), ('dcoh', 'pdcoh')) for g in ('mean', 'max') for b in BANDS]
    A.append(Axis('mediation', 'marginal', 'direct (conditioned on other channels)', tuple(mediation),
        'Conditioning set is every other channel, so scores are comparable only at fixed montage.'))
    band = [(mt(f, g, 'low'), mt(f, g, 'high')) for f in ('cohmag', 'icoh', 'plv', 'ppc', 'dtf', 'dcoh', 'pdcoh', 'ddtf') for g in ('mean', 'max')]
    band += [(mt(f, 'mean', 'low'), mt(f, 'mean', 'high')) for f in ('pli', 'wpli', 'dspli', 'dswpli')]
    band += [(sgc(g, 'low'), sgc(g, 'high')) for g in ('mean', 'max')]
    A.append(Axis('band', 'low band', 'high band', tuple(band),
        'Bands are fixed fractions of the sampling rate (0-fs/4, fs/4-fs/2).'))
    peak = [(mt(f, 'mean', 'broad'), mt(f, 'max', 'broad')) for f in ('cohmag', 'icoh', 'plv', 'ppc', 'dtf', 'dcoh', 'pdcoh', 'ddtf')]
    peak += [(sgc('mean', 'broad'), sgc('max', 'broad')), (sgc('mean', 'broad', 20), sgc('max', 'broad', 20))]
    A.append(Axis('spectral_peak', 'broadband', 'spectrally peaked', tuple(peak)))
    A.append(Axis('phase_amplitude', 'phase', 'amplitude envelope',
        ((mt('plv', 'mean', 'broad'), 'pec'), (mt('ppc', 'mean', 'broad'), 'pec'),
         (mt('plv', 'mean', 'low'), 'pec'), (mt('wpli', 'mean', 'broad'), 'pec_orth'), (mt('cohmag', 'mean', 'broad'), mt('plv', 'mean', 'broad'))),
        'Waveform shape alone can couple phase and envelope statistics; see the hardware calibration record.'))
    A.append(Axis('tail', 'bulk', 'outlier-robust (rare-event) coupling',
        (('cov_EmpiricalCovariance', 'cov_MinCovDet'), ('cov_EmpiricalCovariance', 'cov_EllipticEnvelope'),
         ('cov-sq_EmpiricalCovariance', 'cov-sq_MinCovDet'), ('prec_EmpiricalCovariance', 'prec_MinCovDet'),
         ('pdist_euclidean', 'pdist_chebyshev'), ('pdist_euclidean', 'pdist_cityblock'))))
    A.append(Axis('identity', 'control', 'control',
        (('cov_EmpiricalCovariance', 'pdist_euclidean'), ('cov-sq_EmpiricalCovariance', 'mi_gaussian'),
         ('je_gaussian', 'mi_gaussian'), ('cov_EmpiricalCovariance', 'cov_OAS')),
        'Fixed functions of each other on standardized inputs; |score| should be ~1 and measures numerical floor.', 'identity'))
    A.append(Axis('estimator', 'control', 'control',
        (('cov_EmpiricalCovariance', 'cov_LedoitWolf'), ('cov_EmpiricalCovariance', 'cov_GraphicalLasso'),
         ('mi_kraskov_NN-4', 'mi_kraskov_NN-4_DCE-AUTO'), ('bary_dtw_mean', 'bary_sgddtw_mean'),
         ('te_kraskov_NN-4_k-1_kt-1_l-1_lt-1', 'te_kraskov_NN-4_DCE-AUTO_k-1_kt-1_l-1_lt-1')),
        'Estimator settings, not character; large movement here flags estimation artefacts.', 'nuisance'))
    return tuple(A)


AXES = _axes()

# Level readouts: median over ordered entries of a contrast in shared units.
LEVELS = {
    'form_information': ('mi_kraskov_NN-4', 'mi_gaussian', 'difference'),        # nats of non-Gaussian MI
    'form_monotone': ('spearmanr', 'cov_EmpiricalCovariance', 'abs_difference'),
    'timing_best_shift': ('xcorr-sq_max_sig-True', 'cov-sq_EmpiricalCovariance', 'difference'),
    'phase_lag': (mt('icoh', 'mean', 'broad'), mt('cohmag', 'mean', 'broad'), 'ratio'),
    'band': (mt('cohmag', 'mean', 'high'), mt('cohmag', 'mean', 'low'), 'share'),
    'spectral_peak': (mt('cohmag', 'max', 'broad'), mt('cohmag', 'mean', 'broad'), 'ratio'),
    'timing_warping': ('dtw', 'pdist_euclidean', 'ratio'),
    'tail': ('pdist_chebyshev', 'pdist_euclidean', 'ratio'),
}


def is_antisymmetric(name):
    return name.startswith(ANTISYMMETRIC_PREFIXES)


def edge_profiles(aligned, M, K):
    """(n, K*M(M-1)) bank block -> (n, K, M(M-1)) ordered off-diagonal profiles."""
    aligned = np.asarray(aligned, dtype=np.float64)
    return aligned.reshape(aligned.shape[0], K, M * (M - 1))


def profile_correlation(x, y):
    """Row-wise Pearson correlation; NaN where either profile is nonfinite or constant."""
    x = x - x.mean(-1, keepdims=True)
    y = y - y.mean(-1, keepdims=True)
    nx, ny = np.linalg.norm(x, axis=-1), np.linalg.norm(y, axis=-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.sum(x * y, -1) / (nx * ny)
    bad = ~np.isfinite(r) | (nx < 1e-12) | (ny < 1e-12)
    return np.where(bad, np.nan, np.clip(r, -1, 1))


def axis_scores(profiles, spi_order: Sequence[str], axes=AXES):
    """Return organisation scores, per-pair correlations and the pairs used.

    Pairs naming a statistic absent from ``spi_order`` are dropped and listed.
    """
    index = {name: k for k, name in enumerate(spi_order)}
    pair_values, pairs, missing, scores = [], [], [], []
    for axis in axes:
        values = []
        for a, b in axis.pairs:
            if a not in index or b not in index:
                missing.append((axis.name, a, b)); continue
            x, y = profiles[:, index[a]], profiles[:, index[b]]
            if is_antisymmetric(a) != is_antisymmetric(b):
                x = np.abs(x) if is_antisymmetric(a) else x
                y = np.abs(y) if is_antisymmetric(b) else y
            r = profile_correlation(x, y)
            if axis.role == 'identity':
                r = np.abs(r)
            values.append(r); pairs.append((axis.name, a, b)); pair_values.append(r)
        values = np.array(values).T if values else np.full((len(profiles), 0), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            scores.append(np.nanmean(values, axis=1) if values.size else np.full(len(profiles), np.nan))
    return np.array(scores).T, np.array(pair_values).T, pairs, missing


def level_scores(profiles, spi_order: Sequence[str], levels=LEVELS):
    index = {name: k for k, name in enumerate(spi_order)}
    names, values = [], []
    for axis, (a, b, kind) in levels.items():
        if a not in index or b not in index:
            continue
        x, y = profiles[:, index[a]], profiles[:, index[b]]
        with np.errstate(invalid='ignore', divide='ignore'):
            if kind == 'difference': c = x - y
            elif kind == 'abs_difference': c = np.abs(x) - np.abs(y)
            elif kind == 'ratio': c = np.abs(x) / np.abs(y)
            elif kind == 'share': c = np.abs(x) / (np.abs(x) + np.abs(y))
            else: raise ValueError(kind)
        c = np.where(np.isfinite(c), c, np.nan)
        ok = np.isfinite(c).all(1)
        values.append(np.where(ok, np.nanmedian(c, axis=1), np.nan)); names.append(axis)
    return np.array(values).T, names
