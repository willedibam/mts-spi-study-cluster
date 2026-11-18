"""Critic's screen: which non-correlation z coordinates carry replicated state effects?"""
import numpy as np, re, json, sys
from pathlib import Path

D = Path(__file__).parent / 'banks'


def family(s):
    if re.match(r'(cov|cov-sq)_', s) or s.startswith(('spearmanr', 'kendalltau', 'corr_', 'xcorr')):
        return 'CORR'
    if s in ('mi_gaussian', 'je_gaussian', 'ce_gaussian', 'pdist_euclidean', 'pdist_cosine') or s.startswith('lmfit'):
        return 'CORR'  # fixed functions of zero-lag r on standardised inputs (or near it)
    if s.startswith(('tlmi_gaussian', 'xme_gaussian', 'cce_gaussian', 'si_gaussian', 'di_gaussian', 'gc_gaussian')):
        return 'LINPRED'  # Gaussian lagged/predictive: functions of lagged correlations
    if s.startswith(('prec',)):
        return 'PREC'
    if s.startswith(('mi_kraskov', 'mi_kernel', 'tlmi_kraskov', 'tlmi_kernel', 'je_k', 'ce_k', 'cce_k', 'xme_k', 'si_k', 'te_')):
        return 'NPINFO'
    if s.startswith(('cohmag', 'icoh', 'plv', 'pli', 'wpli', 'dspli', 'dswpli', 'ppc', 'phase', 'psi', 'gd_')):
        return 'SPEC'
    if s.startswith(('dtf', 'dcoh', 'pdcoh', 'gpdcoh', 'ddtf', 'sgc')):
        return 'SPECDIR'
    if s.startswith(('pdist', 'dtw', 'lcss', 'xpdist', 'bary', 'gwtau')):
        return 'SHAPE'
    if s.startswith('coint'):
        return 'COINT'
    if s.startswith('pec'):
        return 'ENV'
    if s.startswith('phi'):
        return 'PHI'
    return {'dcorr': 'NPDEP', 'dcorr_biased': 'NPDEP', 'ids': 'NPDEP', 'cds': 'CAUSAL', 'reci': 'CAUSAL'}.get(s, 'OTHER')


def load(name):
    b = np.load(D / f'{name}.npz')
    return {k: b[k] for k in b.files}


def pair_index(K):
    return np.triu_indices(K, 1)


def unit_diffs(v, unit, y, a, c):
    """per-unit mean(state a) - mean(state c); v is windows x features"""
    out, names = [], []
    for u in np.unique(unit):
        sa, sc = (unit == u) & (y == a), (unit == u) & (y == c)
        if sa.any() and sc.any():
            out.append(np.nanmean(v[sa], 0) - np.nanmean(v[sc], 0)); names.append(u)
    return np.array(out), np.array(names)


def dstat(diff):
    with np.errstate(all='ignore'):
        return np.nanmean(diff, 0) / np.nanstd(diff, 0, ddof=1)


if __name__ == '__main__':
    pass

