import numpy as np, warnings
warnings.filterwarnings('ignore')
from screen import *
from panel import idx
from helpers import pmeans, hedges
# 1. NT per-unit raw and matched for MI pair
b = load('nt'); spi = [str(x) for x in b['spi_order']]; y = b['y']
u = np.array([f"{x.split('--')[0][8:12]}-{g}" for x, g in zip(b['row_id'], b['group'])])
sp = np.nanstd(np.abs(b['pearson']), 1); sm = load('nt_surr_m200')
for A, B in (('mi_gaussian', 'mi_kraskov_NN-4_DCE-AUTO'), ('mi_gaussian', 'ids'), ('cov-sq_EmpiricalCovariance', 'dcorr')):
    k = idx(spi, A, B); v = b['z'][:, k].astype(float); sv = sm['z'][:, k].astype(float)
    print(f'\nNT {A} | {B}: per unit raw dz, surr dz, matched dz (overlap bins), SD|r| change')
    for uu in np.unique(u):
        a0, a1 = (u == uu) & (y == 0), (u == uu) & (y == 1)
        lo, hi = max(sp[a0].min(), sp[a1].min()), min(sp[a0].max(), sp[a1].max())
        num = den = 0; nums = 0
        if hi > lo:
            for e0, e1 in zip(np.linspace(lo, hi, 5)[:-1], np.linspace(lo, hi, 5)[1:]):
                s0, s1 = a0 & (sp >= e0) & (sp <= e1), a1 & (sp >= e0) & (sp <= e1); w = min(s0.sum(), s1.sum())
                if w: num += w * (v[s1].mean() - v[s0].mean()); nums += w * (sv[s1].mean() - sv[s0].mean()); den += w
        print(f"  {uu:12s} raw {v[a1].mean()-v[a0].mean():+.3f} surr {sv[a1].mean()-sv[a0].mean():+.3f} | matched real {num/den if den else np.nan:+.3f} surr {nums/den if den else np.nan:+.3f} (n={den}) | dSD {sp[a1].mean()-sp[a0].mean():+.3f}")
# 2. AD permutation null, Fisher-z of 1-z (log(1-z)), raw and residual
b = load('adftd'); spi = [str(x) for x in b['spi_order']]; g = b['group'].astype(str); y = b['y']
ss = load('adftd_surr_spec')
pairs = [('mi_gaussian', 'mi_kraskov_NN-4_DCE-AUTO'), ('tlmi_gaussian', 'tlmi_kraskov_NN-4_DCE-AUTO'), ('mi_gaussian', 'ids'), ('cov-sq_EmpiricalCovariance', 'dcorr'), ('cov-sq_EmpiricalCovariance', 'spearmanr-sq')]
K = [idx(spi, a, c) for a, c in pairs]
def block(z):  # log disagreement, mean over the five pairs
    return np.nanmean(np.log(np.clip(1 - z[:, K], 1e-6, None)), 1)
rng = np.random.default_rng(0)
for lab, a, c in (('AD vs HC', 1, 0), ('AD vs FTD', 1, 2), ('FTD vs HC', 2, 0)):
    for nm, w in (('real', block(b['z'].astype(float))), ('real-surr', block(b['z'].astype(float)) - block(ss['z'].astype(float)))):
        Pa, _ = pmeans(w, g, y, a); Pc, _ = pmeans(w, g, y, c); obs = Pa.mean() - Pc.mean()
        allp = np.r_[Pa, Pc]; n = len(Pa)
        null = np.array([(lambda p: p[:n].mean() - p[n:].mean())(rng.permutation(allp)) for _ in range(10000)])
        from scipy.stats import mannwhitneyu
        print(f"{lab:10s} {nm:9s} log(1-z) block diff {obs:+.3f} d {hedges(Pa,Pc):+.2f} perm p(two-sided) {np.mean(np.abs(null) >= abs(obs)):.4f} MWU p {mannwhitneyu(Pa, Pc).pvalue:.4f}")
# 3. barycentre optimiser agreement
for n in ('adftd', 'lemon', 'prop', 'nt'):
    b = load(n); spi = [str(x) for x in b['spi_order']]
    k = idx(spi, 'bary_dtw_mean', 'bary_sgddtw_mean'); k2 = idx(spi, 'bary_dtw_max', 'bary_sgddtw_max')
    print(f"{n:6s} z(bary_dtw_mean, bary_sgddtw_mean) median {np.nanmedian(b['z'][:,k]):+.2f} | max pair {np.nanmedian(b['z'][:,k2]):+.2f}")
