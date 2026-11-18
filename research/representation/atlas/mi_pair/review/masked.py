import numpy as np, warnings
warnings.filterwarnings('ignore')
from screen import *
from panel import idx
from spread import resid_effect
b = load('nt'); spi = [str(x) for x in b['spi_order']]; y = b['y']
u = np.array([f"{x.split('--')[0][8:12]}-{g}" for x, g in zip(b['row_id'], b['group'])])
sp = np.nanstd(np.abs(b['pearson']), 1)
S = {s: load(s) for s in ('nt_surr_m200', 'nt_surr_spec', 'nt_surr_m1000')}
print('SD|r| per unit: awake [min,max] vs anaes [min,max]; overlap share of windows')
for uu in np.unique(u):
    a0, a1 = sp[(u == uu) & (y == 0)], sp[(u == uu) & (y == 1)]
    lo, hi = max(a0.min(), a1.min()), min(a0.max(), a1.max())
    ov = np.mean(np.r_[(a0 >= lo) & (a0 <= hi), (a1 >= lo) & (a1 <= hi)])
    print(f"  {uu:12s} awake [{a0.min():.3f},{a0.max():.3f}] anaes [{a1.min():.3f},{a1.max():.3f}] overlap {ov:.2f}")
pairs = [('mi_gaussian', 'mi_kraskov_NN-4_DCE-AUTO'), ('tlmi_gaussian', 'tlmi_kraskov_NN-4_DCE-AUTO'),
         ('mi_gaussian', 'ids'), ('cov-sq_EmpiricalCovariance', 'dcorr'), ('cov-sq_EmpiricalCovariance', 'spearmanr-sq'),
         ('mi_gaussian', 'mi_kernel_W-0.25'), ('cov_EmpiricalCovariance', 'pdist_euclidean')]
def matched(v, u, y, x, nb=4):
    """within unit: bins of x over the overlap range; mean z difference within bins, weighted by min count"""
    out = []
    for uu in np.unique(u):
        a0, a1 = (u == uu) & (y == 0) & np.isfinite(v), (u == uu) & (y == 1) & np.isfinite(v)
        lo, hi = max(x[a0].min(), x[a1].min()), min(x[a0].max(), x[a1].max())
        if hi <= lo: out.append(np.nan); continue
        edges = np.linspace(lo, hi, nb + 1); num = den = 0
        for e0, e1 in zip(edges[:-1], edges[1:]):
            s0, s1 = a0 & (x >= e0) & (x <= e1), a1 & (x >= e0) & (x <= e1)
            w = min(s0.sum(), s1.sum())
            if w: num += w * (v[s1].mean() - v[s0].mean()); den += w
        out.append(num / den if den else np.nan)
    out = np.array(out); return np.nanmean(out), int(np.isfinite(out).sum()), int((out < 0).sum())
print(f"\n{'pair':62s} raw | ANCOVA(SD|r|) real | matched real (units, neg) | ANCOVA surrogates m200/spec/m1000 | adj real-surr d (m200/spec)")
for A, B in pairs:
    k = idx(spi, A, B); v = b['z'][:, k].astype(float)
    raw = np.nanmean(unit_diffs(v[:, None], u, y, 1, 0)[0])
    e = resid_effect(v, sp, u, y, 1, 0); m = matched(v, u, y, sp)
    ss, rr = [], []
    for s, sb in S.items():
        sv = sb['z'][:, k].astype(float); ssp = np.nanstd(np.abs(sb['pearson']), 1)
        ss.append(resid_effect(sv, ssp, u, y, 1, 0)[0])
        if s != 'nt_surr_m1000': rr.append(resid_effect(v - sv, sp, u, y, 1, 0)[1])
    print(f"{A[:30]:30s} {B[:30]:30s} {raw:+.3f} | {e[0]:+.3f} d{e[1]:+.1f} | {m[0]:+.3f} ({m[1]},{m[2]}) | {ss[0]:+.3f} {ss[1]:+.3f} {ss[2]:+.3f} | {rr[0]:+.1f} {rr[1]:+.1f}")
