import numpy as np, warnings, sys
warnings.filterwarnings('ignore')
from screen import *
MT = lambda f, g, b: f"{f}_multitaper_{g}_fs-1_fmin-{ {'broad':'0','low':'0','high':'0-25'}[b]}_fmax-{ {'broad':'0-5','low':'0-25','high':'0-5'}[b]}"
PANEL = [
 # (label, A, B, one-assumption reading)
 ('form: Gaussian vs DCE-Kraskov MI', 'mi_gaussian', 'mi_kraskov_NN-4_DCE-AUTO'),
 ('form: lagged Gaussian vs DCE-Kraskov TLMI', 'tlmi_gaussian', 'tlmi_kraskov_NN-4_DCE-AUTO'),
 ('form: GC vs DCE-Kraskov TE (k1)', 'gc_gaussian_k-1_kt-1_l-1_lt-1', 'te_kraskov_NN-4_DCE-AUTO_k-1_kt-1_l-1_lt-1'),
 ('form: squared cov vs dcorr', 'cov-sq_EmpiricalCovariance', 'dcorr'),
 ('form: Gaussian MI vs IDS', 'mi_gaussian', 'ids'),
 ('form (uncorrected, control): Gaussian vs Kraskov MI', 'mi_gaussian', 'mi_kraskov_NN-4'),
 ('timing-nonlin: DCE MI vs DCE TLMI', 'mi_kraskov_NN-4_DCE-AUTO', 'tlmi_kraskov_NN-4_DCE-AUTO'),
 ('timing-lin: MI vs TLMI Gaussian', 'mi_gaussian', 'tlmi_gaussian'),
 ('phase vs envelope (leakage-free): icoh vs pec_orth', MT('icoh','mean','broad'), 'pec_orth'),
 ('phase vs envelope: PLV vs PEC', MT('plv','mean','broad'), 'pec'),
 ('envelope leakage: pec vs pec_orth', 'pec', 'pec_orth'),
 ('zero vs lagged phase: |coh| vs icoh', MT('cohmag','mean','broad'), MT('icoh','mean','broad')),
 ('mediation spectral: dcoh vs pdcoh', MT('dcoh','mean','broad'), MT('pdcoh','mean','broad')),
 ('mediation: cov-sq vs prec-sq', 'cov-sq_EmpiricalCovariance', 'prec-sq_EmpiricalCovariance'),
 ('tail-robust direct: prec-sq Emp vs MCD', 'prec-sq_EmpiricalCovariance', 'prec-sq_MinCovDet'),
 ('tail: euclid vs chebyshev', 'pdist_euclidean', 'pdist_chebyshev'),
 ('warp: rigid shift vs DTW', 'xpdist_euclidean_tau-1_min_rmse', 'dtw'),
 ('trend: cov-sq vs Johansen', 'cov-sq_EmpiricalCovariance', 'coint_johansen_max_eig_stat_order-0_ardiff-10'),
 ('trend: DCE MI vs AEG t', 'mi_kraskov_NN-4_DCE-AUTO', 'coint_aeg_tstat_trend-c_autolag-aic_maxlag-10'),
 ('trend: cov-sq vs AEG t', 'cov-sq_EmpiricalCovariance', 'coint_aeg_tstat_trend-c_autolag-aic_maxlag-10'),
 ('history: SGC order 1 vs 20', 'sgc_parametric_mean_fs-1_fmin-1e-05_fmax-0-5_order-1', 'sgc_parametric_mean_fs-1_fmin-1e-05_fmax-0-5_order-20'),
 ('predictive: MI vs GC (Gaussian)', 'mi_gaussian', 'gc_gaussian_k-1_kt-1_l-1_lt-1'),
 ('predictive nonlin: SI Gaussian vs kernel', 'si_gaussian_k-1', 'si_kernel_W-0.5_k-1'),
 ('band: |coh| low vs high', MT('cohmag','mean','low'), MT('cohmag','mean','high')),
 ('peak: |coh| mean vs max', MT('cohmag','mean','broad'), MT('cohmag','max','broad')),
 ('identity control: cov vs euclid', 'cov_EmpiricalCovariance', 'pdist_euclidean'),
]

def idx(spi, a, b):
    i, j = spi.index(a), spi.index(b)
    i, j = min(i, j), max(i, j); K = len(spi)
    return i * K - i * (i + 1) // 2 + (j - i - 1)

def cohort_rows(name, surrs, unitf, a, c, extra=None):
    b = load(name); spi = [str(x) for x in b['spi_order']]
    S = {s: load(s) for s in surrs}
    u = unitf(b); y = b['y']
    strength = np.nanmean(np.abs(b['pearson']), 1)
    out = []
    for lab, A, B in PANEL:
        k = idx(spi, A, B)
        v = b['z'][:, k].astype(float)
        diff, un = unit_diffs(v[:, None], u, y, a, c); diff = diff[:, 0]
        row = dict(label=lab, base=np.nanmean(v[y == c]), dz=np.nanmean(diff), d=np.nanmean(diff)/np.nanstd(diff, ddof=1),
                   pos=int((diff > 0).sum()), n=int(np.isfinite(diff).sum()))
        for s, sb in S.items():
            sv = sb['z'][:, k].astype(float)
            sd, _ = unit_diffs(sv[:, None], u, y, a, c); rd, _ = unit_diffs((v - sv)[:, None], u, y, a, c)
            row[s] = (np.nanmean(sd), np.nanmean(rd[:, 0]) / np.nanstd(rd[:, 0], ddof=1), int((rd[:, 0] > 0).sum()))
        # strength bleed: within unit-state correlation of z with mean |r| across windows
        rs = []
        for uu in np.unique(u):
            for st in (a, c):
                m = (u == uu) & (y == st) & np.isfinite(v)
                if m.sum() > 5: rs.append(np.corrcoef(v[m], strength[m])[0, 1])
        row['bleed'] = np.nanmean(rs)
        la = unit_diffs(b['spi_mean'][:, [spi.index(A), spi.index(B)]].astype(float), u, y, a, c)[0]
        row['lvl'] = dstat(la)
        out.append(row)
    return out

def show(title, rows, surrs):
    print(f'\n### {title}')
    print(f"{'pair':50s} base   dz     d   pos/n | " + ' | '.join(f'{s[-6:]}: dz_s  d_res pos' for s in surrs) + ' | bleed lvlA lvlB')
    for r in rows:
        ss = ' | '.join(f"{r[s][0]:+.3f} {r[s][1]:+5.1f} {r[s][2]:2d}" for s in surrs)
        print(f"{r['label'][:50]:50s} {r['base']:+.2f} {r['dz']:+.3f} {r['d']:+5.1f} {r['pos']:2d}/{r['n']:2d} | {ss} | {r['bleed']:+.2f} {r['lvl'][0]:+.1f} {r['lvl'][1]:+.1f}")

if __name__ == '__main__':
    which = sys.argv[1:]
    ntu = lambda b: np.array([f"{x.split('--')[0][8:12]}-{g}" for x, g in zip(b['row_id'], b['group'])])
    pu = lambda b: b['group'].astype(str)
    if 'nt' in which:
        s = ['nt_surr_m200', 'nt_surr_spec']; show('NeuroTycho anaesthesia vs awake (6 units)', cohort_rows('nt', s, ntu, 1, 0), s)
    if 'prop' in which:
        s = ['prop_surr_m', 'prop_surr_spec']
        show('Propofol moderate vs baseline (20)', cohort_rows('prop', s, pu, 2, 0), s)
        show('Propofol moderate vs recovery (20)', cohort_rows('prop', s, pu, 2, 3), s)
        show('Propofol mild vs baseline (20)', cohort_rows('prop', s, pu, 1, 0), s)
