import numpy as np, warnings
warnings.filterwarnings('ignore')
from scipy.signal import welch
from screen import unit_diffs, dstat, load
def prof_corr(a, b):
    m = np.isfinite(a) & np.isfinite(b); return np.corrcoef(a[m], b[m])[0, 1]
for name, fs, units, contrasts in (('nt', 250.0, lambda b: np.array([f"{x.split('--')[0][8:12]}-{g}" for x, g in zip(b['row_id'], b['group'])]), [(1, 0)]),
                                   ('prop', 100.0, lambda b: b['group'].astype(str), [(2, 0), (1, 0), (2, 3)])):
    b = np.load(f'banks/{name}_x2.npz'); x = b['x'].astype(float); sub = b['sub'].astype(float); names = [str(s) for s in b['sub_names']]; y = b['y']
    u = units(b); n, T, M = x.shape
    I, J = np.nonzero(~np.eye(M, dtype=bool))
    xs = (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
    f, P = welch(xs, fs=fs, nperseg=min(256, T), axis=1); slow = P[:, f < 4].sum(1) / P.sum(1)
    cen = (f[None, :, None] * P).sum(1) / P.sum(1)
    S = lambda s: sub[:, names.index(s)]
    rows = {}
    for w in range(n):
        c2 = S('cov-sq_EmpiricalCovariance')[w]
        for lab, v in (('z(cov2, slow_i+slow_j)  [pyspi-free]', -cen[w][I] - cen[w][J]),
                       ('z(cov2, AEG)', S('coint_aeg_tstat_trend-c_autolag-aic_maxlag-10')[w]),
                       ('z(cov2, Johansen tr ardiff10)', S('coint_johansen_trace_stat_order-0_ardiff-10')[w]),
                       ('z(DCE MI, AEG)', None),
                       ('z(dtf max, dcoh max high)', None), ('z(dtf max, -cen_i+cen_j) [pyspi-free]', None),
                       ('z(dcoh max high, slowness sum) [pyspi-free]', None),
                       ('z(cov, corr tau5)', None), ('z(lcss, corr kendall tau5)', None), ('z(lcss, cov)', None), ('z(cov, xcorr max)', None)):
            if lab == 'z(DCE MI, AEG)': val = prof_corr(S('mi_kraskov_NN-4_DCE-AUTO')[w], S('coint_aeg_tstat_trend-c_autolag-aic_maxlag-10')[w])
            elif lab.startswith('z(dtf max, dcoh'): val = prof_corr(S('dtf_multitaper_max_fs-1_fmin-0_fmax-0-5')[w], S('dcoh_multitaper_max_fs-1_fmin-0-25_fmax-0-5')[w])
            elif lab.startswith('z(dtf max, -cen'): val = prof_corr(S('dtf_multitaper_max_fs-1_fmin-0_fmax-0-5')[w], -cen[w][I] + cen[w][J])
            elif lab.startswith('z(dcoh max high, slow'): val = prof_corr(S('dcoh_multitaper_max_fs-1_fmin-0-25_fmax-0-5')[w], -cen[w][I] - cen[w][J])
            elif lab == 'z(cov, corr tau5)': val = prof_corr(S('cov_EmpiricalCovariance')[w], S('corr_pearson_tau-5')[w])
            elif lab == 'z(lcss, corr kendall tau5)': val = prof_corr(S('lcss')[w], S('corr_kendall_tau-5')[w])
            elif lab == 'z(lcss, cov)': val = prof_corr(S('lcss')[w], S('cov_EmpiricalCovariance')[w])
            elif lab == 'z(cov, xcorr max)': val = prof_corr(S('cov_EmpiricalCovariance')[w], S('xcorr_max_sig-True')[w])
            else: val = prof_corr(c2, v)
            rows.setdefault(lab, []).append(val)
    print(f'\n### {name}')
    for lab, v in rows.items():
        v = np.array(v); out = []
        for a, c in contrasts:
            d, _ = unit_diffs(v[:, None], u, y, a, c); d = d[:, 0]
            out.append(f"{a}v{c}: base {np.nanmean(v[y==c]):+.2f} dz {np.nanmean(d):+.3f} d {np.nanmean(d)/np.nanstd(d,ddof=1):+.1f} ({int((d>0).sum())}/{int(np.isfinite(d).sum())})")
        print(f"  {lab:46s} " + ' | '.join(out))
