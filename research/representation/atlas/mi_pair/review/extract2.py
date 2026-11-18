import sys, io, numpy as np
p, full = sys.argv[1], sys.argv[2] == '1'
b = np.load(p); spi = [str(s) for s in b['spi_order']]; K = len(spi); M = int(b['M'][0]); n = len(b['y'])
A = b['aligned'].reshape(n, K, M * (M - 1))
mask = ~np.eye(M, dtype=bool); I, J = np.nonzero(mask)
# design: row effect + column effect (directed additive node model)
X = np.zeros((M * (M - 1), 2 * M)); X[np.arange(len(I)), I] = 1; X[np.arange(len(J)), M + J] = 1
P = X @ np.linalg.pinv(X)
r2 = np.full((n, K), np.nan)
for w in range(n):
    V = A[w].T.astype(float)  # edges x K
    ok = np.isfinite(V).all(0) & (np.nanstd(V, 0) > 0)
    Vc = V[:, ok] - V[:, ok].mean(0); fit = P @ Vc
    r2[w, ok] = 1 - ((Vc - fit) ** 2).sum(0) / (Vc ** 2).sum(0)
out = dict(r2=r2.astype(np.float32), spi_order=b['spi_order'], y=b['y'], group=b['group'], row_id=b['row_id'])
if full:
    keep = ['cov_EmpiricalCovariance','cov-sq_EmpiricalCovariance','mi_gaussian','mi_kraskov_NN-4_DCE-AUTO','spearmanr',
      'coint_aeg_tstat_trend-c_autolag-aic_maxlag-10','coint_johansen_trace_stat_order-0_ardiff-10','coint_johansen_max_eig_stat_order-0_ardiff-10',
      'coint_johansen_trace_stat_order-0_ardiff-1','lcss','lcss_constraint-sakoe-chiba','corr_kendall_tau-5','corr_pearson_tau-5','corr_pearson_tau-1',
      'cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-5','cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-25','cohmag_multitaper_mean_fs-1_fmin-0-25_fmax-0-5',
      'dtf_multitaper_max_fs-1_fmin-0_fmax-0-5','dtf_multitaper_mean_fs-1_fmin-0_fmax-0-5','dcoh_multitaper_max_fs-1_fmin-0-25_fmax-0-5',
      'gpdcoh_multitaper_max_fs-1_fmin-0-25_fmax-0-5','dcoh_multitaper_max_fs-1_fmin-0_fmax-0-5','dtf_multitaper_max_fs-1_fmin-0-25_fmax-0-5',
      'pdist_euclidean','dtw','xcorr_max_sig-True']
    ix = [spi.index(s) for s in keep]
    out.update(sub=A[:, ix].astype(np.float32), sub_names=np.array(keep), x=b['x'].astype(np.float32))
buf = io.BytesIO(); np.savez(buf, **out); sys.stdout.buffer.write(buf.getvalue())
