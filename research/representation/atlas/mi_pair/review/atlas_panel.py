import numpy as np, json, warnings; warnings.filterwarnings('ignore')
from scipy.signal import welch
MT = lambda f, g, b: f"{f}_multitaper_{g}_fs-1_fmin-{ {'broad':'0','low':'0','high':'0-25'}[b]}_fmax-{ {'broad':'0-5','low':'0-25','high':'0-5'}[b]}"
PANEL = [('linear vs monotone', 'cov_EmpiricalCovariance', 'spearmanr'),
         ('monotone vs any dependence', 'spearmanr-sq', 'mi_kraskov_NN-4_DCE-AUTO'),
         ('Gaussian vs any dependence', 'mi_gaussian', 'mi_kraskov_NN-4_DCE-AUTO'),
         ('linear vs distance dependence', 'cov-sq_EmpiricalCovariance', 'dcorr'),
         ('zero-lag vs lagged phase', MT('cohmag', 'mean', 'broad'), MT('icoh', 'mean', 'broad')),
         ('phase vs envelope', MT('plv', 'mean', 'broad'), 'pec'),
         ('envelope: all vs leakage-free', 'pec', 'pec_orth'),
         ('nonlinear: zero-lag vs lagged', 'mi_kraskov_NN-4_DCE-AUTO', 'tlmi_kraskov_NN-4_DCE-AUTO'),
         ('predictive: linear vs nonlinear', 'gc_gaussian_k-1_kt-1_l-1_lt-1', 'te_kraskov_NN-4_DCE-AUTO_k-1_kt-1_l-1_lt-1'),
         ('predictive: amplitude vs ordinal', 'te_kraskov_NN-4_DCE-AUTO_k-1_kt-1_l-1_lt-1', 'te_symbolic_k-3')]
b = np.load('banks/atlas_neurotycho-short.npz'); meta = json.loads(str(b['meta'])); spi = [str(s) for s in b['spi']]; K = len(spi)
def zidx(a, c):
    i, j = sorted((spi.index(a), spi.index(c))); return i * K - i * (i + 1) // 2 + (j - i - 1)
Z = b['z'].astype(float)
unit = np.array([m['unit'] for m in meta]); agent = np.array([m['agent'] for m in meta]); state = np.array([m['state'] for m in meta])
# controls from raw windows
W = np.load('raw/nt/windows.npz'); fs = 100.0
ctrl = np.zeros((len(meta), 4))
for k, m in enumerate(meta):
    x = W[m['id']].astype(float); x = (x - x.mean(0)) / x.std(0)
    r = np.corrcoef(x.T)[np.triu_indices(x.shape[1], 1)]; a = np.abs(r)
    f, P = welch(x, fs=fs, nperseg=100, axis=0)
    ctrl[k] = [a.mean(), a.std(), (P[f < 4].sum(0) / P.sum(0)).mean(), ((f[:, None] * P).sum(0) / P.sum(0)).mean()]
np.save('atlas_ctrl.npy', ctrl)
AG = {'KTMD': 'ket-med', 'KT': 'ketamine', 'MD': 'medetom', 'PF': 'propofol', 'SLP': 'sleep'}
def contrast(v, s_a, ag=None, adjust=None):
    out = []
    for u in np.unique(unit):
        if ag and agent[unit == u][0] != ag: continue
        ma, mc = (unit == u) & (state == s_a), (unit == u) & (state == 'awake_ec')
        if ma.any() and mc.any(): out.append(np.nanmean(v[ma]) - np.nanmean(v[mc]))
    return np.array(out)
def adjusted(v, X):
    """remove within-unit-state pooled slopes on controls X, then return adjusted v (ANCOVA)"""
    m = np.isfinite(v); vv, XX = v.copy(), X.copy()
    for u in np.unique(unit):
        for s in np.unique(state[unit == u]):
            g = m & (unit == u) & (state == s); vv[g] -= vv[g].mean(); XX[g] -= XX[g].mean(0)
    beta = np.linalg.lstsq(XX[m], vv[m], rcond=None)[0]
    return v - X @ beta, beta
res = {}
print('Δz = state − own awake (eyes closed); in units of the eyes-open null SD across archives; (positive units / n)')
hdr = f"{'pair':34s} {'awake z':>7s} " + ' '.join(f'{AG[a]:>15s}' for a in AG) + f" {'EO null':>9s}"
for adj in (None, 'strength', 'strength+spectra'):
    print(f'\n### adjustment: {adj or "none"}'); print(hdr)
    for lab, A, B in PANEL:
        v = Z[:, zidx(A, B)].astype(float)
        if adj == 'strength': v, _ = adjusted(v, ctrl[:, :2])
        if adj == 'strength+spectra': v, _ = adjusted(v, ctrl)
        eo = contrast(v, 'awake_eo'); sd0 = np.nanstd(eo, ddof=1)
        cells = []
        for a in AG:
            d = contrast(v, 'sleep' if a == 'SLP' else 'drug', a)
            cells.append(f"{np.nanmean(d):+.3f}({np.nanmean(d)/sd0:+4.1f},{int((d>0).sum())}/{len(d)})")
            res[(adj, lab, a)] = d
        print(f"{lab[:34]:34s} {np.nanmean(v[state=='awake_ec']) if adj is None else np.nan:+7.2f} " + ' '.join(f'{c:>15s}' for c in cells) + f" {np.nanmean(eo):+.3f}±{sd0:.3f}")
# recovery for unadjusted
print('\nrecovery (eyes closed) − awake, unadjusted, as fraction of drug shift:')
for lab, A, B in PANEL:
    v = Z[:, zidx(A, B)].astype(float)
    print(f"  {lab[:34]:34s} " + ' '.join(f"{AG[a]}: {np.nanmean(contrast(v,'recovery_ec',a))/np.nanmean(contrast(v,'drug',a)):+.2f}" for a in ('KTMD','KT','MD','PF')))
print('\ncontrol shifts (drug − awake): mean|r|, SD|r|, slow<4Hz, centroid')
for a in AG:
    print(f"  {AG[a]:9s}", ' '.join(f"{np.nanmean(contrast(ctrl[:,i],'sleep' if a=='SLP' else 'drug',a)):+.3f}" for i in range(4)), '| EO', ' '.join(f"{np.nanmean(contrast(ctrl[:,i],'awake_eo')):+.3f}" for i in range(4)) if a == 'SLP' else '')
