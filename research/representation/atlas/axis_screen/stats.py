"""Pyspi-free edge profiles for the axis screen (prereg.md), one window (T x M) at a time.

Every statistic is a pyspi SPI of the atlas set (benchmarked_p90_pcorr_symbolic at 0c798b4) re-implemented so that
nulls with several draws and all 126 channels are affordable at T = 1000; `validate` checks each against the stored
pyspi MPIs of the same raw 10-s windows. Channels are z-scored first, as pyspi's normalise=True does.

  python -m research.representation.atlas.axis_screen.stats validate <val.npz> [<out.json>]
"""
import sys, json, warnings
import numpy as np
from scipy.signal import hilbert
from scipy.signal.windows import dpss
from scipy.stats import kendalltau, rankdata

PYSPI = {  # key -> pyspi identifier
    'r': 'cov_EmpiricalCovariance', 'r2': 'cov-sq_EmpiricalCovariance',
    'rho': 'spearmanr', 'rho2': 'spearmanr-sq', 'tau': 'kendalltau', 'tau2': 'kendalltau-sq',
    'mcd': 'cov_MinCovDet', 'mcd2': 'cov-sq_MinCovDet',
    'pcorr': 'pcorr_EmpiricalCovariance', 'pcorr_mcd': 'pcorr_MinCovDet',
    'l1': 'pdist_cityblock', 'l2': 'pdist_euclidean', 'linf': 'pdist_chebyshev',
    'dcorr': 'dcorr', 'mi_g': 'mi_gaussian',
    'pec': 'pec', 'pec_log': 'pec_log',
    'coh_lo': 'cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-25', 'coh_hi': 'cohmag_multitaper_mean_fs-1_fmin-0-25_fmax-0-5',
    'coh_bb': 'cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-5', 'icoh_lo': 'icoh_multitaper_mean_fs-1_fmin-0_fmax-0-25',
    'icoh_bb': 'icoh_multitaper_mean_fs-1_fmin-0_fmax-0-5',
    'plv': 'plv_multitaper_mean_fs-1_fmin-0_fmax-0-5', 'plv_lo': 'plv_multitaper_mean_fs-1_fmin-0_fmax-0-25',
    'plv_hi': 'plv_multitaper_mean_fs-1_fmin-0-25_fmax-0-5',
    'ppc': 'ppc_multitaper_mean_fs-1_fmin-0_fmax-0-5', 'ppc_lo': 'ppc_multitaper_mean_fs-1_fmin-0_fmax-0-25',
    'ppc_hi': 'ppc_multitaper_mean_fs-1_fmin-0-25_fmax-0-5',
    'dswpli': 'dswpli_multitaper_mean_fs-1_fmin-0_fmax-0-5', 'dswpli_lo': 'dswpli_multitaper_mean_fs-1_fmin-0_fmax-0-25',
    'dswpli_hi': 'dswpli_multitaper_mean_fs-1_fmin-0-25_fmax-0-5',
    'xc2max': 'xcorr-sq_max_sig-True', 'xcmax': 'xcorr_max_sig-True',
}
KEYS = tuple(PYSPI)
BANDS = {'lo': lambda f: (f > 0) & (f <= .25), 'hi': lambda f: (f > .25) & (f < .5), 'bb': lambda f: (f > 0) & (f < .5)}


def zscore(x):
    return (x - x.mean(0)) / x.std(0)


def _cov_family(x):
    from sklearn.covariance import MinCovDet
    T, M = x.shape
    C = x.T @ x / T
    P = np.linalg.inv(C)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mcd = MinCovDet(random_state=42).fit(x)
    pc = lambda Q: -Q / np.sqrt(np.outer(np.diag(Q), np.diag(Q)))
    return C, pc(P), mcd.covariance_, pc(mcd.precision_)


def _dcorr_all(x, chunk=16):
    """Unbiased (U-centred) distance correlation for every pair, as hyppo Dcorr(bias=False).statistic: the ratio
    sum(A~ B~) / sqrt(sum(A~^2) sum(B~^2)) of U-centred distance matrices with zero diagonals, via one Gram matrix."""
    T, M = x.shape
    U = np.empty((M, T * T), np.float32)
    for m in range(M):
        d = np.abs(x[:, m][:, None] - x[:, m][None, :])
        c = d - d.sum(0)[None, :] / (T - 2) - d.sum(1)[:, None] / (T - 2) + d.sum() / ((T - 1) * (T - 2))
        np.fill_diagonal(c, 0)
        U[m] = c.ravel()
    G = np.zeros((M, M))
    for i in range(0, M, chunk):
        G[i:i + chunk] = (U[i:i + chunk] @ U.T).astype(np.float64)
    del U
    v = np.diag(G).copy()
    return G / np.sqrt(np.outer(v, v))


def _multitaper(x, nw=3, k=5):
    """Per-taper Fourier coefficients (k x F x M) and frequencies (fs = 1)."""
    T = len(x)
    return np.fft.rfft(dpss(T, nw, k)[:, :, None] * x[None], axis=1), np.fft.rfftfreq(T)


def profiles(x, keys=KEYS, mcd=True, dcorr=True):
    """Upper-triangle edge profile of each statistic for one window (T x M), channels z-scored first."""
    x = zscore(np.asarray(x, np.float64))
    T, M = x.shape
    a, b = np.triu_indices(M, 1)
    out = {}
    C, PC, MC, MPC = _cov_family(x) if mcd else (x.T @ x / T, None, None, None)
    if not mcd:
        PC = (lambda P: -P / np.sqrt(np.outer(np.diag(P), np.diag(P))))(np.linalg.inv(C))
    r = C[a, b]
    out.update(r=r, r2=r ** 2, mi_g=-.5 * np.log(1 - r ** 2), pcorr=PC[a, b])
    if mcd:
        out.update(mcd=MC[a, b], mcd2=MC[a, b] ** 2, pcorr_mcd=MPC[a, b])
    R = zscore(np.apply_along_axis(rankdata, 0, x))
    rho = (R.T @ R / T)[a, b]
    out.update(rho=rho, rho2=rho ** 2)
    if 'tau' in keys:
        tau = np.array([kendalltau(x[:, i], x[:, j])[0] for i, j in zip(a, b)])
        out.update(tau=tau, tau2=tau ** 2)
    D = x[:, a] - x[:, b]
    out.update(l1=np.abs(D).sum(0), l2=np.sqrt((D ** 2).sum(0)), linf=np.abs(D).max(0))
    del D
    if dcorr:
        out['dcorr'] = _dcorr_all(x)[a, b]
    h = hilbert(x, axis=0)
    env = np.abs(h)
    out['pec'] = np.corrcoef(env.T)[a, b]
    out['pec_log'] = np.corrcoef(np.log(env ** 2).T)[a, b]
    Xk, f = _multitaper(x)
    nk = Xk.shape[0]
    for band, sel in BANDS.items():
        s = sel(f)
        S = np.einsum('kfi,kfj->fij', Xk[:, s], Xk[:, s].conj()) / nk
        d = np.real(np.einsum('fii->fi', S))
        Cc = S[:, a, b] / np.sqrt(d[:, a] * d[:, b])
        out[f'coh_{band}'] = (np.abs(Cc) ** 2).mean(0)
        out[f'icoh_{band}'] = np.abs(Cc.imag).mean(0)
        del S, Cc
        P = Xk[:, s][:, :, a] * Xk[:, s][:, :, b].conj()
        U = P / np.abs(P)
        out[f'plv_{band}'] = np.abs(U.mean(0)).mean(0)
        out[f'ppc_{band}'] = ((np.abs(U.sum(0)) ** 2 - nk) / (nk * (nk - 1))).mean(0)
        im = P.imag
        sq = (im ** 2).sum(0)
        with np.errstate(all='ignore'):
            out[f'dswpli_{band}'] = np.nanmean((im.sum(0) ** 2 - sq) / (np.abs(im).sum(0) ** 2 - sq), 0)
        del P, U, im
    for k in ('coh', 'icoh', 'plv', 'ppc', 'dswpli'):
        out[k] = out[f'{k}_bb']
    F = np.fft.rfft(x, 2 * T, axis=0)
    L = T // 4
    xc = np.empty((2 * L + 1, len(a)))
    for i0 in range(0, len(a), 2000):
        sl = slice(i0, i0 + 2000)
        rr = np.fft.irfft(F[:, a[sl]] * F[:, b[sl]].conj(), axis=0) / T
        xc[:, sl] = np.concatenate([rr[-L:], rr[:L + 1]])
    thr = np.abs(xc) > 1.96 / np.sqrt(T)
    out['xcmax'] = np.where(thr.any(0), np.where(thr, xc, -np.inf).max(0), 0.)
    out['xc2max'] = np.where(thr.any(0), np.where(thr, xc ** 2, -np.inf).max(0), 0.)
    return {k: out[k] for k in keys if k in out}


def validate(path, output=None):
    """Per statistic and window: Pearson and Spearman of my profile against pyspi's stored MPI upper triangle, and
    max |difference|; then for every pair of statistics, my z against pyspi's z on the same window."""
    import time
    V = np.load(path, allow_pickle=False)
    spis = [str(s) for s in V['spis']]
    res = {}
    for key in [k for k in V.files if k.startswith('x:')]:
        wid = key[2:]
        x = V[key].astype(np.float64)
        mp = V['mpi:' + wid].astype(np.float64)
        M = x.shape[1]
        a, b = np.triu_indices(M, 1)
        t0 = time.time()
        P = profiles(x)
        el = time.time() - t0
        rows, mine, theirs = {}, {}, {}
        for k, name in PYSPI.items():
            if k not in P or name not in spis:
                rows[k] = dict(missing=name not in spis)
                continue
            q = mp[spis.index(name)][a, b]
            p = P[k]
            ok = np.isfinite(q) & np.isfinite(p)
            from scipy.stats import spearmanr
            rows[k] = dict(pearson=float(np.corrcoef(p[ok], q[ok])[0, 1]), spearman=float(spearmanr(p[ok], q[ok])[0]),
                           maxabs=float(np.max(np.abs(p[ok] - q[ok]))), ratio=float(np.median(p[ok] / q[ok])))
            mine[k], theirs[k] = p, q
        ks = list(mine)
        zd = []
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                zm = np.corrcoef(mine[ks[i]], mine[ks[j]])[0, 1]
                zt = np.corrcoef(theirs[ks[i]], theirs[ks[j]])[0, 1]
                zd.append((ks[i], ks[j], float(zm), float(zt)))
        dz = np.array([abs(m - t) for *_, m, t in zd])
        res[wid] = dict(seconds=el, stats=rows, z_absdiff_median=float(np.nanmedian(dz)), z_absdiff_max=float(np.nanmax(dz)),
                        z_worst=sorted(zd, key=lambda r: -abs(r[2] - r[3]))[:15])
        print(wid, f'{el:.0f}s', 'z |diff| median', round(res[wid]['z_absdiff_median'], 5), 'max', round(res[wid]['z_absdiff_max'], 4))
        for k, r in rows.items():
            print(f'  {k:10s}', {kk: (round(v, 5) if isinstance(v, float) else v) for kk, v in r.items()})
        for w in res[wid]['z_worst'][:8]:
            print('   worst z', w)
    if output:
        open(output, 'w').write(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    if sys.argv[1] == 'validate':
        validate(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
