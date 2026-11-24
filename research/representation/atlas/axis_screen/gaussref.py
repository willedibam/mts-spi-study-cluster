"""Gaussian-referenced readouts (criteria in result.md, committed before the run; calibration in gaussref_calib.py).

Primary readout env_E (the calibration, gaussref_calib.py, found the others non-specific; result.md). Per window and version (real and the screen's 3 IAAFT draws, same seeds) of the prepared window, all channels, prep as
mechanism.py ('z', 'z-m', 'ns'; '-m' = marginal-final IAAFT). Each statistic is computed on the z-scored prepared window:
  can_E     mean over edges of C - f_G(r): C = mean_t |x - y| / (|x| + |y|) (pyspi pdist_canberra / T), f_G its exact
            Gaussian value (an angular integral, below). A gain shared by both channels at a sample leaves C unchanged.
  can_z     z(f_G(r), C) across edges
  env_E     mean over edges of pec - g_amp(|gamma|^2): pec = corr(|a|, |b|) of Hilbert amplitudes (pyspi pec), gamma the
            zero-lag complex correlation of the analytic signals, g_amp the exact Gaussian relation
  env_z     z(g_amp(|gamma|^2), pec)
  envlog_z  z(g_log(|gamma|^2), pec_log), pec_log = corr(log|a|^2, log|b|^2), g_log = 6 Li2(.) / pi^2
  cob_z     z(r, bary-sq max): r against max_t ((x + y) / 2)^2 (pyspi bary-sq_euclidean_max)
  asym_z    z(bary max, bary-sq max): signed against two-sided barycentre maximum (pyspi bary_euclidean_max)
  envdir_z  z(E, E^T) over edges i < j, E_ij = corr(|a_i(t)|, |a_j(t + LAG)|): reciprocity of lagged amplitude coupling
  kurt      mean excess kurtosis over channels (diagnostic)
Scored by mechanism.score.

  python -m research.representation.atlas.axis_screen.gaussref compute <bank> <out dir> <workers> <z|z-m|ns>
  python -m research.representation.atlas.axis_screen.gaussref collect <out dir> <out.npz>
"""
import sys, glob, time
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from functools import lru_cache
from scipy.signal import hilbert
from scipy.special import hyp2f1, spence

METRICS = ('env_E', 'env_z', 'envlog_z', 'can_E', 'can_z', 'cob_z', 'asym_z', 'envdir_z', 'kurt')  # env_E primary
LAG = 10  # samples (100 ms at 100 Hz)


@lru_cache(1)
def _fg_table(n_r=2001, n_t=20000):
    """f_G(r) = E|x - y| / (|x| + |y|) for a standard bivariate normal: the ratio depends only on the angle, whose density
    is sqrt(1 - r^2) / (2 pi (1 - r sin 2t))."""
    t = (np.arange(n_t) + .5) * 2 * np.pi / n_t
    h = np.abs(np.cos(t) - np.sin(t)) / (np.abs(np.cos(t)) + np.abs(np.sin(t)))
    r = np.linspace(-.999, .999, n_r)
    dens = np.sqrt(1 - r[:, None] ** 2) / (2 * np.pi * (1 - r[:, None] * np.sin(2 * t)[None]))
    return r, (dens * h[None]).sum(1) * 2 * np.pi / n_t


def f_g(r):
    R, F = _fg_table()
    return np.interp(r, R, F)


def g_amp(G):
    return np.pi / 4 * (hyp2f1(-.5, -.5, 1, G) - 1) / (1 - np.pi / 4)


def g_log(G):
    return 6 * spence(1 - G) / np.pi ** 2


def _z(u, v):
    ok = np.isfinite(u) & np.isfinite(v)
    return float(np.corrcoef(u[ok], v[ok])[0, 1])


def metrics(v, chunk=1500):
    from scipy.stats import kurtosis
    from research.representation.atlas.axis_screen.stats import zscore
    x = zscore(np.asarray(v, np.float64))
    T, M = x.shape
    a, b = np.triu_indices(M, 1)
    r = (x.T @ x / T)[a, b]
    can, bmax, bsq = (np.empty(len(a)) for _ in range(3))
    for i in range(0, len(a), chunk):
        s = slice(i, i + chunk)
        xa, xb = x[:, a[s]], x[:, b[s]]
        with np.errstate(invalid='ignore', divide='ignore'):
            can[s] = np.nanmean(np.abs(xa - xb) / (np.abs(xa) + np.abs(xb)), 0)
        bc = (xa + xb) / 2
        bmax[s], bsq[s] = bc.max(0), (bc ** 2).max(0)
    h = hilbert(x, axis=0)
    g = h.T @ h.conj() / T
    d = np.sqrt(np.real(np.diag(g)))
    G = np.clip((np.abs(g) / np.outer(d, d))[a, b] ** 2, 0, 1 - 1e-12)
    env = np.abs(h)
    pec = np.corrcoef(env.T)[a, b]
    pec_log = np.corrcoef(np.log(env ** 2).T)[a, b]
    e0 = env[:-LAG] - env[:-LAG].mean(0); e1 = env[LAG:] - env[LAG:].mean(0)
    E = (e0.T @ e1) / np.outer(np.linalg.norm(e0, axis=0), np.linalg.norm(e1, axis=0))
    fg, ga = f_g(r), g_amp(G)
    return np.array([np.mean(pec - ga), _z(ga, pec), _z(g_log(G), pec_log), np.mean(can - fg), _z(fg, can),
                     _z(r, bsq), _z(bmax, bsq), _z(E[a, b], E[b, a]), float(kurtosis(x).mean())])


def window(args):
    bank, out, wid, prep = args
    from research.representation.atlas.extract import seed
    from research.representation.atlas.axis_screen.run import _ns
    from research.representation.atlas.axis_screen.stats import zscore
    from mtsspi.surrogates import multivariate_iaaft
    target = Path(out) / f'{wid}.npz'
    if target.exists():
        return wid, 0.
    t0 = time.time()
    with np.load(Path(bank) / 'windows.npz') as W:
        v = np.asarray(W[wid], np.float64)
    v = _ns(v) if prep.split('-')[0] == 'ns' else zscore(v)
    final = 'marginal' if prep.endswith('-m') else 'spectrum'
    X = [v] + [multivariate_iaaft(v, np.random.default_rng([seed(wid), 10, k]), final=final)[0] for k in range(3)]
    np.savez(target, m=np.stack([metrics(x) for x in X]))
    return wid, time.time() - t0


def compute(bank, out, workers, prep):
    from research.representation.atlas.axis_screen.run import jobs
    Path(out).mkdir(parents=True, exist_ok=True)
    J = [(str(bank), str(out), i, prep) for j in jobs(bank, out) for i, _ in j[4]]
    print('windows', len(J), prep, flush=True)
    with Pool(workers, maxtasksperchild=50) as p:
        for k, (wid, sec) in enumerate(p.imap_unordered(window, J, chunksize=2)):
            if k % 100 == 0:
                print(f'{k:5d} {sec:6.1f}s {wid}', flush=True)


def collect(out, target):
    from research.representation.atlas.axis_screen.mechanism import VERS
    fs = sorted(glob.glob(f'{out}/*.npz'))
    m = np.stack([np.load(f)['m'] for f in fs])
    np.savez(target, ids=np.array([Path(f).stem for f in fs]), m=m, metrics=np.array(METRICS), versions=np.array(VERS))
    print('collected', len(fs), 'windows ->', target)


def link(path, screen, manifest):
    """Secondary readout stated with the criteria: within drug sessions, session-demeaned window-level r of each metric's
    net with the anchor net (z(mi_g, KSG), 32-channel subset) and the MinCovDet net (z(pcorr, pcorr_mcd)) of the screen."""
    import json
    from research.representation.atlas.axis_screen.score import DRUGS
    D, S = np.load(path), np.load(screen)
    rows = {r['id']: r for r in json.load(open(manifest))['rows']}
    keys = [str(k) for k in S['keys']]
    sid = {str(i): w for w, i in enumerate(S['ids'])}
    ids = [str(i) for i in D['ids']]
    w = np.array([sid[i] for i in ids])
    net = lambda Z: Z[:, 0] - Z[:, 1:4].mean(1)
    mi, pc, mcd = keys.index('mi_g'), keys.index('pcorr'), keys.index('pcorr_mcd')
    ref = dict(anchor=net(S['Z32'][w, :, mi, len(keys)]), mcd=net(S['Z'][w, :, pc, mcd]))
    m = D['m'][:, 0] - D['m'][:, 1:].mean(1)
    unit = np.array([rows[i]['unit'] for i in ids])
    dw = np.array([rows[i]['state'] == 'drug' and rows[i].get('agent', '') in DRUGS for i in ids])
    dm = lambda y: y[dw] - np.array([y[dw & (unit == u)].mean() for u in unit[dw]])
    out = {str(n): {k: round(float(np.corrcoef(dm(m[:, j]), dm(v))[0, 1]), 3) for k, v in ref.items()}
           for j, n in enumerate(D['metrics'])}
    print(json.dumps(dict(windows=int(dw.sum()), r=out), indent=1))
    return out


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'compute':
        compute(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
    elif cmd == 'collect':
        collect(*sys.argv[2:4])
    elif cmd == 'link':
        link(*sys.argv[2:5])
