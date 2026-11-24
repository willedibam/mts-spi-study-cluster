"""Axis screen `l1|linf` preprocessing sensitivity at 2 s (post hoc; criteria in result.md, committed before this ran).

Per window and version (real and the screen's 3 IAAFT draws, same seeds) of the prepared window, prep as mechanism.py
('ns', 'z', 'ns-m', 'z-m'):
  z_l1linf  the screen's z(l1, linf): correlation over all edges of the cityblock and Chebyshev distance profiles of the
            z-scored window (stats.profiles)
  kurt      mean excess kurtosis over channels of the prepared window
Scored by mechanism.score (net = real - mean of the draws; animal t(3), split-half null SD, agents, eyes-open, sleep).

  python -m research.representation.atlas.axis_screen.linf_prep compute <bank> <out dir> <workers> [ns|z|ns-m|z-m]
  python -m research.representation.atlas.axis_screen.linf_prep collect <out dir> <out.npz>
  python -m research.representation.atlas.axis_screen.mechanism score <out.npz> <manifest.json> <out.json>
"""
import sys, glob, time
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from research.representation.atlas.axis_screen.mechanism import VERS

METRICS = ('z_l1linf', 'kurt')


def metrics(v):
    from scipy.stats import kurtosis
    from research.representation.atlas.axis_screen.stats import zscore
    x = zscore(np.asarray(v, np.float64))
    a, b = np.triu_indices(x.shape[1], 1)
    D = np.abs(x[:, a] - x[:, b])  # as stats.profiles
    return np.array([np.corrcoef(D.sum(0), D.max(0))[0, 1], kurtosis(x).mean()])


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


def compute(bank, out, workers, prep='ns'):
    from research.representation.atlas.axis_screen.run import jobs
    Path(out).mkdir(parents=True, exist_ok=True)
    J = [(str(bank), str(out), i, prep) for j in jobs(bank, out) for i, _ in j[4]]
    print('windows', len(J), prep, flush=True)
    with Pool(workers, maxtasksperchild=50) as p:
        for k, (wid, sec) in enumerate(p.imap_unordered(window, J, chunksize=4)):
            if k % 100 == 0:
                print(f'{k:5d} {sec:6.1f}s {wid}', flush=True)


def collect(out, target):
    fs = sorted(glob.glob(f'{out}/*.npz'))
    m = np.stack([np.load(f)['m'] for f in fs])
    np.savez(target, ids=np.array([Path(f).stem for f in fs]), m=m, metrics=np.array(METRICS), versions=np.array(VERS))
    print('collected', len(fs), 'windows ->', target)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'compute':
        compute(sys.argv[2], sys.argv[3], int(sys.argv[4]), *sys.argv[5:6])
    elif cmd == 'collect':
        collect(*sys.argv[2:4])
