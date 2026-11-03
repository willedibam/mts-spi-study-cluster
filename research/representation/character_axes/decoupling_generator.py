"""Which mechanism decouples direct from pairwise dependence patterns? Calibration against a known direct network.

NeuroTycho anaesthesia lowers the edge-pattern alignment of robust direct dependence (pyspi prec-sq_MinCovDet) with
pairwise dependence (Spearman r^2, IDS); about two-thirds is reproduced by spectra-preserving surrogates and robust
and classical direct-coupling patterns diverge beyond them (findings.md, "Direct versus pairwise dependence").
Generator: M=16, T=2000 sparse VAR(1) direct network (density 0.2, |weights| 0.1-0.3, random sign), Gaussian
innovations. One knob varies at a time from the shared base (level 0):
  common   slow common drive (AR(1), phi .995) with heterogeneous loadings U(.2, 1), gain 0-4
  slowing  intrinsic channel memory (diagonal AR coefficient) 0-.9, network rescaled to keep coupling share
  state    slow two-state process (25% up, mean dwell 200): up states amplify innovations (gain 1-4) and switch
           to a second, independent direct network
  burst    the same amplification without the network switch (amplitude intermittency only)
  heavy    Student-t innovations (df inf, 10, 5, 3, 2.5): non-Gaussian marginals without any dependence change
Readouts per window, on the data and on a multivariate phase-randomised, rank-remapped surrogate (same phases in all
channels; keeps auto- and cross-spectra approximately and marginals exactly): alignment over edges of robust and
classical prec-sq with Spearman r^2 and with IDS, robust-versus-classical prec-sq alignment, common-mode share, mean
r^2 and mean partial r^2, and recovery of the true direct network (alignment of prec-sq with |A + A^T|).
Run on Gadi (jobs/decoupling_generator.pbs).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import rankdata, spearmanr
from sklearn.covariance import EmpiricalCovariance, MinCovDet

from pyspi.lib.ids.dependence import compute_IDS

M, T, BURN = 16, 2000, 500
LEVELS = dict(common=[0, .5, 1, 2, 4], slowing=[0, .3, .6, .8, .9], state=[1, 1.5, 2, 3, 4], burst=[1, 1.5, 2, 3, 4],
              heavy=[np.inf, 10, 5, 3, 2.5])
NAMES = ['align_robust_spearman', 'align_classical_spearman', 'align_robust_ids', 'align_classical_ids', 'robust_vs_classical',
         'common_share', 'mean_r2', 'mean_partial_r2', 'truth_robust', 'truth_classical']


def network(rng):
    A = np.where(rng.random((M, M)) < .2, rng.uniform(.1, .3, (M, M)) * rng.choice([-1, 1], (M, M)), 0.)
    np.fill_diagonal(A, 0)
    return A


def stable(A, cap=.95):
    r = np.max(np.abs(np.linalg.eigvals(A)))
    return A * (cap / r) if r > cap else A


def simulate(knob, level, seed):
    rng = np.random.default_rng(seed)
    A1, A2 = network(rng), network(rng)
    diag = level if knob == 'slowing' else .3
    B1 = stable(diag * np.eye(M) + (1 - diag) / .7 * A1)
    B2 = stable(diag * np.eye(M) + (1 - diag) / .7 * A2)
    n = T + BURN
    if knob == 'heavy' and np.isfinite(level):
        e = rng.standard_t(level, (n, M)) / np.sqrt(level / (level - 2))
    else:
        e = rng.standard_normal((n, M))
    up = np.zeros(n, bool)
    if knob in ('state', 'burst'):
        p_exit = 1 / 200
        p_enter = p_exit * .25 / .75
        for t in range(1, n):
            up[t] = (rng.random() > p_exit) if up[t - 1] else (rng.random() < p_enter)
        e[up] *= level
    x = np.zeros((n, M))
    for t in range(1, n):
        B = B2 if (knob == 'state' and up[t]) else B1
        x[t] = B @ x[t - 1] + e[t]
    if knob == 'common':
        s = np.zeros(n)
        for t in range(1, n):
            s[t] = .995 * s[t - 1] + rng.standard_normal() * np.sqrt(1 - .995 ** 2)
        x += level * np.outer(s, rng.uniform(.2, 1, M)) * x[BURN:].std()
    return x[BURN:], np.abs(A1) + np.abs(A1.T)


def surrogate(x, rng):
    f = np.fft.rfft(x - x.mean(0), axis=0)
    phase = np.exp(1j * rng.uniform(0, 2 * np.pi, (f.shape[0], 1)))
    phase[0] = 1
    y = np.fft.irfft(f * phase, n=len(x), axis=0)
    out = np.empty_like(x)
    for i in range(x.shape[1]):
        out[np.argsort(y[:, i]), i] = np.sort(x[:, i])
    return out


def readouts(x, truth):
    x = (x - x.mean(0)) / x.std(0)
    off = ~np.eye(M, dtype=bool)
    pr = MinCovDet(random_state=42).fit(x).precision_[off] ** 2
    pc_mat = EmpiricalCovariance().fit(x).precision_
    pc = pc_mat[off] ** 2
    sp = spearmanr(x).statistic[off] ** 2
    ids = np.asarray(compute_IDS(x))[off]
    c = np.corrcoef(x, rowvar=False)
    d = np.sqrt(np.diag(pc_mat))
    corr = lambda a, b: np.corrcoef(a, b)[0, 1]
    return [corr(pr, sp), corr(pc, sp), corr(pr, ids), corr(pc, ids), corr(pr, pc), np.linalg.eigvalsh(c)[-1] / M,
            np.mean(c[off] ** 2), np.mean((pc_mat / np.outer(d, d))[off] ** 2), corr(pr, truth[off]), corr(pc, truth[off])]


def run(knob, level, seed):
    x, truth = simulate(knob, level, seed)
    return readouts(x, truth), readouts(surrogate(x, np.random.default_rng(seed + 1)), truth)


def main(output, seeds, jobs):
    tasks = [(k, lv, s) for k, lvs in LEVELS.items() for lv in lvs for s in range(seeds)]
    res = Parallel(n_jobs=jobs)(delayed(run)(k, lv, 261104 + s) for k, lv, s in tasks)
    report = dict(names=NAMES, levels={k: [float(v) for v in lv] for k, lv in LEVELS.items()}, results={})
    for k, lvs in LEVELS.items():
        report['results'][k] = []
        for lv in lvs:
            real = np.array([r for (kk, l, s), (r, _) in zip(tasks, res) if kk == k and l == lv])
            surr = np.array([u for (kk, l, s), (_, u) in zip(tasks, res) if kk == k and l == lv])
            report['results'][k].append(dict(level=float(lv), real_mean=real.mean(0).tolist(), real_sd=real.std(0).tolist(),
                                             surr_mean=surr.mean(0).tolist(), resid_mean=(real - surr).mean(0).tolist()))
            print(f'{k:8s} {lv:5}', ' '.join(f'{n[:14]}={m:+.3f}({u:+.3f})' for n, m, u in zip(NAMES, real.mean(0), surr.mean(0))), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, default=20)
    p.add_argument('--jobs', type=int, default=16)
    a = p.parse_args()
    main(a.output, a.seeds, a.jobs)
