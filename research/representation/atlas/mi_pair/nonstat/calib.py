"""Known-answer calibration of the non-stationary nulls (nulls.py) for z(Gaussian MI, KSG MI) on 32-channel, T = 1000 windows
(the anchor's 10-s pipeline: normal-scored, edges.edge_mi, one draw per null).

Generators: K = 4 band-limited AR(2) latent sources mixed into M = 32 channels with heterogeneous loadings, plus an AR floor.
  linear        stationary mixing, independent floor
  burst         shared on/off gain (on 40% of the time, mean dwell 0.8 s; off level .15) on signal and floor, plus a .15 white floor
  burst_slow    the same with mean dwell 3 s
  gain_cmfloor  shared slow log-normal gain (~2 s) on the signal over a floor with common-mode share .5 (broke the amplitude gate)
  gain_local    each channel's whole signal under its own slow log-normal gain (~2 s)
  switch        mixing switches between two loading matrices, mean dwell 2 s (slow piecewise-linear coupling)
  decouple      half the channels: in local episodes (~20% of time, ~250 ms) the shared part drops to .3 and a 2.5x local burst enters
  quadratic     half the channels carry b (s1^2 - 1) of latent source 1, b ~ U(.5, 1): a static nonlinear (curved) coupling
  quad_weak     the same with b ~ U(.15, .3), near the real effect size
N1 (envelope remodulation) was dropped after a pilot: calibrated on linear (+.000 at sigma >= 100) but +.039 to +.125 on shared gain
over an independent floor (sigma 10-100), i.e. it injects non-Gaussian dependence when the floor is not gained.
Residual per null = z(real) - z(null surrogate). The decision rule that uses these numbers is in ../preregs/nonstat-prereg.md.

  python calib.py <n seeds> <n workers> <out.npz>
"""
import sys, time, numpy as np, warnings
from pathlib import Path
from joblib import Parallel, delayed
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[4] / 'src')]
from edges import edge_mi, zc
from nulls import ns, n0, n2
warnings.filterwarnings('ignore')
M, T, K, PH = 32, 1000, 4, (1.6, -.7)
KINDS = ('linear', 'burst', 'burst_slow', 'gain_cmfloor', 'gain_local', 'switch', 'decouple', 'quadratic', 'quad_weak')
NULLS = (('N0', None), ('N2 L500', 500), ('N2 L200', 200), ('N2 L100', 100), ('N2 L50', 50))


def ar(n, phi, rng, burn=200):
    e = rng.standard_normal(n + burn); z = np.zeros(n + burn)
    for t in range(2, n + burn): z[t] = phi[0] * z[t - 1] + phi[1] * z[t - 2] + e[t]
    return z[burn:] / z[burn:].std()


def episodes(n, frac, mean_len, rng):
    m = np.zeros(n, bool); t = 0
    while t < n:
        t += int(rng.exponential(mean_len * (1 - frac) / frac)); L = max(5, int(rng.exponential(mean_len))); m[t:t + L] = True; t += L
    return m


def gain(rho, rng):
    g = np.exp(.8 * ar(T, (rho, 0), rng)); return g / np.sqrt((g ** 2).mean())


def synth(kind, rng):
    s = np.column_stack([ar(T, PH, rng) for _ in range(K)]); A = rng.normal(size=(M, K)) * rng.uniform(.3, 1.5, (M, 1))
    sh = s @ A.T; sh /= sh.std(0); fl = np.column_stack([ar(T, PH, rng) for _ in range(M)])
    if kind == 'gain_cmfloor': fl = np.sqrt(.5) * ar(T, PH, rng)[:, None] + np.sqrt(.5) * fl
    if kind == 'gain_cmfloor': sh = sh * gain(.995, rng)[:, None]
    if kind == 'switch':
        B = rng.normal(size=(M, K)) * rng.uniform(.3, 1.5, (M, 1)); sh2 = s @ B.T; sh2 /= sh2.std(0)
        state = np.zeros(T, bool); t = 0; on = False
        while t < T: L = max(20, int(rng.exponential(200))); state[t:t + L] = on; on = not on; t += L
        sh = np.where(state[:, None], sh2, sh)
    if kind == 'decouple':
        for c in range(M // 2):
            ep = episodes(T, .2, 25, rng)[:T]; sh[:, c] = np.where(ep, .3 * sh[:, c] + 2.5 * ar(T, PH, rng), sh[:, c])
    if kind in ('quadratic', 'quad_weak'):
        q = s[:, 0] ** 2 - 1; q /= q.std(); lo, hi = (.5, 1.) if kind == 'quadratic' else (.15, .3)
        for c in range(M // 2): sh[:, c] = sh[:, c] + rng.uniform(lo, hi) * q
    x = sh + .6 * fl
    if kind in ('burst', 'burst_slow'):
        g = np.where(episodes(T, .4, 80 if kind == 'burst' else 300, rng)[:T], 1., .15)[:, None]; x = g * x + .15 * rng.standard_normal((T, M))
    if kind == 'gain_local': x = x * np.column_stack([gain(.995, rng) for _ in range(M)])
    return x


def one(kind, seed):
    rng = np.random.default_rng(seed); v = ns(synth(kind, rng)); zr = zc(*edge_mi(v)); out = []
    for k, (name, p) in enumerate(NULLS):
        r = np.random.default_rng([seed, k])
        s = n0(v, r) if p is None else n2(v, r, p)
        out.append(zr - zc(*edge_mi(s)))
    return out


if __name__ == '__main__':
    B, nw, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]; t = time.time(); R = {}
    for kind in KINDS:
        R[kind] = np.array(Parallel(n_jobs=nw)(delayed(one)(kind, 1000 * (KINDS.index(kind) + 1) + s) for s in range(B)))
        m, sd = R[kind].mean(0), R[kind].std(0, ddof=1)
        print(f'{kind:13s} ' + '  '.join(f'{n} {a:+.4f} (t {a / b * np.sqrt(B):+.1f})' for (n, _), a, b in zip(NULLS, m, sd)), f'[{time.time() - t:.0f} s]', flush=True)
    np.savez(out, nulls=[n for n, _ in NULLS], **R)
