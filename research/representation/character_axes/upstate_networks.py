"""Do high-amplitude slow-wave epochs carry a different direct network than the rest of the window?

The decoupling generator singles out state-dependent coupling (a different direct network in high-amplitude epochs)
as the mechanism matching NeuroTycho's robust-versus-classical divergence (findings.md). This tests it directly,
without the robust estimator: per window (channels standardized), up epochs are the top quarter of the channel-mean
delta-band (0.5-4 Hz at 250 Hz) amplitude envelope smoothed over 0.4 s; the rest is the comparison set. The null keeps
the mask's block structure: the same mask circularly shifted by random offsets (20 shifts, at least 10% of T).
Readouts: similarity over edges of partial r^2 in up versus rest, the shifted-mask mean, their gap, the same for r^2,
and the up-minus-rest mean r^2 and partial r^2. Calibrated on the generator (band None: broadband power) and run on
NeuroTycho real data and both IAAFT variants. Run on Gadi (jobs/upstate_networks.pbs).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt

from research.representation.character_axes.decoupling_generator import simulate, surrogate
from research.representation.character_axes.direct_marginal import COHORTS, units
from research.representation.character_axes.evaluate_cohorts import effect

NAMES = ['sim_partial_up', 'sim_partial_shift', 'sim_partial_gap', 'sim_r2_up', 'sim_r2_shift', 'sim_r2_gap', 'up_minus_rest_r2', 'up_minus_rest_partial_r2']
FS, SMOOTH, SHIFTS = 250, 100, 20


def mats(x, off):
    c = np.corrcoef(x, rowvar=False)
    p = np.linalg.pinv(c)
    d = np.sqrt(np.abs(np.diag(p)))
    return c[off] ** 2, (p / np.outer(d, d))[off] ** 2


def mask(x, band):
    s = x if band is None else sosfiltfilt(butter(4, band, 'bandpass', fs=FS, output='sos'), x, axis=0)
    env = np.abs(hilbert(s, axis=0)).mean(1) if band is not None else (s ** 2).mean(1)
    env = uniform_filter1d(env, SMOOTH)
    return env >= np.quantile(env, .75)


def split(x, m, off):
    (ra, pa), (rb, pb) = mats(x[m], off), mats(x[~m], off)
    return np.corrcoef(pa, pb)[0, 1], np.corrcoef(ra, rb)[0, 1], ra.mean() - rb.mean(), pa.mean() - pb.mean()


def window(x, band, rng):
    x = (x - x.mean(0)) / x.std(0)
    off = ~np.eye(x.shape[1], dtype=bool)
    m = mask(x, band)
    sp, sr, dr, dp = split(x, m, off)
    T = len(x)
    sh = np.array([split(x, np.roll(m, rng.integers(T // 10, T - T // 10)), off)[:2] for _ in range(SHIFTS)]).mean(0)
    return [sp, sh[0], sp - sh[0], sr, sh[1], sr - sh[1], dr, dp]


def calibrate(seeds):
    out = {}
    for k, lv in [('state', 1), ('state', 2), ('state', 3), ('burst', 3), ('common', 2), ('slowing', .9), ('heavy', 3)]:
        R, S = [], []
        for s in range(seeds):
            x, _ = simulate(k, lv, 261104 + s)
            R.append(window(x, None, np.random.default_rng(s)))
            S.append(window(surrogate(x, np.random.default_rng(s + 1)), None, np.random.default_rng(s)))
        out[f'{k}-{lv}'] = dict(real=np.mean(R, 0).tolist(), surrogate=np.mean(S, 0).tolist())
        print('generator', k, lv, 'real', np.round(np.mean(R, 0), 3), 'surr', np.round(np.mean(S, 0), 3), flush=True)
    return out


def readouts(bank):
    rng = np.random.default_rng(261104)
    return np.array([window(np.asarray(x, np.float64), (.5, 4), rng) for x in bank['x']])


def main(root, output, seeds):
    report = dict(names=NAMES, generator=calibrate(seeds))
    rel, srel, variants = COHORTS['neurotycho']
    bank = np.load(root / rel / 'bank.npz')
    X, y, g = readouts(bank), bank['y'], units(bank, 'neurotycho')
    sur = {v: readouts(np.load(root / srel.format(v) / 'bank.npz')) for v in variants}
    unit_mean = lambda V, c: np.mean([np.nanmean(V[(g == u) & (y == c)], 0) for u in np.unique(g[y == c])], 0).tolist()
    report['neurotycho'] = dict(real={c: unit_mean(X, c) for c in (0, 1)}, surrogate={v: {c: unit_mean(S, c) for c in (0, 1)} for v, S in sur.items()},
                                effect=dict(real=effect(X, y, g), **{f'resid{v}': effect(X - S, y, g) for v, S in sur.items()},
                                            **{f'surr{v}': effect(S, y, g) for v, S in sur.items()}))
    f = lambda e: f"{e['standardized']:+6.2f} {e['positive']}/{e['units']}" if e['standardized'] is not None else 'n/a'
    E = report['neurotycho']['effect']
    for k, n in enumerate(NAMES):
        print(f'neurotycho {n:26s} awake {report["neurotycho"]["real"][0][k]:+.3f} anaes {report["neurotycho"]["real"][1][k]:+.3f} d {f(E["real"][k])}',
              *(f'| surr{v} {report["neurotycho"]["surrogate"][v][0][k]:+.3f}->{report["neurotycho"]["surrogate"][v][1][k]:+.3f} res {f(E[f"resid{v}"][k])}' for v in sur), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, default=float) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, default=20)
    a = p.parse_args()
    main(a.root, a.output, a.seeds)
