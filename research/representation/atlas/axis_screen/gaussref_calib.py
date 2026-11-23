"""Synthetic calibration of the gaussref readouts (run before the criteria are frozen; result.md).

32 channels, AR(1) a .9 driven by 3-factor correlated innovations (r about .1-.6), 10-s (T 1000) and 2-s (T 200) windows.
Generators (each z-scored, then prepared as 'z', 'z-m' or 'ns' and set against 3 IAAFT draws as in gaussref.window):
  gaussian           the linear-Gaussian process
  static_heavy       per-channel static monotone transform (heavier tails, Gaussian copula: no dependence change)
  shared_gain        one slow gain multiplies every channel (scale mixture; correlation unchanged)
  separate_gains     an independent slow gain per channel
  gain_over_floor    the gain multiplies the shared factors only, not the private noise (coupling rises with amplitude)
  amp_decoupling     shared gain, and the factor loadings fall where the gain is high (coupling falls with amplitude)
  amp_lead           channels 0-15 carry a slow gain that channels 16-31 carry LAG samples later (directed amplitude)
Slow gain: exp(.6 u), u a unit-variance AR(1) with time constant 50 samples (0.5 s).

  python -m research.representation.atlas.axis_screen.gaussref_calib <out.json> [workers]
"""
import sys, json
import numpy as np
from multiprocessing import Pool

GENS = ('gaussian', 'static_heavy', 'shared_gain', 'separate_gains', 'gain_over_floor', 'amp_decoupling', 'amp_lead')
PREPS = ('z', 'z-m', 'ns')
M, A, REPS = 32, .9, 8


def _ar(e, a=A):
    x = np.zeros_like(e)
    for t in range(1, len(e)):
        x[t] = a * x[t - 1] + e[t]
    return x


def _gain(rng, n, k=1, tau=50):
    a = np.exp(-1 / tau)
    u = _ar(rng.standard_normal((n, k)) * np.sqrt(1 - a ** 2), a)
    return np.exp(.6 * u / u.std(0))


def generate(kind, T, rng):
    from research.representation.atlas.axis_screen.gaussref import LAG
    B = 300
    n = T + B
    L = rng.uniform(.2, .8, (M, 3)) * rng.choice([-1, 1], (M, 3)) * (rng.random((M, 3)) < .6)
    f = _ar(rng.standard_normal((n, 3)))
    p = _ar(rng.standard_normal((n, M)))
    if kind == 'gain_over_floor':
        x = _gain(rng, n) * f @ L.T + p
    elif kind == 'amp_decoupling':
        g = _gain(rng, n)
        w = np.where(g > np.median(g), .3, 1.)  # loadings shrink at high gain
        x = g * ((w * f) @ L.T + p)
    elif kind == 'amp_lead':
        g = _gain(rng, n + LAG)
        G = np.hstack([np.repeat(g[LAG:], M // 2, 1), np.repeat(g[:-LAG], M - M // 2, 1)])
        x = G * (f @ L.T + p)
    else:
        x = f @ L.T + p
        if kind == 'shared_gain':
            x = _gain(rng, n) * x
        elif kind == 'separate_gains':
            x = _gain(rng, n, M) * x
        elif kind == 'static_heavy':
            z = (x - x.mean(0)) / x.std(0)
            x = np.sinh(.9 * z)
    x = x[B:]
    return (x - x.mean(0)) / x.std(0)


def one(args):
    kind, T, rep = args
    from research.representation.atlas.axis_screen.gaussref import metrics
    from research.representation.atlas.axis_screen.run import _ns
    from mtsspi.surrogates import multivariate_iaaft
    rng = np.random.default_rng([rep, T, GENS.index(kind)])
    x = generate(kind, T, rng)
    out = {}
    for prep in PREPS:
        v = _ns(x) if prep == 'ns' else x
        final = 'marginal' if prep.endswith('-m') else 'spectrum'
        m = [metrics(v)] + [metrics(multivariate_iaaft(v, np.random.default_rng([rep, k, 7]), final=final)[0]) for k in range(3)]
        out[prep] = (np.array(m[0]), np.mean(m[1:], 0))
    return kind, T, out


def main(output, workers=8):
    from research.representation.atlas.axis_screen.gaussref import METRICS
    J = [(k, T, r) for k in GENS for T in (1000, 200) for r in range(REPS)]
    with Pool(workers) as p:
        R = p.map(one, J)
    res = {}
    for kind in GENS:
        for T in (1000, 200):
            for prep in PREPS:
                rr = [o[prep] for k, t, o in R if k == kind and t == T]
                real = np.array([a for a, _ in rr]); net = np.array([a - b for a, b in rr])
                res[f'{kind}|T{T}|{prep}'] = {m: dict(real=round(float(real[:, i].mean()), 4), net=round(float(net[:, i].mean()), 4),
                                                      se=round(float(net[:, i].std(ddof=1) / np.sqrt(len(net))), 4))
                                              for i, m in enumerate(METRICS)}
    json.dump(dict(design=__doc__.split('\n\n')[1], reps=REPS, results=res), open(output, 'w'), indent=1)
    for T in (1000, 200):
        for prep in PREPS:
            print(f'--- T {T} prep {prep}: net (real - IAAFT) / se')
            print(f"{'':16s}" + ''.join(f'{m:>15s}' for m in METRICS))
            for kind in GENS:
                r = res[f'{kind}|T{T}|{prep}']
                print(f'{kind:16s}' + ''.join(f"{r[m]['net']:+8.3f}({r[m]['se']:.3f})" for m in METRICS))


if __name__ == '__main__':
    main(sys.argv[1], *map(int, sys.argv[2:3]))
