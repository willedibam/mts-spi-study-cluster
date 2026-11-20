"""Numerical checks of the Gaussian functional relations the pair catalogue relies on (pyspi-free, seconds).

Each check compares a closed form with a Monte Carlo estimate on synthetic Gaussian data. A relation that holds
fixes z(A, B) under a Gaussian null up to saturation curvature and estimator noise, so a departure beyond a
linear-Gaussian surrogate reads the non-Gaussian feature the more flexible statistic targets.

    python -m research.representation.atlas.pair_catalogue.identities --output research/representation/atlas/pair_catalogue/identities.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import hilbert, lfilter
from scipy.special import hyp2f1, spence
from scipy.stats import kendalltau, spearmanr


def li2(x):
    return spence(1 - x)


def bivariate(rho, n, rng):
    x = rng.standard_normal(n)
    return x, rho * x + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)


def dcor2(x, y):
    """Squared (V-statistic) distance correlation, Szekely et al. 2007."""
    def centred(v):
        d = np.abs(v[:, None] - v[None])
        return d - d.mean(0) - d.mean(1)[:, None] + d.mean()
    A, B = centred(x), centred(y)
    return (A * B).mean() / np.sqrt((A * A).mean() * (B * B).mean())


def dcor2_gaussian(rho):
    """Szekely, Rizzo & Bakirov 2007, Theorem 7 (bivariate normal)."""
    r = abs(rho)
    num = r * np.arcsin(r) + np.sqrt(1 - r ** 2) - r * np.arcsin(r / 2) - np.sqrt(4 - r ** 2) + 1
    return num / (1 + np.pi / 3 - np.sqrt(3))


def ar1(n, a, rng):
    return lfilter([1.0], [1.0, -a], rng.standard_normal(n))


def canberra_check(rng, T=20000, reps=20, rho=.6):
    """Canberra distance mean |x-y|/(|x|+|y|) is invariant to a gain shared by both channels at each sample. Excess
    over its Gaussian value f_G(r), on z-scored and on normal-scored channels, for five generators."""
    from scipy.stats import norm, rankdata
    grid = np.linspace(-.95, .95, 39)
    curve = []
    for g in grid:
        u, v = bivariate(g, 200000, rng)
        curve.append(np.mean(np.abs(u - v) / (np.abs(u) + np.abs(v))))
    f_g = lambda r: np.interp(r, grid, curve)
    zs = lambda x: (x - x.mean()) / x.std()
    ns = lambda x: norm.ppf(rankdata(x) / (len(x) + 1))
    gain = lambda: np.exp(.8 * rng.standard_normal(T // 50).repeat(50))

    def generate(kind):
        u, e = rng.standard_normal(T), rng.standard_normal(T)
        if kind == 'gaussian':
            return u, rho * u + np.sqrt(1 - rho ** 2) * e
        if kind == 'shared_gain':
            g = gain()
            return g * u, g * (rho * u + np.sqrt(1 - rho ** 2) * e)
        if kind == 'separate_gains':
            return gain() * u, gain() * (rho * u + np.sqrt(1 - rho ** 2) * e)
        if kind == 'amplitude_linked_decoupling':  # correlation .85 at low gain, .2 at high gain
            g = gain()
            c = np.where(g > 1, .2, .85)
            return g * u, g * (c * u + np.sqrt(1 - c ** 2) * e)
        if kind == 'gain_over_fixed_floor':  # gain scales the shared signal, not the independent floor
            g, s = gain(), rng.standard_normal(T)
            return g * s + rng.standard_normal(T), g * s + rng.standard_normal(T)
        return u, rho * u + .5 * (u ** 2 - 1) + np.sqrt(1 - rho ** 2) * e  # static quadratic

    out = {}
    for kind in ('gaussian', 'shared_gain', 'separate_gains', 'amplitude_linked_decoupling', 'gain_over_fixed_floor', 'static_quadratic'):
        for tag, f in (('z_scored', zs), ('normal_scored', ns)):
            ex = []
            for _ in range(reps):
                x, y = generate(kind)
                x, y = f(x), f(y)
                r = np.mean(x * y)
                ex.append([r, np.mean(np.abs(x - y) / (np.abs(x) + np.abs(y))) - f_g(r)])
            ex = np.array(ex)
            out[f'{kind}:{tag}'] = dict(r=float(ex[:, 0].mean()), excess=float(ex[:, 1].mean()),
                                        se=float(ex[:, 1].std() / np.sqrt(reps)))
    return out


def main(output):
    rng = np.random.default_rng(0)
    out = {}
    rhos = [.1, .3, .5, .7, .9]

    # 1-2. Rank correlations of a bivariate normal (Kendall: Greiner; Spearman: Pearson 1907).
    rows = []
    for rho in rhos:
        x, y = bivariate(rho, 20000, rng)
        rows.append(dict(rho=rho, kendall=float(kendalltau(x[:4000], y[:4000])[0]), kendall_formula=2 / np.pi * np.arcsin(rho),
                         spearman=float(spearmanr(x, y)[0]), spearman_formula=6 / np.pi * np.arcsin(rho / 2)))
    out['rank_correlations'] = rows

    # 3. Distance correlation of a bivariate normal (Theorem 7); dCor^2 is monotone in |rho| and below rho^2 near 0? (report)
    rows = []
    for rho in rhos:
        x, y = bivariate(rho, 2500, rng)
        rows.append(dict(rho=rho, dcor2=float(dcor2(x, y)), formula=float(dcor2_gaussian(rho)), rho2=rho ** 2))
    out['distance_correlation'] = rows

    # 4. Envelopes of jointly Gaussian broadband processes with a lag: the analytic signals are proper complex
    # Gaussian with complex correlation gamma; |gamma|^2 = r^2 + r_H^2 with r_H = corr(x, H[y]).
    rows = []
    for lag, c in [(0, .6), (2, .6), (5, .6), (10, .6), (3, .3)]:
        n = 400000
        s = ar1(n + 20, .9, rng)
        x = s[20:] + .8 * ar1(n, .9, rng)
        y = c * s[20 - lag:n + 20 - lag] + ar1(n, .9, rng)
        x, y = (x - x.mean()) / x.std(), (y - y.mean()) / y.std()
        ax, ay = hilbert(x), hilbert(y)
        g = np.mean(ax * np.conj(ay)) / np.sqrt(np.mean(np.abs(ax) ** 2) * np.mean(np.abs(ay) ** 2))
        g2 = float(np.abs(g) ** 2)
        r = float(np.mean(x * y))
        amp = np.corrcoef(np.abs(ax), np.abs(ay))[0, 1]
        rows.append(dict(lag=lag, r2=r ** 2, gamma2=g2,
                         power_corr=float(np.corrcoef(np.abs(ax) ** 2, np.abs(ay) ** 2)[0, 1]), power_formula=g2,
                         amp_corr=float(amp), amp_formula=float(np.pi / 4 * (hyp2f1(-.5, -.5, 1, g2) - 1) / (1 - np.pi / 4)),
                         logpower_corr=float(np.corrcoef(np.log(np.abs(ax) ** 2), np.log(np.abs(ay) ** 2))[0, 1]),
                         logpower_formula=float(6 * li2(g2) / np.pi ** 2)))
    out['envelope'] = rows

    # 5. Phase locking of a proper complex Gaussian pair: |E exp(i dphi)| = (pi/4)|g| 2F1(1/2,1/2;2;|g|^2).
    rows = []
    for g in [.2, .4, .6, .8, .95]:
        n = 400000
        a = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
        b = g * a + np.sqrt(1 - g ** 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
        plv = np.abs(np.mean(np.exp(1j * (np.angle(a) - np.angle(b)))))
        rows.append(dict(abs_gamma=g, plv=float(plv), formula=float(np.pi / 4 * g * hyp2f1(.5, .5, 2, g ** 2))))
    out['phase_locking'] = rows

    # 6. L1/L2 of the difference signal: Gaussian x - y gives mean|d| / rms(d) = sqrt(2/pi) whatever r.
    # Chebyshev: the expected max|d| / rms(d) depends on T and on the autocorrelation of d (fewer effective
    # samples), so linf|l2 is fixed by r only once the spectrum of x - y is fixed (as joint IAAFT fixes it).
    rows = []
    for T in (200, 1000):
        for a in (0., .9, .98):
            ratios, l1 = [], []
            for _ in range(400):
                d = ar1(T + 200, a, rng)[200:]
                d /= np.sqrt(np.mean(d ** 2))
                ratios.append(np.max(np.abs(d)))
                l1.append(np.mean(np.abs(d)))
            rows.append(dict(T=T, ar=a, max_over_rms=float(np.mean(ratios)), max_sd=float(np.std(ratios)),
                             l1_over_rms=float(np.mean(l1)), sqrt_2_over_pi=float(np.sqrt(2 / np.pi))))
    out['difference_signal'] = rows

    # 7. Static statistics on a Gaussian network: z across edges for a 16-channel Gaussian process with
    # heterogeneous spectra (AR(1) poles .5-.98) and random mixing. Static pairs should be near their
    # fixed-relation value; Chebyshev against L2 is not, because spectra differ across edges.
    M, T = 16, 1000
    zs = {k: [] for k in ('r2|dcor2', 'r|spearman', 'r|kendall', 'l1|l2', 'linf|l2', 'linf|l2_white')}
    for rep in range(10):
        A = rng.standard_normal((M, M)) * (rng.random((M, M)) < .3) + np.eye(M) * 2
        poles = rng.uniform(.5, .98, M)
        S = np.stack([ar1(T + 200, p, rng)[200:] for p in poles])
        X = A @ S
        W = A @ rng.standard_normal((M, T))
        prof = {k: [] for k in ('r', 'r2', 'dcor2', 'spearman', 'kendall', 'l1', 'l2', 'linf', 'linf_white', 'l2_white')}
        for Z, tag in ((X, ''), (W, '_white')):
            Z = (Z - Z.mean(1, keepdims=True)) / Z.std(1, keepdims=True)
            for i in range(M):
                for j in range(i + 1, M):
                    d = Z[i] - Z[j]
                    if tag:
                        prof['linf_white'].append(np.max(np.abs(d)))
                        prof['l2_white'].append(np.sqrt(np.sum(d ** 2)))
                        continue
                    r = np.mean(Z[i] * Z[j])
                    prof['r'].append(r); prof['r2'].append(r ** 2)
                    prof['dcor2'].append(dcor2(Z[i][::4], Z[j][::4]))
                    prof['spearman'].append(spearmanr(Z[i], Z[j])[0]); prof['kendall'].append(kendalltau(Z[i], Z[j])[0])
                    prof['l1'].append(np.sum(np.abs(d))); prof['l2'].append(np.sqrt(np.sum(d ** 2))); prof['linf'].append(np.max(np.abs(d)))
        c = lambda a, b: float(np.corrcoef(prof[a], prof[b])[0, 1])
        zs['r2|dcor2'].append(c('r2', 'dcor2')); zs['r|spearman'].append(c('r', 'spearman')); zs['r|kendall'].append(c('r', 'kendall'))
        zs['l1|l2'].append(c('l1', 'l2')); zs['linf|l2'].append(c('linf', 'l2')); zs['linf|l2_white'].append(c('linf_white', 'l2_white'))
    out['gaussian_network_z'] = {k: dict(mean=float(np.mean(v)), sd=float(np.std(v))) for k, v in zs.items()}
    out['canberra'] = canberra_check(rng)
    out['notes'] = ('dcor2 uses every 4th sample (250 points); kendall in check 1 uses the first 4000 samples. '
                    'Envelope check: AR(1) carrier .9, shared source with a lag in samples. Canberra: excess = mean '
                    'canberra - f_G(r), f_G tabulated on Gaussian pairs; 20 reps of T 20000 iid samples, gain in 50-sample blocks.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=1) + '\n')
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    o = main(p.parse_args().output)
    print(json.dumps(o, indent=1))
