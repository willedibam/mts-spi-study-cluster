"""Pair-level mechanism readouts for excess (kNN − Gaussian) MI beyond IAAFT, and their known-answer calibration.
Readouts on z-scored x, y (T samples, 0.5-s blocks of BL samples):
 exmi   : KSG MI − Gaussian MI (the target)
 switch : variance across blocks of Fisher-z block r          (coupling changes within the window)
 gate   : corr(block Fisher r, block mean log-RMS)             (+: coupling rises with shared amplitude)
 imbal  : corr(block Fisher r, block |log-RMS_x − log-RMS_y|)  (−: coupling breaks when one channel has a local event)
 curv   : out-of-half ΔR² of [x², x³] over linear, both directions, clipped ±4 (static curved function)
 cokurt : mean(x²y²) − (1+2r²)                                  (joint tail co-activation)
 rkurt  : kurtosis of the linear-fit residual, both directions   (heavy-tailed coupling errors)"""
import numpy as np
from surr_pairs import acf, ksg, multivariate_iaaft
BL = 50
NAMES = ['exmi', 'switch', 'gate', 'imbal', 'curv', 'cokurt', 'rkurt']


def r2(X, y, Xt, yt):
    b = np.linalg.lstsq(X, y, rcond=None)[0]; return 1 - ((yt - Xt @ b) ** 2).mean() / yt.var()


def feats(x, y, w=None):
    x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std(); T = len(x); nb = T // BL
    r = np.corrcoef(x, y)[0, 1]
    if w is None: w = min(int(2 * np.dot(acf(x), acf(y))), T // 10)
    exmi = ksg(x, y, w) + 0.5 * np.log(1 - r ** 2)
    bx, by = x[:nb * BL].reshape(nb, BL), y[:nb * BL].reshape(nb, BL)
    rb = np.arctanh(np.clip([np.corrcoef(a, b)[0, 1] for a, b in zip(bx, by)], -.999, .999))
    lx, ly = np.log(bx.std(1)), np.log(by.std(1))
    gate = np.corrcoef(rb, (lx + ly) / 2)[0, 1]; imbal = np.corrcoef(rb, np.abs(lx - ly))[0, 1]
    xc, yc = x.clip(-4, 4), y.clip(-4, 4); h = T // 2; o = np.ones(h); cv = 0
    for a, b in ((xc, yc), (yc, xc)):
        for tr, te in ((slice(0, h), slice(h, 2 * h)), (slice(h, 2 * h), slice(0, h))):
            L = lambda z: np.c_[o, z]; C = lambda z: np.c_[o, z, z ** 2, z ** 3]
            cv += (r2(C(a[tr]), b[tr], C(a[te]), b[te]) - r2(L(a[tr]), b[tr], L(a[te]), b[te])) / 4
    ck = (x ** 2 * y ** 2).mean() - (1 + 2 * r ** 2)
    rk = 0
    for a, b in ((x, y), (y, x)):
        e = b - r * a; e = (e - e.mean()) / e.std(); rk += (e ** 4).mean() / 2
    return np.array([exmi, rb.var(), gate, imbal, cv, ck, rk])


def ar(T, phi, rng, burn=200):
    e = rng.standard_normal(T + burn); z = np.zeros(T + burn)
    for t in range(2, T + burn): z[t] = phi[0] * z[t - 1] + phi[1] * z[t - 2] + e[t]
    return z[burn:] / z[burn:].std()


def episodes(T, frac, mean_len, rng):           # boolean mask with random episodes covering about frac of T
    m = np.zeros(T, bool); t = 0
    while t < T:
        gap = int(rng.exponential(mean_len * (1 - frac) / frac)); t += gap
        L = max(5, int(rng.exponential(mean_len))); m[t:t + L] = True; t += L
    return m


def synth(kind, T, rng, a=.85):
    PH = (1.6, -.7)                                   # band-limited (~6 Hz at 100 Hz), like low-band ECoG
    x = ar(T, PH, rng); n = ar(T, PH, rng); c = np.sqrt(1 - a ** 2)
    if kind == 'linear': y = a * x + c * n
    elif kind == 'marginals':                         # linear Gaussian, then static monotone transforms
        y = a * x + c * n; x, y = np.sinh(1.2 * x), np.sign(y) * np.abs(y) ** 1.6
    elif kind == 'curved': y = a * x + .45 * (x ** 2 - 1) + c * n
    elif kind == 'switch':                            # coupling off in random 0.3-s episodes, ~25% of the time
        off = episodes(T, .25, 30, rng); y = np.where(off, n, a * x + c * n)
    elif kind == 'local':                             # independent bursts in y break the coupling while raising y's amplitude
        bur = episodes(T, .2, 25, rng) * 2.5 * ar(T, PH, rng); y = a * x + c * n + bur
    elif kind == 'gain':                              # shared source with slow bursty envelope
        env = np.exp(1.0 * ar(T, (.995, -.0), rng)); s = env * x / env.std()
        x = s + .5 * n; y = s + .5 * ar(T, PH, rng)
    return x, y


def zscore_vs_surr(x, y, rng, B=10):
    f = feats(x, y); S = np.array([feats(*multivariate_iaaft(np.c_[x, y], rng, final='spectrum')[0].T) for _ in range(B)])
    return f - S.mean(0), S.std(0, ddof=1)


if __name__ == '__main__':
    from joblib import Parallel, delayed
    KINDS = ['linear', 'marginals', 'curved', 'switch', 'local', 'gain']
    def run(kind, s):
        rng = np.random.default_rng(s); x, y = synth(kind, 1000, rng); d, sd = zscore_vs_surr(x, y, rng)
        return d, sd, np.corrcoef(x, y)[0, 1]
    print('mean (real − IAAFT) / mean surrogate SD over 16 windows, T=1000;  r  |  ' + '  '.join(f'{n:>7s}' for n in NAMES))
    for k in KINDS:
        R = Parallel(n_jobs=6)(delayed(run)(k, s) for s in range(16))
        D = np.array([q[0] for q in R]); SD = np.array([q[1] for q in R]); rr = np.mean([q[2] for q in R])
        print(f'{k:10s} {rr:.2f} | ' + '  '.join(f'{v:+7.1f}' for v in D.mean(0) / SD.mean(0)))
