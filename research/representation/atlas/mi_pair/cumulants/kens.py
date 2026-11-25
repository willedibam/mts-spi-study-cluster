"""Surrogate-ensemble fourth-order readouts of the MI pair (rule ../preregs/kens-prereg.md, post hoc, 25 November).

Per window and pair, in the standardised principal axes u = (a + b)/sqrt(2(1 + r)), v = (a - b)/sqrt(2(1 - r)) of kdiff.py:
k40, k22, k04 and Mardia's k40 + 2 k22 + k04, each also cross-only (the marginal kurtosis terms removed, so only the joint
cumulants c31, c22, c13 of the pair remain), and the L-kurtosis of u and of v. Exact identities on the standardised pair:
  k40 = (ka + kb + 4 c31 + 4 c13 + 6 c22) / (4 (1 + r)^2), k04 = (ka + kb - 4 c31 - 4 c13 + 6 c22) / (4 (1 - r)^2),
  k22 = (ka + kb - 2 c22) / (4 (1 - r^2)),  ka, kb the marginal excess kurtoses.
Nulls, all spectrum-preserving: B joint IAAFT draws (final='spectrum'; draw 0 is the anchor's own, seed k), B multivariate
phase-randomised (Gaussian) draws, and the random-phase expectation computed from the window's own Fourier coefficients.
  python kens.py <bank dir> <out.npz> <agent> <rowmode: first|row> <n workers> <B>
"""
import sys, json, numpy as np, warnings
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
warnings.filterwarnings('ignore')
ROWS = ('r', 'k40', 'k22', 'k04', 'k40x', 'k22x', 'k04x', 'M', 'Mx', 'tau_u', 'tau_v')


def lkurt(y):
    """Sample L-kurtosis tau4 of each column (unbiased probability-weighted moments; Gaussian 0.1226)."""
    y = np.sort(y, 0); n = len(y); i = np.arange(n)[:, None]
    b = [np.mean(w * y, 0) for w in (1, i / (n - 1), i * (i - 1) / ((n - 1) * (n - 2)), i * (i - 1) * (i - 2) / ((n - 1) * (n - 2) * (n - 3)))]
    return (20 * b[3] - 30 * b[2] + 12 * b[1] - b[0]) / (2 * b[1] - b[0])


def stats(x):
    """(11, E) ROWS per pair i < j and (M,) marginal excess kurtosis, on the standardised window."""
    x = (x - x.mean(0)) / x.std(0); T, M = x.shape; iu, ju = np.triu_indices(M, 1)
    r = (x.T @ x / T)[iu, ju]; km = np.mean(x ** 4, 0) - 3; a, b = x[:, iu], x[:, ju]
    c31, c13 = np.mean(a ** 3 * b, 0) - 3 * r, np.mean(a * b ** 3, 0) - 3 * r; c22 = np.mean(a * a * b * b, 0) - 1 - 2 * r * r
    m, dp, dm, d2 = km[iu] + km[ju], 4 * (1 + r) ** 2, 4 * (1 - r) ** 2, 4 * (1 - r * r)
    k40x, k04x, k22x = (4 * c31 + 4 * c13 + 6 * c22) / dp, (6 * c22 - 4 * c31 - 4 * c13) / dm, -2 * c22 / d2
    k40, k04, k22 = k40x + m / dp, k04x + m / dm, k22x + m / d2
    u, v = (a + b) / np.sqrt(2 * (1 + r)), (a - b) / np.sqrt(2 * (1 - r))
    return np.array([r, k40, k22, k04, k40x, k22x, k04x, k40 + 2 * k22 + k04, k40x + 2 * k22x + k04x, lkurt(u), lkurt(v)]), km


def randphase(x):
    """Random-phase expectation (fixed Fourier amplitudes and cross-phases, uniform common phases) of k40, k22, k04 per pair
    (3, E) and of the marginal excess kurtosis (M,). Sum over 0 < f < T/2 of each frequency's joint fourth cumulant."""
    x = (x - x.mean(0)) / x.std(0); T, M = x.shape; iu, ju = np.triu_indices(M, 1)
    r = (x.T @ x / T)[iu, ju]; a, b = x[:, iu], x[:, ju]
    zu, zv, zx = (2 / T * np.fft.rfft(w, axis=0)[1:(T + 1) // 2] for w in ((a + b) / np.sqrt(2 * (1 + r)), (a - b) / np.sqrt(2 * (1 - r)), x))
    R = lambda p, q, s, t: np.real(p * q * np.conj(s) * np.conj(t)); P = lambda p, q: np.real(p * np.conj(q))
    c4 = lambda p, q, s, t: np.sum((R(p, q, s, t) + R(p, s, q, t) + R(p, t, q, s)) / 8 - (P(p, q) * P(s, t) + P(p, s) * P(q, t) + P(p, t) * P(q, s)) / 4, 0)
    mu, mv = np.mean(((a + b) / np.sqrt(2 * (1 + r))) ** 2, 0), np.mean(((a - b) / np.sqrt(2 * (1 - r))) ** 2, 0)
    return np.array([c4(zu, zu, zu, zu) / mu ** 2, c4(zu, zu, zv, zv) / (mu * mv), c4(zv, zv, zv, zv) / mv ** 2]), c4(zx, zx, zx, zx)


def one(args):
    from run import ns
    from mtsspi.surrogates import multivariate_iaaft, phase_randomized
    x, seed, B = args; v = ns(x); re, kre = stats(v); pr, kpr = randphase(v); out = {}
    for nm, draw in (('ia', lambda b: multivariate_iaaft(v, np.random.default_rng(seed if b == 0 else [seed, b]), final='spectrum')[0]),
                     ('ft', lambda b: phase_randomized(v, np.random.default_rng([seed, b, 7])))):
        S1, S2, below, W, K = 0, 0, 0, [], 0
        for b in range(B):
            s, ks = stats(draw(b)); S1, S2, below, K = S1 + s, S2 + s * s, below + (s < re), K + ks; W.append(s.mean(1))
        mu = S1 / B; out[nm] = (mu, np.sqrt(np.maximum(S2 / B - mu * mu, 0) * B / (B - 1)), below.astype(np.uint8), np.array(W), K / B)
    f = lambda q: np.asarray(q, np.float32)
    return (f(re), f(pr), *(f(q) if q.dtype != np.uint8 else q for k in ('ia', 'ft') for q in out[k][:3]), f(out['ia'][3]), f(out['ft'][3]),
            f(np.array([kre, out['ia'][4], out['ft'][4], kpr])))


KEYS = ('real', 'pred', 'ia_mean', 'ia_sd', 'ia_below', 'ft_mean', 'ft_sd', 'ft_below', 'ia_win', 'ft_win', 'kmarg')

if __name__ == '__main__':
    from multiprocessing import Pool
    from edges import sub32, meta, STATES
    d, out, agent, rowmode, nw, B = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
    man = json.load(open(f'{d}/manifest.json')); Wn = np.load(f'{d}/windows.npz')
    rows = [r for r in man['rows'] if r['state'] in STATES and r['agent'] in agent.split(',')]
    S0 = sub32(rows[0]['M']); sub = lambda r: S0 if rowmode == 'first' else sub32(r['M'])
    with Pool(nw) as p: res = p.map(one, [(Wn[r['id']].astype(float)[:, sub(r)], k, B) for k, r in enumerate(rows)], chunksize=1)
    m, _ = meta(rows, man)
    np.savez(out, rows=np.array(ROWS), B=B, **{k: np.array([q[i] for q in res]) for i, k in enumerate(KEYS)}, **m); print('done', out, flush=True)
