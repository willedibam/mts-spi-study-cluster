"""Surrogate-ensemble fourth-order readouts of the MI pair (rule ../preregs/kens-prereg.md, post hoc, 25 November).

Per window and pair, in the standardised principal axes u = (a + b)/sqrt(2(1 + r)), v = (a - b)/sqrt(2(1 - r)) of kdiff.py:
k40, k22, k04 and Mardia's k40 + 2 k22 + k04, and the L-kurtosis of u and of v. Exact identities on the standardised pair
(ka, kb the marginal excess kurtoses, c the joint cumulants):
  k40 = (ka + kb + 4 c31 + 4 c13 + 6 c22) / (4 (1 + r)^2), k04 = (ka + kb - 4 c31 - 4 c13 + 6 c22) / (4 (1 - r)^2),
  k22 = (ka + kb - 2 c22) / (4 (1 - r^2)).
Nulls, all spectrum-preserving: B joint IAAFT draws (final='spectrum'; draw 0 is the anchor's own, seed k) as drawn ('ia') and
normal-scored ('im': marginals matched to the real window's exactly, so the (ka + kb) terms cancel in the net; spectrum then
approximate), B multivariate phase-randomised (Gaussian) draws ('ft'), and the random-phase expectation from the window's own
Fourier coefficients. Not cross-only terms: the difference kurtosis is bounded, so as r -> 1 the marginal and cross terms of a
spectrum-final surrogate cancel and either alone inverts the marginal mismatch (kens-prereg amendment).
  python kens.py <bank dir> <out.npz> <agent> <rowmode: first|row> <n workers> <B>
"""
import sys, json, numpy as np, warnings
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
warnings.filterwarnings('ignore')
ROWS = ('r', 'k40', 'k22', 'k04', 'M', 'tau_u', 'tau_v')


def lkurt(y):
    """Sample L-kurtosis tau4 of each column (unbiased probability-weighted moments; Gaussian 0.1226)."""
    y = np.sort(y, 0); n = len(y); i = np.arange(n)[:, None]
    b = [np.mean(w * y, 0) for w in (1, i / (n - 1), i * (i - 1) / ((n - 1) * (n - 2)), i * (i - 1) * (i - 2) / ((n - 1) * (n - 2) * (n - 3)))]
    return (20 * b[3] - 30 * b[2] + 12 * b[1] - b[0]) / (2 * b[1] - b[0])


def stats(x):
    """(7, E) ROWS per pair i < j and (M,) marginal excess kurtosis, on the standardised window."""
    x = (x - x.mean(0)) / x.std(0); T, M = x.shape; iu, ju = np.triu_indices(M, 1)
    r = (x.T @ x / T)[iu, ju]; km = np.mean(x ** 4, 0) - 3; a, b = x[:, iu], x[:, ju]
    c31, c13 = np.mean(a ** 3 * b, 0) - 3 * r, np.mean(a * b ** 3, 0) - 3 * r; c22 = np.mean(a * a * b * b, 0) - 1 - 2 * r * r
    m = km[iu] + km[ju]; k40 = (m + 4 * c31 + 4 * c13 + 6 * c22) / (4 * (1 + r) ** 2)
    k04, k22 = (m - 4 * c31 - 4 * c13 + 6 * c22) / (4 * (1 - r) ** 2), (m - 2 * c22) / (4 * (1 - r * r))
    u, v = (a + b) / np.sqrt(2 * (1 + r)), (a - b) / np.sqrt(2 * (1 - r))
    return np.array([r, k40, k22, k04, k40 + 2 * k22 + k04, lkurt(u), lkurt(v)]), km


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
    x, seed, B = args; v = ns(x); re, kre = stats(v); pr, kpr = randphase(v); A = {a: [0, 0, 0, [], 0] for a in ARMS}
    for b in range(B):
        s = multivariate_iaaft(v, np.random.default_rng(seed if b == 0 else [seed, b]), final='spectrum')[0]
        for a, y in (('ia', s), ('im', ns(s)), ('ft', phase_randomized(v, np.random.default_rng([seed, b, 7])))):
            q, kq = stats(y); t = A[a]; t[0] += q; t[1] += q * q; t[2] += q < re; t[3].append(q.mean(1)); t[4] += kq
    f = lambda q: np.asarray(q, np.float32); out = [f(re), f(pr)]
    for a in ARMS:
        mu = A[a][0] / B; out += [f(mu), f(np.sqrt(np.maximum(A[a][1] / B - mu * mu, 0) * B / (B - 1))), A[a][2].astype(np.uint8)]
    return (*out, *(f(np.array(A[a][3])) for a in ARMS), f(np.array([kre, *(A[a][4] / B for a in ARMS), kpr])))


ARMS = ('ia', 'im', 'ft')
KEYS = ('real', 'pred', *(f'{a}_{k}' for a in ARMS for k in ('mean', 'sd', 'below')), *(f'{a}_win' for a in ARMS), 'kmarg')

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
