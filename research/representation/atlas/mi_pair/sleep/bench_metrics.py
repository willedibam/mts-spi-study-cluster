"""Conventional metrics per window, copied verbatim from bench.py at 3b0f0259 (branch claude/mi-alt-stats-dec35d, unmerged;
rule ../../preregs/bench-prereg.md there): FS, lz76, ordinal, wsmi, one, COLS. Only this header and the imports are new.
one(x) takes a raw (pre-normal-scoring) window (T, M) at 100 Hz and returns (16 metrics in COLS order, 0.5-s RMS series)."""
import numpy as np
FS = 100
PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
OPP = np.array([PERMS.index(p[::-1]) for p in PERMS])              # pattern of -x: the reversed ranking


def lz76(s):
    """Kaspar-Schuster LZ76 phrase count of a binary sequence."""
    n = len(s); i, k, l, c, km = 0, 1, 1, 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1; break
        else:
            km = max(km, k); i += 1
            if i == l:
                c += 1; l += km
                if l + 1 > n:
                    break
                i, k, km = 0, 1, 1
            else:
                k = 1
    return c


def ordinal(x, tau):
    """(T - 2 tau, M) ordinal-pattern index (order 3) per channel."""
    e = np.stack([x[:-2 * tau], x[tau:-tau], x[2 * tau:]], -1); r = np.argsort(e, -1)
    code = r[..., 0] * 9 + r[..., 1] * 3 + r[..., 2]
    lut = np.full(27, -1); lut[[p[0] * 9 + p[1] * 3 + p[2] for p in PERMS]] = range(6)
    return lut[code]


def wsmi(sym):
    M = sym.shape[1]; iu = np.triu_indices(M, 1); w = np.ones((6, 6)); w[np.arange(6), np.arange(6)] = 0; w[np.arange(6), OPP] = 0
    px = np.stack([np.bincount(sym[:, m], minlength=6) for m in range(M)]) / len(sym); out = []
    for i, j in zip(*iu):
        pj = np.bincount(sym[:, i] * 6 + sym[:, j], minlength=36).reshape(6, 6) / len(sym)
        q = pj > 0; out.append(np.sum((w * pj)[q] * np.log(pj[q] / np.outer(px[i], px[j])[q])) / np.log(6))
    return np.mean(out)


def one(x):
    from scipy.signal import welch, butter, sosfiltfilt, hilbert
    x = x - x.mean(0); T, M = x.shape; iu = np.triu_indices(M, 1)
    f, p = welch(x, fs=FS, nperseg=4 * FS, axis=0); P = p.sum(1); band = lambda a, b: P[(f >= a) & (f <= b)].sum()
    sw = P[(f >= .5) & (f < 4)].sum() / band(.5, 40); alpha = band(8, 13) / band(.5, 40)
    br = np.log(band(30, 40) / band(11, 20)); c = np.cumsum(P * ((f >= .5) & (f <= 40))); sef = f[np.searchsorted(c, .95 * c[-1])]
    m = (f >= .8) & (f <= 32); q = p[m] / p[m].sum(0); se = np.mean(-np.sum(q * np.log(q), 0) / np.log(m.sum()))
    ma = (f >= 20) & (f <= 40); ap = np.polyfit(np.log10(f[ma]), np.log10(P[ma]), 1)[0]
    seg = np.sqrt(np.array([np.mean(x[s:s + 50] ** 2) for s in range(0, T, 50)]))           # 0.5-s RMS, for BSR at score time
    lzc = np.mean([lz76((x[:, j] > np.median(x[:, j])).astype(np.int8).tolist()) for j in range(M)]) * np.log2(T) / T
    sy1 = ordinal(x, 1); pe = np.mean([-np.sum((h := np.bincount(sy1[:, j], minlength=6) / len(sy1))[h > 0] * np.log(h[h > 0])) / np.log(6) for j in range(M)])
    C = np.corrcoef(x.T); absr = np.abs(C[iu]).mean(); eig = np.linalg.eigvalsh(C)[-1] / M
    n = 2 * FS; st = range(0, T - n + 1, n // 2); hw = np.hanning(n)[:, None]
    X = np.stack([np.fft.rfft((x[s:s + n] - x[s:s + n].mean(0)) * hw, axis=0) for s in st]); fr = np.fft.rfftfreq(n, 1 / FS)
    out = []
    for a, b in ((1, 4), (8, 13)):
        Xb = X[:, (fr >= a) & (fr <= b)]; cs = np.einsum('sfi,sfj->sfij', Xb, Xb.conj())
        S = cs.mean(0); d = np.real(np.einsum('fii->fi', S)); coh = np.abs(S) ** 2 / (d[:, :, None] * d[:, None, :])
        im = cs.imag; wp = np.abs(im.mean(0)) / np.maximum(np.abs(im).mean(0), 1e-300)
        out += [coh[:, iu[0], iu[1]].mean(), wp[:, iu[0], iu[1]].mean()]
    ph = np.angle(hilbert(sosfiltfilt(butter(4, [8, 13], btype='band', fs=FS, output='sos'), x, axis=0), axis=0))
    kura = np.abs(np.exp(1j * ph).mean(1)).mean()
    return np.array([sw, alpha, br, sef, se, ap, lzc, pe, absr, eig, *out, kura, wsmi(ordinal(x, 4))], float), seg


COLS = ('SW', 'ALPHA', 'BR', 'SEF95', 'SE', 'AP', 'LZC', 'PE', 'ABSR', 'EIG', 'COHD', 'WPLID', 'COHA', 'WPLIA', 'KURA', 'WSMI')
