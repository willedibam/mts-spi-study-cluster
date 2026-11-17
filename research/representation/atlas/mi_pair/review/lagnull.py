"""Is the drug rise of z(cohmag, icoh) a phase-noise-floor artefact?
Per unit x state: real windows vs a zero-phase Gaussian null (state's mean cross-spectrum with phases removed,
same T, same count). If the null's drug-minus-wake shift matches the real one, the 'lag on strong pairs' reading fails."""
import numpy as np, json, sys
from scipy.signal.windows import dpss

FS, T, NW, K, FMAX = 100, 200, 3, 5, 25
TAP = dpss(T, NW, K)                       # K x T
FR = np.fft.rfftfreq(T, 1 / FS); BAND = (FR > 0) & (FR <= FMAX)


def xspec(x):                              # x: T x M -> F x M x M cross-spectrum (taper-averaged), band only
    X = np.fft.rfft(TAP[:, :, None] * x[None], axis=1)[:, BAND]   # K x F x M
    return np.einsum('kfi,kfj->fij', X, X.conj()) / K


def feats(S):
    d = np.real(np.einsum('fii->fi', S)); C = S / np.sqrt(d[:, :, None] * d[:, None, :])
    iu = np.triu_indices(S.shape[1], 1)
    c2 = (np.abs(C) ** 2).mean(0)[iu]; ic = np.abs(C.imag).mean(0)[iu]
    mag = np.abs(C).mean(0)[iu]; sphi = np.abs(np.sin(np.angle(C))).mean(0)[iu]
    return dict(z=np.corrcoef(c2, ic)[0, 1], zphase=np.corrcoef(mag, sphi)[0, 1], share=ic.mean() / np.abs(C).mean(0)[iu].mean(),
                cv=c2.std() / c2.mean())


def simulate(Smean, n, rng):               # stationary Gaussian windows with spectral matrix Smean on full rfft grid
    F, M = len(FR), Smean.shape[-1]
    L = np.zeros((F, M, M), complex)
    for f in range(F):
        w, V = np.linalg.eigh(Smean[f]); L[f] = V * np.sqrt(np.clip(w, 0, None))
    out = []
    for _ in range(n):
        Z = (rng.standard_normal((F, M)) + 1j * rng.standard_normal((F, M))) / np.sqrt(2)
        Z[0] = Z[0].real * np.sqrt(2); Z[-1] = Z[-1].real * np.sqrt(2)
        out.append(np.fft.irfft(np.einsum('fij,fj->fi', L, Z), n=T, axis=0))
    return out


def fullspec(xs):                          # mean periodogram-type cross-spectrum on full grid (taper-averaged)
    S = 0
    for x in xs:
        X = np.fft.rfft(TAP[:, :, None] * x[None], axis=1)
        S = S + np.einsum('kfi,kfj->fij', X, X.conj()) / K
    return S / len(xs)


if __name__ == '__main__':
    W = np.load('raw/nt/windows.npz'); rows = json.load(open('raw/nt/manifest.json'))['rows']
    sub = np.round(np.arange(int(sys.argv[1]) if len(sys.argv) > 1 else 32) * 125 / ((int(sys.argv[1]) if len(sys.argv) > 1 else 32) - 1)).astype(int)
    rng = np.random.default_rng(0); res = []
    keys = sorted({(r['unit'], r['state'], r['agent']) for r in rows if r['state'] in ('awake_ec', 'awake_eo', 'drug', 'sleep', 'recovery_ec')})
    for u, s, a in keys:
        xs = [W[r['id']][:, sub] for r in rows if r['unit'] == u and r['state'] == s]
        xs = [(x - x.mean(0)) / x.std(0) for x in xs]
        real = [feats(xspec(x)) for x in xs]
        S = fullspec(xs); S0 = S.real.astype(complex)            # zero-phase: keep magnitudes' real part (PSD)
        nul0 = [feats(xspec(x)) for x in simulate(S0, len(xs), rng)]
        nulS = [feats(xspec(x)) for x in simulate(S, len(xs), rng)]   # full-phase stationary null
        m = lambda L, k: float(np.mean([d[k] for d in L]))
        res.append(dict(unit=u, state=s, agent=a, **{f'{n}_{k}': m(L, k) for n, L in (('real', real), ('null0', nul0), ('nullS', nulS)) for k in ('z', 'zphase', 'share', 'cv')}))
        print(a, s, u[:8], ' '.join(f"{k}={res[-1][k]:+.3f}" for k in ('real_z', 'null0_z', 'nullS_z', 'real_zphase', 'null0_zphase')), flush=True)
    json.dump(res, open(f'lagnull_{len(sub)}.json', 'w'))

