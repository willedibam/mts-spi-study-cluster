import numpy as np, warnings
warnings.filterwarnings('ignore')
from scipy.signal import welch
from scipy.stats import kurtosis
for name, fs in (('nt', 250.0), ('prop', 100.0)):
    b = np.load(f'banks/{name}_x2.npz'); x = b['x'].astype(float); sub = b['sub'].astype(float); names = [str(s) for s in b['sub_names']]; y = b['y']
    n = len(y); print(f'\n### {name}: x {x.shape}')
    if x.shape[1] < x.shape[2]: x = x.transpose(0, 2, 1)  # -> (n, T, M)
    T, M = x.shape[1], x.shape[2]
    mask = ~np.eye(M, dtype=bool); I, J = np.nonzero(mask)
    X = np.zeros((len(I), 2 * M)); X[np.arange(len(I)), I] = 1; X[np.arange(len(J)), M + J] = 1; Pinv = np.linalg.pinv(X)
    feats = {}
    xs = (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
    feats['rho1'] = np.mean(xs[:, 1:] * xs[:, :-1], 1)
    f, P = welch(xs, fs=fs, nperseg=min(256, T), axis=1)
    feats['centroid'] = (f[None, :, None] * P).sum(1) / P.sum(1)
    feats['slow<4Hz'] = P[:, f < 4].sum(1) / P.sum(1)
    feats['kurt'] = kurtosis(xs, axis=1)
    # integrated autocorrelation time via ACF up to 0.5 s
    L = int(0.5 * fs); acf = np.stack([np.mean(xs[:, k:] * xs[:, :T - k], 1) for k in range(1, L)], 1)
    feats['tau_int'] = 1 + 2 * acf.sum(1)
    print(f"{'SPI':48s} R2add | corr of source effect / target effect with channel features (pooled within window)")
    for k, s in enumerate(names):
        src, tgt, r2 = [], [], []
        for w in range(n):
            v = sub[w, k]
            if not np.isfinite(v).all(): continue
            coef = Pinv @ (v - v.mean()); src.append(coef[:M]); tgt.append(coef[M:])
            fit = X @ coef; r2.append(1 - ((v - v.mean() - fit) ** 2).sum() / ((v - v.mean()) ** 2).sum())
        src, tgt = np.array(src), np.array(tgt)
        line = []
        for fn, fv in feats.items():
            fv = fv[:len(src)] if len(src) == n else fv
            def wc(a): return np.nanmean([np.corrcoef(a[w], fv[w])[0, 1] for w in range(len(a))])
            line.append(f"{fn} {wc(src):+.2f}/{wc(tgt):+.2f}")
        print(f"{s[:48]:48s} {np.nanmedian(r2):.2f} | " + '  '.join(line))
