"""EXPLORATORY (not in the prereg): isoflurane residual predicted at the mouse's own awake burst index, minus awake mean.
Per mouse: OLS res_iso ~ BI; Δ_matched = fit(median awake BI) − mean awake residual. Also window-level pooled fit with mouse intercepts."""
import sys, json, numpy as np
from scipy.stats import binom
Z = np.load(sys.argv[1]); rows = json.loads(str(Z['rows'])); res = Z['ns'][:, 0] - Z['ns'][:, 1:].mean(1)
sess = np.array([r['session'] for r in rows]); st = np.array([r['state'] for r in rows]); bi = np.array([r['burst'] for r in rows])
P = [s for s in dict.fromkeys(sess) if s != 'sub-569073_ses-20210416']; out = []
for s in P:
    i, a = (sess == s) & (st == 'iso'), (sess == s) & (st == 'awake')
    b = np.polyfit(bi[i], res[i], 1); out.append((np.polyval(b, np.median(bi[a])) - res[a].mean(), b[0], np.median(bi[a]), bi[i].max()))
o = np.array(out); n = len(o); neg = int((o[:, 0] < 0).sum())
boot = np.random.default_rng(0).choice(o[:, 0], (10000, n)).mean(1)
print(f'Δ at awake BI: mean {o[:, 0].mean():+.4f} [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}] = {o[:, 0].mean() / .0026:+.1f} null SD; negative {neg}/{n} (p={binom.sf(neg - 1, n, .5):.3f})')
print(f'slope d res / d BI: median {np.median(o[:, 1]):+.3f}, positive {(o[:, 1] > 0).sum()}/{n}; extrapolation: max iso BI {np.median(o[:, 3]):.2f} vs awake BI {np.median(o[:, 2]):.2f} (median over mice)')
print('per mouse Δ_matched', np.round(o[:, 0], 4).tolist())
