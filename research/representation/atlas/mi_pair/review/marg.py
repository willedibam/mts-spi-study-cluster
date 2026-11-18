"""Per-channel marginal shape by state (10-s, M=32 columns as in decide10b): excess kurtosis and |skew|, session means."""
import numpy as np, json
from scipy.stats import kurtosis, skew
for d in ('raw/nt10', 'raw/nt10b'):
    R = [r for r in json.load(open(f'{d}/manifest.json'))['rows'] if r['state'] in ('awake_ec', 'awake_eo', 'drug', 'sleep')]
    W = np.load(f'{d}/windows.npz'); sub = np.round(np.arange(32) * (R[0]['M'] - 1) / 31).astype(int)
    acc = {}
    for r in R:
        x = W[r['id']].astype(float)[:, sub]; k = kurtosis(x, 0); s = np.abs(skew(x, 0))
        acc.setdefault((r['agent'], r['animal'], r['unit'][:8], r['state']), []).append((np.median(k), np.percentile(k, 90), np.median(s)))
    for key in sorted(acc):
        a = np.mean(acc[key], 0); print(d[-5:], *key, f'kurt med {a[0]:+.2f} p90 {a[1]:+.2f} |skew| {a[2]:.2f}')
