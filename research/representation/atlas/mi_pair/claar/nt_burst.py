"""Burst index (median over channels of 1.4826·MAD/SD) for the macaque windows behind zns_*, and its relation to the residual."""
import glob, json, numpy as np
from scipy.stats import spearmanr
burst = lambda c: float(np.median(1.4826 * np.median(np.abs(c - np.median(c, 0)), 0) / c.std(0)))
S = lambda m: np.round(np.arange(32) * (m - 1) / 31).astype(int)
rows = []
for stage, d in (('A', 'raw/nt10'), ('B', 'raw/nt10b')):
    W = np.load(f'{d}/windows.npz'); man = {r['id']: r for r in json.load(open(f'{d}/manifest.json'))['rows']}
    for f in sorted(glob.glob(f'zns_{stage}_*.npz')):
        Z = np.load(f)
        for k, i in enumerate(Z['id']):
            w = W[i].astype(float); rows.append((str(Z['unit'][k]), str(Z['state'][k]), str(Z['agent'][k]), burst(w[:, S(man[i]['M'])]), burst(w), Z['V'][k, 1, 0] - Z['V'][k, 1, 1]))
u, st, ag = (np.array([r[j] for r in rows]) for j in range(3)); bi, bia, res = (np.array([r[j] for r in rows], float) for j in (3, 4, 5))
print('corr BI(32 ch) vs BI(all ch)', round(np.corrcoef(bi, bia)[0, 1], 3))
for a in ('KT', 'KTMD', 'MD', 'PF', 'SLP'):
    s = 'sleep' if a == 'SLP' else 'drug'; m = ag == a
    print(f'{a:5s} BI awake_ec {np.median(bi[m & (st == "awake_ec")]):.2f} [{np.percentile(bi[m & (st == "awake_ec")], 5):.2f}] | {s} {np.median(bi[m & (st == s)]):.2f} '
          f'[5-95% {np.percentile(bi[m & (st == s)], 5):.2f}-{np.percentile(bi[m & (st == s)], 95):.2f}] | frac {s} BI≥.85 {np.mean(bi[m & (st == s)] >= .85):.2f}')
    rho, dd, d85 = [], [], []
    for x in np.unique(u[m]):
        dm, am = (u == x) & (st == s), (u == x) & (st == 'awake_ec')
        if dm.sum() >= 5: rho.append(spearmanr(bi[dm], res[dm])[0])
        dd.append(res[dm].mean() - res[am].mean())
        k = dm & (bi >= .85)
        if k.sum() >= 3: d85.append(res[k].mean() - res[am].mean())
    print(f'      within-{s} Spearman(res, BI) median {np.nanmedian(rho):+.2f} pos {np.sum(np.array(rho) > 0)}/{len(rho)} | Δ all {np.mean(dd):+.4f} | Δ BI≥.85 only {np.mean(d85) if d85 else np.nan:+.4f} ({len(d85)} units, {sum(x < 0 for x in d85)} neg)')
print('\nPer-unit regression (as claar_bimatch): drug residual ~ BI, evaluated at the unit\'s median awake_ec BI, minus awake_ec mean')
for a in ('KT', 'KTMD', 'MD', 'PF', 'SLP'):
    s = 'sleep' if a == 'SLP' else 'drug'; o = []
    for x in np.unique(u[ag == a]):
        dm, am = (u == x) & (st == s), (u == x) & (st == 'awake_ec')
        b = np.polyfit(bi[dm], res[dm], 1); o.append((np.polyval(b, np.median(bi[am])) - res[am].mean(), res[dm].mean() - res[am].mean(), b[0], np.median(bi[am]) - np.median(bi[dm])))
    o = np.array(o)
    print(f'{a:5s} n={len(o):2d} Δ raw {o[:, 1].mean():+.4f} → Δ at awake BI {o[:, 0].mean():+.4f} ({(o[:, 0] < 0).sum()}/{len(o)} neg) | slope median {np.median(o[:, 2]):+.3f} | BI gap awake−drug {np.median(o[:, 3]):+.3f}')
