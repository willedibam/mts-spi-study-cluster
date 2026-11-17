import numpy as np, warnings
warnings.filterwarnings('ignore')
from screen import *
from panel import PANEL, idx
def within_corr(v, x, u, y):
    rs = []
    for uu in np.unique(u):
        for st in np.unique(y):
            m = (u == uu) & (y == st) & np.isfinite(v) & np.isfinite(x)
            if m.sum() > 5 and np.std(x[m]) > 0: rs.append(np.corrcoef(v[m], x[m])[0, 1])
    return np.nanmean(rs)
def resid_effect(v, x, u, y, a, c):
    """state effect after removing a within-unit, within-state pooled slope on covariate x (ANCOVA, unit FE)"""
    m = np.isfinite(v) & np.isfinite(x) & np.isin(y, [a, c])
    vv, xx = v.copy(), x.copy()
    for uu in np.unique(u):
        for st in (a, c):
            s = m & (u == uu) & (y == st); vv[s] -= vv[s].mean(); xx[s] -= xx[s].mean()
    beta = np.sum(vv[m] * xx[m]) / np.sum(xx[m] ** 2)
    adj = v - beta * x
    dd, _ = unit_diffs(adj[:, None], u, y, a, c); dd = dd[:, 0]
    return np.nanmean(dd), np.nanmean(dd) / np.nanstd(dd, ddof=1), int((dd > 0).sum())
for name, uf, a, c in (('nt', lambda b: np.array([f"{x.split('--')[0][8:12]}-{g}" for x, g in zip(b['row_id'], b['group'])]), 1, 0),
                       ('prop', lambda b: b['group'].astype(str), 2, 0)):
    b = load(name); spi = [str(x) for x in b['spi_order']]; u = uf(b); y = b['y']
    ar = np.abs(b['pearson']); spread = np.nanstd(ar, 1); mean = np.nanmean(ar, 1)
    cvs = spread / mean
    print(f'\n### {name}: state change in strength: mean|r| d={dstat(unit_diffs(mean[:,None],u,y,a,c)[0])[0]:+.1f}, SD|r| d={dstat(unit_diffs(spread[:,None],u,y,a,c)[0])[0]:+.1f}')
    print(f"{'pair':50s} r(z,SD|r|) r(z,mean|r|) | raw dz  d | adj(SD|r|) dz d pos")
    for lab, A, B in PANEL:
        v = b['z'][:, idx(spi, A, B)].astype(float)
        dd, _ = unit_diffs(v[:, None], u, y, a, c); dd = dd[:, 0]
        r1, r2 = within_corr(v, spread, u, y), within_corr(v, mean, u, y)
        e = resid_effect(v, spread, u, y, a, c)
        print(f"{lab[:50]:50s} {r1:+.2f} {r2:+.2f} | {np.nanmean(dd):+.3f} {np.nanmean(dd)/np.nanstd(dd,ddof=1):+5.1f} | {e[0]:+.3f} {e[1]:+5.1f} {e[2]:2d}")
