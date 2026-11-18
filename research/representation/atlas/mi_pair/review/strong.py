import numpy as np, warnings
warnings.filterwarnings('ignore')
from screen import *
from helpers import pmeans, hedges
def within(name, uf, a, c):
    b = load(name); u = uf(b); y = b['y']; spi = [str(x) for x in b['spi_order']]; K = len(spi); I, J = pair_index(K)
    lv = dstat(unit_diffs(b['spi_mean'].astype(float), u, y, a, c)[0])
    zd = dstat(unit_diffs(b['z'].astype(float), u, y, a, c)[0])
    return spi, lv, zd, I, J
def between(name, a, c):
    b = load(name); g = b['group'].astype(str); y = b['y']; spi = [str(x) for x in b['spi_order']]; I, J = pair_index(len(spi))
    A, _ = pmeans(b['spi_mean'].astype(float), g, y, a); C, _ = pmeans(b['spi_mean'].astype(float), g, y, c)
    Az, _ = pmeans(b['z'].astype(float), g, y, a); Cz, _ = pmeans(b['z'].astype(float), g, y, c)
    return spi, hedges(A, C), hedges(Az, Cz), I, J
ntu = lambda b: np.array([f"{x.split('--')[0][8:12]}-{g}" for x, g in zip(b['row_id'], b['group'])])
R = {'NT anaes': within('nt', ntu, 1, 0), 'prop mod': within('prop', lambda b: b['group'].astype(str), 2, 0),
     'LEMON age': between('lemon', 1, 0), 'AD vs HC': between('adftd', 1, 0)}
spi = R['NT anaes'][0]; fam = np.array([family(s) for s in spi])
tab = {}
for k, (s, lv, zd, I, J) in R.items():
    # z involvement: 90th percentile of |d| over the SPI's 282 coordinates
    inv = np.array([np.nanpercentile(np.abs(np.r_[zd[(I == i)], zd[(J == i)]]), 90) for i in range(len(s))])
    tab[k] = (lv, inv)
for k, (lv, inv) in tab.items():
    print(f'\n## {k}: top SPIs by LEVEL effect (|d| of edge-mean)')
    o = np.argsort(-np.nan_to_num(np.abs(lv)))[:12]
    print('   ' + '; '.join(f'{spi[i][:38]} {lv[i]:+.1f}' for i in o))
    print(f'## {k}: top SPIs by z-involvement (90th pct |d| over its z coords)')
    o = np.argsort(-np.nan_to_num(inv))[:12]
    print('   ' + '; '.join(f'{spi[i][:38]} {inv[i]:.2f}' for i in o))
# family summary of z-involvement ranks
print('\n## family median z-involvement rank (1 = strongest) per cohort')
for k, (lv, inv) in tab.items():
    rk = np.argsort(np.argsort(-np.nan_to_num(inv))) + 1
    print(k, sorted([(f, int(np.median(rk[fam == f]))) for f in np.unique(fam)], key=lambda t: t[1]))
