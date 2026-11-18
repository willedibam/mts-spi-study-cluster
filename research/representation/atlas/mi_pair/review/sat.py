import numpy as np, warnings, collections
warnings.filterwarnings('ignore')
from screen import *
for name, ref in (('nt', 0), ('prop', 0), ('lemon', 0), ('adftd', 0)):
    try: b = load(name)
    except Exception: continue
    spi = [str(x) for x in b['spi_order']]; K = len(spi); I, J = pair_index(K)
    fam = np.array([family(x) for x in spi])
    zb = np.nanmean(np.abs(b['z'][b['y'] == ref].astype(float)), 0)
    sym_time = np.isin(fam, ['CORR', 'PREC', 'NPINFO', 'NPDEP', 'SHAPE', 'LINPRED'])
    m = sym_time[I] & sym_time[J] & (fam[I] != fam[J]) & np.isfinite(zb)
    print(f"{name:6s} M={b['M'][0]} T={b['T'][0]}  cross-family time-domain pairs: median |z| {np.median(zb[m]):.2f}, share |z|>.9 {np.mean(zb[m]>.9):.2f}, >.95 {np.mean(zb[m]>.95):.2f};  all coords median {np.nanmedian(zb):.2f}")
    for F, G in (('CORR', 'NPINFO'), ('CORR', 'NPDEP'), ('CORR', 'SHAPE'), ('CORR', 'PREC'), ('CORR', 'SPEC'), ('SPEC', 'SPECDIR'), ('CORR','ENV')):
        mm = ((fam[I] == F) & (fam[J] == G)) | ((fam[I] == G) & (fam[J] == F))
        print(f"    {F}-{G}: median |z| {np.nanmedian(zb[mm]):.2f}  share>.9 {np.nanmean(zb[mm]>.9):.2f}")
