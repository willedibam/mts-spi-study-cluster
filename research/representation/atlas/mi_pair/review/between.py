import numpy as np, warnings, collections, sys
warnings.filterwarnings('ignore')
from screen import *
from panel import PANEL, idx
def pmeans(v, g, y, cls):
    ps = [p for p in np.unique(g) if (y[g == p] == cls).all()]
    return np.array([np.nanmean(v[g == p], 0) for p in ps]), np.array(ps)
def hedges(a, b):
    s = np.sqrt(((len(a)-1)*np.nanvar(a, 0, ddof=1) + (len(b)-1)*np.nanvar(b, 0, ddof=1)) / (len(a)+len(b)-2))
    return (np.nanmean(a, 0) - np.nanmean(b, 0)) / s
name, a, c = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); surr = sys.argv[4:]
b = load(name); spi = [str(x) for x in b['spi_order']]; I, J = pair_index(len(spi))
fam = np.array([family(x) for x in spi]); pf = np.array(['|'.join(sorted((fam[i], fam[j]))) for i, j in zip(I, J)])
z = b['z'].astype(float); g = b['group'].astype(str); y = b['y']
print(name, 'classes', np.unique(y, return_counts=True))
A, pa = pmeans(z, g, y, a); C, pc = pmeans(z, g, y, c)
d = hedges(A, C)
rng = np.random.default_rng(3); ia = rng.permutation(len(A)); ic = rng.permutation(len(C))
d1 = hedges(A[ia[:len(A)//2]], C[ic[:len(C)//2]]); d2 = hedges(A[ia[len(A)//2:]], C[ic[len(C)//2:]])
rep = (np.sign(d1) == np.sign(d2)) & (np.abs(d1) > .4) & (np.abs(d2) > .4)
res = []
for s in surr:
    sz = load(s)['z'].astype(float)
    Ar, _ = pmeans(z - sz, g, y, a); Cr, _ = pmeans(z - sz, g, y, c); res.append(hedges(Ar, Cr))
    As, _ = pmeans(sz, g, y, a); Cs, _ = pmeans(sz, g, y, c); res.append(hedges(As, Cs))
print(f'{len(A)} vs {len(C)} participants; split-half replicated coords (|d|>.4 both halves):', rep.sum())
print('by family pair:', collections.Counter(pf[rep]).most_common(15))
if surr:
    beyond = rep & np.all([(np.sign(r) == np.sign(d)) & (np.abs(r) > .3) for r in res[0::2]], 0)
    print('replicated and residual beyond every surrogate variant (|d|>.3):', beyond.sum(), collections.Counter(pf[beyond]).most_common(10))
nc = ~np.isin(pf, ['CORR|CORR', 'CORR|LINPRED', 'LINPRED|LINPRED'])
o = np.argsort(-np.abs(np.nan_to_num(d * rep * nc)))
print(f"\ntop non-corr replicated: d (halves) | " + ' '.join(f'{s[-6:]}: d_res d_surr' for s in surr))
for k in o[:25]:
    rr = ' '.join(f"{res[2*i][k]:+.2f} {res[2*i+1][k]:+.2f}" for i in range(len(surr)))
    print(f"{spi[I[k]][:42]:42s} {spi[J[k]][:42]:42s} {pf[k][:14]:14s} {d[k]:+.2f} ({d1[k]:+.2f},{d2[k]:+.2f}) dz={np.nanmean(A[:,k])-np.nanmean(C[:,k]):+.3f} | {rr}")
print('\nPANEL:')
for lab, X, Y in PANEL:
    k = idx(spi, X, Y)
    rr = ' '.join(f"{res[2*i][k]:+.2f} {res[2*i+1][k]:+.2f}" for i in range(len(surr)))
    print(f"{lab[:50]:50s} base {np.nanmean(C[:,k]):+.2f} d {d[k]:+.2f} ({d1[k]:+.2f},{d2[k]:+.2f}) | {rr}")
