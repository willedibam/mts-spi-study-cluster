import numpy as np, warnings, collections, sys
warnings.filterwarnings('ignore')
from screen import *
from between import pmeans, hedges
name, a, c = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); surr = sys.argv[4:]
b = load(name); spi = [str(x) for x in b['spi_order']]; I, J = pair_index(len(spi))
fam = np.array([family(x) for x in spi]); pf = np.array(['|'.join(sorted((fam[i], fam[j]))) for i, j in zip(I, J)])
z = b['z'].astype(float); g = b['group'].astype(str); y = b['y']
A, _ = pmeans(z, g, y, a); C, _ = pmeans(z, g, y, c); d = hedges(A, C)
rng = np.random.default_rng(3); ia = rng.permutation(len(A)); ic = rng.permutation(len(C))
R = []
for s in surr:
    sz = load(s)['z'].astype(float)
    Ar, _ = pmeans(z - sz, g, y, a); Cr, _ = pmeans(z - sz, g, y, c)
    r = hedges(Ar, Cr); r1 = hedges(Ar[ia[:len(Ar)//2]], Cr[ic[:len(Cr)//2]]); r2 = hedges(Ar[ia[len(Ar)//2:]], Cr[ic[len(Cr)//2:]])
    R.append((r, r1, r2, np.nanmean(Ar, 0) - np.nanmean(Cr, 0)))
# residual itself replicates in both halves, both variants, same sign
ok = np.ones(len(d), bool)
for r, r1, r2, _ in R: ok &= (np.sign(r1) == np.sign(r2)) & (np.abs(r1) > .4) & (np.abs(r2) > .4) & (np.sign(r) == np.sign(R[0][0]))
print('residual replicated in both halves and variants:', ok.sum()); print(collections.Counter(pf[ok]).most_common(20))
# per-SPI involvement count
cnt = collections.Counter([spi[i] for i in I[ok]] + [spi[j] for j in J[ok]])
print('SPIs most involved:', cnt.most_common(20))
o = np.argsort(-np.nan_to_num(np.min([np.abs(r[0]) for r in R], 0) * ok))
print(f"\n{'A':44s} {'B':44s} fam | d_real | d_res (variants) | dz_res | base")
for k in o[:40]:
    print(f"{spi[I[k]][:44]:44s} {spi[J[k]][:44]:44s} {pf[k][:13]:13s} | {d[k]:+.2f} | {' '.join(f'{r[0][k]:+.2f}' for r in R)} | {R[0][3][k]:+.3f} | {np.nanmean(C[:,k]):+.2f}")
