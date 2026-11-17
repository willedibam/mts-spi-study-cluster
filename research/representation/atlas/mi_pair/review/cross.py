import numpy as np, collections, warnings
warnings.filterwarnings('ignore')
from screen import *
nt = load('nt'); pr = load('prop'); nts = np.load('nt_screen.npz', allow_pickle=True)
spi = [str(x) for x in pr['spi_order']]; assert spi == [str(x) for x in nt['spi_order']]
K = len(spi); I, J = pair_index(K); fam = np.array([family(x) for x in spi])
pf = nts['pf']
z = pr['z'].astype(float); y = pr['y']; g = pr['group'].astype(str)
print('propofol windows', z.shape, np.unique(y, return_counts=True), len(np.unique(g)))
people = np.unique(g); rng = np.random.default_rng(1); perm = rng.permutation(people); A, B = perm[:10], perm[10:]
dm, un = unit_diffs(z, g, y, 2, 0)
dA = dstat(dm[np.isin(un, A)]); dB = dstat(dm[np.isin(un, B)])
drec, _ = unit_diffs(z, g, y, 2, 3); drd = dstat(drec)
dmild, _ = unit_diffs(z, g, y, 1, 0); dmd = dstat(dmild)
dall = dstat(dm); mz = np.nanmean(dm, 0); npos = (dm > 0).sum(0)
ntd, ntpos, ntm = nts['d'], nts['npos'], nts['mdiff']
rep = (np.sign(dA) == np.sign(dB)) & (np.abs(dA) > .5) & (np.abs(dB) > .5) & (np.sign(drd) == np.sign(dall)) & (np.abs(drd) > .5)
cross = rep & ((ntpos == 6) | (ntpos == 0)) & (np.sign(ntd) == np.sign(dall)) & (np.abs(ntm) > .03) & (np.abs(mz) > .03)
print('propofol replicated:', rep.sum(), ' + NT 6/6 same sign:', cross.sum())
print(collections.Counter(pf[cross]).most_common(40))
np.savez('cross.npz', dall=dall, dA=dA, dB=dB, drd=drd, dmd=dmd, mz=mz, rep=rep, cross=cross)
nc = ~np.isin(pf, ['CORR|CORR', 'CORR|LINPRED', 'LINPRED|LINPRED'])
o = np.argsort(-np.abs(np.nan_to_num(dall * cross * nc)))
print('\nTop cross-species non-corr: A B fam | prop dz d(mod) d(rec) d(mild) base | NT dz d')
for k in o[:70]:
    print(f"{spi[I[k]][:42]:42s} {spi[J[k]][:42]:42s} {pf[k]:15s} | {mz[k]:+.3f} {dall[k]:+5.2f} {drd[k]:+5.2f} {dmd[k]:+5.2f} {np.nanmean(z[y==0,k]):+.2f} | {ntm[k]:+.3f} {ntd[k]:+5.1f}")
