import numpy as np, collections
from screen import *
b = load('nt'); s = {k: load(k) for k in ('nt_surr_m200',)}
spi = [str(x) for x in b['spi_order']]; K = len(spi); I, J = pair_index(K)
fam = np.array([family(x) for x in spi])
pf = np.array(['|'.join(sorted((fam[i], fam[j]))) for i, j in zip(I, J)])
unit = np.array([f"{x.split('--')[0][8:12]}-{g}" for x, g in zip(b['row_id'], b['group'])])
y = b['y']; z = b['z'].astype(float)
diff, un = unit_diffs(z, unit, y, 1, 0)
d = dstat(diff); npos = (diff > 0).sum(0); nfin = np.isfinite(diff).sum(0)
kt = np.array([u.startswith('ktmd') for u in un])
sign_all = (npos == 6) | (npos == 0)
# surrogate
rd, _ = unit_diffs(z - s['nt_surr_m200']['z'].astype(float), unit, y, 1, 0)
dr = dstat(rd); rpos = (rd > 0).sum(0)
sd_, _ = unit_diffs(s['nt_surr_m200']['z'].astype(float), unit, y, 1, 0); ds = dstat(sd_)
ok = (nfin == 6) & sign_all
print('coords with 6/6 sign:', ok.sum(), 'of', np.isfinite(d).sum())
print('pair-family counts among 6/6 & |d|>3:')
sel = ok & (np.abs(d) > 3)
print(collections.Counter(pf[sel]).most_common(30))
print('\n6/6 & residual 6/6 same sign & |dr|>1.5, by pair family:')
sel2 = ok & ((rpos == 6) | (rpos == 0)) & (np.sign(dr) == np.sign(d)) & (np.abs(dr) > 1.5)
print(collections.Counter(pf[sel2]).most_common(30))
nc = ~np.isin(pf, ['CORR|CORR', 'CORR|LINPRED', 'LINPRED|LINPRED'])
o = np.argsort(-np.abs(np.nan_to_num(dr * sel2 * nc)))
print('\nTop non-corr pairs with beyond-IAAFT residual (mean diff real, d real, d surr, d resid):')
for k in o[:45]:
    print(f"{spi[I[k]]:48s} {spi[J[k]]:48s} {pf[k]:16s} dz={np.nanmean(diff[:,k]):+.3f} d={d[k]:+6.1f} ds={ds[k]:+6.1f} dr={dr[k]:+5.1f} awake={np.nanmean(z[y==0,k]):+.2f}")
np.savez('nt_screen.npz', d=d, dr=dr, ds=ds, npos=npos, rpos=rpos, pf=pf, mdiff=np.nanmean(diff,0))
