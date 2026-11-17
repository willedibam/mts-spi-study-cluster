import numpy as np, warnings
warnings.filterwarnings('ignore')
from screen import *
from helpers import pmeans, hedges
from panel import idx
b = load('adftd'); spi = [str(x) for x in b['spi_order']]; g = b['group'].astype(str); y = b['y']
sm, ss = load('adftd_surr_m'), load('adftd_surr_spec')
ar = np.abs(b['pearson']); cov = np.c_[np.nanstd(ar, 1), np.nanmean(ar, 1)]
pairs = [('mi_gaussian', 'mi_kraskov_NN-4_DCE-AUTO'), ('tlmi_gaussian', 'tlmi_kraskov_NN-4_DCE-AUTO'), ('mi_gaussian', 'ids'), ('cov-sq_EmpiricalCovariance', 'dcorr'), ('cov-sq_EmpiricalCovariance', 'spearmanr-sq')]
for lab, a, c in (('AD vs HC', 1, 0), ('FTD vs HC', 2, 0), ('AD vs FTD', 1, 2)):
    Ca, _ = pmeans(cov, g, y, a); Cc, _ = pmeans(cov, g, y, c)
    print(f'\n## {lab}: SD|r| d {hedges(Ca[:,0],Cc[:,0]):+.2f}, mean|r| d {hedges(Ca[:,1],Cc[:,1]):+.2f}')
    for A, B in pairs:
        k = idx(spi, A, B); v = b['z'][:, k].astype(float)
        out = []
        for nm, s in (('real', 0), ('m', sm), ('spec', ss)):
            w = v if nm == 'real' else v - s['z'][:, k].astype(float)
            Pa, _ = pmeans(w, g, y, a); Pc, _ = pmeans(w, g, y, c)
            # adjust participant means for SD|r| and mean|r| (pooled within-group slopes)
            X = np.r_[Ca, Cc]; W = np.r_[Pa, Pc]; grp = np.r_[np.ones(len(Pa)), np.zeros(len(Pc))]
            Xc = X - np.where(grp[:, None] == 1, Ca.mean(0), Cc.mean(0)); Wc = W - np.where(grp == 1, Pa.mean(), Pc.mean())
            beta = np.linalg.lstsq(Xc, Wc, rcond=None)[0]; Wa = W - X @ beta
            out.append(f"{nm}: dz {Pa.mean()-Pc.mean():+.4f} d {hedges(Pa,Pc):+.2f} adj d {hedges(Wa[grp==1],Wa[grp==0]):+.2f}")
        print(f"  {A[:26]:26s} {B[:26]:26s} base {np.nanmean(v[y==c]):.3f} | " + ' | '.join(out))
