import numpy as np, warnings, collections
warnings.filterwarnings('ignore')
from screen import *
c = np.load('cross.npz'); nts = np.load('nt_screen.npz', allow_pickle=True); pf = nts['pf']
nt = load('nt'); pr = load('prop'); spi = [str(x) for x in pr['spi_order']]; I, J = pair_index(len(spi))
ntu = np.array([f"{x.split('--')[0][8:12]}-{g}" for x, g in zip(nt['row_id'], nt['group'])])
def resid(bank, surr, u, a, cc):
    out = []
    for s in surr:
        sb = load(s)
        rd, _ = unit_diffs(bank['z'].astype(float) - sb['z'].astype(float), u, bank['y'], a, cc)
        sd, _ = unit_diffs(sb['z'].astype(float), u, bank['y'], a, cc)
        out.append((dstat(rd), np.nanmean(sd, 0), (rd > 0).sum(0), np.isfinite(rd).sum(0)))
    return out
R_nt = resid(nt, ['nt_surr_m200', 'nt_surr_spec'], ntu, 1, 0)
R_pr = resid(pr, ['prop_surr_m', 'prop_surr_spec'], pr['group'].astype(str), 2, 0)
cross = c['cross']; sgn = np.sign(c['dall'])
ok = cross.copy()
for d, _, _, _ in R_nt + R_pr: ok &= (np.sign(d) == sgn) & (np.abs(d) > .5)
print('cross-species replicated:', cross.sum(), ' of which residual beyond both surrogates in both species (|d|>.5 same sign):', ok.sum())
print(collections.Counter(pf[ok]).most_common())
o = np.argsort(-np.abs(np.nan_to_num(c['dall'] * ok)))
print(f"\n{'A':44s} {'B':44s} fam | prop dz [surr m, spec] dres m/spec | NT dz [surr] dres")
for k in o[:40]:
    print(f"{spi[I[k]][:44]:44s} {spi[J[k]][:44]:44s} {pf[k][:13]:13s} | {c['mz'][k]:+.3f} [{R_pr[0][1][k]:+.3f} {R_pr[1][1][k]:+.3f}] {R_pr[0][0][k]:+.1f}/{R_pr[1][0][k]:+.1f} | {nts['mdiff'][k]:+.3f} [{R_nt[0][1][k]:+.3f} {R_nt[1][1][k]:+.3f}] {R_nt[0][0][k]:+.1f}/{R_nt[1][0][k]:+.1f}")
