import numpy as np, json, collections, warnings; warnings.filterwarnings('ignore')
from screen import family
from trajplot import centre
b = np.load('banks/atlas_neurotycho-short.npz'); meta = json.loads(str(b['meta'])); spi = [str(s) for s in b['spi']]; K = len(spi)
Z = b['z'].astype(float); keep = np.isfinite(Z).all(0); I, J = np.triu_indices(K, 1); I, J = I[keep], J[keep]; Z = np.clip(Z[:, keep], -.995, .995)
lv = b['level'].astype(float)
unit = np.array([m['unit'] for m in meta]); agent = np.array([m['agent'] for m in meta])
state = np.array(['drug' if m['state'] == 'sleep' else m['state'] for m in meta])
Zc = centre(Z, unit, state, 'awake_ec')
drug_units = [u for u in np.unique(unit) if agent[unit == u][0] != 'SLP']
D = np.array([Zc[(unit == u) & (state == 'drug')].mean(0) for u in drug_units])
axis = D.mean(0); axis /= np.linalg.norm(axis)
fam = np.array([family(s) for s in spi]); pf = np.array(['|'.join(sorted((fam[i], fam[j]))) for i, j in zip(I, J)])
w = axis ** 2
print('share of the shared axis (squared loading) by family pair:')
tot = collections.defaultdict(float)
for p, x in zip(pf, w): tot[p] += x
for p, x in sorted(tot.items(), key=lambda t: -t[1])[:10]: print(f'  {p:18s} {x:.2f}  (pairs {np.sum(pf == p)})')
o = np.argsort(-np.abs(axis))[:12]
print('top coordinates:'); [print(f'  {spi[I[k]][:40]:40s} {spi[J[k]][:40]:40s} {axis[k]:+.3f}') for k in o]
# strength: score each window on the axis; within-unit correlation with mean r^2 level and slow-power proxy (cohmag low/high level)
score = Zc @ axis
s2 = lv[:, spi.index('cov-sq_EmpiricalCovariance')]
lo = lv[:, spi.index('cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-25')]; hi = lv[:, spi.index('cohmag_multitaper_mean_fs-1_fmin-0-25_fmax-0-5')]
def within(a, c):
    rs = []
    for u in np.unique(unit):
        for st in np.unique(state[unit == u]):
            m = (unit == u) & (state == st)
            if m.sum() > 5: rs.append(np.corrcoef(a[m], c[m])[0, 1])
    return np.nanmean(rs)
print('within unit-state corr(score, mean r^2):', round(within(score, s2), 2), '; corr(score, low-band coherence):', round(within(score, lo), 2), '; corr(score, high-band coherence):', round(within(score, hi), 2))
# between states: does the mean-level change predict score change across units?
ds = np.array([score[(unit == u) & (state == 'drug')].mean() for u in drug_units])
dl = np.array([s2[(unit == u) & (state == 'drug')].mean() - s2[(unit == u) & (state == 'awake_ec')].mean() for u in drug_units])
print('across drug units: corr(shift score, change in mean r^2)', round(np.corrcoef(ds, dl)[0, 1], 2), '; mean r^2 change', dl.round(3))
# level-picture comparison: do the K mean levels shift in a shared direction too?
L = lv.copy(); L = (L - np.nanmean(L, 0)) / np.nanstd(L, 0); L = np.nan_to_num(L)
Lc = centre(L, unit, state, 'awake_ec')
LD = np.array([Lc[(unit == u) & (state == 'drug')].mean(0) for u in drug_units])
cos = lambda a, c: a @ c / np.linalg.norm(a) / np.linalg.norm(c)
print('consistency of the shared direction, leave-one-out cosine median: z', round(np.median([cos(D[i], np.delete(D, i, 0).mean(0)) for i in range(len(D))]), 2),
      '| mean-SPI levels', round(np.median([cos(LD[i], np.delete(LD, i, 0).mean(0)) for i in range(len(LD))]), 2))
# eyes-open and sleep on the axis, in drug-shift units
eo = np.array([score[(unit == u) & (state == 'awake_eo')].mean() for u in np.unique(unit) if ((unit == u) & (state == 'awake_eo')).any()])
sl = np.array([score[(unit == u) & (state == 'drug')].mean() for u in np.unique(unit) if agent[unit == u][0] == 'SLP'])
print('score on shared axis: drug units', ds.round(1), '| sleep', sl.round(1), '| eyes open', eo.round(1))
