import numpy as np, json, warnings; warnings.filterwarnings('ignore')
import style, matplotlib.pyplot as plt
from trajplot import centre
def bank(p, ds):
    b = np.load(p); return dict(z=b['z'].astype(float), meta=json.loads(str(b['meta'])), ds=ds)
B = [bank('banks/atlas_neurotycho-short.npz', 'nt'), bank('banks/atlas_fly-short.npz', 'fly')]
Z = np.vstack([b['z'] for b in B]); keep = np.isfinite(Z).all(0); Z = np.clip(Z[:, keep], -.995, .995)
lab = {'KTMD': 'ketamine-medetomidine', 'KT': 'ketamine', 'MD': 'medetomidine', 'PF': 'propofol', 'SLP': 'natural sleep'}
unit, state, cond = [], [], []
for b in B:
    for m in b['meta']:
        unit.append(m['unit'])
        if b['ds'] == 'fly': state.append({'awake': 'awake_ec', 'isoflurane': 'drug'}[m['state']]); cond.append('fly isoflurane')
        else: state.append('drug' if m['state'] == 'sleep' else m['state']); cond.append(lab[m['agent']])
unit, state, cond = map(np.array, (unit, state, cond))
colours = {'ketamine-medetomidine': '#cb181d', 'propofol': '#fd8d3c', 'medetomidine': '#6a51a3', 'ketamine': '#238b45', 'natural sleep': '#2171b5', 'fly isoflurane': '#8c510a'}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7))
# raw PCA, windows only
mu = Z.mean(0); _, s, Vt = np.linalg.svd(Z - mu, full_matrices=False); ev = s ** 2 / (s ** 2).sum(); P = (Z - mu) @ Vt[:2].T
ax = axes[0]
for c, col in colours.items(): ax.plot(*P[cond == c].T, 'o', color=col, alpha=.25, ms=2, mec='none')
ax.set_xlabel(f'PC1 ({ev[0]:.0%})'); ax.set_ylabel(f'PC2 ({ev[1]:.0%})'); ax.set_title('raw z (all windows)'); ax.set_box_aspect(1)
# centred: unit-state means relative to awake (eyes closed); PCA axes from these shifts, no re-centring
Zc = centre(Z, unit, state, 'awake_ec')
shifts, sc, su = [], [], []
for u in np.unique(unit):
    for st in ('drug', 'awake_eo', 'recovery_ec'):
        m = (unit == u) & (state == st)
        if m.any(): shifts.append(Zc[m].mean(0)); sc.append(cond[unit == u][0] if st == 'drug' else st); su.append(st)
shifts, sc, su = np.array(shifts), np.array(sc), np.array(su)
fitm = su == 'drug'
_, s2, V2 = np.linalg.svd(shifts[fitm], full_matrices=False); ev2 = s2 ** 2 / (s2 ** 2).sum(); Q = shifts @ V2[:2].T
ax = axes[1]
for c, col in colours.items():
    q = Q[sc == c]
    if not len(q): continue
    ax.plot(*q.T, 'o', color=col, alpha=.35, ms=3, mec='none')
    m = q.mean(0); ax.annotate('', m, (0, 0), arrowprops=dict(arrowstyle='-|>', color=col, lw=1.7, shrinkA=0, shrinkB=0))
m = Q[sc == 'awake_eo'].mean(0); ax.annotate('', m, (0, 0), arrowprops=dict(arrowstyle='-|>', color='#969696', lw=1.7, shrinkA=0, shrinkB=0))
ax.plot(0, 0, 'k+', ms=7)
ax.set_xlabel('shift PC1'); ax.set_ylabel('shift PC2'); ax.set_title('change from own awake (eyes closed)'); ax.set_box_aspect(1)
h = [plt.Line2D([], [], color=c, label=k) for k, c in colours.items()] + [plt.Line2D([], [], color='#969696', label='eyes open (null)')]
fig.legend(handles=h, loc='outside lower center', ncol=4)
fig.savefig('fig/r13_atlas_short.png'); fig.savefig('fig/r13_atlas_short.svg')
print('raw PCA: share of PC1 variance between datasets', round(np.var([P[cond == 'fly isoflurane', 0].mean(), P[cond != 'fly isoflurane', 0].mean()]) , 1))
# recovery: fraction of drug shift remaining, per condition (projection of recovery shift on drug shift)
for c in colours:
    us = np.unique(unit[(cond == c)])
    r = [np.dot(Zc[(unit == u) & (state == 'recovery_ec')].mean(0), Zc[(unit == u) & (state == 'drug')].mean(0)) / np.linalg.norm(Zc[(unit == u) & (state == 'drug')].mean(0)) ** 2
         for u in us if ((unit == u) & (state == 'recovery_ec')).any()]
    if r: print(f'{c:22s} recovery keeps {np.mean(r):.0%} of the drug shift (units {len(r)})')
# per-unit consistency: cosine of each unit's shift with the leave-one-out mean of the macaque drug shifts
mac = fitm & (sc != 'fly isoflurane') & (sc != 'natural sleep')
cos = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)
idx = np.flatnonzero(mac)
loo = [cos(shifts[i], shifts[idx[idx != i]].mean(0)) for i in idx]
print('macaque drug units: cosine with leave-one-out mean drug shift: median', round(np.median(loo), 2), 'min', round(min(loo), 2))
for c in ('natural sleep', 'fly isoflurane'):
    print(c, 'units vs macaque drug mean: median cos', round(np.median([cos(shifts[i], shifts[idx].mean(0)) for i in np.flatnonzero(fitm & (sc == c))]), 2))
print('eyes-open units vs macaque drug mean: median cos', round(np.median([cos(shifts[i], shifts[idx].mean(0)) for i in np.flatnonzero(su == 'awake_eo')]), 2))
print('shift PCs explained', ev2[:3].round(2))
