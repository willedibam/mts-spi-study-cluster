import numpy as np, json, sys, warnings; warnings.filterwarnings('ignore')
from trajplot import *
def bank(p, dataset):
    b = np.load(p); meta = json.loads(str(b['meta']))
    return dict(z=b['z'].astype(float), spi=[str(s) for s in b['spi']], meta=meta, dataset=dataset)
banks = [bank('banks/atlas_fly-short.npz', 'fly')]
try: banks.insert(0, bank('banks/atlas_neurotycho-short.npz', 'nt'))
except Exception as e: print('NeuroTycho bank not ready:', e)
spi = banks[0]['spi']; assert all(b['spi'] == spi for b in banks)
Z = np.vstack([b['z'] for b in banks]); keep = np.isfinite(Z).all(0); Z = np.clip(Z[:, keep], -.995, .995)
print('windows', Z.shape[0], 'common finite coordinates', keep.sum(), 'of', len(keep))
unit, state, cond = [], [], []
lab = {'KTMD': 'ketamine-medetomidine', 'KT': 'ketamine', 'MD': 'medetomidine', 'PF': 'propofol', 'SLP': 'natural sleep'}
for b in banks:
    for m in b['meta']:
        if b['dataset'] == 'fly':
            unit.append(m['unit']); state.append({'awake': 'awake_ec', 'isoflurane': 'drug'}[m['state']]); cond.append('fly isoflurane')
        else:
            unit.append(m['unit']); state.append('drug' if m['state'] == 'sleep' else m['state']); cond.append(lab[m['agent']])
unit, state, cond = map(np.array, (unit, state, cond))
colours = {'ketamine-medetomidine': '#cb181d', 'propofol': '#fd8d3c', 'medetomidine': '#6a51a3', 'ketamine': '#238b45',
           'natural sleep': '#2171b5', 'fly isoflurane': '#8c510a'}
paths, endpoint = {}, {}
for c in colours:
    if c not in cond: continue
    st = set(state[cond == c])
    paths[c] = ['awake_ec', 'drug'] + (['recovery_ec'] if 'recovery_ec' in st else []); endpoint[c] = 'drug'
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7))
P, ev = pca(Z)
for c, col in colours.items():
    if c in cond: axes[0].plot(*P[cond == c].T, 'o', color=col, alpha=.12, ms=2, mec='none')
panel(axes[0], P, ev, unit, state, cond, paths, colours, 'raw z', faint=False)
Zc = centre(Z, unit, state, 'awake_ec'); P, ev = pca(Zc)
# eyes-open null: grey arrow per condition group mean
eo = state == 'awake_eo'
if eo.any():
    m = np.nanmean([P[(unit == u) & eo].mean(0) for u in np.unique(unit[eo])], 0)
    axes[1].annotate('', m, (0, 0), arrowprops=dict(arrowstyle='-|>', color='#969696', lw=1.7, shrinkA=0, shrinkB=0))
panel(axes[1], P, ev, unit, state, cond, paths, colours, "centred on each unit's awake (eyes closed) mean", endpoint=endpoint)
h = [plt.Line2D([], [], color=colours[c], label=c) for c in paths] + ([plt.Line2D([], [], color='#969696', label='eyes open (null)')] if eo.any() else [])
fig.legend(handles=h, loc='outside lower center', ncol=4)
fig.savefig('fig/r13_atlas_short.png'); fig.savefig('fig/r13_atlas_short.svg')
# numbers: shift directions per condition
cos = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)
shift = {c: np.nanmean([Zc[(unit == u) & (state == 'drug')].mean(0) for u in np.unique(unit[cond == c])], 0) for c in paths}
if eo.any(): shift['eyes open'] = np.nanmean([Zc[(unit == u) & eo].mean(0) for u in np.unique(unit[eo])], 0)
ks = list(shift)
print('norm of mean shift:', {k: round(float(np.linalg.norm(v)), 1) for k, v in shift.items()})
print('cosine between mean shifts:'); print('      ' + ' '.join(f'{k[:6]:>7s}' for k in ks))
for a in ks: print(f'{a[:6]:6s}' + ' '.join(f'{cos(shift[a], shift[b]):+7.2f}' for b in ks))
