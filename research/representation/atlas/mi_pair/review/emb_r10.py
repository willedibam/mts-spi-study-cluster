import numpy as np, warnings; warnings.filterwarnings('ignore')
import style, matplotlib.pyplot as plt
from screen import load
nt, pr = load('nt'), load('prop')
Z = np.vstack([nt['z'], pr['z']]).astype(float)
keep = np.isfinite(Z).all(0); Z = np.clip(Z[:, keep], -.995, .995)
sp = np.r_[np.zeros(len(nt['y'])), np.ones(len(pr['y']))]
ntu = np.array([f"{x.split('--')[0][8:12]}-{g}" for x, g in zip(nt['row_id'], nt['group'])])
unit = np.r_[ntu, pr['group'].astype(str)]
state = np.r_[np.where(nt['y'] == 1, 'anaesthesia', 'awake'), np.array(['awake', 'mild', 'moderate', 'recovery'])[pr['y']]]
ref = 'awake'
# within-unit centring on the unit's reference (awake / baseline) mean
Zc = Z.copy()
for u in np.unique(unit):
    m = unit == u; Zc[m] -= Z[m & (state == ref)].mean(0)
def pca(X):
    mu = X.mean(0); U, s, Vt = np.linalg.svd(X - mu, full_matrices=False); return Vt[:2], s ** 2 / np.sum(s ** 2), mu
col = {'awake': '#4d4d4d', 'mild': '#9ecae1', 'moderate': '#2171b5', 'recovery': '#bdbdbd', 'anaesthesia': '#cb181d'}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
for ax, X, title in ((axes[0], Z, 'z, raw'), (axes[1], Zc, 'z, centred on each unit\'s awake mean')):
    V, ev, mu = pca(X); P = (X - mu) @ V.T
    for u in np.unique(unit):
        mk = 'o' if u in pr['group'] else 's'
        cents = {s: P[(unit == u) & (state == s)].mean(0) for s in np.unique(state[unit == u])}
        order = ['awake', 'mild', 'moderate', 'recovery'] if 'mild' in cents else ['awake', 'anaesthesia']
        pts = np.array([cents[s] for s in order])
        ax.plot(pts[:, 0], pts[:, 1], '-', color='#d9d9d9' if mk == 'o' else '#fcbba1', lw=0.8, zorder=1)
        for s in order: ax.plot(*cents[s], mk, color=col[s], ms=4 if mk == 'o' else 6, zorder=2, mec='none')
    ax.set_xlabel(f'PC1 ({ev[0]:.0%})'); ax.set_ylabel(f'PC2 ({ev[1]:.0%})'); ax.set_title(title); ax.set_box_aspect(1)
h = [plt.Line2D([], [], marker='o', ls='', color=col[s], label=s) for s in ('awake', 'mild', 'moderate', 'recovery')]
h += [plt.Line2D([], [], marker='s', ls='', color=col['anaesthesia'], label='anaesthesia (macaque)')]
h += [plt.Line2D([], [], marker='o', ls='', color='k', mfc='none', label='human (propofol)'), plt.Line2D([], [], marker='s', ls='', color='k', mfc='none', label='macaque (KTMD, propofol)')]
fig.legend(handles=h, loc='outside lower center', ncol=4)
fig.savefig('fig/embedding_r10_nt_propofol.png'); fig.savefig('fig/embedding_r10_nt_propofol.svg')
# direction agreement
d_mac = np.array([Zc[(unit == u) & (state == 'anaesthesia')].mean(0) for u in np.unique(ntu)])
d_hum = np.array([Zc[(unit == u) & (state == 'moderate')].mean(0) for u in np.unique(pr['group'])])
cos = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)
print('coords', Z.shape[1]); print('cos(mean macaque anaesthesia shift, mean human moderate shift):', round(cos(d_mac.mean(0), d_hum.mean(0)), 3))
print('within macaque units mean pairwise cos:', round(np.mean([cos(a, b) for i, a in enumerate(d_mac) for b in d_mac[i+1:]]), 3), '; within humans:', round(np.mean([cos(a, b) for i, a in enumerate(d_hum) for b in d_hum[i+1:]]), 3))
print('cross mean pairwise cos:', round(np.mean([cos(a, b) for a in d_mac for b in d_hum]), 3))
print('shift norm macaque', np.linalg.norm(d_mac, axis=1).round(1), 'human median', np.median(np.linalg.norm(d_hum, axis=1)).round(1))
