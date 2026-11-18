import numpy as np, warnings; warnings.filterwarnings('ignore')
from trajplot import *
from screen import load
nt, pr = load('nt'), load('prop')
Z = np.vstack([nt['z'], pr['z']]).astype(float); keep = np.isfinite(Z).all(0); Z = np.clip(Z[:, keep], -.995, .995)
ntu = np.array([f"{x.split('--')[0][8:12]}-{g}" for x, g in zip(nt['row_id'], nt['group'])])
unit = np.r_[ntu, pr['group'].astype(str)]
state = np.r_[np.where(nt['y'] == 1, 'anaesthesia', 'awake'), np.array(['awake', 'mild', 'moderate', 'recovery'])[pr['y']]]
cond = np.r_[np.where(np.char.startswith(ntu, 'ktmd'), 'macaque KTMD', 'macaque propofol'), np.full(len(pr['y']), 'human propofol')]
paths = {'macaque KTMD': ['awake', 'anaesthesia'], 'macaque propofol': ['awake', 'anaesthesia'], 'human propofol': ['awake', 'mild', 'moderate', 'recovery']}
colours = {'macaque KTMD': '#cb181d', 'macaque propofol': '#fd8d3c', 'human propofol': '#2171b5'}
fig, axes = plt.subplots(1, 2, figsize=(7, 3.6))
P, ev = pca(Z); 
for c, col in colours.items():
    axes[0].plot(*P[cond == c].T, 'o', color=col, alpha=.15, ms=2, mec='none')
panel(axes[0], P, ev, unit, state, cond, paths, colours, 'raw z', faint=False)
Zc = centre(Z, unit, state, 'awake'); P, ev = pca(Zc)
panel(axes[1], P, ev, unit, state, cond, paths, colours, 'centred on each unit\'s awake mean')
h = [plt.Line2D([], [], color=c, label=k) for k, c in colours.items()]
fig.legend(handles=h, loc='outside lower center', ncol=3)
fig.savefig('fig/r10_nt_propofol.png'); fig.savefig('fig/r10_nt_propofol.svg')
