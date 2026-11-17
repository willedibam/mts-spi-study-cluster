import json, numpy as np, warnings; warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
exec(open('surr_pairs.py').read().split("if __name__")[0])
from mtsspi.surrogates import multivariate_iaaft
def dev(x):
    x = (x - x.mean(0)) / x.std(0); T, M = x.shape; iu = np.triu_indices(M, 1)
    r = np.corrcoef(x.T)[iu]; mig = -0.5 * np.log(1 - r ** 2); A = [acf(x[:, m]) for m in range(M)]
    mik = np.array([ksg(x[:, i], x[:, j], min(int(2 * np.dot(A[i], A[j])), T // 10)) for i, j in zip(*iu)])
    b = np.polyfit(mig, mik, 1); e = mik - np.polyval(b, mig)          # excess of nonparametric over Gaussian, per edge
    X = np.zeros((len(e), M)); X[np.arange(len(e)), iu[0]] = 1; X[np.arange(len(e)), iu[1]] = 1
    fit = X @ np.linalg.lstsq(X, e - e.mean(), rcond=None)[0]
    node = np.abs(e) @ X / (M - 1)                                     # per-channel mean |excess|
    return dict(r2_node=1 - np.sum((e - e.mean() - fit) ** 2) / np.sum((e - e.mean()) ** 2), top_share=node.max() / node.sum(), top=int(node.argmax()), sd=e.std())
def both(x, seed):
    return dict(real=dev(x), surr=dev(multivariate_iaaft(x, np.random.default_rng(seed), final='spectrum')[0]))
d = json.load(open('surr_pairs_nt.json')); rows = d['rows']; sub = d['subset']; W = np.load('raw/nt/windows.npz')
res = Parallel(n_jobs=8)(delayed(both)(W[r['id']].astype(float)[:, sub], k) for k, r in enumerate(rows))
unit = np.array([r['unit'] for r in rows]); agent = np.array([r['agent'] for r in rows]); state = np.array([r['state'] for r in rows])
def contrast(v, s, ag=None):
    out = []
    for u in np.unique(unit):
        if ag and agent[unit == u][0] != ag: continue
        a, c = (unit == u) & (state == s), (unit == u) & (state == 'awake_ec')
        if a.any() and c.any(): out.append(np.nanmean(v[a]) - np.nanmean(v[c]))
    return np.array(out)
AG = {'KTMD': 'ket-med', 'KT': 'ketamine', 'MD': 'medetom', 'PF': 'propofol', 'SLP': 'sleep'}
for k, lab in (('r2_node', 'node-additive share of the excess (R2)'), ('top_share', 'largest single-channel share of |excess|'), ('sd', 'spread of excess across edges')):
    for v in ('real', 'surr'):
        a = np.array([x[v][k] for x in res])
        print(f"{lab:42s} {v:5s} awake {np.nanmean(a[state=='awake_ec']):.3f} | " + ' '.join(f"{AG[g]} {np.nanmean(contrast(a,'sleep' if g=='SLP' else 'drug',g)):+.3f}({int((contrast(a,'sleep' if g=='SLP' else 'drug',g)>0).sum())}/{len(contrast(a,'sleep' if g=='SLP' else 'drug',g))})" for g in AG) + f" | EO {np.nanmean(contrast(a,'awake_eo')):+.3f}")
top = np.array([x['real']['top'] for x in res])
for g in ('KTMD', 'PF'):
    for s in ('awake_ec', 'drug'):
        m = (agent == g) & (state == s); c = np.bincount(top[m], minlength=16) / m.sum()
        print(g, s, 'most frequent top channel share', c.max().round(2), 'channel idx', c.argmax(), '(uniform .06)')
