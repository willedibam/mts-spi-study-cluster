import json, numpy as np, warnings; warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
from mtsspi.surrogates import multivariate_iaaft
exec(open('calib.py').read().split("def one(")[0])     # gen(), run(), helpers
def edge_moments(x):
    x = (x - x.mean(0)) / x.std(0); T, M = x.shape; curv, het = [], []
    X1 = lambda v: np.c_[np.ones(T), v]; X3 = lambda v: np.c_[np.ones(T), v, v ** 2, v ** 3]
    for i in range(M):
        for j in range(M):
            if i == j: continue
            y = x[:, j]; v = x[:, i]
            b1 = np.linalg.lstsq(X1(v), y, rcond=None)[0]; e1 = y - X1(v) @ b1
            b3 = np.linalg.lstsq(X3(v), y, rcond=None)[0]; e3 = y - X3(v) @ b3
            curv.append((e1 @ e1 - e3 @ e3) / (y @ y))
            het.append(np.corrcoef(v ** 2, e3 ** 2)[0, 1])
    return np.mean(curv), np.mean(het)
def mrun(x, seed):
    s = multivariate_iaaft(x, np.random.default_rng(seed), final='spectrum')[0]
    a, b = edge_moments(x), edge_moments(s)
    return dict(curv=a[0], het=a[1], curv_s=b[0], het_s=b[1])
import sys
if sys.argv[1] == 'synth':
    for kind, s in (('linear', 0), ('nonlinear', .6), ('nonlinear', 1.2), ('gain', .5), ('gain', 1.0)):
        r = Parallel(n_jobs=8)(delayed(lambda k: mrun(gen(kind, s, 7000 + k), k))(k) for k in range(40))
        c = np.array([q['curv'] - q['curv_s'] for q in r]); h = np.array([q['het'] - q['het_s'] for q in r])
        print(f"{kind+' '+str(s):16s} curvature real−IAAFT {c.mean():+.4f} (se {c.std()/np.sqrt(40):.4f}) | heteroscedasticity real−IAAFT {h.mean():+.4f} (se {h.std()/np.sqrt(40):.4f})")
else:
    d = json.load(open('surr_pairs_nt.json')); rows = d['rows']; sub = d['subset']; W = np.load('raw/nt/windows.npz')
    res = Parallel(n_jobs=8)(delayed(mrun)(W[r['id']].astype(float)[:, sub], k) for k, r in enumerate(rows))
    json.dump(res, open('moments_nt.json', 'w'))
    unit = np.array([r['unit'] for r in rows]); agent = np.array([r['agent'] for r in rows]); state = np.array([r['state'] for r in rows])
    def contrast(v, s, ag=None):
        out = []
        for u in np.unique(unit):
            if ag and agent[unit == u][0] != ag: continue
            a, cc = (unit == u) & (state == s), (unit == u) & (state == 'awake_ec')
            if a.any() and cc.any(): out.append(np.nanmean(v[a]) - np.nanmean(v[cc]))
        return np.array(out)
    AG = {'KTMD': 'ket-med', 'KT': 'ketamine', 'MD': 'medetom', 'PF': 'propofol', 'SLP': 'sleep'}
    g = lambda k: np.array([x[k] for x in res])
    for lab, v in (('curvature real', g('curv')), ('curvature real−IAAFT', g('curv') - g('curv_s')), ('heterosc. real', g('het')), ('heterosc. real−IAAFT', g('het') - g('het_s'))):
        eo = contrast(v, 'awake_eo')
        print(f"{lab:22s} awake {np.nanmean(v[state=='awake_ec']):+.4f} | " + ' '.join(f"{AG[a]} {np.nanmean(contrast(v,'sleep' if a=='SLP' else 'drug',a)):+.4f}({int((contrast(v,'sleep' if a=='SLP' else 'drug',a)>0).sum())}/{len(contrast(v,'sleep' if a=='SLP' else 'drug',a))})" for a in AG) + f" | EO {np.nanmean(eo):+.4f}±{np.nanstd(eo,ddof=1):.4f}")
