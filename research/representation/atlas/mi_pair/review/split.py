import json, numpy as np, warnings; warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
from scipy.stats import rankdata
from scipy.special import ndtri
exec(open('surr_pairs.py').read().split("if __name__")[0])
from mtsspi.surrogates import multivariate_iaaft
def zpair(x, keep=None):
    x = (x - x.mean(0)) / x.std(0); T, M = x.shape; iu = np.triu_indices(M, 1)
    A = [acf(x[:, m]) for m in range(M)]; Wt = [[min(int(2 * np.dot(A[i], A[j])), T // 10) for j in range(M)] for i in range(M)]
    if keep is not None: x = x[keep]
    ns = ndtri((np.apply_along_axis(rankdata, 0, x) - .5) / len(x))
    r = np.corrcoef(x.T)[iu]; rn = np.corrcoef(ns.T)[iu]
    mik = np.array([ksg(x[:, i], x[:, j], Wt[i][j]) for i, j in zip(*iu)])
    g = lambda q: -0.5 * np.log(1 - q ** 2); c = lambda a, b: np.corrcoef(a, b)[0, 1]
    return c(g(r), mik), c(g(rn), mik)
def run(x, seed):
    s = multivariate_iaaft(x, np.random.default_rng(seed), final='spectrum')[0]
    out = {}
    for nm, arr in (('real', x), ('surr', s)):
        xs = (arr - arr.mean(0)) / arr.std(0); amp = (xs ** 2).sum(1); keep = amp <= np.quantile(amp, .95)
        full = zpair(arr); trim = zpair(arr, keep)
        out[nm] = dict(gauss=full[0], copula=full[1], gauss_trim=trim[0], copula_trim=trim[1])
    return out
d = json.load(open('surr_pairs_nt.json')); rows = d['rows']; sub = d['subset']; W = np.load('raw/nt/windows.npz')
res = Parallel(n_jobs=8)(delayed(run)(W[r['id']].astype(float)[:, sub], k) for k, r in enumerate(rows))
json.dump(res, open('split_nt.json', 'w'))
unit = np.array([r['unit'] for r in rows]); agent = np.array([r['agent'] for r in rows]); state = np.array([r['state'] for r in rows])
def contrast(v, s, ag=None):
    out = []
    for u in np.unique(unit):
        if ag and agent[unit == u][0] != ag: continue
        a, c = (unit == u) & (state == s), (unit == u) & (state == 'awake_ec')
        if a.any() and c.any(): out.append(np.nanmean(v[a]) - np.nanmean(v[c]))
    return np.array(out)
AG = {'KTMD': 'ket-med', 'KT': 'ketamine', 'MD': 'medetom', 'PF': 'propofol', 'SLP': 'sleep'}
print('real − IAAFT(spectrum) of z, Δ vs own awake; (units negative/n); EO null mean±SD')
for k, lab in (('gauss', 'Gaussian MI (Pearson) vs KSG'), ('copula', 'Gaussian-copula MI (normal scores) vs KSG'), ('gauss_trim', 'Gaussian vs KSG, top 5% joint amplitude removed'), ('copula_trim', 'copula vs KSG, top 5% removed')):
    v = np.array([x['real'][k] - x['surr'][k] for x in res]); eo = contrast(v, 'awake_eo'); sd = np.nanstd(eo, ddof=1)
    print(f"{lab:48s} " + ' '.join(f"{AG[g]} {np.nanmean(contrast(v,'sleep' if g=='SLP' else 'drug',g)):+.3f}({int((contrast(v,'sleep' if g=='SLP' else 'drug',g)<0).sum())}/{len(contrast(v,'sleep' if g=='SLP' else 'drug',g))})" for g in AG) + f" | EO {np.nanmean(eo):+.3f}±{sd:.3f}")
