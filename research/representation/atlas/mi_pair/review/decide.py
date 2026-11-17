import json, numpy as np, warnings; warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
from scipy.ndimage import uniform_filter1d
exec(open('surr_pairs.py').read().split("if __name__")[0])
from mtsspi.surrogates import multivariate_iaaft
def zgk(x):
    x = (x - x.mean(0)) / x.std(0); T, M = x.shape; iu = np.triu_indices(M, 1)
    r = np.corrcoef(x.T)[iu]; A = [acf(x[:, m]) for m in range(M)]
    mik = np.array([ksg(x[:, i], x[:, j], min(int(2 * np.dot(A[i], A[j])), T // 10)) for i, j in zip(*iu)])
    return np.corrcoef(-0.5 * np.log(1 - r ** 2), mik)[0, 1]
def gain(x, n):
    p = uniform_filter1d(x ** 2, size=n, axis=0, mode='reflect'); return x / np.sqrt(p + 1e-12 * p.mean())
def block_iaaft(x, nb, rng):
    T = len(x); edges = np.linspace(0, T, nb + 1).astype(int)
    return np.vstack([multivariate_iaaft(x[a:b], rng, final='spectrum')[0] for a, b in zip(edges[:-1], edges[1:])])
def run(x, seed):
    rng = lambda k: np.random.default_rng([seed, k])
    g = gain(x, 50)
    return dict(real=zgk(x), glob=zgk(multivariate_iaaft(x, rng(0), final='spectrum')[0]),
                half=zgk(block_iaaft(x, 2, rng(1))), quarter=zgk(block_iaaft(x, 4, rng(2))),
                gain_real=zgk(g), gain_glob=zgk(multivariate_iaaft(g, rng(3), final='spectrum')[0]))
if __name__ == '__main__':
    d = json.load(open('surr_pairs_nt.json')); rows = d['rows']; sub = d['subset']; W = np.load('raw/nt/windows.npz')
    res = Parallel(n_jobs=8)(delayed(run)(W[r['id']].astype(float)[:, sub], k) for k, r in enumerate(rows))
    json.dump(res, open('decide_nt.json', 'w'))
    unit = np.array([r['unit'] for r in rows]); agent = np.array([r['agent'] for r in rows]); state = np.array([r['state'] for r in rows])
    def contrast(v, s, ag=None):
        out = []
        for u in np.unique(unit):
            if ag and agent[unit == u][0] != ag: continue
            a, c = (unit == u) & (state == s), (unit == u) & (state == 'awake_ec')
            if a.any() and c.any(): out.append(np.nanmean(v[a]) - np.nanmean(v[c]))
        return np.array(out)
    AG = {'KTMD': 'ket-med', 'KT': 'ketamine', 'MD': 'medetom', 'PF': 'propofol', 'SLP': 'sleep'}
    get = lambda k: np.array([x[k] for x in res])
    print('Δ vs own eyes-closed wake (units negative/n) | EO null mean±SD')
    for lab, v in (('real', get('real')), ('real − global IAAFT', get('real') - get('glob')),
                   ('real − half-block IAAFT', get('real') - get('half')), ('real − quarter-block IAAFT', get('real') - get('quarter')),
                   ('gain-normalised real − its global IAAFT', get('gain_real') - get('gain_glob'))):
        eo = contrast(v, 'awake_eo')
        print(f"{lab:42s} " + ' '.join(f"{AG[g]} {np.nanmean(contrast(v,'sleep' if g=='SLP' else 'drug',g)):+.3f}({int((contrast(v,'sleep' if g=='SLP' else 'drug',g)<0).sum())}/{len(contrast(v,'sleep' if g=='SLP' else 'drug',g))})" for g in AG) + f" | EO {np.nanmean(eo):+.3f}±{np.nanstd(eo,ddof=1):.3f}")
    print('baseline (awake) level of each surrogate z:', {k: round(float(np.nanmean(get(k)[state == 'awake_ec'])), 3) for k in ('real', 'glob', 'half', 'quarter', 'gain_real', 'gain_glob')})
