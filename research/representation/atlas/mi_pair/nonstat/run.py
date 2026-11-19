"""Non-stationary nulls on the macaque 10-s MI-pair windows (rule: ../preregs/nonstat-prereg.md).

Same rows, 32-channel subsets, normal-scoring and seeds as estimators.py / edges.py recompute, so N0 is the anchor's own IAAFT draw
(reproduction gate against the stored zns z). Per window: z(Gaussian MI, KSG k=4) on the real window, N0 and N2 at every L.

  python run.py <bank dir> <out.npz> <agent> <rowmode: first|row> <n workers>
  python run.py score <dir with ns_[AB]_*.npz>          (descriptive: no primary L qualified in calibration)
"""
import sys, glob, json, numpy as np, warnings
from pathlib import Path
from multiprocessing import Pool
from scipy.stats import ttest_1samp
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from edges import edge_mi, zc, sub32, meta, STATES
from nulls import ns, n2
from mtsspi.surrogates import multivariate_iaaft
warnings.filterwarnings('ignore')
LS = (500, 200, 100, 50)


def one(args):
    x, seed = args; v = ns(x); zr = zc(*edge_mi(v))
    z0 = zc(*edge_mi(multivariate_iaaft(v, np.random.default_rng(seed), final='spectrum')[0]))  # = edges.one_recompute, ns branch
    return [zr, z0] + [zc(*edge_mi(n2(v, np.random.default_rng([seed, L]), L))) for L in LS]


def score(d):
    F = sorted(glob.glob(f'{d}/ns_[AB]_*.npz')); Q = [np.load(f) for f in F]; cat = lambda k: np.concatenate([q[k] for q in Q])
    Z = cat('Z'); Vz = cat('Vzns'); unit, state, agent, animal = cat('unit'), cat('state'), cat('agent'), cat('animal')
    gap = np.nanmax(np.abs(Z[:, :2] - Vz)); print(f'reproduction gate: max |(real, N0) z - stored| = {gap:.2e} ->', 'PASS' if gap < 1e-4 else 'FAIL')
    if gap >= 1e-4: return
    names = ['N0'] + [f'N2 L{L}' for L in LS]; res = Z[:, :1] - Z[:, 1:]  # (W, 5): real - null
    out = {}
    for c, nm in enumerate(names):
        S = []
        for u in np.unique(unit):
            m = lambda s: (unit == u) & (state == s); ag = agent[unit == u][0]; s = 'sleep' if ag == 'SLP' else 'drug'
            if m(s).any() and m('awake_ec').any():
                S.append((ag, animal[unit == u][0], res[m(s), c].mean() - res[m('awake_ec'), c].mean(),
                          res[m('awake_eo'), c].mean() - res[m('awake_ec'), c].mean() if m('awake_eo').any() else np.nan))
        ag, an = np.array([q[0] for q in S]), np.array([q[1] for q in S]); dz, eo = np.array([q[2] for q in S]), np.array([q[3] for q in S])
        dr = ag != 'SLP'; A = np.array([dz[dr & (an == x)].mean() for x in np.unique(an[dr])]); p = ttest_1samp(A, 0, alternative='less').pvalue
        agm = {a: (dz[ag == a].mean(), int(np.sum(dz[ag == a] < 0)), int(np.sum(ag == a))) for a in ('KT', 'KTMD', 'MD', 'PF', 'SLP')}
        out[nm] = dict(pooled=float(dz[dr].mean()), animals=A.tolist(), neg=int((A < 0).sum()), p=float(p), agents=agm, eo_sd=float(np.nanstd(eo, ddof=1)))
        print(f'{nm:8s} pooled drug Δ {dz[dr].mean():+.4f} | animal means {np.round(A, 4)} ({(A < 0).sum()}/4 < 0, one-sided t p = {p:.4f}) | eyes-open SD {np.nanstd(eo, ddof=1):.4f}')
        print('          ' + '  '.join(f'{a} {v[0]:+.4f} ({v[1]}/{v[2]} down)' for a, v in agm.items()))
    print('\nretention R(L) = pooled drug Δ under N2(L) / under N0: ' + '  '.join(f'L{L} {out[f"N2 L{L}"]["pooled"] / out["N0"]["pooled"]:.2f}' for L in LS))
    Path(d, 'nonstat-score.json').write_text(json.dumps(dict(retention={L: out[f'N2 L{L}']['pooled'] / out['N0']['pooled'] for L in LS}, nulls=out), indent=1) + '\n')

if __name__ == '__main__':
    if sys.argv[1] == 'score': score(sys.argv[2]); sys.exit()
    d, out, agent, rowmode, nw = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    man = json.load(open(f'{d}/manifest.json')); Wn = np.load(f'{d}/windows.npz')
    rows = [r for r in man['rows'] if r['state'] in STATES and r['agent'] in agent.split(',')]
    S0 = sub32(rows[0]['M']); sub = lambda r: S0 if rowmode == 'first' else sub32(r['M'])
    with Pool(nw) as p: Z = p.map(one, [(Wn[r['id']].astype(float)[:, sub(r)], k) for k, r in enumerate(rows)], chunksize=1)
    m, _ = meta(rows, man); zns = out.replace('/nonstat/ns_', '/zns/zns_')
    np.savez(out, Z=np.array(Z), Vzns=np.load(zns)['V'][:, 1], L=np.array(LS), **m)
    print('done', out, flush=True)
