"""Gain-over-floor null on the macaque 10-s MI-pair windows (rule: ../preregs/envelope-prereg.md, real-data section).

Same rows, 32-channel subsets, normal-scoring and N0 seeds as nonstat/run.py (reproduction gate against the stored zns z).
Per window: z on the real window, N0, NDRAW GOF draws of the confirmed variant, 1 DECOR draw.

  python run.py <bank dir> <out.npz> <agent> <rowmode: first|row> <n workers> <variant, e.g. 40-diag>
  python run.py score <dir with env_[AB]_*.npz>
"""
import sys, glob, json, numpy as np, warnings
from pathlib import Path
from multiprocessing import Pool
from scipy.stats import ttest_1samp
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from edges import edge_mi, zc, sub32, meta, STATES
from nulls_env import ns, gof
from mtsspi.surrogates import multivariate_iaaft
warnings.filterwarnings('ignore')
NDRAW = 3


def one(args):
    x, seed, sg, fl = args; v = ns(x); zr = zc(*edge_mi(v))
    z0 = zc(*edge_mi(multivariate_iaaft(v, np.random.default_rng(seed), final='spectrum')[0]))  # the anchor's draw
    zg = [zc(*edge_mi(gof(x, np.random.default_rng([seed, 10 + d]), sg, fl, 'keep'))) for d in range(NDRAW)]
    zd = zc(*edge_mi(gof(x, np.random.default_rng([seed, 20]), sg, fl, 'decor')))
    return [zr, z0, float(np.mean(zg)), zd] + zg


def score(d):
    F = sorted(glob.glob(f'{d}/env_[AB]_*.npz')); Q = [np.load(f) for f in F]; cat = lambda k: np.concatenate([q[k] for q in Q])
    Z = cat('Z'); Vz = cat('Vzns'); unit, state, agent, animal = cat('unit'), cat('state'), cat('agent'), cat('animal')
    gap = np.nanmax(np.abs(Z[:, :2] - Vz)); print(f'reproduction gate: max |(real, N0) z - stored| = {gap:.2e} ->', 'PASS' if gap < 1e-4 else 'FAIL')
    if gap >= 1e-4: return
    names = ['N0', 'GOF', 'DECOR']; res = Z[:, :1] - Z[:, 1:4]; out = {}
    for c, nm in enumerate(names):
        S = []
        for u in np.unique(unit):
            m = lambda s: (unit == u) & (state == s); ag = agent[unit == u][0]; s = 'sleep' if ag == 'SLP' else 'drug'
            if m(s).any() and m('awake_ec').any():
                S.append((ag, animal[unit == u][0], res[m(s), c].mean() - res[m('awake_ec'), c].mean(),
                          res[m('awake_eo'), c].mean() - res[m('awake_ec'), c].mean() if m('awake_eo').any() else np.nan))
        ag, an = np.array([q[0] for q in S]), np.array([q[1] for q in S]); dz, eo = np.array([q[2] for q in S]), np.array([q[3] for q in S])
        dr = ag != 'SLP'; ans = np.unique(an[dr]); A = np.array([dz[dr & (an == x)].mean() for x in ans]); p = ttest_1samp(A, 0, alternative='less').pvalue
        agm = {a: (float(dz[ag == a].mean()), int(np.sum(dz[ag == a] < 0)), int(np.sum(ag == a))) for a in ('KT', 'KTMD', 'MD', 'PF', 'SLP')}
        out[nm] = dict(pooled=float(dz[dr].mean()), animals=dict(zip(ans.tolist(), A.tolist())), neg=int((A < 0).sum()), p=float(p), agents=agm,
                       eo_mean=float(np.nanmean(eo)), eo_sd=float(np.nanstd(eo, ddof=1)))
        print(f'{nm:6s} pooled drug Δ {dz[dr].mean():+.4f} | animals {dict(zip(ans, np.round(A, 4)))} ({(A < 0).sum()}/{len(A)} < 0, one-sided p = {p:.4f}) | eyes-open mean {np.nanmean(eo):+.4f} SD {np.nanstd(eo, ddof=1):.4f}')
        print('        ' + '  '.join(f'{a} {v[0]:+.4f} ({v[1]}/{v[2]} down)' for a, v in agm.items()))
    g, n0 = out['GOF'], out['N0']; R = g['pooled'] / n0['pooled']
    Ran = {a: g['animals'][a] / n0['animals'][a] for a in n0['animals']}; Rag = {a: g['agents'][a][0] / n0['agents'][a][0] for a in ('KT', 'KTMD', 'MD', 'PF')}
    v = ('COUPLING SURVIVES' if R >= .6 and g['neg'] == 4 and g['p'] < .05 else
         'GAIN EXPLAINS' if R <= .3 and (g['p'] >= .05 or g['neg'] < 4) else 'PARTIAL')
    print(f'\nretained R (GOF / N0) = {R:.2f}; by animal ' + '  '.join(f'{a} {r:.2f}' for a, r in Ran.items()) + '; by agent ' + '  '.join(f'{a} {r:.2f}' for a, r in Rag.items()))
    print(f'DECOR retained {out["DECOR"]["pooled"] / n0["pooled"]:.2f}  (descriptive)\nVERDICT (frozen rule): {v}')
    Path(d, 'env-score.json').write_text(json.dumps(dict(verdict=v, retained=R, by_animal=Ran, by_agent=Rag, nulls=out), indent=1) + '\n')


if __name__ == '__main__':
    if sys.argv[1] == 'score': score(sys.argv[2]); sys.exit()
    d, out, agent, rowmode, nw, variant = sys.argv[1:7]; sg, fl = variant.split('-')
    man = json.load(open(f'{d}/manifest.json')); Wn = np.load(f'{d}/windows.npz')
    rows = [r for r in man['rows'] if r['state'] in STATES and r['agent'] in agent.split(',')]
    S0 = sub32(rows[0]['M']); sub = lambda r: S0 if rowmode == 'first' else sub32(r['M'])
    with Pool(int(nw)) as p: Z = p.map(one, [(Wn[r['id']].astype(float)[:, sub(r)], k, int(sg), fl) for k, r in enumerate(rows)], chunksize=1)
    m, _ = meta(rows, man); zns = out.replace('/envnull/env_', '/zns/zns_')
    np.savez(out, Z=np.array(Z), Vzns=np.load(zns)['V'][:, 1], variant=variant, ndraw=NDRAW, **m)
    print('done', out, flush=True)
