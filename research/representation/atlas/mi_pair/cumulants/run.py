"""Cross-cumulant decomposition of the MI pair on the macaque 10-s windows (rule: ../preregs/cumulants-prereg.md).

Same rows, 32-channel subsets, normal-scoring and IAAFT seed as estimators.py / nonstat/run.py, so the surrogate is the anchor's
own draw; Gaussian and kNN MI per edge are read from the stored zns arrays (E[:, 1]), after a gate that recomputes G.
  python run.py <bank dir> <out.npz> <agent> <rowmode: first|row> <n workers>
  python run.py score <dir with cum_[AB]_*.npz> <zns dir>
"""
import sys, glob, json, numpy as np, warnings
from pathlib import Path
from multiprocessing import Pool
from scipy.stats import ttest_1samp
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from edges import zc, sub32, meta, STATES
from stats import edge_cum
from mtsspi.surrogates import multivariate_iaaft
from scipy.stats import norm, rankdata
warnings.filterwarnings('ignore')
ns = lambda x: norm.ppf((np.apply_along_axis(rankdata, 0, x) - .5) / len(x))
F = ('G', 'J3', 'J4', 'p', 'g2')


def one(args):
    x, seed = args; v = ns(x); s = multivariate_iaaft(v, np.random.default_rng(seed), final='spectrum')[0]
    return np.array([[edge_cum(u)[f] for f in F] for u in (v, s)], np.float32)      # (2 real/surr, 5, E)


def pz(g, k, C):
    X = np.column_stack([np.ones(len(g)), C]); res = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return zc(res(g), res(k))


KEYS = ('zGK', 'zJ', 'zJ3', 'zJ4', 'zP', 'link3', 'link4', 'L3', 'L4', 'Lenv', 'zenv')


def window(C, E):
    """C (2, 5, E) cumulant fields, E (2, 2, E) stored [real, surr] x [G, K]: the calibration's net readouts."""
    R = []
    for a in (0, 1):
        g, k = E[a, 0].astype(float), E[a, 1].astype(float); J3, J4, p, g2 = (C[a, i].astype(float) for i in (1, 2, 3, 4))
        R.append(dict(zGK=zc(g, k), zJ=zc(g, g + J3 + J4), zJ3=zc(g, g + J3), zJ4=zc(g, g + J4),
                      zP=pz(g, k, np.column_stack([J3, J4])), zenv=zc(g2, p), k=k, J3=J3, J4=J4, env=p - g2))
    r, s = R; ek = r['k'] - s['k']
    out = {q: r[q] - s[q] for q in ('zGK', 'zJ', 'zJ3', 'zJ4', 'zP', 'zenv')}
    out.update(link3=zc(ek, r['J3'] - s['J3']), link4=zc(ek, r['J4'] - s['J4']), L3=np.mean(r['J3'] - s['J3']),
               L4=np.mean(r['J4'] - s['J4']), Lenv=np.mean(r['env'] - s['env']))
    return [out[q] for q in KEYS]


def score(d, zd):
    Fs = sorted(glob.glob(f'{d}/cum_[AB]_*.npz')); Q = [np.load(f) for f in Fs]
    Zs = [np.load(f.replace(d, zd).replace('/cum_', '/zns_')) for f in Fs]
    C = np.concatenate([q['C'] for q in Q]); E = np.concatenate([z['E'][:, 1] for z in Zs])
    cat = lambda k: np.concatenate([q[k] for q in Q]); unit, state, agent, animal = cat('unit'), cat('state'), cat('agent'), cat('animal')
    gap = np.max(np.abs(C[:, :, 0] - E[:, :, 0])); print(f'reproduction gate: max |G recomputed - G stored| = {gap:.2e} ->', 'PASS' if gap < 1e-4 else 'FAIL')
    if gap >= 1e-4: return
    W = np.array([window(c, e) for c, e in zip(C, E)]); out = {}
    for c, nm in enumerate(KEYS):
        S = []
        for u in np.unique(unit):
            m = lambda s: (unit == u) & (state == s); ag = agent[unit == u][0]; s = 'sleep' if ag == 'SLP' else 'drug'
            if m(s).any() and m('awake_ec').any():
                eo = W[m('awake_eo'), c].mean() - W[m('awake_ec'), c].mean() if m('awake_eo').any() else np.nan
                rmi = np.corrcoef(W[m(s), c], W[m(s), 0])[0, 1] if m(s).sum() > 3 else np.nan
                S.append((ag, animal[unit == u][0], W[m(s), c].mean() - W[m('awake_ec'), c].mean(), eo, W[m('awake_ec'), c].mean(), W[m(s), c].mean(), rmi))
        ag, an = np.array([q[0] for q in S]), np.array([q[1] for q in S]); dz, eo, aw, st, rmi = (np.array([q[i] for q in S]) for i in (2, 3, 4, 5, 6))
        dr = ag != 'SLP'; A = np.array([dz[dr & (an == x)].mean() for x in np.unique(an[dr])]); t = ttest_1samp(A, 0)
        agm = {a: (float(dz[ag == a].mean()), int(np.sum(dz[ag == a] < 0)), int(np.sum(ag == a))) for a in ('KT', 'KTMD', 'MD', 'PF', 'SLP')}
        out[nm] = dict(pooled=float(dz[dr].mean()), animals=A.tolist(), t=float(t.statistic), p_two=float(t.pvalue), eo_sd=float(np.nanstd(eo, ddof=1)),
                       awake=float(aw[dr].mean()), drug=float(st[dr].mean()), r_with_anchor=float(np.nanmedian(rmi[dr])), agents=agm)
        print(f'{nm:6s} awake {aw[dr].mean():+.4f} drug {st[dr].mean():+.4f} | Δ {dz[dr].mean():+.4f} animals {np.round(A, 4)} t(3) {t.statistic:+.2f} p {t.pvalue:.4f} '
              f'| EO SD {np.nanstd(eo, ddof=1):.4f} | median within-drug r with anchor net {np.nanmedian(rmi[dr]):+.2f}')
        print('        ' + '  '.join(f'{a} {v[0]:+.4f} ({v[1]}/{v[2]} down)' for a, v in agm.items()))
    R = out['zP']['pooled'] / out['zGK']['pooled']; print(f'\nretention of the anchor change after partialling J3, J4: {R:.2f}')
    a = lambda k: np.array(out[k]['animals']); sz, sl = a('zJ3') / a('zJ'), a('L3') / (a('L3') + a('L4'))
    Sz, SL = out['zJ3']['pooled'] / out['zJ']['pooled'], out['L3']['pooled'] / (out['L3']['pooled'] + out['L4']['pooled'])
    print(f'J3 share of the zJ change S_z {Sz:.2f} (animals {np.round(sz, 2)}); of the level change S_L {SL:.2f} (animals {np.round(sl, 2)})')
    Path(d, 'cum-score.json').write_text(json.dumps(dict(retention_partial=R, S_z=Sz, S_L=SL, S_z_animals=sz.tolist(), S_L_animals=sl.tolist(), readouts=out), indent=1) + '\n')


if __name__ == '__main__':
    if sys.argv[1] == 'score': score(sys.argv[2], sys.argv[3]); sys.exit()
    d, out, agent, rowmode, nw = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    man = json.load(open(f'{d}/manifest.json')); Wn = np.load(f'{d}/windows.npz')
    rows = [r for r in man['rows'] if r['state'] in STATES and r['agent'] in agent.split(',')]
    S0 = sub32(rows[0]['M']); sub = lambda r: S0 if rowmode == 'first' else sub32(r['M'])
    with Pool(nw) as p: C = p.map(one, [(Wn[r['id']].astype(float)[:, sub(r)], k) for k, r in enumerate(rows)], chunksize=1)
    m, _ = meta(rows, man); np.savez(out, C=np.array(C), F=np.array(F), **m); print('done', out, flush=True)
