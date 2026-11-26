"""Surrogate-ensemble KSG MI per pair (rule preregs/mens-prereg.md): Δz in null-SD units and the per-pair MI excess z.

run:   per window, the normal-scored 32-channel subset of edges.py recompute (same rows, same subset), Gaussian and KSG MI
       per pair for the recording and for B spectrum-final joint IAAFT draws (draw 0 seeded k, as zns; draw b > 0 [k, b], as kens).
       python mens.py run <bank dir> <out.npz> <agent> <rowmode: first|row> <n workers> <B>
score: python mens.py score <dir with mn_*.npz> <zns dir>   (gate against zns, then the readouts of the prereg)
"""
import sys, json, numpy as np, warnings
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
warnings.filterwarnings('ignore')


def one(args):
    from edges import edge_mi, ns
    from mtsspi.surrogates import multivariate_iaaft
    x, seed, B = args; v = ns(x); g, k = edge_mi(v); G, K = [], []
    for b in range(B):
        s = multivariate_iaaft(v, np.random.default_rng(seed if b == 0 else [seed, b]), final='spectrum')[0]
        gs, ks = edge_mi(s); G.append(gs); K.append(ks)
    return np.array(g, np.float32), np.array(k, np.float32), np.array(G, np.float32), np.array(K, np.float32)


def readouts(g, k, G, K):
    """Per window: Fisher z of r(G, K) real and per draw, and per-pair excess z = (K - mean Ks) / sd Ks."""
    fz = lambda a, b: np.arctanh(np.corrcoef(a, b)[0, 1])
    zr, zs = fz(g, k), np.array([fz(a, b) for a, b in zip(G, K)])
    e = (k - K.mean(0)) / K.std(0, ddof=1)
    return dict(zr=zr, zs0=zs[0], dz0=zr - zs[0], dz_ens=(zr - zs.mean()) / zs.std(ddof=1), dz_raw=zr - zs.mean(),
                lvl=e.mean(), scat=e.std(ddof=1), exc=np.mean(np.abs(e) > 1.96), rho_eG=np.corrcoef(e, g)[0, 1])


def score(d, zd):
    import pandas as pd
    from scipy import stats
    rows = []
    for p in sorted(Path(d).glob('mn_*.npz')):
        with np.load(p) as a, np.load(Path(zd) / p.name.replace('mn_', 'zns_')) as z:
            assert (a['id'] == z['id']).all(), p
            Vz = np.arctanh(z['V'][:, 1].astype(float))
            for i in range(len(a['id'])):
                r = readouts(a['g'][i].astype(float), a['k'][i].astype(float), a['G'][i].astype(float), a['K'][i].astype(float))
                assert abs(r['zr'] - Vz[i, 0]) < 1e-4 and abs(r['zs0'] - Vz[i, 1]) < 1e-4, ('gate', a['id'][i], r['zr'], Vz[i])
                rows.append(dict(id=a['id'][i], unit=a['unit'][i], state=a['state'][i], agent=a['agent'][i], animal=a['animal'][i],
                                 delta=float(a['delta_fraction'][i]), **r))
    df = pd.DataFrame(rows).drop_duplicates('id'); print('gate passed:', len(df), 'windows; draw 0 equals zns real and surrogate z')
    df['group'] = np.where(df.state == 'drug', df.agent, np.where(df.state == 'sleep', 'SLP', df.state))
    R = ('dz0', 'dz_ens', 'dz_raw', 'lvl', 'scat', 'exc', 'rho_eG')
    for c in R:  # window minus its session's eyes-closed mean, in eyes-open SD units (panel (c) convention)
        dd = df[c] - df.unit.map(df[df.state == 'awake_ec'].groupby('unit')[c].mean()); df[c + '_sd'] = dd / dd[df.state == 'awake_eo'].std(ddof=1)
    print('\nwindow means by group (raw):'); print(df.groupby('group')[list(R)].mean().round(3).to_string())
    print('\nby group, eyes-open SD units vs own session EC:'); print(df.groupby('group')[[c + '_sd' for c in R]].mean().round(2).to_string())
    S = df.groupby(['unit', 'animal', 'group'])[[c + '_sd' for c in R]].mean().reset_index()
    dr = S[S.group.isin(['KT', 'KTMD', 'MD', 'PF'])]; sl = S[S.group == 'SLP']; out = {}
    print('\nprimary contrasts (sessions -> animal means):')
    for c in R:
        am = dr.groupby('animal')[c + '_sd'].mean(); t = stats.ttest_1samp(am, 0)
        sm, se = sl[c + '_sd'].mean(), sl[c + '_sd'].std(ddof=1) / np.sqrt(len(sl)); q = stats.t.ppf(.95, len(sl) - 1)
        rng = np.random.default_rng(0); bs = []
        for _ in range(10000):
            a_ = sl[c + '_sd'].values[rng.integers(0, len(sl), len(sl))].mean(); b_ = dr[c + '_sd'].values[rng.integers(0, len(dr), len(dr))].mean(); bs.append(a_ / b_)
        lo, hi = np.percentile(bs, [5, 95])
        out[c] = dict(drug_animals=am.round(3).to_dict(), drug_t=float(t.statistic), drug_p1=float(t.pvalue / 2 if t.statistic < 0 else 1 - t.pvalue / 2),
                      neg=int((am < 0).sum()), sleep_mean=float(sm), sleep_ci90=[float(sm - q * se), float(sm + q * se)], ratio=float(sm / dr[c + '_sd'].mean()), ratio_ci90=[float(lo), float(hi)])
        o = out[c]; print(f"{c:7s} drug animals {o['drug_animals']} t({len(am) - 1}) {o['drug_t']:+.2f} neg {o['neg']}/{len(am)} | sleep {sm:+.2f} [{o['sleep_ci90'][0]:+.2f}, {o['sleep_ci90'][1]:+.2f}] | sleep/drug {o['ratio']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print('\nby agent, session means (sd units):'); print(S.groupby('group')[[c + '_sd' for c in R]].agg(['mean', 'count']).round(2).to_string())
    json.dump(out, open(Path(d) / 'mn-score.json', 'w'), indent=1); df.to_csv(Path(d) / 'mn-windows.csv.gz', index=False)


if __name__ == '__main__':
    if sys.argv[1] == 'score':
        score(sys.argv[2], sys.argv[3]); sys.exit()
    from multiprocessing import Pool
    from edges import STATES, sub32, meta
    _, _, d, out, agent, rowmode, nw, B = sys.argv; nw, B = int(nw), int(B)
    man = json.load(open(f'{d}/manifest.json')); W = np.load(f'{d}/windows.npz')
    rows = [r for r in man['rows'] if r['state'] in STATES and r['agent'] in agent.split(',')]
    S0 = sub32(rows[0]['M']); sub = lambda r: S0 if rowmode == 'first' else sub32(r['M'])
    jobs = [(W[r['id']].astype(float)[:, sub(r)], k, B) for k, r in enumerate(rows)]
    with Pool(nw) as p: res = p.map(one, jobs, chunksize=1)
    m, kept = meta(rows, man)
    np.savez(out, **{n: np.array([q[i] for q in res]) for i, n in enumerate('gkGK')}, B=B, rowmode=rowmode, **m)
    print('done', out, flush=True)
