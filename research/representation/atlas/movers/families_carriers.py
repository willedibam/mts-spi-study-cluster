"""POST HOC, descriptive: what the named families' z changes are made of.
1. Carriers of the precision x phase family: the top-1% carrying edges (movers edges output) against all edges, by
   awake level and drug change of each partner SPI, and by electrode-label distance.
2. Levels against z: per session, the drug change in each partner SPI's mean edge value, next to the change in z and in
   double-centred z (channel effects removed), for every family member.
Usage: python families_carriers.py <atlas dir> <movers edges dir> <movers.json> <out.json>"""
import sys, json, glob, numpy as np
from pathlib import Path
from movers import REF, PRIMARY

atlas, edir, mj, out = sys.argv[1:5]
A = Path(atlas); D = A / 'discovery'
with np.load(D / 'neurotycho-short-reduced.npz') as d:
    spis = [str(s) for s in d['spis']]; ids = [str(x) for x in d['ids']]; rows = json.loads(str(d['rows']))
    Z, ZDC, MEAN = d['z'], d['zdc'], d['mean']
K = len(spis); a, b = np.triu_indices(K, 1); co = {(spis[i], spis[j]): c for c, (i, j) in enumerate(zip(a, b))}
coord = lambda p, q: co.get((p, q), co.get((q, p)))
by = lambda pre: [s for s in spis if s.startswith(pre)]
BROAD = [s for s in spis if s.endswith('_fs-1_fmin-0_fmax-0-5') and s.split('_')[0] in ('cohmag', 'plv', 'ppc')]
fam = {'precision': [(p, s) for p in ('prec-sq_LedoitWolf', 'prec-sq_OAS', 'prec-sq_ShrunkCovariance', 'prec-sq_GraphicalLassoCV') for s in BROAD],
       'Chebyshev': [('pdist_chebyshev', s) for s in by('pdist_') if s != 'pdist_chebyshev'],
       'kernel': [('tlmi_kernel_W-0.25', 'te_kernel_W-0.25_k-1')], 'MI': [('mi_gaussian', 'mi_kraskov_NN-4_DCE-AUTO')],
       'mediation': [(f'cov_{e}', f'pcorr_{e}') for e in ('EmpiricalCovariance', 'MinCovDet', 'LedoitWolf', 'OAS', 'ShrunkCovariance')]}
un = np.array([r['unit'] for r in rows]); st = np.array([r['state'] for r in rows]); ag = {r['unit']: r['agent'] for r in rows}
prim = sorted({u for u in un if ag[u] in PRIMARY})
d_ = lambda X, u: np.nanmean(X[(un == u) & (st == 'drug')], 0) - np.nanmean(X[(un == u) & (st == REF)], 0)
lv = {}
for f, pairs in fam.items():
    for p, q in pairs:
        c = coord(p, q)
        if c is None: continue
        ip, iq = spis.index(p), spis.index(q)
        dz = np.array([d_(Z[:, c], u) for u in prim]); ddc = np.array([d_(ZDC[:, c], u) for u in prim])
        mp = np.array([d_(MEAN[:, ip], u) for u in prim]); mq = np.array([d_(MEAN[:, iq], u) for u in prim])
        base = lambda i: np.array([np.nanmean(MEAN[(un == u) & (st == REF), i]) for u in prim])
        lv[f'{p}|{q}'] = dict(family=f, dz=float(dz.mean()), dz_sessions_neg=int((dz < 0).sum()), dz_dc=float(ddc.mean()), dz_dc_sessions_neg=int((ddc < 0).sum()),
                              rel_level_change=[float(np.mean(mp / np.abs(base(ip)))), float(np.mean(mq / np.abs(base(iq))))],
                              level_sessions_up=[int((mp > 0).sum()), int((mq > 0).sum())],
                              corr_dz_with_level_change=[float(np.corrcoef(dz, mp)[0, 1]), float(np.corrcoef(dz, mq)[0, 1])])
# carriers of the precision family from the pyspi shards
m = json.loads(Path(mj).read_text()); E = {Path(f).stem: dict(np.load(f, allow_pickle=True)) for f in glob.glob(f'{edir}/*.npz')}
coords = [int(c) for c in E[next(iter(E))]['coords']]
P = [(p, s) for p, s in fam['precision'] if coord(p, s) in coords]
pyspi_dir = A / 'pyspi' / 'neurotycho-short'; files = sorted(glob.glob(f'{pyspi_dir}/shard-*.npz')); allids = sorted(ids)
size = -(-len(allids) // len(files))
need = sorted({x for pq in P for x in pq})
def profiles(u, s):
    out = []
    for r in rows:
        if r['unit'] == u and r['state'] == s:
            with np.load(files[allids.index(r['id']) // size]) as d:
                names = [str(x) for x in d['__spis__']]; X = d[r['id']]
                iu = np.triu_indices(X.shape[1], 1); out.append([X[names.index(n)][iu] for n in need])
    return np.nanmean(out, 0)
car = {}
for u in prim:
    e = E[u]; ch = [int(x) for x in e['chan']]; M = len(ch); iu = np.triu_indices(M, 1); dist = np.abs(np.array(ch)[iu[0]] - np.array(ch)[iu[1]])
    aw, dr = profiles(u, REF), profiles(u, 'drug'); k = int(np.ceil(.01 * len(dist)))
    pct = lambda v: (np.argsort(np.argsort(v)) + .5) / len(v)
    for p, s in P:
        n = coords.index(coord(p, s)); sg = np.sign(m['net_h0_by_coord'][str(coord(p, s))]); top = np.argsort(-sg * e['drug_all'][n])[:k]
        ip, is_ = need.index(p), need.index(s)
        x = car.setdefault(f'{p}|{s}', dict(awake_pct_prec=[], awake_pct_phase=[], dprec_top=[], dprec_all=[], dphase_top=[], dphase_all=[],
                                             label_dist_top=[], label_dist_all=[], adjacent_top=[], adjacent_all=[]))
        x['awake_pct_prec'].append(float(pct(aw[ip])[top].mean())); x['awake_pct_phase'].append(float(pct(aw[is_])[top].mean()))
        x['dprec_top'].append(float((dr[ip] - aw[ip])[top].mean() / np.abs(aw[ip]).mean())); x['dprec_all'].append(float((dr[ip] - aw[ip]).mean() / np.abs(aw[ip]).mean()))
        x['dphase_top'].append(float((dr[is_] - aw[is_])[top].mean())); x['dphase_all'].append(float((dr[is_] - aw[is_]).mean()))
        x['label_dist_top'].append(float(np.median(dist[top]))); x['label_dist_all'].append(float(np.median(dist)))
        x['adjacent_top'].append(float(np.mean(dist[top] == 1))); x['adjacent_all'].append(float(np.mean(dist == 1)))
car = {k: {kk: float(np.mean(vv)) for kk, vv in v.items()} for k, v in car.items()}
Path(out).write_text(json.dumps(dict(levels=lv, carriers=car, n_sessions=len(prim)), indent=1))
for k, v in lv.items(): print(f"{k[:70]:70s} dz {v['dz']:+.3f} ({v['dz_sessions_neg']}/19 neg) dc {v['dz_dc']:+.3f} ({v['dz_dc_sessions_neg']}) level {v['rel_level_change'][0]:+.2f},{v['rel_level_change'][1]:+.2f} r(dz,lvl) {v['corr_dz_with_level_change'][0]:+.2f},{v['corr_dz_with_level_change'][1]:+.2f}")
for k, v in car.items(): print(k[:60], {kk: round(vv, 3) for kk, vv in v.items()})
