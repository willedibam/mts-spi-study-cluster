"""POST HOC checks on phase 1 (after scoring): estimator dependence of the partial-correlation movers, a control for the
sleep sign, the usable-coordinate accounting, and hub-SPI tables for the compound movers.
Usage: python movers_posthoc.py <atlas dir> <movers.json> <out.json>"""
import sys, json, numpy as np
from pathlib import Path
from movers import BANKS, REF, PRIMARY, halves, net_contrast, target

atlas, mj, out = sys.argv[1:4]
D = Path(atlas) / 'discovery'; Z = {}
for k, name in BANKS.items():
    with np.load(D / f'{name}-reduced.npz') as d:
        Z[k] = d['z']
        if k == 'real': rows = json.loads(str(d['rows'])); spis = [str(x) for x in d['spis']]
rel = {k: v['reliability'] for k, v in json.loads((D / 'neurotycho-short-reliability-all.json').read_text())['reference_state'].items()}
m = json.loads(Path(mj).read_text()); K = len(spis); a, b = np.triu_indices(K, 1); H = halves(rows)
units = sorted({r['unit'] for r in rows}); agent = {r['unit']: r['agent'] for r in rows}; animal = {r['unit']: r['animal'] for r in rows}
ref = [i for i, r in enumerate(rows) if r['state'] == REF]; z0 = np.nanmean(Z['real'][ref], 0)
fin = np.all([np.isfinite(v).all(0) for v in Z.values()], 0); relok = np.array([rel.get(spis[i], 0) >= .3 and rel.get(spis[j], 0) >= .3 for i, j in zip(a, b)])
ceil = np.abs(z0) < .98
acc = dict(total=int(len(a)), finite_every_window=int(fin.sum()), plus_reliability=int((fin & relok).sum()), plus_ceiling=int((fin & relok & ceil).sum()))
spi_fin = {s: bool(fin[(a == i) | (b == i)].any()) for i, s in enumerate(spis)}
acc['spis_with_no_finite_coordinate'] = [s for s, v in spi_fin.items() if not v]
acc['spis_failing_reliability'] = sorted(s for s in spis if rel.get(s, 0) < .3)
N = {(u, s): net_contrast(Z, H, u, s, 'all') for u in units for s in (target(agent[u]), 'awake_eo') if (u, s, 'all') in H and (u, REF, 'all') in H}
prim = [u for u in units if agent[u] in PRIMARY]; slp = [u for u in units if agent[u] == 'SLP']
idx = {(spis[i], spis[j]): n for n, (i, j) in enumerate(zip(a, b))}
co = lambda p, q: idx.get((p, q), idx.get((q, p)))
mean = lambda us, s, c: float(np.mean([N[(u, s(u))][c] for u in us]))
# 1. partial-correlation estimators
pc = [s for s in spis if s.startswith('pcorr')]; covs = {s.split('_', 1)[1]: s for s in spis if s.startswith('cov_')}
part = {}
for p in pc:
    est = p.split('_', 1)[1]; out_ = {}
    for partner in (covs.get(est), 'cce_kozachenko_n-5', 'si_kernel_W-0.5_k-1'):
        c = co(p, partner) if partner else None
        if c is None: continue
        out_[partner] = dict(usable=bool(fin[c] & relok[c] & ceil[c]), finite=bool(fin[c]), reliability=(rel.get(p), rel.get(partner)),
                             drug_net=mean(prim, lambda u: 'drug', c), eo_net=mean(prim, lambda u: 'awake_eo', c),
                             sessions_neg=int(sum(N[(u, 'drug')][c] < 0 for u in prim)))
    part[p] = out_
# 2. sign control: movers' direction, ratio to their drug mean, for EO in drug sessions, sleep, EO in sleep sessions
mv = m['movers']; C = np.array([x['coord'] for x in mv]); sg = np.sign([x['net_h1'] for x in mv])
dm = np.array([mean(prim, lambda u: 'drug', c) for c in C]) * sg
def ratio(us, s):
    v = np.array([mean(us, s, c) for c in C]) * sg; return dict(median_ratio=float(np.median(v / dm)), frac_same_sign=float(np.mean(v > 0)), median_signed=float(np.median(v)))
sleep_sessions = {u: float(np.mean([np.sign(N[(u, 'sleep')][c]) * s_ for c, s_ in zip(C, sg)])) for u in slp}
fam = lambda x: next((k for k in ('pcorr', 'cce_kozachenko', 'cohmag_multitaper_mean_fs-1_fmin-0-25', 'pdist_chebyshev', 'coint', 'kernel', 'bary', 'xcorr')
                      if k in x['a'] or k in x['b']), 'other')
fams = {}
for i, x in enumerate(mv):
    fams.setdefault(fam(x), []).append(i)
sleep_by_family = {k: dict(n=len(v), frac_same_sign=float(np.mean([np.mean([N[(u, 'sleep')][C[i]] for u in slp]) * sg[i] > 0 for i in v])),
                           eo_in_sleep_frac_same=float(np.mean([np.mean([N[(u, 'awake_eo')][C[i]] for u in slp]) * sg[i] > 0 for i in v])))
                   for k, v in fams.items()}
signs = dict(eo_in_drug_sessions=ratio(prim, lambda u: 'awake_eo'), sleep=ratio(slp, lambda u: 'sleep'), eo_in_sleep_sessions=ratio(slp, lambda u: 'awake_eo'),
             sleep_session_mean_sign_agreement=sleep_sessions, sleep_animals=sorted({animal[u] for u in slp}), by_family=sleep_by_family)
# 4. hub SPIs: movers / usable partners, sign split, top chains
usable = fin & relok & ceil
hubs = {}
for s, n in list(m['per_spi'].items())[:8]:
    i = spis.index(s); partners = int(usable[(a == i) | (b == i)].sum()); L = [x for x in mv if s in (x['a'], x['b'])]
    hubs[s] = dict(movers=n, usable_partners=partners, rate=n / max(partners, 1), up=sum(x['net_h1'] > 0 for x in L), down=sum(x['net_h1'] < 0 for x in L),
                   top=[dict(pair=f"{x['a']}|{x['b']}", net=round(x['net_h1'], 3), chain=x['chain']) for x in sorted(L, key=lambda x: -abs(x['net_h1']))[:3]])
res = dict(accounting=acc, partial_correlation=part, sign_control=signs, hubs=hubs)
Path(out).write_text(json.dumps(res, indent=1, default=float)); print(json.dumps(res, indent=1, default=float)[:9000])
