"""Macaque full-pyspi screen against per-window joint IAAFT (the PRIOS construction), and its test on PRIOS. Rules: matched-prereg.md.

bank:  python matched.py bank <atlas dir> <workers>                     -> <atlas>/banks/nt-ns, nt-iaaft (from neurotycho-short)
score: python matched.py score <atlas dir> <out.json>                   (macaque tests; also writes <out stem>-deltas.npz)
cross: python matched.py cross <atlas dir> <matched.json> <out.json>    (test on PRIOS; needs prios-{ns,iaaft}-reduced, reliability)
pyspi runs between bank and score through ../jobs/farm.pbs, then discovery.py reduce and assumptions.py reliability (jobs/matched.pbs).
"""
import sys, json, numpy as np
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
for p in (ROOT, ROOT / 'src', HERE): sys.path.insert(0, str(p))
from prios import bh, holm, flips, macaque, ANCHOR, Q, FIN, REL, CEIL, NEED

PRIMARY, REF, STATES = ('KT', 'KTMD', 'PF'), 'awake_ec', ('awake_ec', 'awake_eo', 'drug')
SPEC = {'MD drug': ('MD', 'drug'), 'sleep': ('SLP', 'sleep'), 'recovery': (PRIMARY, 'recovery_ec')}


def window(args):
    """Normal-scored window and its spectrum-final joint IAAFT draw, as families_iaaft.variants iaaft_s0."""
    from scipy.stats import norm, rankdata
    from mtsspi.surrogates import multivariate_iaaft
    from research.representation.atlas.extract import seed
    wid, x = args
    v = norm.ppf((np.apply_along_axis(rankdata, 0, x) - .5) / len(x))
    return v, multivariate_iaaft(v, np.random.default_rng([seed(wid), 0, 1]), final='spectrum')[0]


def bank(atlas, workers):
    src = Path(atlas) / 'banks' / 'neurotycho-short'; man = json.loads((src / 'manifest.json').read_text())
    with np.load(src / 'windows.npz') as W:
        jobs = [(r['id'], W[r['id']].astype(float)) for r in man['rows']]
    with Pool(workers) as p:
        out = p.map(window, jobs, chunksize=4)
    for k, name in enumerate(('nt-ns', 'nt-iaaft')):
        d = Path(atlas) / 'banks' / name; d.mkdir(parents=True, exist_ok=True)
        np.savez(d / 'windows.npz', **{wid: o[k] for (wid, _), o in zip(jobs, out)})
        m = dict({k_: v for k_, v in man.items() if k_ != 'rows'}, rows=man['rows'],
                 preprocessing=dict(source=str(src), rule='neurotycho-short windows, normal-scored per channel'))
        if k: m['null'] = dict(kind='iaaft_spectrum_final_joint', draws=1, seed='[extract.seed(id), 0, 1], as families_iaaft iaaft_s0')
        (d / 'manifest.json').write_text(json.dumps(m, indent=1) + '\n')
    print('bank', len(jobs), 'windows', flush=True)


def _load(atlas, a_, b_):
    D = Path(atlas) / 'discovery'
    R, S = (np.load(D / f'{n}-reduced.npz') for n in (a_, b_))
    spis = [str(x) for x in R['spis']]; assert spis == [str(x) for x in S['spis']] and list(R['ids']) == list(S['ids'])
    return R, S, spis, json.loads(str(R['rows']))


def _minimal(spis):
    from research.representation.atlas.assumptions import edges
    K = len(spis); a, b = np.triu_indices(K, 1); idx = {(spis[i], spis[j]): c for c, (i, j) in enumerate(zip(a, b))}
    co = lambda p, q: idx.get((p, q), idx.get((q, p)))
    mins = np.zeros(len(a), bool); cls = {}
    for c_, p, q in edges(spis):
        c = co(p, q); mins[c] = True; cls[c] = c_
    return a, b, co, mins, cls


def _rel(atlas, name, spis, a, b):
    rel = {k: v['reliability'] for k, v in json.loads((Path(atlas) / 'discovery' / f'{name}-reliability-all.json').read_text())['reference_state'].items()}
    return rel, np.array([rel.get(spis[i], 0) >= REL and rel.get(spis[j], 0) >= REL for i, j in zip(a, b)])


def score(atlas, out):
    from scipy.stats import ttest_1samp
    from research.representation.atlas.assumptions import READINGS
    R, S, spis, rows = _load(atlas, 'nt-ns', 'nt-iaaft'); zr, zs, dr, ds = R['z'], S['z'], R['zdc'], S['zdc']
    a, b, co, mins, cls = _minimal(spis)
    unit = np.array([r['unit'] for r in rows]); st = np.array([r['state'] for r in rows])
    agent = {r['unit']: r['agent'] for r in rows}; animal = {r['unit']: r['animal'] for r in rows}
    prim = sorted(u for u in set(unit) if agent[u] in PRIMARY and all(((unit == u) & (st == s)).any() for s in STATES))
    G = lambda u, s: np.flatnonzero((unit == u) & (st == s))
    m = lambda Z, ix: np.nanmean(Z[ix], 0)
    net = lambda u, s, P=zr, N=zs: m(P, G(u, s)) - m(N, G(u, s))
    fin = np.ones(len(a), bool)
    for u in prim:
        for s in STATES:
            for Z in (zr, zs): fin &= np.isfinite(Z[G(u, s)]).mean(0) >= FIN
    z0 = np.mean([m(zr, G(u, REF)) for u in prim], 0)
    rel, relok = _rel(atlas, 'nt-ns', spis, a, b)
    base = fin & (np.abs(z0) < CEIL); usable = base & relok
    Dn = np.array([net(u, 'drug') - net(u, REF) for u in prim]); Cn = np.array([net(u, 'awake_eo') - net(u, REF) for u in prim])
    Dr = np.array([m(zr, G(u, 'drug')) - m(zr, G(u, REF)) for u in prim])
    Ddc = np.array([net(u, 'drug', dr, ds) - net(u, REF, dr, ds) for u in prim])
    an = np.array([animal[u] for u in prim]); animals = sorted(set(an))
    nr2 = np.nanmean(R['node_r2'][np.concatenate([G(u, REF) for u in prim])], 0)
    spec = {}
    for k, (ag, s) in SPEC.items():
        us = sorted(u for u in set(unit) if (agent[u] in ag if isinstance(ag, tuple) else agent[u] == ag) and len(G(u, s)) and len(G(u, REF)))
        spec[k] = (len(us), np.array([net(u, s) - net(u, REF) for u in us]).mean(0) if us else None)
    try:
        old, old_ok = macaque(atlas, spis)
    except FileNotFoundError:
        old, old_ok = np.full(len(a), np.nan), np.zeros(len(a), bool)
    n = len(prim)

    def screen(mask, X=Dn):
        c = np.flatnonzero(mask); x = X[:, c]; mu = x.mean(0)
        p = ttest_1samp(x, 0).pvalue; q = bh(p)
        every = np.all([np.sign(x[an == k].mean(0)) == np.sign(mu) for k in animals], 0)
        return c, p, q, (q <= Q) & every

    def row(c, p=None, q=None):
        x = Dn[:, c]; i, j = a[c], b[c]; rd = READINGS.get(cls.get(c), (None, None, None))
        am = [float(x[an == k].mean()) for k in animals]
        return dict(pair=f'{spis[i]}|{spis[j]}', cls=cls.get(c), character=rd[0], literal=rd[1], confound=rd[2],
                    mean=float(x.mean()), sessions_same_sign=int((np.sign(x) == np.sign(x.mean())).sum()), n=n,
                    p=None if p is None else float(p), q=None if q is None else float(q), animals=dict(zip(animals, am)),
                    t_animals=float(ttest_1samp(am, 0).statistic), d_real=float(Dr[:, c].mean()), d_surr=float((Dr[:, c] - x).mean()),
                    d_dc=float(Ddc[:, c].mean()), eyes_open=float(Cn[:, c].mean()), z0=float(z0[c]),
                    node_r2=[float(nr2[i]), float(nr2[j])], reliability=[rel.get(spis[i]), rel.get(spis[j])], usable=bool(usable[c]),
                    specificity={k: (None if v is None else float(v[c])) for k, (_, v) in spec.items()},
                    old_net=None if not np.isfinite(old[c]) else float(old[c]))

    res = dict(sessions=prim, animals=animals, spis=len(spis), coordinates=int(len(a)),
               specificity_sessions={k: v[0] for k, v in spec.items()},
               accounting=dict(finite=int(fin.sum()), plus_ceiling=int(base.sum()), plus_reliability=int(usable.sum()),
                               minimal=int(mins.sum()), minimal_usable=int((mins & usable).sum()),
                               spis_failing_reliability=sorted(s for s in spis if rel.get(s, 0) < REL)))
    c, p, q, disc = screen(mins & usable); _, _, _, ctrl = screen(mins & usable, Cn)
    fdp = float(ctrl.sum() / max(disc.sum(), 1))
    res['primary'] = dict(tested=int(len(c)), undefined_p=int((~np.isfinite(p)).sum()), discoveries=int(disc.sum()), control_discoveries=int(ctrl.sum()), fdp_estimate=fdp,
                          claims=bool(fdp <= .2), rising=int((Dn[:, c[disc]].mean(0) > 0).sum()),
                          rows=sorted([row(c[k], p[k], q[k]) for k in np.flatnonzero(disc)], key=lambda r: r['p']),
                          top_not_passing=sorted([row(c[k], p[k], q[k]) for k in np.argsort(p)[:30] if not disc[k]], key=lambda r: r['p']))
    c, p, q, disc = screen(mins & base); _, _, _, ctrl = screen(mins & base, Cn)
    res['minimal_no_reliability'] = dict(tested=int(len(c)), discoveries=int(disc.sum()), control_discoveries=int(ctrl.sum()),
                                         rows=sorted([row(c[k], p[k], q[k]) for k in np.flatnonzero(disc & ~usable[c])], key=lambda r: r['p']))
    c, p, q, disc = screen(usable); _, _, _, ctrl = screen(usable, Cn)
    part = {}
    for k in c[disc]:
        for s_ in (spis[a[k]], spis[b[k]]): part[s_] = part.get(s_, 0) + 1
    res['all_usable'] = dict(tested=int(len(c)), discoveries=int(disc.sum()), control_discoveries=int(ctrl.sum()),
                             rising=int((Dn[:, c[disc]].mean(0) > 0).sum()), by_spi=dict(sorted(part.items(), key=lambda kv: -kv[1])[:25]))
    res['anchor'] = row(co(*ANCHOR))
    Path(out).write_text(json.dumps(res, indent=1, default=float) + '\n')
    np.savez(Path(out).with_name(Path(out).stem + '-deltas.npz'), Dn=Dn, usable=usable, spis=np.array(spis), sessions=np.array(prim))
    pr = res['primary']
    print(f"sessions {n}, animals {len(animals)}; usable minimal {pr['tested']}; discoveries {pr['discoveries']} "
          f"(control {pr['control_discoveries']}, fdp {fdp:.2f}, rising {pr['rising']})")
    for r in pr['rows']:
        print(f"  {r['pair']} [{r['cls']}] {r['mean']:+.4f} {r['sessions_same_sign']}/{n} q {r['q']:.3g} t3 {r['t_animals']:+.1f} "
              f"dc {r['d_dc']:+.4f} eo {r['eyes_open']:+.4f} nodeR2 {r['node_r2'][0]:.2f},{r['node_r2'][1]:.2f}")
    print('all usable', {k: v for k, v in res['all_usable'].items() if k != 'by_spi'}, 'anchor', round(res['anchor']['mean'], 4),
          res['anchor']['sessions_same_sign'])


def prios_deltas(atlas):
    """PRIOS patient Delta and usability, exactly as prios.score."""
    R, S, spis, rows = _load(atlas, 'prios-ns', 'prios-iaaft'); zr, zs = R['z'], S['z']
    a, b, co, mins, cls = _minimal(spis)
    subj = np.array([r['unit'] for r in rows]); st = np.array([r['state'] for r in rows])
    keep = [s for s in sorted(set(subj)) if min(((subj == s) & (st == x)).sum() for x in ('awake', 'anaes')) >= NEED]
    G = {(s, x): np.flatnonzero((subj == s) & (st == x)) for s in keep for x in ('awake', 'anaes')}
    fin = np.ones(len(a), bool)
    for ix in G.values():
        for Z in (zr, zs): fin &= np.isfinite(Z[ix]).mean(0) >= FIN
    m = lambda Z, ix: np.nanmean(Z[ix], 0)
    net = lambda s, x: m(zr, G[s, x]) - m(zs, G[s, x])
    Dn = np.array([net(s, 'anaes') - net(s, 'awake') for s in keep])
    z0 = np.mean([m(zr, G[s, 'awake']) for s in keep], 0)
    _, relok = _rel(atlas, 'prios-ns', spis, a, b)
    return spis, keep, Dn, fin & (np.abs(z0) < CEIL) & relok, mins, co


def cross(atlas, matched_json, out):
    from scipy.stats import ttest_1samp
    spis, keep, Dn, usable, mins, co = prios_deltas(atlas)
    M = json.loads(Path(matched_json).read_text())
    with np.load(Path(matched_json).with_name(Path(matched_json).stem + '-deltas.npz')) as d:
        assert [str(x) for x in d['spis']] == spis
        mq, mq_ok = d['Dn'].mean(0), d['usable']
    n = len(keep); F = flips(n); res = dict(patients=keep, macaque_claims=M['primary']['claims'])
    if M['primary']['claims']:
        pred = [(co(*r['pair'].split('|')), np.sign(r['mean']), r['pair']) for r in M['primary']['rows']]
        res['not_usable_in_prios'] = [nm for c, s, nm in pred if not usable[c]]
        pred = [(c, s, nm) for c, s, nm in pred if usable[c]]
        if pred:
            C_, S_ = np.array([x[0] for x in pred]), np.array([x[1] for x in pred])
            stat = lambda e: float(np.mean(S_ * (e[:, None] * Dn[:, C_]).mean(0)))
            null = np.array([stat(e) for e in F]); obs = stat(np.ones(n))
            res['A_aggregate'] = dict(tested=len(pred), signed_mean=obs, p_signflip=float(np.mean(null >= obs - 1e-12)),
                                      frac_same_sign=float(np.mean(S_ * Dn[:, C_].mean(0) > 0)))
            pp = [ttest_1samp(s * Dn[:, c], 0, alternative='greater').pvalue for c, s, _ in pred]
            res['B_per_pair'] = [dict(pair=nm, predicted=int(s), mean=float(Dn[:, c].mean()), same_sign=int((s * Dn[:, c] > 0).sum()),
                                      p_one=float(p1), p_holm=float(h1)) for (c, s, nm), p1, h1 in zip(pred, pp, holm(pp))]
    for name, mask in (('minimal', mins), ('all', np.ones(len(mins), bool))):
        cc = np.flatnonzero(mask & usable & mq_ok)
        if len(cc) > 2:
            r_ = lambda e: float(np.corrcoef((e[:, None] * Dn[:, cc]).mean(0), mq[cc])[0, 1])
            null = np.array([r_(e) for e in F]); obs = r_(np.ones(n))
            res[f'C_concordance_{name}'] = dict(coordinates=int(len(cc)), r=obs, p_signflip=float(np.mean(null >= obs - 1e-12)),
                                                same_sign=float(np.mean(np.sign(Dn[:, cc].mean(0)) == np.sign(mq[cc]))))
    Path(out).write_text(json.dumps(res, indent=1, default=float) + '\n')
    print({k: v for k, v in res.items() if k != 'B_per_pair'})
    for r in res.get('B_per_pair', []): print(f"  {r['pair']} pred {r['predicted']:+d} {r['mean']:+.4f} {r['same_sign']}/{n} holm {r['p_holm']:.3g}")


if __name__ == '__main__':
    if sys.argv[1] == 'bank':
        bank(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == 'score':
        score(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'cross':
        cross(sys.argv[2], sys.argv[3], sys.argv[4])
