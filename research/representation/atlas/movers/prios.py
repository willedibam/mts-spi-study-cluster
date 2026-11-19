"""PRIOS full-pyspi screen: minimal pairs beyond a per-window joint IAAFT null, propofol against awake. Rules: prios-prereg.md.

bank:  python prios.py bank <S-full windows dir> <atlas dir> <workers>      -> <atlas>/banks/prios-ns, prios-iaaft
channels: python prios.py channels <S-full windows dir> <atlas dir>  (per-window channel indices; bank also writes them)
score: python prios.py score <atlas dir> <macaque movers.json> <out.json>
pyspi runs between them through ../jobs/farm.pbs, then discovery.py reduce and assumptions.py reliability (jobs/prios.pbs).
"""
import sys, os, glob, json, zlib, itertools, numpy as np
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
for p in (ROOT, ROOT / 'src', HERE.parent / 'mi_pair', HERE): sys.path.insert(0, str(p))
from mtsspi.surrogates import multivariate_iaaft
from edges import ns

STATES, NEED, Q, FIN, REL, CEIL = ('awake', 'anaes'), 20, .05, .95, .3, .98
ANCHOR = ('mi_gaussian', 'mi_kraskov_NN-4_DCE-AUTO')


def window(args):
    """As score.py one() with every good channel: CAR over kept channels, kept ones minus the last, normal-scored; draw 0."""
    w, stim, seed = args; w = w.astype(float); keep = ~np.isin(np.arange(w.shape[1]), stim)
    v = ns((w - w[:, keep].mean(1, keepdims=True))[:, [j for j in range(w.shape[1] - 1) if keep[j]]])
    return v, multivariate_iaaft(v, np.random.default_rng(seed), final='spectrum')[0]


def bank(src, atlas, workers):
    jobs, rows = [], []
    for fn in sorted(glob.glob(f'{src}/sub-*.npz')):
        Z = np.load(fn); s = os.path.basename(fn)[:-4]
        for i, r in enumerate(json.loads(str(Z['rows']))):
            rows.append(dict({k: v for k, v in r.items() if k != 'stim'}, id=f'{s}:{r["state"]}:{i:04d}', unit=s, animal=s, agent='PF'))
            jobs.append((Z['W'][i], r.get('stim', []), zlib.crc32(f'{s}:{i}'.encode()) * 10))
    with Pool(workers) as p:
        out = p.map(window, jobs, chunksize=4)
    for k, name in enumerate(('prios-ns', 'prios-iaaft')):
        d = Path(atlas) / 'banks' / name; d.mkdir(parents=True, exist_ok=True)
        np.savez(d / 'windows.npz', **{r['id']: o[k] for r, o in zip(rows, out)})
        man = dict(dataset='prios', arm='short', fs=100, T=200, rows=[dict(r, M=int(o[k].shape[1])) for r, o in zip(rows, out)],
                   preprocessing=dict(source=str(src), rule='S-full windows; CAR over kept channels, last dropped; normal-scored'))
        if k: man['null'] = dict(kind='iaaft_spectrum_final_joint', draws=1, seed='crc32(subject:index) * 10, as mi_pair/human/score.py draw 0')
        (d / 'manifest.json').write_text(json.dumps(man, indent=1) + '\n')
    print('bank', len(rows), 'windows', flush=True)
    channels(src, atlas)


def channels(src, atlas):
    """<bank>/channels.json: each window's original channel indices (window() keeps j < M - 1 not stimulated), which
    discovery.reliability uses to align windows whose channel sets differ (Deviation 1 in prios-prereg.md)."""
    ch = {}
    for fn in sorted(glob.glob(f'{src}/sub-*.npz')):
        Z = np.load(fn); s = os.path.basename(fn)[:-4]; M = Z['W'].shape[2]
        for i, r in enumerate(json.loads(str(Z['rows']))):
            ch[f'{s}:{r["state"]}:{i:04d}'] = [j for j in range(M - 1) if j not in set(r.get('stim', []))]
    for name in ('prios-ns', 'prios-iaaft'):
        d = Path(atlas) / 'banks' / name; man = json.loads((d / 'manifest.json').read_text())
        assert all(len(ch[r['id']]) == r['M'] for r in man['rows']), name
        (d / 'channels.json').write_text(json.dumps({r['id']: ch[r['id']] for r in man['rows']}) + '\n')
    print('channels', len(ch), 'windows', flush=True)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); a = np.empty(n)
    a[o] = np.maximum.accumulate(np.minimum(p[o] * (n - np.arange(n)), 1)); return a


def flips(n):
    return np.array(list(itertools.product((1, -1), repeat=n)), float)


def macaque(atlas, spis):
    """Macaque drug net per coordinate (KT, KTMD, PF pooled; movers.py net_contrast, all windows) and its usable mask."""
    import movers as mv
    D = Path(atlas) / 'discovery'; Z = {}
    for k, name in mv.BANKS.items():
        with np.load(D / f'{name}-reduced.npz') as d:
            Z[k] = d['z']
            if k == 'real': rows = json.loads(str(d['rows'])); assert [str(x) for x in d['spis']] == spis
    rel = {k: v['reliability'] for k, v in json.loads((D / 'neurotycho-short-reliability-all.json').read_text())['reference_state'].items()}
    H = mv.halves(rows); a, b = np.triu_indices(len(spis), 1)
    prim = sorted({r['unit'] for r in rows if r['agent'] in mv.PRIMARY})
    z0 = np.nanmean(Z['real'][[i for i, r in enumerate(rows) if r['state'] == mv.REF]], 0)
    ok = np.all([np.isfinite(v).all(0) for v in Z.values()], 0) & (np.abs(z0) < CEIL)
    ok &= np.array([rel.get(spis[i], 0) >= REL and rel.get(spis[j], 0) >= REL for i, j in zip(a, b)])
    net = np.mean([mv.net_contrast(Z, H, u, 'drug', 'all') for u in prim], 0)
    return net, ok


def score(atlas, movers_json, out):
    from scipy.stats import ttest_1samp, t as tdist
    from research.representation.atlas.assumptions import edges, READINGS
    D = Path(atlas) / 'discovery'
    R, S = (np.load(D / f'prios-{b}-reduced.npz') for b in ('ns', 'iaaft'))
    spis = [str(x) for x in R['spis']]; assert spis == [str(x) for x in S['spis']] and list(R['ids']) == list(S['ids'])
    rows = json.loads(str(R['rows'])); zr, zs, dr, ds = R['z'], S['z'], R['zdc'], S['zdc']
    K = len(spis); a, b = np.triu_indices(K, 1); idx = {(spis[i], spis[j]): c for c, (i, j) in enumerate(zip(a, b))}
    co = lambda p, q: idx.get((p, q), idx.get((q, p)))
    subj = np.array([r['unit'] for r in rows]); st = np.array([r['state'] for r in rows])
    keep = [s for s in sorted(set(subj)) if min(((subj == s) & (st == x)).sum() for x in STATES) >= NEED]
    G = {(s, x): np.flatnonzero((subj == s) & (st == x)) for s in keep for x in STATES}
    fin = np.ones(len(a), bool)
    for ix in G.values():
        for Z in (zr, zs): fin &= np.isfinite(Z[ix]).mean(0) >= FIN
    m = lambda Z, ix: np.nanmean(Z[ix], 0)
    net = lambda s, x, P=zr, N=zs: m(P, G[s, x]) - m(N, G[s, x])
    Dn = np.array([net(s, 'anaes') - net(s, 'awake') for s in keep])
    Dr = np.array([m(zr, G[s, 'anaes']) - m(zr, G[s, 'awake']) for s in keep])
    Ddc = np.array([net(s, 'anaes', dr, ds) - net(s, 'awake', dr, ds) for s in keep])

    def half(s):
        ix = G[s, 'awake']; h = len(ix) // 2
        return (m(zr, ix[:h]) - m(zs, ix[:h])) - (m(zr, ix[h:2 * h]) - m(zs, ix[h:2 * h]))
    Cn = np.array([half(s) for s in keep])
    z0 = np.mean([m(zr, G[s, 'awake']) for s in keep], 0)
    rel = {k: v['reliability'] for k, v in json.loads((D / 'prios-ns-reliability-all.json').read_text())['reference_state'].items()}
    relok = np.array([rel.get(spis[i], 0) >= REL and rel.get(spis[j], 0) >= REL for i, j in zip(a, b)])
    base = fin & (np.abs(z0) < CEIL); usable = base & relok
    aw = np.concatenate([G[s, 'awake'] for s in keep]); nr2 = np.nanmean(R['node_r2'][aw], 0)
    mins = np.zeros(len(a), bool); cls = {}
    for c_, p, q in edges(spis):
        c = co(p, q); mins[c] = True; cls[c] = c_
    n = len(keep); F = flips(n)
    try:
        mq, mq_ok = macaque(atlas, spis)
    except FileNotFoundError:
        mq, mq_ok = np.full(len(a), np.nan), np.zeros(len(a), bool)

    def screen(mask, X=Dn):
        c = np.flatnonzero(mask); x = X[:, c]; mu = x.mean(0)
        p = ttest_1samp(x, 0).pvalue; q = bh(p)
        agree = (np.sign(x) == np.sign(mu)).sum(0) >= n - 1
        return c, p, q, (q <= Q) & agree

    def row(c, p=None, q=None):
        x = Dn[:, c]; se = x.std(ddof=1) / np.sqrt(n); h = tdist.ppf(.975, n - 1) * se; i, j = a[c], b[c]
        rd = READINGS.get(cls.get(c), (None, None, None))
        return dict(pair=f'{spis[i]}|{spis[j]}', cls=cls.get(c), character=rd[0], literal=rd[1], confound=rd[2],
                    mean=float(x.mean()), ci95=[float(x.mean() - h), float(x.mean() + h)], negative=int((x < 0).sum()), n=n,
                    p=None if p is None else float(p), q=None if q is None else float(q), subjects=[round(float(v), 4) for v in x],
                    d_real=float(Dr[:, c].mean()), d_surr=float((Dr[:, c] - x).mean()), d_dc=float(Ddc[:, c].mean()),
                    z0=float(z0[c]), node_r2=[float(nr2[i]), float(nr2[j])], reliability=[rel.get(spis[i]), rel.get(spis[j])],
                    usable=bool(usable[c]), macaque_drug_net=None if not np.isfinite(mq[c]) else float(mq[c]), macaque_usable=bool(mq_ok[c]))

    res = dict(subjects=keep, windows={f'{s}:{x}': int(len(G[s, x])) for s, x in G}, spis=K, coordinates=int(len(a)),
               accounting=dict(finite=int(fin.sum()), plus_ceiling=int(base.sum()), plus_reliability=int(usable.sum()),
                               minimal=int(mins.sum()), minimal_usable=int((mins & usable).sum()),
                               spis_failing_reliability=sorted(s for s in spis if rel.get(s, 0) < REL)))
    # 1. primary discovery: usable minimal pairs, with the awake split-half control
    c, p, q, disc = screen(mins & usable); _, _, _, ctrl = screen(mins & usable, Cn)
    fdp = float(ctrl.sum() / max(disc.sum(), 1))
    res['primary'] = dict(tested=int(len(c)), discoveries=int(disc.sum()), control_discoveries=int(ctrl.sum()), fdp_estimate=fdp,
                          claims=bool(fdp <= .2), rows=sorted([row(c[k], p[k], q[k]) for k in np.flatnonzero(disc)], key=lambda r: r['p']),
                          top_not_passing=sorted([row(c[k], p[k], q[k]) for k in np.argsort(p)[:30] if not disc[k]], key=lambda r: r['p']))
    # 2. macaque predictions
    mv = json.loads(Path(movers_json).read_text())['movers']
    m11 = [(co(x['a'], x['b']), np.sign(x['net_h1'])) for x in mv if x['cls']]
    tested = [(c_, s_) for c_, s_ in m11 if usable[c_]]
    pp = [ttest_1samp(s_ * Dn[:, c_], 0, alternative='greater').pvalue for c_, s_ in tested]
    res['movers_minimal'] = dict(tested=len(tested), rows=[dict(row(c_), predicted=int(s_), p_one=float(p1), p_holm=float(h1))
                                                           for (c_, s_), p1, h1 in zip(tested, pp, holm(pp) if pp else [])],
                                 not_usable=[f'{spis[a[c_]]}|{spis[b[c_]]}' for c_, s_ in m11 if not usable[c_]])
    allm = [(co(x['a'], x['b']), np.sign(x['net_h1'])) for x in mv]; allm = [(c_, s_) for c_, s_ in allm if usable[c_]]
    if allm:
        C_, S_ = np.array([x[0] for x in allm]), np.array([x[1] for x in allm])
        stat = lambda e: float(np.mean(S_ * (e[:, None] * Dn[:, C_]).mean(0)))
        null = np.array([stat(e) for e in F]); obs = stat(np.ones(n))
        res['movers_all'] = dict(tested=len(allm), signed_mean=obs, frac_same_sign=float(np.mean(S_ * Dn[:, C_].mean(0) > 0)),
                                 p_signflip=float(np.mean(null >= obs - 1e-12)))
    for name, mask in (('minimal', mins), ('all', np.ones(len(a), bool))):
        cc = np.flatnonzero(mask & usable & mq_ok)
        if len(cc) > 2:
            r_ = lambda e: float(np.corrcoef((e[:, None] * Dn[:, cc]).mean(0), mq[cc])[0, 1])
            null = np.array([r_(e) for e in F]); obs = r_(np.ones(n))
            res[f'concordance_{name}'] = dict(coordinates=int(len(cc)), r=obs, p_signflip=float(np.mean(null >= obs - 1e-12)),
                                              same_sign=float(np.mean(np.sign(Dn[:, cc].mean(0)) == np.sign(mq[cc]))))
    # 3. secondary
    c, p, q, disc = screen(mins & base); _, _, _, ctrl = screen(mins & base, Cn)
    res['minimal_no_reliability'] = dict(tested=int(len(c)), discoveries=int(disc.sum()), control_discoveries=int(ctrl.sum()),
                                         rows=sorted([row(c[k], p[k], q[k]) for k in np.flatnonzero(disc & ~usable[c])], key=lambda r: r['p']))
    c, p, q, disc = screen(usable); _, _, _, ctrl = screen(usable, Cn)
    part = {}
    for k in c[disc]:
        for s_ in (spis[a[k]], spis[b[k]]): part[s_] = part.get(s_, 0) + 1
    res['all_usable'] = dict(tested=int(len(c)), discoveries=int(disc.sum()), control_discoveries=int(ctrl.sum()),
                             rising=int((Dn[:, c[disc]].mean(0) > 0).sum()), by_spi=dict(sorted(part.items(), key=lambda kv: -kv[1])[:25]))
    an = co(*ANCHOR); res['anchor'] = row(an)
    Path(out).write_text(json.dumps(res, indent=1, default=float) + '\n')
    pr = res['primary']
    print(f"subjects {n}; usable minimal {pr['tested']}; discoveries {pr['discoveries']} (control {pr['control_discoveries']}, fdp {fdp:.2f})")
    for r in pr['rows']: print(f"  {r['pair']} [{r['cls']}] {r['mean']:+.4f} {r['negative']}/{n} neg q {r['q']:.3g} dc {r['d_dc']:+.4f} nodeR2 {r['node_r2'][0]:.2f},{r['node_r2'][1]:.2f} macaque {r['macaque_drug_net']}")
    print('movers_minimal', [(r['pair'], round(r['mean'], 4), round(r['p_holm'], 3)) for r in res['movers_minimal']['rows']])
    print({k: res.get(k) for k in ('movers_all', 'concordance_minimal', 'concordance_all')})
    print('all usable', {k: v for k, v in res['all_usable'].items() if k != 'by_spi'}, 'anchor', round(res['anchor']['mean'], 4), res['anchor']['negative'])


if __name__ == '__main__':
    if sys.argv[1] == 'bank':
        bank(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif sys.argv[1] == 'channels':
        channels(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'score':
        score(sys.argv[2], sys.argv[3], sys.argv[4])
