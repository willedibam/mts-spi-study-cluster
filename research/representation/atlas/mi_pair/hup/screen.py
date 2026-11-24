"""HUP full-pyspi screen: SOZ set against matched non-SOZ sets, beyond per-window joint IAAFT (rules: ../preregs/hup-screen-prereg.md).

bank:  python screen.py bank <windows dir> <MI edges dir> <atlas dir> <workers>  -> <atlas>/banks/hup-ns, hup-iaaft
score: python screen.py score <atlas dir> <windows dir> <out.json>
pyspi runs between them through ../../jobs/farm.pbs, then discovery.py reduce and assumptions.py reliability (../jobs/hup-screen.pbs).

Sets are the MI test's own (score.matched, same pool, seed and |r| input): S and its first N_SETS accepted primary
matched sets. Only the Gaussian column of the MI edges files is read (|r| for matching); no kNN MI is read.
"""
import sys, os, glob, json, zlib, numpy as np
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[4]
for p in (ROOT, ROOT / 'src', HERE.parent, HERE): sys.path.insert(0, str(p))
from mtsspi.surrogates import multivariate_iaaft
from edges import ns
from score import matched

N_SETS, MIN_S, Q, FIN, REL, CEIL, MIN_WIN = 20, 8, .05, .95, .3, .98, 10
ANCHOR = ('mi_gaussian', 'mi_kraskov_NN-4_DCE-AUTO')


def subject_sets(src, edir, s):
    """(S, [N_1..N_20]) as score.sets builds them for the primary pool, or None if the subject is not included."""
    X0 = np.load(f'{edir}/{s}.npz')['X'][:, 0].astype(float)  # Gaussian MI only
    m = json.loads(str(np.load(f'{src}/{s}.npz')['meta'])); M = len(m['names'])
    iu = np.triu_indices(M, 1); E = -np.ones((M, M), int); E[iu] = np.arange(len(iu[0]))
    eidx = lambda A: E[np.ix_(A, A)][np.triu_indices(len(A), 1)]
    xyz = np.array([p if p else [np.nan] * 3 for p in m['xyz']], float); D = np.linalg.norm(xyz[:, None] - xyz[None], axis=2)
    R = np.sqrt(1 - np.exp(-2 * X0)).mean(0); has = np.isfinite(xyz).all(1); soz = np.array(m['soz']); rsc = np.array(m['resect'])
    S = [i for i in range(M) if soz[i] and has[i]]
    if len(S) < MIN_S: return None
    _, N, _ = matched(S, [i for i in range(M) if not soz[i] and not rsc[i] and has[i]], D, R, eidx, zlib.crc32(s.encode()))
    return (S, N[:N_SETS]) if len(N) >= N_SETS else None


def window(args):
    """Normal-scored full-implant window and its joint IAAFT draw 0 (identical to score.one's draw 0), sliced per set."""
    w, seed, sets = args; v = ns(w.astype(float)); u = multivariate_iaaft(v, np.random.default_rng(seed), final='spectrum')[0]
    return [(v[:, A], u[:, A]) for A in sets]


def bank(src, edir, atlas, workers):
    jobs, rows, sel = [], [], {}
    for f in sorted(glob.glob(f'{edir}/sub-*.npz')):
        s = os.path.basename(f)[:-4]; r = subject_sets(src, edir, s)
        if r is None: continue
        S, N = r; sets = [S] + N; labels = ['S'] + [f'N{j:02d}' for j in range(1, len(N) + 1)]; sel[s] = dict(zip(labels, sets))
        Z = np.load(f'{src}/{s}.npz'); meta = json.loads(str(Z['meta'])); spike = Z['spike']
        for i, rw in enumerate(json.loads(str(Z['rows']))):
            jobs.append((Z['W'][i], zlib.crc32(f'{s}:{i}'.encode()) * 10, sets))
            rows.append([dict(id=f'{s}:{lab}:{i:04d}', unit=f'{s}:{lab}', state='soz' if lab == 'S' else 'matched', subject=s,
                              set=lab, window=i, run=rw['run'], spike=bool(spike[i, A].any()), implant=meta['implant'],
                              engel=meta['engel'], animal=s, agent='hup') for lab, A in zip(labels, sets)])
    with Pool(workers) as p:
        out = p.map(window, jobs, chunksize=2)
    flat = [(r, o) for rr, oo in zip(rows, out) for r, o in zip(rr, oo)]
    for k, name in enumerate(('hup-ns', 'hup-iaaft')):
        d = Path(atlas) / 'banks' / name; d.mkdir(parents=True, exist_ok=True)
        np.savez(d / 'windows.npz', **{r['id']: o[k] for r, o in flat})
        man = dict(dataset='hup', arm='long', fs=100, T=1000, rows=[dict(r, M=int(o[k].shape[1])) for r, o in flat],
                   preprocessing=dict(source=str(src), rule='hup-prereg.md windows; CAR, one channel dropped; full-implant normal score; set columns'))
        if k: man['null'] = dict(kind='iaaft_spectrum_final_joint', draws=1, scope='full implant, then set columns',
                                 seed='crc32(subject:index) * 10, as score.py draw 0')
        (d / 'manifest.json').write_text(json.dumps(man, indent=1) + '\n')
    (Path(atlas) / 'banks' / 'hup-sets.json').write_text(json.dumps(sel) + '\n')
    print('bank', len(sel), 'subjects', len(flat), 'recordings per bank', flush=True)


def reliability(atlas, chunk=40, jobs=16):
    """assumptions.reliability_all with one change: an SPI enters if its profile is valid in >= FIN of the real bank's
    recordings (reliability_all requires every window, which one bad 8-contact recording would fail for the whole screen).
    Reference state: the matched sets; per unit (subject:set), the mean correlation between windows' edge profiles."""
    from research.representation.atlas.discovery import reliability as rel_
    A = Path(atlas); r = np.load(A / 'discovery' / 'hup-ns-reduced.npz', allow_pickle=False)
    spis = [str(s) for i, s in enumerate(r['spis']) if r['valid'][:, i].mean() >= FIN]
    out = A / 'discovery' / 'hup-ns-reliability-all.json'; res = dict(reference='matched', reference_state={}, spis_valid=len(spis))
    for i in range(0, len(spis), chunk):
        part = out.with_name(f'{out.stem}.part{i}.json')
        rel_(A / 'banks' / 'hup-ns', A / 'pyspi' / 'hup-ns', part, 'matched', named=[('self', s, s) for s in spis[i:i + chunk]], jobs=jobs)
        res['reference_state'].update(json.loads(part.read_text())['reference_state']); part.unlink()
    out.write_text(json.dumps(res, indent=1) + '\n')


def bh(p):
    p = np.where(np.isfinite(p), p, 1.); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def score(atlas, src, out):
    from scipy.stats import ttest_1samp, binomtest, t as tdist
    from research.representation.atlas.assumptions import edges, READINGS
    Dd = Path(atlas) / 'discovery'
    R, S_ = (np.load(Dd / f'hup-{b}-reduced.npz') for b in ('ns', 'iaaft'))
    spis = [str(x) for x in R['spis']]; assert spis == [str(x) for x in S_['spis']] and list(R['ids']) == list(S_['ids'])
    rows = json.loads(str(R['rows'])); zr, zs, dr, ds = R['z'], S_['z'], R['zdc'], S_['zdc']
    K = len(spis); a, b = np.triu_indices(K, 1); idx = {(spis[i], spis[j]): c for c, (i, j) in enumerate(zip(a, b))}
    co = lambda p, q: idx.get((p, q), idx.get((q, p)))
    subj = np.array([r['subject'] for r in rows]); lab = np.array([r['set'] for r in rows]); run = np.array([r['run'] for r in rows])
    spk = np.array([r['spike'] for r in rows])
    keep = sorted(set(subj)); labs = ['S'] + [f'N{j:02d}' for j in range(1, N_SETS + 1)]
    G = {(s, l): np.flatnonzero((subj == s) & (lab == l)) for s in keep for l in labs}
    fin = np.ones(len(a), bool)
    for s in keep:
        for grp in (G[s, 'S'], np.concatenate([G[s, l] for l in labs[1:]])):
            for Z in (zr, zs): fin &= np.isfinite(Z[grp]).mean(0) >= FIN
    m = lambda Z, ix: np.nanmean(Z[ix], 0) if len(ix) >= MIN_WIN else np.full(Z.shape[1], np.nan)
    def net(s, l, P=zr, N=zs, w=None):
        ix = G[s, l] if w is None else G[s, l][w[G[s, l]]]
        return m(P, ix) - m(N, ix)
    def delta(P=zr, N=zs, w=None, subs=None):
        subs = keep if subs is None else subs
        D, C = [], []
        for s in subs:
            n_ = np.array([net(s, l, P, N, w) for l in labs])
            D.append(n_[0] - np.nanmean(n_[1:], 0)); C.append(n_[1] - np.nanmean(n_[2:], 0))
        return np.array(D), np.array(C)
    Dn, Cn = delta()
    Dr = np.array([m(zr, G[s, 'S']) - np.nanmean([m(zr, G[s, l]) for l in labs[1:]], 0) for s in keep])
    Ddc, _ = delta(dr, ds)
    z0 = np.nanmean([m(zr, np.concatenate([G[s, l] for l in labs[1:]])) for s in keep], 0)
    rel = {k: v['reliability'] for k, v in json.loads((Dd / 'hup-ns-reliability-all.json').read_text())['reference_state'].items()}
    relok = np.array([(rel.get(spis[i]) or 0) >= REL and (rel.get(spis[j]) or 0) >= REL for i, j in zip(a, b)])
    base = fin & (np.abs(z0) < CEIL); usable = base & relok
    mt = np.concatenate([G[s, l] for s in keep for l in labs[1:]]); nr2 = np.nanmean(R['node_r2'][mt], 0)
    mins = np.zeros(len(a), bool); cls = {}
    for c_, p, q in edges(spis):
        c = co(p, q); mins[c] = True; cls[c] = c_
    n = len(keep)

    def screen(mask, X=Dn):
        c = np.flatnonzero(mask); x = X[:, c]; mu = x.mean(0)
        p = ttest_1samp(x, 0).pvalue; q = bh(p)
        pos = (x > 0).sum(0); ps = np.array([binomtest(int(k), n, .5).pvalue for k in pos])
        return c, p, q, (q <= Q) & (ps < .05) & (np.sign(pos - n / 2) == np.sign(mu))

    imp = {r['subject']: r['implant'] for r in rows}; eng = {r['subject']: str(r['engel']) for r in rows}
    D1, _ = delta(w=run == 1); D2, _ = delta(w=run == 2); Dsp, _ = delta(w=~spk)
    strata = dict(ecog=[s for s in keep if 'ecog' in imp[s].lower()], seeg=[s for s in keep if 'seeg' in imp[s].lower()],
                  engel1=[s for s in keep if eng[s].strip().startswith('1')], engel_other=[s for s in keep if not eng[s].strip().startswith('1')])
    Dst = {k: (delta(subs=v)[0] if v else None) for k, v in strata.items()}

    def row(c, p=None, q=None):
        x = Dn[:, c]; se = x.std(ddof=1) / np.sqrt(n); h = tdist.ppf(.975, n - 1) * se; i, j = a[c], b[c]
        rd = READINGS.get(cls.get(c), (None, None, None)); f = lambda v: None if not np.isfinite(v) else float(v)
        r1, r2 = D1[:, c], D2[:, c]; ok = np.isfinite(r1) & np.isfinite(r2)
        return dict(pair=f'{spis[i]}|{spis[j]}', cls=cls.get(c), character=rd[0], literal=rd[1], confound=rd[2],
                    mean=float(x.mean()), ci95=[float(x.mean() - h), float(x.mean() + h)], positive=int((x > 0).sum()), n=n,
                    p=None if p is None else float(p), q=None if q is None else float(q), null_sd=float(Cn[:, c].std(ddof=1)),
                    mean_in_null_sd=float(x.mean() / Cn[:, c].std(ddof=1)), subjects=[round(float(v), 4) for v in x],
                    d_real=float(Dr[:, c].mean()), d_surr=float((Dr[:, c] - x).mean()), d_dc=f(np.nanmean(Ddc[:, c])),
                    z0=float(z0[c]), node_r2=[float(nr2[i]), float(nr2[j])], reliability=[rel.get(spis[i]), rel.get(spis[j])],
                    usable=bool(usable[c]), spike_free=f(np.nanmean(Dsp[:, c])), run1=f(np.nanmean(r1)), run2=f(np.nanmean(r2)),
                    run_same_sign=f(np.mean(np.sign(r1[ok]) == np.sign(r2[ok]))) if ok.any() else None,
                    strata={k: (None if v is None else dict(n=len(v), mean=f(np.nanmean(v[:, c])))) for k, v in Dst.items()})

    res = dict(subjects=keep, n=n, windows={s: int(len(G[s, 'S'])) for s in keep}, spis=K, coordinates=int(len(a)),
               accounting=dict(finite=int(fin.sum()), plus_ceiling=int(base.sum()), plus_reliability=int(usable.sum()),
                               minimal=int(mins.sum()), minimal_usable=int((mins & usable).sum()),
                               spis_failing_reliability=sorted(s for s in spis if (rel.get(s) or 0) < REL)))
    c, p, q, disc = screen(mins & usable); _, _, _, ctrl = screen(mins & usable, Cn)
    fdp = float(ctrl.sum() / max(disc.sum(), 1))
    res['primary'] = dict(tested=int(len(c)), undefined_p=int((~np.isfinite(p)).sum()), discoveries=int(disc.sum()),
                          control_discoveries=int(ctrl.sum()), fdp_estimate=fdp, claims=bool(fdp <= .2),
                          rows=sorted([row(c[k], p[k], q[k]) for k in np.flatnonzero(disc)], key=lambda r: r['p']),
                          top_not_passing=sorted([row(c[k], p[k], q[k]) for k in np.argsort(np.where(np.isfinite(p), p, 1))[:30] if not disc[k]], key=lambda r: r['p']))
    c, p, q, disc = screen(mins & base); _, _, _, ctrl = screen(mins & base, Cn)
    res['minimal_no_reliability'] = dict(tested=int(len(c)), discoveries=int(disc.sum()), control_discoveries=int(ctrl.sum()),
                                         rows=sorted([row(c[k], p[k], q[k]) for k in np.flatnonzero(disc & ~usable[c])], key=lambda r: r['p']))
    c, p, q, disc = screen(usable); _, _, _, ctrl = screen(usable, Cn)
    part = {}
    for k in c[disc]:
        for s_ in (spis[a[k]], spis[b[k]]): part[s_] = part.get(s_, 0) + 1
    res['all_usable'] = dict(tested=int(len(c)), discoveries=int(disc.sum()), control_discoveries=int(ctrl.sum()),
                             rising=int((Dn[:, c[disc]].mean(0) > 0).sum()), by_spi=dict(sorted(part.items(), key=lambda kv: -kv[1])[:25]))
    an = co(*ANCHOR); res['anchor'] = None if an is None else row(an)
    Path(out).write_text(json.dumps(res, indent=1, default=float) + '\n')
    pr = res['primary']
    print(f"subjects {n}; usable minimal {pr['tested']}; discoveries {pr['discoveries']} (control {pr['control_discoveries']}, fdp {fdp:.2f}, claims {pr['claims']})")
    for r in pr['rows']: print(f"  {r['pair']} [{r['cls']}] {r['mean']:+.4f} ({r['mean_in_null_sd']:+.1f} null SD) {r['positive']}/{n} pos q {r['q']:.3g} dc {r['d_dc']} nodeR2 {r['node_r2'][0]:.2f},{r['node_r2'][1]:.2f} spike-free {r['spike_free']} runs same sign {r['run_same_sign']}")
    print('all usable', {k: v for k, v in res['all_usable'].items() if k != 'by_spi'})
    if res['anchor']: print('anchor', round(res['anchor']['mean'], 4), f"{res['anchor']['positive']}/{n} positive")


if __name__ == '__main__':
    if sys.argv[1] == 'bank':
        bank(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    elif sys.argv[1] == 'reliability':
        reliability(sys.argv[2])
    elif sys.argv[1] == 'score':
        score(sys.argv[2], sys.argv[3], sys.argv[4])
