"""Whole-atlas movers beyond the pyspi spectral null, and their edge carriers. Rules: prereg.md.

phase 1:  python movers.py select <atlas dir> <out movers.json>
phase 2:  python movers.py edges  <atlas dir> <movers.json> <out dir> <workers>
"""
import sys, json, glob, zlib, numpy as np
from pathlib import Path
from multiprocessing import Pool

PRIMARY, REF, ANCHOR, TOP_N, CAP = ('KT', 'KTMD', 'PF'), 'awake_ec', ('mi_gaussian', 'mi_kraskov_NN-4_DCE-AUTO'), 60, 3
SEL, CONF, FRAC, DRAWS = .04, .02, 15, 20  # |H0 mean net|, |H1 mean net|, sessions of one sign out of 19
BANKS = {'real': 'neurotycho-short', 'stationary': 'null-stationary', 'bias': 'null-bias'}


def target(agent): return 'sleep' if agent == 'SLP' else 'drug'


def halves(rows):
    """{(unit, state, h): row indices}, h in 0, 1 by row order within unit and state, plus h = 'all'."""
    out = {}
    for i, r in enumerate(rows):
        out.setdefault((r['unit'], r['state'], 'all'), []).append(i)
    for (u, s, _), idx in list(out.items()):
        out[(u, s, 0)], out[(u, s, 1)] = idx[0::2], idx[1::2]
    return out


def net_contrast(Z, H, u, s, h):
    """Z: {bank: (windows, ...)}; net = (real - stationary) + (bias - stationary), contrast s against REF."""
    d = {k: np.nanmean(v[H[(u, s, h)]], 0) - np.nanmean(v[H[(u, REF, h)]], 0) for k, v in Z.items()}
    return d['real'] - 2 * d['stationary'] + d['bias']


def select(atlas, out):
    D = Path(atlas) / 'discovery'; Z = {}
    for k, name in BANKS.items():
        with np.load(D / f'{name}-reduced.npz') as d:
            Z[k] = d['z']; ids = [str(x) for x in d['ids']]; spis = [str(x) for x in d['spis']]
            if k == 'real': rows = json.loads(str(d['rows'])); ids0, spis0, modules = ids, spis, [str(x) for x in d['modules']]
            assert ids == ids0 and spis == spis0, k
    rel = {k: v['reliability'] for k, v in json.loads((D / 'neurotycho-short-reliability-all.json').read_text())['reference_state'].items()}
    K = len(spis); a, b = np.triu_indices(K, 1); H = halves(rows)
    units = sorted({r['unit'] for r in rows}); agent = {r['unit']: r['agent'] for r in rows}; animal = {r['unit']: r['animal'] for r in rows}
    ref_rows = [i for i, r in enumerate(rows) if r['state'] == REF]
    z0 = np.nanmean(Z['real'][ref_rows], 0)
    ok = np.all([np.isfinite(v).all(0) for v in Z.values()], 0) & (np.abs(z0) < .98)
    ok &= np.array([rel.get(spis[i], 0) >= .3 and rel.get(spis[j], 0) >= .3 for i, j in zip(a, b)])
    prim = [u for u in units if agent[u] in PRIMARY]
    N = {}
    for u in units:
        for s in (target(agent[u]), 'awake_eo'):
            if (u, s, 0) in H and (u, REF, 0) in H:
                for h in (0, 1):
                    N[(u, s, h)] = net_contrast(Z, H, u, s, h)

    def run(state):
        X0 = np.array([N[(u, state(u), 0)] for u in prim]); X1 = np.array([N[(u, state(u), 1)] for u in prim])
        m0, m1 = X0.mean(0), X1.mean(0); sg = np.sign(m0)
        an = {}
        for i, u in enumerate(prim): an.setdefault(animal[u], []).append(X0[i])
        every_animal = np.all([np.sign(np.mean(v, 0)) == sg for v in an.values()], 0)
        sel = ok & (np.abs(m0) >= SEL) & ((np.sign(X0) == sg).sum(0) >= FRAC) & every_animal
        conf = sel & (np.sign(m1) == sg) & (np.abs(m1) >= CONF) & ((np.sign(X1) == sg).sum(0) >= FRAC)
        return sel, conf, m0, m1, X1
    sel_d, conf_d, m0, m1, X1 = run(lambda u: 'drug')
    sel_e, conf_e, *_ = run(lambda u: 'awake_eo')
    fdp = float(conf_e.sum() / max(conf_d.sum(), 1))
    order = np.flatnonzero(conf_d)[np.argsort(-np.abs(m1[conf_d]))]
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from research.representation.atlas.assumptions import classify, links

    def h1(c, ag):
        us = [u for u in units if agent[u] == ag and (u, target(ag), 1) in N]
        return float(np.mean([N[(u, target(ag), 1)][c] for u in us])) if us else None
    movers = []
    for c in order:
        A, B = spis[a[c]], spis[b[c]]; ch = links(spis, A, B)
        movers.append(dict(coord=int(c), a=A, b=B, cls=classify(A, B), chain=[list(x) for x in ch] if ch else None,
                           modules=f'{modules[a[c]]}|{modules[b[c]]}', z0=float(z0[c]), net_h0=float(m0[c]), net_h1=float(m1[c]),
                           h1_by_agent={ag: h1(c, ag) for ag in ('KT', 'KTMD', 'PF', 'MD', 'SLP')}))
    idx = {(spis[i], spis[j]): n for n, (i, j) in enumerate(zip(a, b))}
    anchor = idx.get(ANCHOR, idx.get(ANCHOR[::-1]))
    # decomposition set: every confirmed minimal pair, then the rest by |H1 net| with at most CAP per SPI, TOP_N in all
    per_spi = {}
    for mv in movers:
        for s in (mv['a'], mv['b']): per_spi[s] = per_spi.get(s, 0) + 1
    dec, used = [mv['coord'] for mv in movers if mv['cls'] is not None][:TOP_N], {}
    for c in dec:
        for s in (spis[a[c]], spis[b[c]]): used[s] = used.get(s, 0) + 1
    for mv in movers:
        if len(dec) >= TOP_N: break
        if mv['coord'] in dec or used.get(mv['a'], 0) >= CAP or used.get(mv['b'], 0) >= CAP: continue
        dec.append(mv['coord']); used[mv['a']] = used.get(mv['a'], 0) + 1; used[mv['b']] = used.get(mv['b'], 0) + 1
    if anchor is not None and anchor not in dec: dec.append(int(anchor))
    res = dict(usable=int(ok.sum()), selected=int(sel_d.sum()), confirmed=int(conf_d.sum()),
               eo_selected=int(sel_e.sum()), eo_confirmed=int(conf_e.sum()), fdp_estimate=fdp,
               confirmed_positive=int((m1[conf_d] > 0).sum()), movers=movers,
               decompose=[int(c) for c in dec], per_spi=dict(sorted(per_spi.items(), key=lambda kv: -kv[1])),
               by_class=dict(sorted({k: sum(1 for mv in movers if (mv['cls'] or 'compound') == k) for k in {mv['cls'] or 'compound' for mv in movers}}.items(), key=lambda kv: -kv[1])),
               anchor=int(anchor) if anchor is not None else None, spis=spis,
               net_h0_by_coord={str(int(c)): float(m0[c]) for c in dec})
    Path(out).write_text(json.dumps(res, indent=1))
    print('usable', res['usable'], 'confirmed', res['confirmed'], 'eo_confirmed', res['eo_confirmed'], 'fdp', round(fdp, 3), flush=True)


def _load(shard, bank, wid):
    with np.load(shard[bank][wid], allow_pickle=True) as d:
        return d[wid]


def _profiles(X, need, M):
    off = ~np.eye(M, dtype=bool); Y, R = {}, {}
    for k in need:
        A = np.asarray(X[k], np.float64); p = A[off]; c = p - p.mean(); n = np.linalg.norm(c)
        R[k] = A
        if np.isfinite(n) and n >= 1e-12:
            Y[k] = np.zeros((M, M)); Y[k][off] = c / n
        else:
            Y[k] = np.full((M, M), np.nan)
    return Y, R


def _zdrop(A, B, sets):
    """corr over the ordered off-diagonal entries of A, B, after removing each unordered edge set (I, J) both ways."""
    M = len(A); off = ~np.eye(M, dtype=bool); a, b = A[off], B[off]; n = a.size
    T = np.array([a.sum(), b.sum(), (a * a).sum(), (b * b).sum(), (a * b).sum()])
    out = []
    for I, J in sets:
        da = np.concatenate([A[I, J], A[J, I]]); db = np.concatenate([B[I, J], B[J, I]])
        S = T - np.array([da.sum(), db.sum(), (da * da).sum(), (db * db).sum(), (da * db).sum()]); m = n - da.size
        cov = S[4] / m - S[0] * S[1] / m ** 2; va = S[2] / m - (S[0] / m) ** 2; vb = S[3] / m - (S[1] / m) ** 2
        out.append(cov / np.sqrt(va * vb))
    return np.array(out)


def unit_job(args):
    atlas, unit, wins, need, pairs, M, shard, signs, target_state, draws = args
    iu = np.triu_indices(M, 1); acc = {}
    for bank in BANKS:                                   # pass 1: mean per-edge contributions per state and half
        for _, wid, st, h in wins:
            Y, R = _profiles(_load(shard, bank, wid), need, M)
            q = np.array([(Y[i] * Y[j])[iu] + (Y[i] * Y[j]).T[iu] for i, j in pairs], np.float32)
            for key in ((bank, st, h), (bank, st, 'all')):
                s, c = acc.get(key, (0, 0)); acc[key] = (s + q, c + 1)
    mean = {k: s / c for k, (s, c) in acc.items()}
    net = lambda c, h: sum(w * (mean[(k, c, h)] - mean[(k, REF, h)]) for k, w in (('real', 1), ('stationary', -2), ('bias', 1)))
    lev = np.abs(mean[('real', REF, 'all')]); E = lev.shape[1]; ke = max(1, int(np.ceil(.01 * E))); kn = max(1, int(round(.05 * M)))
    rng = np.random.default_rng(zlib.crc32(unit.encode()))
    sets = {}                                            # per selection half: per feature list of (I, J) edge sets
    for hs in (0, 1):
        dq = net(target_state, hs); S = []
        for f in range(len(pairs)):
            sg = signs[f]; hot = np.argsort(-sg * dq[f])[:ke]
            node = np.zeros(M); np.add.at(node, iu[0], dq[f] / 2); np.add.at(node, iu[1], dq[f] / 2)
            hn = np.argsort(-sg * node)[:kn]; nodeset = lambda s_: np.flatnonzero(np.isin(iu[0], s_) | np.isin(iu[1], s_))
            bins = np.minimum((np.argsort(np.argsort(lev[f])) * 10) // E, 9)
            L = [np.array([], int), hot, nodeset(hn)]
            L += [np.concatenate([rng.choice(np.flatnonzero(bins == b_), np.sum(bins[hot] == b_), replace=False) for b_ in range(10)]) for _ in range(draws)]
            L += [nodeset(rng.choice(M, kn, replace=False)) for _ in range(draws)]
            S.append([(iu[0][e], iu[1][e]) for e in L])
        sets[hs] = S
    zsum = {}                                            # pass 2: held-out z with each set removed
    for bank in BANKS:
        for _, wid, st, h in wins:
            if st not in (target_state, REF): continue
            hs = 1 - h; _, R = _profiles(_load(shard, bank, wid), need, M)
            Z = np.array([_zdrop(R[i], R[j], sets[hs][f]) for f, (i, j) in enumerate(pairs)])
            s, c = zsum.get((bank, st, h), (0, 0)); zsum[(bank, st, h)] = (s + Z, c + 1)
    zm = {k: s / c for k, (s, c) in zsum.items()}
    drop = {h: sum(w * (zm[(k, target_state, h)] - zm[(k, REF, h)]) for k, w in (('real', 1), ('stationary', -2), ('bias', 1))) for h in (0, 1)}
    return unit, mean, drop


SHARD = {}


def edges(atlas, movers_json, out, workers):
    A = Path(atlas); m = json.loads(Path(movers_json).read_text()); spis = m['spis']; K = len(spis)
    if m['fdp_estimate'] > .2:
        print('STOP: eyes-open false-discovery estimate', m['fdp_estimate'], '> .2 (prereg)', flush=True); return
    a, b = np.triu_indices(K, 1); coords = m['decompose']; pairs = [(int(a[c]), int(b[c])) for c in coords]
    signs = [float(np.sign(m['net_h0_by_coord'][str(c)])) for c in coords]
    need = sorted({i for p in pairs for i in p})
    with np.load(A / 'discovery' / 'neurotycho-short-reduced.npz') as d:
        rows = json.loads(str(d['rows'])); ids = [str(x) for x in d['ids']]
    for bank, name in BANKS.items():
        SHARD[bank] = {}
        for f in sorted(glob.glob(str(A / 'pyspi' / name / 'shard-*.npz'))):
            with np.load(f, allow_pickle=True) as d:
                spi0 = [str(s) for s in d['__spis__']]
                assert spi0 == spis, (bank, f)
                for k in d.files:
                    if not k.startswith('__'): SHARD[bank][k] = f
        assert set(ids) <= set(SHARD[bank]), bank
    man = json.loads((A / 'banks' / 'neurotycho-short' / 'manifest.json').read_text())
    kept = {u['unit']: u['kept_channels'] for u in man['preprocessing']['units']}
    H = halves(rows); jobs = []
    for u in sorted({r['unit'] for r in rows}):
        wins = [(i, ids[i], rows[i]['state'], h) for h in (0, 1) for s in ('awake_ec', 'awake_eo', 'drug', 'sleep')
                for i in H.get((u, s, h), [])]
        M = next(r['M'] for r in rows if r['unit'] == u)
        ag = next(r['agent'] for r in rows if r['unit'] == u)
        jobs.append((atlas, u, wins, need, pairs, M, {k: {w[1]: SHARD[k][w[1]] for w in wins} for k in BANKS}, signs, target(ag), DRAWS))
    with Pool(workers) as p:
        res = p.map(unit_job, jobs, chunksize=1)
    Path(out).mkdir(parents=True, exist_ok=True)
    agent = {r['unit']: r['agent'] for r in rows}; animal = {r['unit']: r['animal'] for r in rows}
    for u, acc, drop in res:
        M = next(r['M'] for r in rows if r['unit'] == u); s = target(agent[u]); D = {}
        for c in (s, 'awake_eo'):
            for h in (0, 1, 'all'):
                if ('real', c, h) not in acc or ('real', REF, h) not in acc: continue
                d = {k: acc[(k, c, h)] - acc[(k, REF, h)] for k in BANKS}
                D[f'{c}_{h}'] = (d['real'] - 2 * d['stationary'] + d['bias']).astype(np.float32)
        np.savez(Path(out) / f'{u}.npz', coords=np.array(coords), chan=np.array(kept[u][:M]), agent=agent[u], animal=animal[u],
                 drop_0=drop[0], drop_1=drop[1], **D)  # drop_h: (features, 3 + 2*DRAWS) held-out net dz measured on half h
    print('done edges', len(res), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'select':
        select(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'edges':
        edges(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
