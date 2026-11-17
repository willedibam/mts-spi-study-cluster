"""Whole-atlas movers beyond the pyspi spectral null, and their edge carriers. Rules: prereg.md.

phase 1:  python movers.py select <atlas dir> <out movers.json>
phase 2:  python movers.py edges  <atlas dir> <movers.json> <out dir> <workers>
"""
import sys, json, glob, numpy as np
from pathlib import Path
from multiprocessing import Pool

PRIMARY, REF, ANCHOR, TOP_N, CAP = ('KT', 'KTMD', 'PF'), 'awake_ec', ('mi_gaussian', 'mi_kraskov_NN-4_DCE-AUTO'), 60, 3
SEL, CONF, FRAC = .04, .02, 15  # |H0 mean net|, |H1 mean net|, sessions of one sign out of 19
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
               anchor=int(anchor) if anchor is not None else None, spis=spis)
    Path(out).write_text(json.dumps(res, indent=1))
    print('usable', res['usable'], 'confirmed', res['confirmed'], 'eo_confirmed', res['eo_confirmed'], 'fdp', round(fdp, 3), flush=True)


def unit_job(args):
    atlas, unit, wins, need, pairs, M, shard = args  # wins: [(row index, id, state, half)]; shard: {bank: {id: file}}
    acc = {}
    for bank, name in BANKS.items():
        for _, wid, st, h in wins:
            f = shard[bank][wid]
            with np.load(f, allow_pickle=True) as d:
                X = d[wid]
            off = ~np.eye(M, dtype=bool); iu = np.triu_indices(M, 1)
            Y = {}
            for k in need:
                p = np.asarray(X[k], np.float64)[off]; p = p - p.mean(); n = np.linalg.norm(p)
                Y[k] = np.full((M, M), np.nan) if not np.isfinite(n) or n < 1e-12 else np.zeros((M, M))
                if np.isfinite(n) and n >= 1e-12:
                    Y[k][off] = p / n
            q = np.array([(Y[i] * Y[j])[iu] + (Y[i] * Y[j]).T[iu] for i, j in pairs], np.float32)
            for key in ((bank, st, h), (bank, st, 'all')):
                s, c = acc.get(key, (0, 0)); acc[key] = (s + q, c + 1)
    return unit, {k: s / c for k, (s, c) in acc.items()}


SHARD = {}


def edges(atlas, movers_json, out, workers):
    A = Path(atlas); m = json.loads(Path(movers_json).read_text()); spis = m['spis']; K = len(spis)
    a, b = np.triu_indices(K, 1); coords = m['decompose']; pairs = [(int(a[c]), int(b[c])) for c in coords]
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
        jobs.append((atlas, u, wins, need, pairs, M, {k: {w[1]: SHARD[k][w[1]] for w in wins} for k in BANKS}))
    with Pool(workers) as p:
        res = p.map(unit_job, jobs, chunksize=1)
    Path(out).mkdir(parents=True, exist_ok=True)
    agent = {r['unit']: r['agent'] for r in rows}; animal = {r['unit']: r['animal'] for r in rows}
    for u, acc in res:
        M = next(r['M'] for r in rows if r['unit'] == u); s = target(agent[u]); D = {}
        for c in (s, 'awake_eo'):
            for h in (0, 1, 'all'):
                if ('real', c, h) not in acc or ('real', REF, h) not in acc: continue
                d = {k: acc[(k, c, h)] - acc[(k, REF, h)] for k in BANKS}
                D[f'{c}_{h}'] = (d['real'] - 2 * d['stationary'] + d['bias']).astype(np.float32)
        np.savez(Path(out) / f'{u}.npz', coords=np.array(coords), chan=np.array(kept[u][:M]), agent=agent[u], animal=animal[u], **D)
    print('done edges', len(res), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'select':
        select(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'edges':
        edges(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
