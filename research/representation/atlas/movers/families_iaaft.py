"""POST HOC: do the named mover families survive per-window joint IAAFT and normal-scoring? (result.md, "Named families").

The pyspi stationary null has each state's MEAN cross-spectrum, so "net beyond the null" mixes window-to-window
non-stationarity with within-window non-Gaussian structure. Per-window surrogates separate them. Variants per window:
  raw        the window;                        iaaft_m<b>  marginal-final joint IAAFT of raw (exact marginals);
  ns         rank-to-normal per channel;        iaaft_s<b>  spectrum-final joint IAAFT of ns (exact cross-spectra).
Families (all SPIs from families.yaml, the atlas pyspi snapshot; same run_pyspi call and seed as extract.py):
  Chebyshev   pdist_chebyshev against the other five pdist metrics;
  precision   prec-sq_{LedoitWolf, OAS, ShrunkCovariance, GraphicalLassoCV} against broadband cohmag, plv, ppc;
  kernel      tlmi_kernel_W-0.25 against te_kernel_W-0.25_k-1 (IAAFT keeps autocorrelation: an autocorrelation test).
Readings fixed before outputs:
  - Chebyshev. raw - iaaft_m survives and ns - iaaft_s survives: pair-specific co-excursions beyond each window's
    linear structure and marginals. raw survives, ns does not: channel tails. Neither: per-window linear structure.
  - precision. Spectrum-final IAAFT keeps the covariance, so prec-sq profiles are (nearly) unchanged and only the
    multitaper side can differ. Reproduced by the surrogate: the change is in each window's own second-order
    structure (a linear description, "beyond the null" from between-window variation). Not reproduced: within-window
    phase structure that tapers see and the periodogram does not.
  - kernel. Reproduced by the surrogate: an autocorrelation (spectral) effect. Not reproduced: beyond it.
  A family survives a null if its member-mean residual (real - surrogate) drug contrast has the sign of the original
  net, |mean| >= .02, >= 15/19 primary sessions of that sign and every animal of that sign; reproduced if |mean| < .01
  or < 12/19 sessions; partial otherwise.

compute: python families_iaaft.py compute <bank> <atlas pyspi dir> <config> <out.npz> <shard i> <n shards> <workers>
score:   python families_iaaft.py score <dir of npz> <atlas dir> <out.json>
"""
import sys, json, glob, numpy as np, warnings
from pathlib import Path
from multiprocessing import Pool
from scipy.stats import norm, rankdata

warnings.filterwarnings('ignore')
B, STATES = 3, ('awake_ec', 'awake_eo', 'drug', 'sleep')
PRIMARY, REF = ('KT', 'KTMD', 'PF'), 'awake_ec'
ns = lambda x: norm.ppf((np.apply_along_axis(rankdata, 0, x) - .5) / len(x))


def variants(x, s):
    from mtsspi.surrogates import multivariate_iaaft
    v = {'raw': x, 'ns': ns(x)}
    for b in range(B):
        v[f'iaaft_m{b}'] = multivariate_iaaft(x, np.random.default_rng([s, b, 0]), final='marginal')[0]
        v[f'iaaft_s{b}'] = multivariate_iaaft(v['ns'], np.random.default_rng([s, b, 1]), final='spectrum')[0]
    return v


def window(args):
    wid, x, config = args
    from mtsspi.compute import run_pyspi
    from research.representation.atlas.extract import seed
    from research.representation.atlas.discovery import offdiag_stats, corr_upper
    out = {}
    for name, v in variants(x, seed(wid)).items():
        r = run_pyspi(v, config_path=config, normalise=True, n_jobs=1, random_seed=seed(wid))
        spis = list(r.matrices); P, Q, R2 = [], [], []
        for s in spis:
            p, c, r2 = offdiag_stats(np.asarray(r.matrices[s], dtype=np.float64)); P.append(p); Q.append(c); R2.append(r2)
        out[name] = dict(z=corr_upper(np.array(P))[0], zdc=corr_upper(np.array(Q))[0], r2=np.array(R2, np.float32),
                         raw=np.array(P, np.float32) if name == 'raw' else None)
    return wid, spis, out


def gate(bank, pyspi_dir, wid, spis, P):
    """The raw profiles must equal the atlas's stored pyspi output for this window."""
    from research.representation.atlas.extract import window_ids
    ids = window_ids(bank); size = -(-len(ids) // len(glob.glob(f'{pyspi_dir}/shard-*.npz'))); k = ids.index(wid) // size
    with np.load(Path(pyspi_dir) / f'shard-{k:05d}.npz', allow_pickle=False) as d:
        names = [str(s) for s in d['__spis__']]; A = d[wid]
    M = A.shape[1]; off = ~np.eye(M, dtype=bool)
    for i, s in enumerate(spis):
        ref = A[names.index(s)][off]
        if not np.allclose(P[i], ref, rtol=1e-5, atol=1e-7, equal_nan=True):
            raise SystemExit(f'GATE FAILED {wid} {s}: max diff {np.nanmax(np.abs(P[i] - ref))}')
    print('gate ok', wid, len(spis), flush=True)


def compute(bank, pyspi_dir, config, out, i, n, nw):
    bank = Path(bank); man = json.loads((bank / 'manifest.json').read_text())
    rows = sorted([r for r in man['rows'] if r['state'] in STATES], key=lambda r: r['id'])
    rows = [r for g, r in enumerate(rows) if g % n == i]
    with np.load(bank / 'windows.npz') as W:
        jobs = [(r['id'], np.asarray(W[r['id']], dtype=np.float64), config) for r in rows]
    wid, spis, o = window(jobs[0])  # gate before spending the shard
    gate(bank, pyspi_dir, wid, spis, o['raw']['raw'])
    with Pool(nw) as p:
        res = [(wid, spis, o)] + p.map(window, jobs[1:], chunksize=1)
    names = list(res[0][2])
    save = dict(ids=np.array([w for w, _, _ in res]), spis=np.array(spis), variants=np.array(names),
                rows=np.array(json.dumps([r for r in rows])))
    for key in ('z', 'zdc', 'r2'):
        save[key] = np.array([[o[v][key] for v in names] for _, _, o in res])  # (windows, variants, ...)
    np.savez(out, **save); print('done', out, len(res), flush=True)


def score(indir, atlas, out):
    F = [np.load(f, allow_pickle=False) for f in sorted(glob.glob(f'{indir}/*.npz'))]
    spis = [str(s) for s in F[0]['spis']]; V = [str(v) for v in F[0]['variants']]
    rows = sum([json.loads(str(f['rows'])) for f in F], []); Z = np.concatenate([f['z'] for f in F]); R2 = np.concatenate([f['r2'] for f in F])
    ZDC = np.concatenate([f['zdc'] for f in F])
    K = len(spis); a, b = np.triu_indices(K, 1); co = {(spis[i], spis[j]): c for c, (i, j) in enumerate(zip(a, b))}
    coord = lambda p, q: co.get((p, q), co.get((q, p)))
    by = lambda pre: [s for s in spis if s.startswith(pre)]
    fam = {'Chebyshev': [('pdist_chebyshev', s) for s in by('pdist_') if s != 'pdist_chebyshev'],
           'precision': [(p, s) for p in by('prec-sq_') for s in by('cohmag') + by('plv') + by('ppc')],
           'kernel': [(by('tlmi_kernel')[0], by('te_kernel')[0])]}
    iv = lambda pre: [k for k, v in enumerate(V) if v.startswith(pre)]
    q = {'raw': Z[:, V.index('raw')], 'ns': Z[:, V.index('ns')], 'iaaft_m': np.nanmean(Z[:, iv('iaaft_m')], 1),
         'iaaft_s': np.nanmean(Z[:, iv('iaaft_s')], 1)}
    q['res_raw'] = q['raw'] - q['iaaft_m']; q['res_ns'] = q['ns'] - q['iaaft_s']
    q['raw_dc'] = ZDC[:, V.index('raw')]; q['res_ns_dc'] = ZDC[:, V.index('ns')] - np.nanmean(ZDC[:, iv('iaaft_s')], 1)
    # the atlas bank's net for the same coordinates (and a check that raw z equals the bank's real z)
    D = Path(atlas) / 'discovery'; bank = {}
    for k, name in {'real': 'neurotycho-short', 'stationary': 'null-stationary', 'bias': 'null-bias'}.items():
        with np.load(D / f'{name}-reduced.npz') as d:
            ids = [str(x) for x in d['ids']]; bs = [str(x) for x in d['spis']]; ba, bb = np.triu_indices(len(bs), 1)
            bco = {(bs[i], bs[j]): c for c, (i, j) in enumerate(zip(ba, bb))}
            pos = [ids.index(r['id']) for r in rows]
            cols = [bco.get((spis[i], spis[j]), bco.get((spis[j], spis[i]))) for i, j in zip(a, b)]
            bank[k] = np.array([d['z'][pos, c] if c is not None else np.full(len(pos), np.nan) for c in cols]).T
    q['bank_net'] = bank['real'] - 2 * bank['stationary'] + bank['bias']
    check = float(np.nanmax(np.abs(bank['real'] - q['raw'])))
    units = sorted({r['unit'] for r in rows}); ag = {r['unit']: r['agent'] for r in rows}; an = {r['unit']: r['animal'] for r in rows}
    st = np.array([r['state'] for r in rows]); un = np.array([r['unit'] for r in rows])
    tgt = lambda u: 'sleep' if ag[u] == 'SLP' else 'drug'
    def contrast(x, u, s):
        return np.nanmean(x[(un == u) & (st == s)], 0) - np.nanmean(x[(un == u) & (st == REF)], 0)
    C = {(k, u, s): contrast(x, u, s) for k, x in q.items() for u in units for s in (tgt(u), 'awake_eo')}
    prim = [u for u in units if ag[u] in PRIMARY]
    res = dict(check_raw_vs_bank_max_abs=check, n_windows=len(rows), variants=V, families={})
    for fname, pairs in fam.items():
        cs = [coord(p, s) for p, s in pairs]; sg = np.sign(np.mean([C[('bank_net', u, 'drug')][cs] for u in prim]))
        entry = dict(members=[f'{p}|{s}' for p, s in pairs], sign=float(sg), quantities={})
        for k in q:
            m = lambda us, s: float(np.mean([np.mean(C[(k, u, s(u))][cs]) for u in us]))
            per = np.array([np.mean(C[(k, u, 'drug')][cs]) for u in prim])
            anim = {}
            for u, v in zip(prim, per): anim.setdefault(an[u], []).append(v)
            e = dict(drug={g: m([u for u in prim if ag[u] == g], lambda u: 'drug') for g in PRIMARY},
                     primary=float(per.mean()), sessions_same_sign=int(np.sum(np.sign(per) == sg)), n_sessions=len(per),
                     animals={k_: float(np.mean(v)) for k_, v in anim.items()},
                     MD=m([u for u in units if ag[u] == 'MD'], tgt), sleep=m([u for u in units if ag[u] == 'SLP'], tgt),
                     eyes_open=m(prim, lambda u: 'awake_eo'),
                     eo_sd_sessions=float(np.std([np.mean(C[(k, u, 'awake_eo')][cs]) for u in prim])),
                     members={f'{p}|{s}': float(np.mean([C[(k, u, 'drug')][c] for u in prim])) for (p, s), c in zip(pairs, cs)})
            if k.startswith('res'):
                ok_sign = np.sign(e['primary']) == sg
                e['verdict'] = ('SURVIVES' if ok_sign and abs(e['primary']) >= .02 and e['sessions_same_sign'] >= 15
                                and all(np.sign(v) == sg for v in e['animals'].values())
                                else 'REPRODUCED' if abs(e['primary']) < .01 or e['sessions_same_sign'] < 12 else 'PARTIAL')
            entry['quantities'][k] = e
        res['families'][fname] = entry
    # node-additivity (channel-driven share) of each SPI, awake reference, raw and normal-scored
    ref = st == REF
    res['node_r2_awake'] = {s: dict(raw=float(np.nanmean(R2[ref, V.index('raw'), k])), ns=float(np.nanmean(R2[ref, V.index('ns'), k])),
                                    iaaft_s=float(np.nanmean(R2[ref][:, iv('iaaft_s'), k]))) for k, s in enumerate(spis)}
    Path(out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({f: {k: (e['primary'], e['sessions_same_sign'], e.get('verdict')) for k, e in v['quantities'].items()}
                      for f, v in res['families'].items()}, indent=1), '\ncheck', check)


if __name__ == '__main__':
    if sys.argv[1] == 'compute':
        compute(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]))
    else:
        score(sys.argv[2], sys.argv[3], sys.argv[4])
