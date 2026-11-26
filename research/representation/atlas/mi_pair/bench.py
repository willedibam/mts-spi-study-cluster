"""Conventional anaesthesia, complexity and coupling metrics against Δz (rule preregs/bench-prereg.md, frozen before output).

run:   per window, the raw 32-channel subset of edges.py recompute (same rows, same subsets), every metric of the prereg.
       python bench.py run <bank dir> <out.npz> <agent> <rowmode: first|row> <n workers>
score: python bench.py score <dir with bn_*.npz> <dir holding zns/> <out dir>
"""
import sys, json, numpy as np, warnings
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
warnings.filterwarnings('ignore')
FS = 100
PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
OPP = np.array([PERMS.index(p[::-1]) for p in PERMS])              # pattern of -x: the reversed ranking


def lz76(s):
    """Kaspar-Schuster LZ76 phrase count of a binary sequence."""
    n = len(s); i, k, l, c, km = 0, 1, 1, 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1; break
        else:
            km = max(km, k); i += 1
            if i == l:
                c += 1; l += km
                if l + 1 > n:
                    break
                i, k, km = 0, 1, 1
            else:
                k = 1
    return c


def ordinal(x, tau):
    """(T - 2 tau, M) ordinal-pattern index (order 3) per channel."""
    e = np.stack([x[:-2 * tau], x[tau:-tau], x[2 * tau:]], -1); r = np.argsort(e, -1)
    code = r[..., 0] * 9 + r[..., 1] * 3 + r[..., 2]
    lut = np.full(27, -1); lut[[p[0] * 9 + p[1] * 3 + p[2] for p in PERMS]] = range(6)
    return lut[code]


def wsmi(sym):
    M = sym.shape[1]; iu = np.triu_indices(M, 1); w = np.ones((6, 6)); w[np.arange(6), np.arange(6)] = 0; w[np.arange(6), OPP] = 0
    px = np.stack([np.bincount(sym[:, m], minlength=6) for m in range(M)]) / len(sym); out = []
    for i, j in zip(*iu):
        pj = np.bincount(sym[:, i] * 6 + sym[:, j], minlength=36).reshape(6, 6) / len(sym)
        q = pj > 0; out.append(np.sum((w * pj)[q] * np.log(pj[q] / np.outer(px[i], px[j])[q])) / np.log(6))
    return np.mean(out)


def one(x):
    from scipy.signal import welch, butter, sosfiltfilt, hilbert
    x = x - x.mean(0); T, M = x.shape; iu = np.triu_indices(M, 1)
    f, p = welch(x, fs=FS, nperseg=4 * FS, axis=0); P = p.sum(1); band = lambda a, b: P[(f >= a) & (f <= b)].sum()
    sw = P[(f >= .5) & (f < 4)].sum() / band(.5, 40); alpha = band(8, 13) / band(.5, 40)
    br = np.log(band(30, 40) / band(11, 20)); c = np.cumsum(P * ((f >= .5) & (f <= 40))); sef = f[np.searchsorted(c, .95 * c[-1])]
    m = (f >= .8) & (f <= 32); q = p[m] / p[m].sum(0); se = np.mean(-np.sum(q * np.log(q), 0) / np.log(m.sum()))
    ma = (f >= 20) & (f <= 40); ap = np.polyfit(np.log10(f[ma]), np.log10(P[ma]), 1)[0]
    seg = np.sqrt(np.array([np.mean(x[s:s + 50] ** 2) for s in range(0, T, 50)]))           # 0.5-s RMS, for BSR at score time
    lzc = np.mean([lz76((x[:, j] > np.median(x[:, j])).astype(np.int8).tolist()) for j in range(M)]) * np.log2(T) / T
    sy1 = ordinal(x, 1); pe = np.mean([-np.sum((h := np.bincount(sy1[:, j], minlength=6) / len(sy1))[h > 0] * np.log(h[h > 0])) / np.log(6) for j in range(M)])
    C = np.corrcoef(x.T); absr = np.abs(C[iu]).mean(); eig = np.linalg.eigvalsh(C)[-1] / M
    n = 2 * FS; st = range(0, T - n + 1, n // 2); hw = np.hanning(n)[:, None]
    X = np.stack([np.fft.rfft((x[s:s + n] - x[s:s + n].mean(0)) * hw, axis=0) for s in st]); fr = np.fft.rfftfreq(n, 1 / FS)
    out = []
    for a, b in ((1, 4), (8, 13)):
        Xb = X[:, (fr >= a) & (fr <= b)]; cs = np.einsum('sfi,sfj->sfij', Xb, Xb.conj())
        S = cs.mean(0); d = np.real(np.einsum('fii->fi', S)); coh = np.abs(S) ** 2 / (d[:, :, None] * d[:, None, :])
        im = cs.imag; wp = np.abs(im.mean(0)) / np.maximum(np.abs(im).mean(0), 1e-300)
        out += [coh[:, iu[0], iu[1]].mean(), wp[:, iu[0], iu[1]].mean()]
    ph = np.angle(hilbert(sosfiltfilt(butter(4, [8, 13], btype='band', fs=FS, output='sos'), x, axis=0), axis=0))
    kura = np.abs(np.exp(1j * ph).mean(1)).mean()
    return np.array([sw, alpha, br, sef, se, ap, lzc, pe, absr, eig, *out, kura, wsmi(ordinal(x, 4))], float), seg


COLS = ('SW', 'ALPHA', 'BR', 'SEF95', 'SE', 'AP', 'LZC', 'PE', 'ABSR', 'EIG', 'COHD', 'WPLID', 'COHA', 'WPLIA', 'KURA', 'WSMI')
METRICS = ('DZ', 'SW', 'ALPHA', 'BR', 'SEF95', 'SE', 'AP', 'BSR', 'LZC', 'PE', 'ABSR', 'EIG', 'COHD', 'COHA', 'WPLID', 'WPLIA',
           'KURA', 'WSMI', 'MG', 'MK', 'MKG')
DRUGS = ('KT', 'KTMD', 'MD', 'PF')


def auc(a, b):
    a, b = np.asarray(a), np.asarray(b); return (np.sum(a[:, None] > b[None, :]) + .5 * np.sum(a[:, None] == b[None, :])) / (len(a) * len(b))


def score(d, root, outd):
    import pandas as pd
    from plot_character_draft import load                                       # read only
    fr = []
    for p in sorted(Path(d).glob('bn_*.npz')):
        with np.load(p) as a:
            f = pd.DataFrame(a['X'], columns=COLS); f['seg'] = list(a['seg'])
            for k in ('id', 'unit', 'state', 'agent', 'animal'): f[k] = a[k]
            fr.append(f)
    df = pd.concat(fr, ignore_index=True).drop_duplicates('id')
    zl, edges = load(Path(root)); zl = zl.set_index('id')
    df['DZ'] = df.id.map(zl.sd); assert df.DZ.notna().all()
    e = {w: np.asarray(edges[w][0]) for w in df.id}                              # normal-scored real (G, K) per edge
    df['MG'] = [e[w][0].mean() for w in df.id]; df['MK'] = [e[w][1].mean() for w in df.id]; df['MKG'] = df.MK - df.MG
    ecm = df[df.state == 'awake_ec'].groupby('unit').seg.apply(lambda s: np.median(np.concatenate(list(s))))
    df['BSR'] = [np.mean(s < .25 * ecm[u]) for s, u in zip(df.seg, df.unit)]
    df['group'] = np.where(df.state == 'drug', df.agent, np.where(df.state == 'sleep', 'SLP', df.state))
    U = {}
    for c in METRICS:
        dd = df[c] - df.unit.map(df[df.state == 'awake_ec'].groupby('unit')[c].mean()); sd = dd[df.state == 'awake_eo'].std(ddof=1)
        U[c] = dd / sd if sd > 0 else dd * 0
    U = pd.DataFrame(U); S = pd.concat([df[['unit', 'animal', 'group']], U], axis=1).groupby(['unit', 'animal', 'group']).mean().reset_index()
    dr, sl = S[S.group.isin(DRUGS)], S[S.group == 'SLP']
    sign = np.sign(dr[list(METRICS)].mean()); U, S = U * sign, S.assign(**{c: S[c] * sign[c] for c in METRICS})
    dr, sl = S[S.group.isin(DRUGS)], S[S.group == 'SLP']
    rng = np.random.default_rng(0); bi = [(rng.integers(0, len(sl), len(sl)), rng.integers(0, len(dr), len(dr))) for _ in range(10000)]
    W = pd.concat([df[['id', 'unit', 'state', 'animal', 'group']], U, df[['SW']].rename(columns={'SW': 'sw_raw'})], axis=1)
    res, boot = {}, {}
    for c in METRICS:
        cells = dr.groupby(['animal', 'group'])[c].mean(); sv, dv = sl[c].values, dr[c].values
        bs = np.array([sv[i].mean() / dv[j].mean() for i, j in bi]); boot[c] = bs
        r = dict(cells_pos=int((cells > 0).sum()), cells=len(cells), min_cell=float(cells.min()), drug=float(dv.mean()),
                 sleep=float(sv.mean()), ratio=float(sv.mean() / dv.mean()), ratio_ci90=[float(x) for x in np.percentile(bs, [5, 95])],
                 by_agent={g: float(dr[dr.group == g][c].mean()) for g in DRUGS})
        r['C1'] = r['cells_pos'] == r['cells']; r['C2'] = r['ratio_ci90'][1] < .25
        for an in ('Chibi', 'George'):
            w = W[W.animal == an]; wd, ws = w[w.state == 'drug'], w[w.group == 'SLP']; we = w[(w.state == 'awake_ec') & w.unit.isin(ws.unit)]
            lo, hi = max(wd.sw_raw.min(), ws.sw_raw.min()), min(wd.sw_raw.max(), ws.sw_raw.max())
            md, ms = wd[wd.sw_raw.between(lo, hi)][c].values, ws[ws.sw_raw.between(lo, hi)][c].values
            ab = [auc(md[rng.integers(0, len(md), len(md))], ms[rng.integers(0, len(ms), len(ms))]) for _ in range(2000)] if len(md) and len(ms) else [np.nan]
            r[an] = dict(drug_vs_sleep=auc(wd[c], ws[c]), sleep_vs_ec=auc(ws[c], we[c]), matched=auc(md, ms) if len(md) and len(ms) else None,
                         matched_ci90=[float(x) for x in np.nanpercentile(ab, [5, 95])], matched_n=[len(md), len(ms)], sw_overlap=[float(lo), float(hi)])
        res[c] = r
    for c in METRICS:
        dfx = boot[c] - boot['DZ']; res[c]['ratio_minus_dz_ci90'] = [float(x) for x in np.percentile(dfx, [5, 95])]
    Path(outd).mkdir(parents=True, exist_ok=True)
    json.dump(dict(sign={c: int(sign[c]) for c in METRICS}, res=res), open(Path(outd) / 'bench-score.json', 'w'), indent=1)
    W.to_csv(Path(outd) / 'bench-windows.csv.gz', index=False)
    print(f"{'metric':6s} sgn C1 cells  min   C2 ratio [90% CI]          r-DZ CI          | Chibi d/s s/ec match(n)        | George d/s s/ec match [CI] (n)")
    for c in METRICS:
        r = res[c]; ch, ge = r['Chibi'], r['George']
        fm = lambda a: f"{a['drug_vs_sleep']:.2f} {a['sleep_vs_ec']:.2f} " + (f"{a['matched']:.2f}[{a['matched_ci90'][0]:.2f},{a['matched_ci90'][1]:.2f}]({a['matched_n'][0]},{a['matched_n'][1]})" if a['matched'] is not None else 'none')
        print(f"{c:6s} {int(sign[c]):+d} {'Y' if r['C1'] else 'n'} {r['cells_pos']:2d}/{r['cells']} {r['min_cell']:+.2f} {'Y' if r['C2'] else 'n'} "
              f"{r['ratio']:+.2f} [{r['ratio_ci90'][0]:+.2f},{r['ratio_ci90'][1]:+.2f}] [{r['ratio_minus_dz_ci90'][0]:+.2f},{r['ratio_minus_dz_ci90'][1]:+.2f}] | {fm(ch)} | {fm(ge)}")
    print('\nby agent (session means, oriented wake SDs):')
    print(pd.DataFrame({c: res[c]['by_agent'] for c in METRICS}).T.round(2).assign(SLP=[round(res[c]['sleep'], 2) for c in METRICS]).to_string())


if __name__ == '__main__':
    if sys.argv[1] == 'score':
        score(*sys.argv[2:5]); sys.exit()
    from multiprocessing import Pool
    from edges import STATES, sub32, meta
    _, _, d, out, agent, rowmode, nw = sys.argv
    man = json.load(open(f'{d}/manifest.json')); W = np.load(f'{d}/windows.npz')
    rows = [r for r in man['rows'] if r['state'] in STATES and r['agent'] in agent.split(',')]
    S0 = sub32(rows[0]['M']); sub = lambda r: S0 if rowmode == 'first' else sub32(r['M'])
    with Pool(int(nw)) as p: res = p.map(one, [W[r['id']].astype(float)[:, sub(r)] for r in rows], chunksize=4)
    m, _ = meta(rows, man); np.savez(out, X=np.array([a for a, _ in res]), seg=np.array([b for _, b in res]), **m); print('done', out, flush=True)
