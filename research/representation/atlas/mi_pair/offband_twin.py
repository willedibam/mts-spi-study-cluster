"""Linear-Gaussian twin of every zns window, run through the off-band count's pipeline (rules: preregs/offband-twin-prereg.md).

Twin: T samples of a stationary Gaussian process whose auto- and cross-covariances are the window's own (biased) sample
covariances. The normal-scored 32-channel window is zero-padded to 2T, each rFFT bin is multiplied by one complex normal
shared by all channels, and the first T samples of the inverse are kept. The padded periodogram's inverse is the window's
aperiodic sample covariance, which is zero beyond lag T - 1, so the 2T-periodic process has exactly that covariance at
lags below T, and the kept segment is not periodic: its ends mismatch as a recording's do. The twin then enters where the
recording does: ns, edge_mi, spectrum-final joint IAAFT draws, count about its own line against draw 0's limit.
Control: the same with no padding (a T-periodic twin, whose surrogate is from nearly its own ensemble).

    python offband_twin.py run   <banks> <zns dir> <out.npz> <shard i> <n shards> <n workers> [<max windows>]
    python offband_twin.py score <zns dir> <offband dir (t1_*.npz)> <twin dir (tw_*.npz)>       > twin-score.txt
"""
import sys, numpy as np, warnings
from multiprocessing import Pool
from mtsspi.surrogates import multivariate_iaaft
from edges import edge_mi, ns
from offband_null import windows, theiler, load
warnings.filterwarnings('ignore')
R, B, BC = 3, 4, 3          # twins per window; IAAFT draws per twin (draw 0 sets the limit); draws for the periodic control


def twin(v, rng, pad=True):
    x = v - v.mean(0); T = len(x); n = 2 * T if pad else T
    X = np.fft.rfft(x, n=n, axis=0); F = len(X)
    z = (rng.standard_normal(F) + 1j * rng.standard_normal(F)) / np.sqrt(2)
    z[0] = rng.standard_normal()
    if n % 2 == 0: z[-1] = rng.standard_normal()                           # real bins get a real normal: E|z|^2 = 1 everywhere
    return np.fft.irfft(X * z[:, None], n=n, axis=0)[:T]


def jump(v):
    """Per channel: squared end-to-start jump over the mean squared one-step increment."""
    return (v[0] - v[-1]) ** 2 / np.mean(np.diff(v, axis=0) ** 2, axis=0)


def through(y, seeds):
    """y enters as the recording does: normal-scored, G and K, then spectrum-final IAAFT draws (K only; G is kept)."""
    v = ns(y); g, k = edge_mi(v); Ks, gd = [], 0.
    for sd in seeds:
        gs, ks = edge_mi(multivariate_iaaft(v, np.random.default_rng(sd), final='spectrum')[0]); Ks.append(ks)
        gd = max(gd, float(np.abs(gs - g).max()))
    return g, k, np.array(Ks), gd, theiler(v), jump(v)


def task(a):
    kind, q, gidx, k, v, zrec, zsur = a[:7]
    if kind == 'gate':                                                     # recording path and draw 0 against the stored zns K
        kr = edge_mi(v)[1]; ks = edge_mi(multivariate_iaaft(v, np.random.default_rng(k), final='spectrum')[0])[1]
        return kind, q, None, (float(np.abs(kr - zrec).max()), float(np.abs(ks - zsur).max()), jump(v))
    if kind == 'twin':
        r = a[7]; y = twin(v, np.random.default_rng([gidx, 31, r]))
        return kind, q, r, through(y, [[gidx, 32, r, b] for b in range(B)])
    y = twin(v, np.random.default_rng([gidx, 33]), pad=False)
    return kind, q, None, through(y, [[gidx, 34, b] for b in range(BC)])


def run(banks, zdir, out, i, n, nw, cap=None):
    ws = windows(banks); sel = [(g, w) for g, w in enumerate(ws) if g % n == i][:cap]
    Z = {}
    for _, w in sel:
        if w[0] not in Z: Z[w[0]] = np.load(f'{zdir}/{w[0]}.npz')['E'][:, 1]
    X = {q: load(w) for q, (_, w) in enumerate(sel)}
    jobs = []
    for q, (g, w) in enumerate(sel):
        e = Z[w[0]][w[1]]; base = (q, g, w[1], X[q], e[0, 1], e[1, 1])
        jobs += [('gate',) + base] + [('twin',) + base + (r,) for r in range(R)] + [('circ',) + base]
    m = len(sel); o = dict(Gt=np.empty((m, R, 496), np.float32), Kt=np.empty((m, R, 496), np.float32),
                           Kts=np.empty((m, R, B, 496), np.float32), Wt=np.empty((m, R, 496), np.int16), jt=np.empty((m, R, 32)),
                           gdt=np.empty((m, R)), Gc=np.empty((m, 496), np.float32), Kc=np.empty((m, 496), np.float32),
                           Kcs=np.empty((m, BC, 496), np.float32), gdc=np.empty(m), gate=np.empty((m, 2)), jr=np.empty((m, 32)))
    with Pool(nw) as p:
        for kind, q, r, res in p.imap_unordered(task, jobs, chunksize=1):
            if kind == 'gate': o['gate'][q] = res[:2]; o['jr'][q] = res[2]
            elif kind == 'twin': o['Gt'][q, r], o['Kt'][q, r], o['Kts'][q, r], o['gdt'][q, r], o['Wt'][q, r], o['jt'][q, r] = res
            else: o['Gc'][q], o['Kc'][q], o['Kcs'][q], o['gdc'][q] = res[:4]
    np.savez(out, **o, zfile=np.array([w[0] for _, w in sel]), k=np.array([w[1] for _, w in sel]),
             id=np.array([w[2] for _, w in sel]), g=np.array([g for g, _ in sel]))
    dev = o['gate'].max(0); ok = dev.max() < 1e-4
    print(f'gate over {m} windows: max |K_rec - zns| = {dev[0]:.3g}, max |K draw0 - zns K_surr| = {dev[1]:.3g};'
          f' max |G draw - G twin| = {max(o["gdt"].max(), o["gdc"].max()):.3g}', 'PASS' if ok else 'FAIL', flush=True)
    return ok


def boot(d, col, n=2000):
    """95% interval of the window mean, resampling sessions (units) with replacement."""
    u = d.unit.unique(); grp = {x: d[d.unit == x][col].values for x in u}; rng = np.random.default_rng(0)
    bs = [np.mean(np.concatenate([grp[x] for x in rng.choice(u, len(u))])) for _ in range(n)]
    return np.percentile(bs, [2.5, 97.5])


def count(G, K, Ks):
    """Count of K about its own line against draw 0's limit, and the null counts of draws 1.. (as offband.step2)."""
    from offband import resid, Q
    e = resid(G, K); eb = np.array([resid(G, x) for x in Ks]); L = np.percentile(np.abs(eb[0]), Q)
    return (np.abs(e) > L).sum(), (np.abs(eb[1:]) > L).sum(1), e.std() / eb[0].std()


def score(zdir, odir, tdir):
    import glob, pandas as pd
    from scipy.stats import spearmanr
    from offband import load, load_null, session, animal_test, GROUPS, DRUGS, Q
    df, E, er, e0 = load(zdir); pos, e, Wr, _ = load_null(df, E, odir, 't1'); L = df.L.values[pos]
    c0 = (np.abs(e[:, 1:]) > L[:, None, None]).sum(2); d = df.iloc[pos].copy(); d['E'] = d.c.values - c0.mean(1)
    d['Wr'] = list(Wr); d = d.set_index('id')
    T = [np.load(f) for f in sorted(glob.glob(f'{tdir}/tw_*.npz'))]; cat = lambda k: np.concatenate([t[k] for t in T])
    ids = cat('id'); assert len(set(ids)) == len(ids) == len(d), (len(ids), len(d)); d = d.loc[ids]
    gate = cat('gate'); print(f'gate: max |K_rec − zns| {gate[:, 0].max():.2e}, max |K draw 0 − zns K_surr| {gate[:, 1].max():.2e},'
                              f' max |G draw − G twin| {max(cat("gdt").max(), cat("gdc").max()):.2e} over {len(ids)} windows')
    assert gate.max() < 1e-4, 'gate failed: stop (prereg)'
    Gt, Kt, Kts, Wt, jt = cat('Gt').astype(float), cat('Kt').astype(float), cat('Kts').astype(float), cat('Wt'), cat('jt')
    Gc, Kc, Kcs, jr = cat('Gc').astype(float), cat('Kc').astype(float), cat('Kcs').astype(float), cat('jr')
    Et, ct, mt, sr = [], [], [], []
    for w in range(len(ids)):
        rr = [count(Gt[w, r], Kt[w, r], Kts[w, r]) for r in range(Kt.shape[1])]
        Et.append(np.mean([c - n.mean() for c, n, _ in rr])); ct.append(np.mean([c for c, _, _ in rr]))
        mt.append(np.mean([n.mean() for _, n, _ in rr])); sr.append(np.median([s for _, _, s in rr]))
    cc = [count(Gc[w], Kc[w], Kcs[w]) for w in range(len(ids))]
    d['Et'], d['ct'], d['mt'], d['srt'] = Et, ct, mt, sr; d['Ec'] = [c - n.mean() for c, n, _ in cc]; d['Ecorr'] = d.E - d.Et
    d['srr'] = er[d.index.map(df.reset_index().set_index('id')['index'])].std(1) / e0[d.index.map(df.reset_index().set_index('id')['index'])].std(1)
    d['jr'] = jr.mean(1); d['jt'] = jt.mean((1, 2))
    Grec = E[df.reset_index().set_index('id').loc[ids, 'index'].values, 0, 0]
    d['rhoG'] = [np.median([spearmanr(Gt[w, r], Grec[w]).statistic for r in range(Gt.shape[1])]) for w in range(len(ids))]
    d['dG'] = [np.mean(np.abs(Gt[w] - Grec[w])) / Grec[w].mean() for w in range(len(ids))]
    d['Wt'] = Wt.mean((1, 2)); d['Wrm'] = [x.mean() for x in d.Wr]
    print('== Linear-Gaussian twin through the pipeline: excess E = count − own null mean (pairs of 496) ==')
    print(f'{"group":12s} {"n":>4s} {"E rec":>6s} {"E twin":>6s} {"95% (sessions)":>15s} {"E periodic":>10s} {"E rec−twin":>10s}'
          f' {"c twin":>6s} {"null twin":>9s} {"SD ratio rec/twin":>17s}')
    for g in GROUPS:
        x = d[d.group == g]; lo, hi = boot(x, 'Et')
        print(f'{g:12s} {len(x):4d} {x.E.mean():+6.1f} {x.Et.mean():+6.2f} [{lo:+5.2f}, {hi:+5.2f}] {x.Ec.mean():+10.2f} {x.Ecorr.mean():+10.1f}'
              f' {x.ct.mean():6.1f} {x.mt.mean():9.1f} {x.srr.median():8.3f}/{x.srt.median():.3f}')
    x = d[d.group == 'awake_eo']; et = x.Et.mean(); lo, hi = boot(x, 'Et'); er_ = x.E.mean()
    v = 'EXCLUDED: pipeline mismatch is at most a quarter of the wake excess' if hi < 2 else \
        'ARTEFACT: at least half of the wake excess is pipeline mismatch' if et >= 4 else 'PARTIAL'
    print(f'decision (awake_eo): twin E {et:+.2f} [{lo:+.2f}, {hi:+.2f}], share of the recording excess {et / er_:.2f} -> {v}')
    ce = d.Ec.mean(); print(f'periodic control, all windows: E {ce:+.2f} -> {"calibrated" if abs(ce) <= 1.5 else "MISCALIBRATED (declare)"}')
    print('secondary: corrected excess E_rec − E_twin')
    S = session(d.reset_index(), 'Ecorr'); r = animal_test(S, 'Δ corrected excess (session state − awake_ec mean)')
    dr = d[d.group.isin(DRUGS)].Ecorr.mean(); V = {g: d[d.group == g].Ecorr.mean() for g in GROUPS}
    print(f'  ratio rule on corrected E: pooled drug {dr:+.1f}, wake {V["awake_eo"]:+.1f} ({dr / V["awake_eo"]:.1f}x), sleep {V["SLP"]:+.1f} ({dr / V["SLP"]:.1f}x)')
    print('fidelity (median per window): Spearman(G twin, G rec), mean |G twin − G rec| / mean G rec, Theiler window twin/rec, end jump rec/twin')
    for g in GROUPS:
        x = d[d.group == g]
        print(f'  {g:12s} {x.rhoG.median():.3f}  {x.dG.median():.3f}  {x.Wt.mean():5.1f}/{x.Wrm.mean():5.1f}  {x.jr.median():.2f}/{x.jt.median():.2f}')
    print('end mismatch, within state: Spearman(E rec, end jump rec) / Spearman(E twin, end jump twin): ' + ', '.join(
        f'{g} {spearmanr(d[d.group == g].E, d[d.group == g].jr).statistic:+.2f}/{spearmanr(d[d.group == g].Et, d[d.group == g].jt).statistic:+.2f}' for g in GROUPS))
    return d


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        a = sys.argv[2:]
        sys.exit(0 if run(a[0], a[1], a[2], int(a[3]), int(a[4]), int(a[5]), int(a[6]) if len(a) > 6 else None) else 3)
    score(*sys.argv[2:5])
