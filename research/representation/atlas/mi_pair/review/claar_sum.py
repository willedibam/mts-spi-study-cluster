"""Frozen decision rule for the Claar test (claar-prereg.md) and its secondaries, from claar_z.py output.
    python claar_sum.py Z.npz [QC.json]"""
import sys, json, numpy as np
from scipy.stats import binom, spearmanr
Z = np.load(sys.argv[1]); rows = json.loads(str(Z['rows'])); SECOND = 'sub-569073_ses-20210416'
qc = {q['session']: q for q in json.load(open(sys.argv[2]))} if len(sys.argv) > 2 else None
sess = np.array([r['session'] for r in rows]); st = np.array([r['state'] for r in rows]); t = np.array([r['start_s'] for r in rows])
bi = np.array([r['burst'] for r in rows]); sp = np.array([r['speed'] for r in rows])
S = [s for s in dict.fromkeys(sess) if (qc is None or qc[s]['included'])]; P = [s for s in S if s != SECOND]


def per_session(res, s, iso_mask=None, awake_mask=None):
    a = (sess == s) & (st == 'awake') & (awake_mask if awake_mask is not None else True)
    i = (sess == s) & (st == 'iso') & (iso_mask if iso_mask is not None else True)
    aw = np.flatnonzero((sess == s) & (st == 'awake')); aw = aw[np.argsort(t[aw])]; h = len(aw) // 2
    return res[i].mean() - res[a].mean(), res[aw[-h:]].mean() - res[aw[:h]].mean(), i.sum(), a.sum()


def report(name, res, **kw):
    D = np.array([per_session(res, s, **kw)[:2] for s in P]); d, nl = D[:, 0], D[:, 1]; n = len(d); neg = int((d < 0).sum())
    k = min(j for j in range(n + 1) if binom.sf(j - 1, n, .5) <= .025); sd = nl.std(ddof=1)
    boot = np.random.default_rng(0).choice(d, (10000, n)).mean(1)
    print(f'{name:28s} n={n} negative {neg}/{n} (need ≥{k}; one-sided p={binom.sf(neg - 1, n, .5):.4f}) mean Δ {d.mean():+.4f} '
          f'[{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}] = {d.mean() / sd:+.1f} null SD (null SD {sd:.4f})')
    return d, nl, neg, k, sd


V = {v: Z[v][:, 0] - Z[v][:, 1:].mean(1) for v in ('ns', 'raw', 'asis') if v in Z.files}
print('sessions included', len(S), '| primary units', len(P)); print('windows awake/iso per session', {s: (int(((sess == s) & (st == 'awake')).sum()), int(((sess == s) & (st == 'iso')).sum())) for s in S})
d, nl, neg, k, sd = report('PRIMARY normal-scored', V['ns'])
lo_fals = max(j for j in range(len(P) + 1) if binom.sf(j - 1, len(P), .5) > .5) if False else None
verdict = 'REPLICATES' if neg >= k and d.mean() < -2 * sd else ('FALSIFIED' if (neg <= len(P) // 2 + (len(P) % 2 == 0 and 0) or abs(d.mean()) < 2 * sd) else 'INCONCLUSIVE')
print('verdict (≥k negative and mean beyond −2 null SD → replicates; ≤ n/2+0.5 negative or within 2 SD → falsified):', verdict)
print('per session Δ:', {s: round(float(x), 4) for s, x in zip(P, d)})
print('real z awake/iso mean', round(Z['ns'][st == 'awake', 0].mean(), 3), round(Z['ns'][st == 'iso', 0].mean(), 3), '| surrogate', round(Z['ns'][st == 'awake', 1:].mean(), 3), round(Z['ns'][st == 'iso', 1:].mean(), 3))
for v in ('raw', 'asis'):
    if v in V: report(f'secondary {v}', V[v])
still = np.zeros(len(rows), bool)
for s in P:
    a = np.flatnonzero((sess == s) & (st == 'awake')); still[a[sp[a] <= np.median(sp[a])]] = True
report('secondary stationary awake', V['ns'], awake_mask=still)
rho = [spearmanr(bi[(sess == s) & (st == 'iso')], V['ns'][(sess == s) & (st == 'iso')])[0] for s in P]
print(f'burst: within-iso Spearman(residual, BI) median {np.median(rho):+.2f}, positive in {sum(r > 0 for r in rho)}/{len(rho)}')
mbi = [np.median(bi[(sess == s) & (st == 'iso')]) for s in P]; r, p = spearmanr(mbi, d)
print(f'burst: across mice Spearman(Δ, median iso BI) {r:+.2f} (p={p:.3f}); median iso BI by mouse', np.round(mbi, 2).tolist())
ok = [s for s in P if ((sess == s) & (st == 'iso') & (bi >= .85)).sum() >= 5]
if ok:
    dd = [per_session(V['ns'], s, iso_mask=bi >= .85)[0] for s in ok]
    print(f'burst: continuous-like iso windows (BI ≥ .85) only: {len(ok)} sessions, negative {sum(x < 0 for x in dd)}, mean Δ {np.mean(dd):+.4f} = {np.mean(dd) / sd:+.1f} null SD')
else: print('burst: no session has ≥ 5 iso windows with BI ≥ .85')
if SECOND in sess: print('second 569073 session Δ', round(float(per_session(V['ns'], SECOND)[0]), 4))

