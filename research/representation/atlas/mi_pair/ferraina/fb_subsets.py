"""Random 32-channel subsets of monkey B's per-array-CAR all-channel run (paall), rescored by the frozen rule (post hoc).
Per-pair G and K are read from paall; the surrogate pairs come from joint IAAFT over all 190 channels, not over the 32.
Gates: (1) the copy of fb_score's rule reproduces fb_score on paall (same rng stream); (2) the real z of the fixed pa
subset rebuilt from paall pairs equals Z_pa from job A.
    python fb_subsets.py secA.npz secB.npz SEL_DIR [N_SUBSETS]"""
import sys, json, numpy as np
A, Bf, d = np.load(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3]; N = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
rows = json.loads(str(Bf['rows'])); st = np.array([r['state'] for r in rows]); kk = np.array([r['k'] for r in rows])
pa = json.load(open(f'{d}/pa.json')); kc = np.array(pa['kept_channels']); M = len(kc); B, NB = 30, 10000
G, K = Bf['G_paall'], Bf['K_paall']; iu = np.triu_indices(M, 1); P = -np.ones((M, M), int); P[iu] = np.arange(len(iu[0]))


def score(Z, seed=20261126):
    """fb_score's PRIMARY rule, vectorised; identical rng use, so identical output for the same Z."""
    rng = np.random.default_rng(seed); res = Z[:, 0] - Z[:, 1:].mean(1)

    def mbb(x):
        n = len(x); nb = int(np.ceil(n / B)); s = rng.integers(0, n - B + 1, (NB, nb))
        return x[(s[:, :, None] + np.arange(B)).reshape(NB, -1)[:, :n]].mean(1)
    w1, dd, w2 = res[st == 'W1'], res[st == 'D'], res[st == 'W2']; bw1, bw2, bd = mbb(w1), mbb(w2), mbb(dd)
    dD, dR = dd.mean() - w1.mean(), w2.mean() - dd.mean(); ciD = np.percentile(bd - bw1, [2.5, 97.5])

    def blocks(s):
        m = st == s; b = kk[m] // B; v = res[m]
        return np.array([v[b == j].mean() for j in np.unique(b) if (b == j).sum() >= 15])
    nsd = blocks('W1').std(ddof=1); fr = (blocks('D') < w1.mean()).mean()
    c = [ciD[1] < 0, dD < -2 * nsd, fr >= .8, dR > 0]; z1 = np.median(Z[st == 'W1', 0])
    v = ('PRESENT' if all(c) else 'REVERSED' if dD > 0 and ciD[0] > 0 else 'UNINFORMATIVE' if z1 >= .97
         else 'ABSENT' if abs(dD) <= 2 * nsd and ciD[0] > -0.010 else 'INCONCLUSIVE')
    return dict(z1=z1, dD=dD, lo=ciD[0], hi=ciD[1], sd=dD / nsd, fr=fr, dR=dR, v=v,
                dreal=np.median(Z[st == 'D', 0]) - z1, dsurr=np.median(Z[st == 'D', 1:]) - np.median(Z[st == 'W1', 1:]))


def zsub(ch):
    a, b = np.triu_indices(len(ch), 1); idx = P[ch[a], ch[b]]; g, k = G[:, :, idx], K[:, :, idx]
    g = g - g.mean(2, keepdims=True); k = k - k.mean(2, keepdims=True)
    return (g * k).sum(2) / np.sqrt((g * g).sum(2) * (k * k).sum(2))


full = score(Bf['Z_paall']); print('gate 1, full paall (fb_score: ΔD -0.0033 [-0.0041, -0.0030], -2.34 SD, 1.00, +0.0015, PRESENT):',
                                    {k: (round(float(v), 4) if not isinstance(v, str) else v) for k, v in full.items()})
s32 = np.array([int(np.where(kc == c)[0][0]) for c in pa['sel32']]); Zp = zsub(s32)
print(f'gate 2, pa subset real z from paall pairs vs Z_pa: max |diff| {np.abs(Zp[:, 0] - A["Z_pa"][:, 0]).max():.2e}; '
      f'surrogate (IAAFT over 190 vs 32 channels, not expected equal) {np.abs(Zp[:, 1:] - A["Z_pa"][:, 1:]).max():.2e}')
pp = score(Zp); print('pa subset with 190-channel surrogates:', {k: (round(float(v), 4) if not isinstance(v, str) else v) for k, v in pp.items()})
rng = np.random.default_rng(20261127); arr = kc <= 96; out = []
for i in range(N):
    ch = np.sort(rng.choice(M, 32, replace=False)); r = score(zsub(ch), seed=20261126 + i); r['n96'] = int(arr[ch].sum()); out.append(r)
for lab, sel in (('all', lambda r: True), ('balanced 13-19 per array', lambda r: 13 <= r['n96'] <= 19)):
    o = [r for r in out if sel(r)]; f = lambda k: np.array([r[k] for r in o])
    q = lambda k: ' '.join(f'{x:+.4f}' for x in np.percentile(f(k), [5, 50, 95]))
    vs = {v: sum(r['v'] == v for r in o) for v in ('PRESENT', 'UNINFORMATIVE', 'ABSENT', 'INCONCLUSIVE', 'REVERSED')}
    print(f'\nrandom 32-channel subsets, {lab}: n {len(o)} (5/50/95%)\n  W1 median real z {q("z1")}\n  ΔD {q("dD")}   CI high < 0 in {np.mean(f("hi") < 0):.2f}'
          f'\n  ΔD / W1 block SD {q("sd")}   beyond −2 SD in {np.mean(f("sd") < -2):.2f}\n  D blocks below W1 mean {q("fr")}\n  Δrec {q("dR")}   > 0 in {np.mean(f("dR") > 0):.2f}'
          f'\n  median Δ real z {q("dreal")}   median Δ surr z {q("dsurr")}\n  W1 z ≥ .97 in {np.mean(f("z1") >= .97):.2f}\n  verdicts {vs}')
