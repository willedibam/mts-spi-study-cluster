"""Verdict, gates and reported rows for the human-sleep MI-pair test (rules: ../preregs/sleep-prereg.md, frozen with this file).

    python analyse.py <dir with sub-*.npz from score.py run (all five states)> <out.json>

Net per window = z_real - mean of the IAAFT draws (raw Pearson z; Fisher row reported). Subject delta = mean net (target) -
mean net (wake). Null SD = SD over subjects of the wake split-half delta (second half minus first by start time, equal counts),
each scaled to that subject's contrast window counts by sqrt((1/n_b + 1/n_t) / (2/n_half)) (transfer protocol).
"""
import sys, os, json, glob, numpy as np
from scipy.stats import t as tdist
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.join(HERE, '..')]
from extract import shared_contact
from edges import zc

WAKE = 'wake:1min'
TARGETS = {'N3': 'sleepN3:10min', 'N2': 'sleepN2:10min', 'N3_1min': 'sleepN3:1min', 'N2_1min': 'sleepN2:1min'}
NEED = {'N3': 10, 'N2': 10, 'N3_1min': 4, 'N2_1min': 4}
MARGIN_SD = .25 * 2.7          # quarter of PRIOS propofol S-full (robust null SD units)
MARGIN_RAW = .25 * .0083       # quarter of PRIOS propofol S-full raw delta (z units)
REF = dict(prios_sd=-2.7, prios_raw=-.0083, macaque_sd=-3.29, macaque_raw=-.0133)
CEIL, FLOOR, NPERM = .98, .3, 200000


def signflip(d, alt='two', seed=0):
    rng = np.random.default_rng(seed); s = rng.choice([-1., 1.], (NPERM, len(d))); m = (s * d).mean(1); o = d.mean()
    if alt == 'two': return (np.sum(np.abs(m) >= abs(o) - 1e-15) + 1) / (NPERM + 1)
    if alt == 'less': return (np.sum(m <= o + 1e-15) + 1) / (NPERM + 1)
    return (np.sum(m >= o - 1e-15) + 1) / (NPERM + 1)


def halves(v):
    h = len(v) // 2; return v[:h], v[len(v) - h:]


def rel(P):
    return np.mean(np.corrcoef(P)[np.triu_indices(len(P), 1)]) if len(P) > 1 else np.nan


def load(d):
    S = {}
    for fn in sorted(glob.glob(f'{d}/sub-*.npz')):
        q = np.load(fn); rows = json.loads(str(q['rows'])); S[os.path.basename(fn)[:-4]] = dict(
            Z=q['Z'], P=q['P'], C=q['C'], B=q['B'], RMS=q['RMS'], state=np.array([r['state'] for r in rows]),
            start=np.array([r['start_s'] for r in rows]), seg=np.array([r['segment'] for r in rows]),
            burst=np.array([r['burst'] for r in rows]), chans=json.loads(str(q['chans'])), cols=json.loads(str(q['cols'])))
    return S


def nets(s, mode='raw', mask=None):
    Z = s['Z']
    if mask is not None:  # recompute z from stored profiles on a pair subset
        P = s['P']; Z = np.array([[zc(P[w, 2 * b][mask], P[w, 2 * b + 1][mask]) for b in range(P.shape[1] // 2)] for w in range(len(P))])
    if mode == 'fisher': Z = np.arctanh(np.clip(Z, -.999999, .999999))
    return Z[:, 0] - Z[:, 1:].mean(1), Z[:, 0], Z[:, 1:].mean(1)


def contrast(S, target, value, need):
    """value(sub, s) -> per-window array. Returns subjects, deltas, scaled split-half deltas, counts."""
    subs, D, H = [], [], []
    for sub, s in S.items():
        b, t = s['state'] == WAKE, s['state'] == target
        if t.sum() < need or b.sum() < 4: continue
        v = value(sub, s); vb = v[b][np.argsort(s['start'][b])]; h1, h2 = halves(vb); nb, nt, nh = b.sum(), t.sum(), len(h1)
        subs.append(sub); D.append(v[t].mean() - v[b].mean()); H.append((h2.mean() - h1.mean()) * np.sqrt((1 / nb + 1 / nt) / (2 / nh)))
    return subs, np.array(D), np.array(H)


def ci90(x):
    n = len(x); se = x.std(ddof=1) / np.sqrt(n); q = tdist.ppf(.95, n - 1); return [float(x.mean() - q * se), float(x.mean() + q * se)]


def summary(D, H):
    sd = H.std(ddof=1); u = D / sd; c = ci90(u)
    return dict(n=len(D), mean_raw=float(D.mean()), ci90_raw=ci90(D), null_sd=float(sd), mean_sd=float(u.mean()), ci90_sd=c,
                neg=int((D < 0).sum()), p_two=float(signflip(D)), split_mean_raw=float(H.mean()))


def verdict(r, zwake, relG, relK):
    m, lo, hi = r['mean_sd'], *r['ci90_sd']; rl, rh = r['ci90_raw']
    if zwake >= CEIL or min(relG, relK) < FLOOR: return 'UNINFORMATIVE (ceiling or reliability floor)'
    if (hi - lo) / 2 > MARGIN_SD: return 'UNINFORMATIVE (90% CI half-width above the null-SD margin)'
    if lo > MARGIN_SD or hi < -MARGIN_SD: return 'MOVES: N3 shifts the MI pair by more than a quarter of the human anaesthesia effect'
    sd_eq, raw_eq = -MARGIN_SD < lo and hi < MARGIN_SD, -MARGIN_RAW < rl and rh < MARGIN_RAW
    if sd_eq and raw_eq: return 'NEAR WAKE: equivalent within a quarter of the human anaesthesia effect (null-SD and raw margins)'
    if sd_eq: return 'NEAR WAKE IN NULL-SD UNITS ONLY (raw 90% CI exceeds the raw margin)'
    return 'INCONCLUSIVE'


def gates(S, target, need, D, H, sd):
    """Protocol gates for the 'sleep moves the MI pair' direction (reported; the verdict is the equivalence rule)."""
    subs = contrast(S, target, lambda sub, s: nets(s)[0], need)[0]; adj, dis = [], []
    for sub in subs:
        s = S[sub]; b, t = s['state'] == WAKE, s['state'] == target; k = b | t; y = nets(s)[0][k]
        X = np.column_stack([np.ones(k.sum()), t[k].astype(float), s['C'][k][:, :3], s['burst'][k]])
        adj.append(np.linalg.lstsq(X, y, rcond=None)[0][1])
        r = lambda m: np.sqrt(max(rel(s['P'][m][:, 0]), 1e-3) * max(rel(s['P'][m][:, 1]), 1e-3))
        dis.append(nets(s)[0][t].mean() / r(t) - nets(s)[0][b].mean() / r(b))
    adj, dis = np.array(adj), np.array(dis); sgn = np.sign(D.mean()); alt = 'less' if sgn < 0 else 'greater'
    return dict(g1_signflip_two=float(signflip(D)), g1=bool(signflip(D) < .05), g2_size_sd=float(D.mean() / sd), g2=bool(abs(D.mean()) >= 3 * sd),
                g3_split_ratio=float(abs(H.mean()) / abs(D.mean())), g3=bool(abs(H.mean()) <= .25 * abs(D.mean())), g4='not applied (no nuisance contrast)',
                g5_adjusted=float(adj.mean()), g5=bool(np.sign(adj.mean()) == sgn and abs(adj.mean()) >= .5 * abs(D.mean()) and signflip(adj, alt) < .05),
                g6_disattenuated=float(dis.mean()), g6=bool(np.sign(dis.mean()) == sgn and signflip(dis, alt) < .05))


def main(d, out):
    S = load(d); R = {}
    wake = [s for s in S.values()]; b = lambda s: s['state'] == WAKE
    zwake = float(np.mean([s['Z'][b(s), 0].mean() for s in wake]))
    relG = float(np.nanmean([rel(s['P'][b(s)][:, 0]) for s in wake])); relK = float(np.nanmean([rel(s['P'][b(s)][:, 1]) for s in wake]))
    R['baseline'] = dict(subjects=len(S), wake_real_z=zwake, rel_G=relG, rel_K=relK, margin_sd=MARGIN_SD, margin_raw=MARGIN_RAW, ref=REF)
    for name, target in TARGETS.items():
        subs, D, H = contrast(S, target, lambda sub, s: nets(s)[0], NEED[name]); r = summary(D, H)
        r['verdict'] = verdict(r, zwake, relG, relK)
        r['ratio_prios_sd'] = r['mean_sd'] / REF['prios_sd']; r['ratio_prios_raw'] = r['mean_raw'] / REF['prios_raw']
        r['ratio_macaque_sd'] = r['mean_sd'] / REF['macaque_sd']; r['ratio_macaque_raw'] = r['mean_raw'] / REF['macaque_raw']
        uns = []  # PRIOS construction: unscaled, first minus second half, 1.4826 MAD
        for sub in subs:
            s = S[sub]; v = nets(s)[0][b(s)][np.argsort(s['start'][b(s)])]; h1, h2 = halves(v); uns.append(h1.mean() - h2.mean())
        uns = np.array(uns); rsd = 1.4826 * np.median(np.abs(uns - np.median(uns)))
        r['prios_construction'] = dict(robust_null_sd=float(rsd), mean_sd=float(D.mean() / rsd), ci90_sd=[x / rsd for x in r['ci90_raw']])
        _, Dr, _ = contrast(S, target, lambda sub, s: nets(s)[1], NEED[name]); _, Ds, _ = contrast(S, target, lambda sub, s: nets(s)[2], NEED[name])
        r['delta_real_z'], r['delta_surr_z'] = float(Dr.mean()), float(Ds.mean())
        _, Df, Hf = contrast(S, target, lambda sub, s: nets(s, 'fisher')[0], NEED[name]); r['fisher'] = summary(Df, Hf)
        r['subjects'] = dict(zip(subs, np.round(D, 5).tolist()))
        if name in ('N3', 'N2'):
            r['gates'] = gates(S, target, NEED[name], D, H, r['null_sd'])
            masks = {sub: ~shared_contact(s['chans']) for sub, s in S.items()}
            Dm, Hm = contrast(S, target, lambda sub, s: nets(s, mask=masks[sub])[0] if masks[sub].sum() >= 10 else nets(s)[0] * np.nan, NEED[name])[1:]
            ok = np.isfinite(Dm); r['no_shared_contact'] = summary(Dm[ok], Hm[ok])
            big = np.array([len(S[x]['chans']) >= 16 for x in subs]); r['M16'] = summary(D[big], H[big])
            r['by_centre'] = {c: summary(D[np.array([S[x]['chans'][0][0] == c for x in subs])], H[np.array([S[x]['chans'][0][0] == c for x in subs])])
                              for c in 'GMN' if sum(S[x]['chans'][0][0] == c for x in subs) >= 3}
            r['bench'] = bench(S, target, NEED[name], r)
        R[name] = r
    json.dump(R, open(out, 'w'), indent=1); report(R)


def bench(S, target, need, r_dz):
    """Reported, not criterion: each conventional metric's target - wake delta in its own split-half null SD."""
    cols = list(S[next(iter(S))]['cols']); out = {}
    def val(k):
        def f(sub, s):
            if k == 'MG': return s['P'][:, 0].mean(1)
            if k == 'MK': return s['P'][:, 1].mean(1)
            if k == 'MKG': return (s['P'][:, 1] - s['P'][:, 0]).mean(1)
            if k == 'DELTA': return s['C'][:, 3]
            if k == 'BSR':
                med = np.median(np.concatenate(list(s['RMS'][s['state'] == WAKE]))); return np.array([np.mean(x < .25 * med) for x in s['RMS']])
            return s['B'][:, cols.index(k)]
        return f
    for k in ['DZ'] + cols + ['MG', 'MK', 'MKG', 'DELTA', 'BSR']:
        if k == 'DZ': out[k] = dict(mean_sd=r_dz['mean_sd'], ci90_sd=r_dz['ci90_sd'], pos=r_dz['n'] - r_dz['neg'], n=r_dz['n']); continue
        subs, D, H = contrast(S, target, val(k), need); sd = H.std(ddof=1)
        out[k] = dict(mean_sd=float(D.mean() / sd) if sd > 0 else None, ci90_sd=[x / sd for x in ci90(D)] if sd > 0 else None,
                      pos=int((D > 0).sum()), n=len(D), mean_raw=float(D.mean()))
    return out


def report(R):
    b = R['baseline']; print(f"baseline: {b['subjects']} subjects; wake real z {b['wake_real_z']:.3f}; reliability G {b['rel_G']:.3f} K {b['rel_K']:.3f}; "
                             f"margins ±{b['margin_sd']:.3f} null SD, ±{b['margin_raw']:.4f} raw")
    for name in TARGETS:
        r = R[name]; print(f"\n{name}: n {r['n']}; Δ raw {r['mean_raw']:+.5f} [{r['ci90_raw'][0]:+.5f}, {r['ci90_raw'][1]:+.5f}]; null SD {r['null_sd']:.5f}; "
                           f"Δ {r['mean_sd']:+.2f} SD [{r['ci90_sd'][0]:+.2f}, {r['ci90_sd'][1]:+.2f}]; {r['neg']}/{r['n']} negative; sign-flip p {r['p_two']:.4f}")
        print(f"  VERDICT: {r['verdict']}")
        print(f"  ratios: PRIOS sd {r['ratio_prios_sd']:+.2f}, raw {r['ratio_prios_raw']:+.2f}; macaque sd {r['ratio_macaque_sd']:+.2f}, raw {r['ratio_macaque_raw']:+.2f}")
        pc = r['prios_construction']; print(f"  PRIOS construction: robust null SD {pc['robust_null_sd']:.5f} -> {pc['mean_sd']:+.2f} [{pc['ci90_sd'][0]:+.2f}, {pc['ci90_sd'][1]:+.2f}]")
        print(f"  Δ real z {r['delta_real_z']:+.5f}, Δ surrogate z {r['delta_surr_z']:+.5f}; Fisher {r['fisher']['mean_sd']:+.2f} SD [{r['fisher']['ci90_sd'][0]:+.2f}, {r['fisher']['ci90_sd'][1]:+.2f}]")
        if 'gates' in r:
            print('  gates:', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r['gates'].items()})
            for k in ('no_shared_contact', 'M16'): print(f"  {k}: n {r[k]['n']}, {r[k]['mean_sd']:+.2f} SD [{r[k]['ci90_sd'][0]:+.2f}, {r[k]['ci90_sd'][1]:+.2f}], raw {r[k]['mean_raw']:+.5f}")
            for c, q in r['by_centre'].items(): print(f"  centre {c}: n {q['n']}, {q['mean_sd']:+.2f} SD, raw {q['mean_raw']:+.5f}")
            print('  bench (target − wake, own null SD; reported, not criterion):')
            for k, q in r['bench'].items():
                if q['mean_sd'] is not None: print(f"    {k:6s} {q['mean_sd']:+7.2f} [{q['ci90_sd'][0]:+7.2f}, {q['ci90_sd'][1]:+7.2f}]  {q['pos']}/{q['n']} up")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
