"""Axis screen PRIOS prediction test (prereg.md, "Prediction test on PRIOS" and the 20 November addendum).

Net per window = real - mean of the 3 IAAFT draws (every axis is F1). Patient delta = mean net (anaes) - mean net
(awake). A: mean over the set of sign-aligned patient deltas, exact one-sided sign-flip over patients. B: per axis
one-sided t in the macaque direction, Holm. Control: the same on the awake split-half (second half minus first, by run
and onset).

  python -m research.representation.atlas.axis_screen.prios_score <screen-prios.npz> <manifest.json> <score-long.json> <out.json>
"""
import sys, json, itertools
import numpy as np
from scipy.stats import ttest_1samp
from research.representation.atlas.axis_screen.score import load, coordinates, V

STATES, NEED, FIN, REL, CEIL = ('awake', 'anaes'), 20, .95, .3, .98
NOT_IAAFT = {'coh_lo', 'coh_hi', 'coh_bb', 'icoh_lo', 'icoh_bb', 'plv', 'plv_lo', 'plv_hi', 'ppc', 'ppc_lo', 'ppc_hi',
             'dswpli', 'dswpli_lo', 'dswpli_hi', 'xcmax', 'xc2max'}  # addendum: IAAFT does not reproduce these
L1LINF = ('l1|linf', -1.)  # addendum: 2-s arm axis


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); out = np.empty(m); run = 0.
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i]); out[i] = min(run, 1.)
    return out


def main(path, manifest, macaque, output):
    D, keys, gi, I = load(path, manifest)
    rows = {r['id']: r for r in json.load(open(manifest))['rows']}
    ids = [str(i) for i in D['ids']]
    run = np.array([rows[i]['run'] for i in ids])
    unit, state = I['unit'], I['state']
    groups = [str(g) for g in D['groups']]
    S = len(keys)
    Z, Z32, rel = D['Z'].astype(float), D['Z32'].astype(float), D['rel']
    patients = sorted(u for u in set(unit) if all(((unit == u) & (state == s)).sum() >= NEED for s in STATES))
    G = lambda u, s: np.flatnonzero((unit == u) & (state == s))
    C = {c[0]: c for c in coordinates(keys)}

    def zs(c, v):
        _, space, i, j, _, _ = c
        return (Z if space == 'all' else Z32)[:, v, i, j]

    def net(c):
        return zs(c, V['real']) - np.nanmean(np.stack([zs(c, v) for v in V['iaaft']], 1), 1)

    def rel_awake(k):
        return float(np.nanmean([np.nanmean([rel[g, 0, k] for g, name in enumerate(groups) if name.startswith(f'{u}__awake__')])
                                 for u in patients]))

    def usable(c):
        _, space, i, j, _, _ = c
        for u in patients:
            for s in STATES:
                ix = G(u, s)
                for v in (V['real'],) + V['iaaft']:
                    if np.isfinite(zs(c, v)[ix]).mean() < FIN:
                        return False, 'finite'
        z0 = np.nanmean([np.nanmean(zs(c, 0)[G(u, 'awake')]) for u in patients])
        r = (rel_awake(i), rel_awake(j if j < S else S))
        if abs(z0) >= CEIL:
            return False, f'ceiling z0 {z0:.3f}'
        if min(r) < REL:
            return False, f'reliability {r[0]:.2f},{r[1]:.2f}'
        return True, ''

    def deltas(x):
        d = np.array([np.nanmean(x[G(u, 'anaes')]) - np.nanmean(x[G(u, 'awake')]) for u in patients])
        sh = []
        for u in patients:
            r_ = G(u, 'awake'); r_ = r_[np.lexsort((I['start'][r_], run[r_]))]; h = len(r_) // 2
            sh.append(np.nanmean(x[r_[-h:]]) - np.nanmean(x[r_[:h]]))
        return d, np.array(sh)

    def lab(p):
        return p if p in C else '|'.join(p.split('|')[::-1])

    mac = json.load(open(macaque))
    mrows = {r['pair']: r for r in mac['families']['F1']['rows']}
    pred = [(p, float(np.sign(mrows[p]['mean']))) for p in mac['families']['F1']['axes']]
    per, vec = {}, {}
    for p, sgn in pred + [L1LINF]:
        c = C[lab(p)]
        ok, why = usable(c)
        x = net(c); d, sh = deltas(x); vec[p] = dict(d=d, sh=sh)
        dr, _ = deltas(zs(c, 0)); dn, _ = deltas(np.nanmean(np.stack([zs(c, v) for v in V['iaaft']], 1), 1))
        t = ttest_1samp(sgn * d, 0, alternative='greater'); tc = ttest_1samp(sgn * sh, 0, alternative='greater')
        per[p] = dict(sign=sgn, usable=ok, why=why, delta=float(d.mean()), patients=dict(zip(patients, d.round(5).tolist())),
                      same_sign=f'{int(np.sum(np.sign(d) == sgn))}/{len(d)}', t=float(t.statistic), p=float(t.pvalue),
                      split_half=float(sh.mean()), split_half_t=float(tc.statistic), split_half_p=float(tc.pvalue),
                      sd_multiple=float(d.mean() / (np.std(sh, ddof=1) / np.sqrt(2))), real_delta=float(dr.mean()), iaaft_delta=float(dn.mean()),
                      z0_awake=float(np.nanmean([np.nanmean(zs(c, 0)[G(u, 'awake')]) for u in patients])),
                      reliability_awake=[rel_awake(c[2]), rel_awake(c[3] if c[3] < S else S)],
                      macaque_delta=mrows[p]['mean'] if p in mrows else None)

    flips = np.array(list(itertools.product((1, -1), repeat=len(patients))))

    def test_a(names, key='d'):
        if not names:
            return None
        s_p = np.mean([per[p]['sign'] * vec[p][key] for p in names], 0); obs = s_p.mean(); null = (flips * s_p).mean(1)
        return dict(axes=len(names), statistic=float(obs), per_patient=dict(zip(patients, s_p.round(5).tolist())),
                    p_signflip=float(np.mean(null >= obs - 1e-12)), patterns=len(flips))

    use = [p for p, _ in pred if per[p]['usable']]
    valid = [p for p in use if not set(p.split('|')) & NOT_IAAFT]
    hb = holm([per[p]['p'] for p in use]) if use else []
    for p, h in zip(use, hb):
        per[p]['p_holm'] = float(h)
    res = dict(patients=patients, prediction_set=[p for p, _ in pred], usable=use,
               A=test_a(use), A_control_split_half=test_a(use, 'sh'),
               B=[dict(pair=p, t=per[p]['t'], p=per[p]['p'], p_holm=per[p]['p_holm']) for p in use],
               secondary=dict(A_valid=test_a(valid), A_valid_control=test_a(valid, 'sh'), valid=valid, l1_linf=per[L1LINF[0]]),
               axes=per)
    json.dump(res, open(output, 'w'), indent=1, default=float)
    print('patients', patients, 'usable', len(use), 'of', len(pred))
    for k in ('A', 'A_control_split_half'):
        print(k, {kk: v for kk, v in (res[k] or {}).items() if kk != 'per_patient'})
    for k in ('A_valid', 'A_valid_control'):
        print(k, {kk: v for kk, v in (res['secondary'][k] or {}).items() if kk != 'per_patient'})
    for p, r in per.items():
        print(f"{p:<18s} use {r['usable']!s:5s} {r['why']:<22s} Δ {r['delta']:+.4f} ({r['same_sign']}) t {r['t']:+5.2f} p {r['p']:.3f} "
              f"holm {r.get('p_holm', float('nan')):.3f} sh {r['split_half']:+.4f} sd× {r['sd_multiple']:+.1f} real {r['real_delta']:+.3f} iaaft {r['iaaft_delta']:+.3f} mac {r['macaque_delta']}")


if __name__ == '__main__':
    main(*sys.argv[1:5])
