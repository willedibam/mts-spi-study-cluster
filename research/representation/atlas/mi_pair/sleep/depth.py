"""Depth-graded test inside human NREM sleep (rules: ../preregs/sleep-depth-prereg.md). Reads the frozen windows of score.py.

    python depth.py <windows dir> <out json>

Per patient, over its 10-min N2 and N3 windows: OLS of the window value on [1, N3 indicator, delta fraction (+ covariates)].
beta = slope on delta (within state), gamma = N3 - N2 offset at equal delta. Patient is the unit: mean, t 90% CI, sign-flip p.
"""
import sys, json, numpy as np
from analyse import load, nets, ci90, signflip

N2, N3 = 'sleepN2:10min', 'sleepN3:10min'
MARGIN_SLOPE = .25 * .0411      # quarter of the macaque session line (Δ per unit rise in delta fraction; human-sleep-result.md)
MARGIN_RAW = .25 * .0083        # quarter of PRIOS propofol S-full raw Δ (the frozen raw margin)
MACAQUE_SLOPE = -.0411
NEED, OVERLAP = 10, 5


def fit(y, X):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def common(d, t3):
    lo, hi = max(d[t3].min(), d[~t3].min()), min(d[t3].max(), d[~t3].max()); m = (d >= lo) & (d <= hi); return m, int((m & ~t3).sum()), int((m & t3).sum())


def rows(S, value, extra=None, state=None, matched=False):
    """value(s) -> per-window array. Returns {sub: (beta, gamma)}; gamma NaN where not identifiable (state given, or < OVERLAP)."""
    out = {}
    for sub, s in S.items():
        k = (s['state'] == N2) | (s['state'] == N3)
        t3 = s['state'][k] == N3
        if t3.sum() < NEED or (~t3).sum() < NEED: continue
        y, d = value(s)[k], s['C'][k, 3]
        m, n2, n3 = common(d, t3); ok = n2 >= OVERLAP and n3 >= OVERLAP
        if state is not None:
            w = t3 if state == N3 else ~t3; X = np.column_stack([np.ones(w.sum()), d[w]] + ([] if extra is None else [extra(s)[k][w]]))
            out[sub] = (fit(y[w], X)[1], np.nan); continue
        if matched:
            if not ok: continue
            y, d, t3, w = y[m], d[m], t3[m], m
        else: w = np.ones(len(y), bool)
        X = np.column_stack([np.ones(len(y)), t3.astype(float), d] + ([] if extra is None else [extra(s)[k][w]]))
        b = fit(y, X); out[sub] = (b[2], b[1] if ok else np.nan)
    return out


def summ(x, margin, kind):
    x = np.asarray([v for v in x if np.isfinite(v)]); lo, hi = ci90(x); r = dict(n=len(x), mean=float(x.mean()), ci90=[lo, hi], neg=int((x < 0).sum()),
                                                                                  p_two=float(signflip(x)), margin=margin)
    if (hi - lo) / 2 > margin: v = 'UNINFORMATIVE (90% CI half-width above the margin)'
    elif hi < -margin: v = 'GRADED: falls with depth' if kind == 'slope' else 'OFFSET: N3 below N2 at equal delta'
    elif lo > margin: v = 'REVERSED: rises with depth' if kind == 'slope' else 'OFFSET: N3 above N2 at equal delta'
    elif -margin < lo and hi < margin: v = 'FLAT' if kind == 'slope' else 'NO OFFSET'
    else: v = 'INCONCLUSIVE'
    r['verdict'] = v; return r


def both(o):
    b = [v[0] for v in o.values()]; g = [v[1] for v in o.values()]
    return dict(beta=summ(b, MARGIN_SLOPE, 'slope'), gamma=summ(g, MARGIN_RAW, 'offset') if np.isfinite(g).sum() >= 3 else None,
                patients=sorted(o), per_patient={k: [round(float(v[0]), 5), None if not np.isfinite(v[1]) else round(float(v[1]), 5)] for k, v in o.items()})


def main(d, out):
    S = load(d); R = {}
    net = lambda s: nets(s)[0]
    cov = lambda s: np.column_stack([s['C'][:, :3], s['burst']])
    R['primary'] = both(rows(S, net))
    R['adjusted'] = both(rows(S, net, extra=cov))
    R['control_draw1'] = both(rows(S, lambda s: s['Z'][:, 1] - s['Z'][:, 2:].mean(1)))
    R['real_z'] = both(rows(S, lambda s: nets(s)[1])); R['surrogate_z'] = both(rows(S, lambda s: nets(s)[2]))
    R['within_N2'] = both(rows(S, net, state=N2)); R['within_N3'] = both(rows(S, net, state=N3))
    R['common_range'] = both(rows(S, net, matched=True))
    R['fisher'] = both(rows(S, lambda s: nets(s, 'fisher')[0]))
    b = R['primary']['beta']['mean']; gap = []
    for sub in R['primary']['patients']:
        s = S[sub]; gap.append(np.median(s['C'][s['state'] == N3, 3]) - np.median(s['C'][s['state'] == N2, 3]))
    obs = [np.mean(net(S[x])[S[x]['state'] == N3]) - np.mean(net(S[x])[S[x]['state'] == N2]) for x in R['primary']['patients']]
    R['descriptive'] = dict(beta_over_macaque=b / MACAQUE_SLOPE, median_delta_gap_N3_N2=float(np.median(gap)),
                            implied_N3_minus_N2=float(b * np.median(gap)), observed_N3_minus_N2=float(np.mean(obs)))
    ctrl = R['control_draw1']['beta']['verdict']
    R['stop'] = ctrl.startswith(('GRADED', 'REVERSED'))
    json.dump(R, open(out, 'w'), indent=1)
    for k, v in R.items():
        if isinstance(v, dict) and 'beta' in v:
            g = v['gamma']; print(f"{k:15s} beta {v['beta']['mean']:+.4f} [{v['beta']['ci90'][0]:+.4f}, {v['beta']['ci90'][1]:+.4f}] n {v['beta']['n']} "
                                  f"neg {v['beta']['neg']} p {v['beta']['p_two']:.2g} {v['beta']['verdict']}"
                                  + ('' if g is None else f" | gamma {g['mean']:+.4f} [{g['ci90'][0]:+.4f}, {g['ci90'][1]:+.4f}] n {g['n']} {g['verdict']}"))
    print('descriptive', {k: round(v, 4) for k, v in R['descriptive'].items()}, 'STOP (control moved)' if R['stop'] else '')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
