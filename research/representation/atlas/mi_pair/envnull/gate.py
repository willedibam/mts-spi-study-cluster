"""Apply the frozen calibration rule (../preregs/envelope-prereg.md) to envcalib.py output.

  python gate.py <shard.npz ...> [--only "GOF <variant>"]     (shards over kinds are merged; --only restricts the candidates)
Written before the development calibration output was read.
"""
import sys, numpy as np
from envcalib import GAIN, COUPLING, KINDS

LIN = ('linear', 'linear_hetspec')


def boot_ratio(a, b, rng, n=4000):
    i = rng.integers(len(a), size=(n, len(a))); r = a[i].mean(1) / b[i].mean(1); return np.percentile(r, [2.5, 97.5])


def main(files, only=None):
    Q = [np.load(f) for f in files]; names = list(Q[0]['nulls']); assert all(list(q['nulls']) == names for q in Q)
    D = {k: q[k] for q in Q for k in q['kinds']}; kinds = [k for k in KINDS if k in D]; rng = np.random.default_rng(0)
    assert set(GAIN + COUPLING + LIN) <= set(kinds), 'missing generators'
    cand = [n for n in names if n.startswith('GOF') and (only is None or n == only)]; t = lambda a: a.mean() / a.std(ddof=1) * np.sqrt(len(a))
    print('retention R = mean(null) / mean(N0) [95% bootstrap]; N0 mean (t)')
    for k in kinds:
        R0 = D[k][:, 0]
        cells = []
        for j, n in enumerate(names[1:], 1):
            lo, hi = boot_ratio(D[k][:, j], R0, rng); cells.append(f'{n} {D[k][:, j].mean() / R0.mean():+.2f} [{lo:+.2f},{hi:+.2f}] ({D[k][:, j].mean():+.4f})')
        print(f'{k:15s} N0 {R0.mean():+.4f} (t {t(R0):+.1f}) | ' + '  '.join(cells))
    gradable = [k for k in COUPLING if t(D[k][:, 0]) <= -3]
    if not {'quadratic', 'decouple'} <= set(gradable): print('calibration FAILS: quadratic or decouple not gradable'); return
    passed = {}
    for c in cand:
        j = names.index(c); why = []
        for k in LIN:
            if abs(D[k][:, j].mean()) > .002: why.append(f'{k} |res| {abs(D[k][:, j].mean()):.4f} > .002')
        for k in GAIN:
            R0, Rv = D[k][:, 0], D[k][:, j]
            if t(R0) > -3:
                if abs(Rv.mean()) > .003: why.append(f'{k} (no N0 fall) |res| {abs(Rv.mean()):.4f} > .003')
            elif abs(Rv.mean() / R0.mean()) > .3: why.append(f'{k} R {Rv.mean() / R0.mean():+.2f} outside ±.3')
        cr = {k: D[k][:, j].mean() / D[k][:, 0].mean() for k in gradable}
        why += [f'{k} R {v:+.2f} < .6' for k, v in cr.items() if v < .6]
        print(f'{c}: ' + ('PASS' if not why else 'FAIL — ' + '; '.join(why)) + f' | min coupling R {min(cr.values()):+.2f}')
        if not why: passed[c] = min(cr.values())
    if not passed: print('\nNO CANDIDATE PASSES: the null cannot separate gain from coupling (stop; no real-data run).'); return
    best = max(passed.values()); tie = [c for c in passed if passed[c] >= best - .05]
    key = lambda c: (int(c.split()[1].split('-')[0]), c.endswith('diag'))
    print('\nselected:', max(tie, key=key), '(min coupling R; ties within .05 -> larger sigma, then diag)')


if __name__ == '__main__':
    a = sys.argv[1:]; only = a[a.index('--only') + 1] if '--only' in a else None
    main([f for f in a if f.endswith('.npz')], only)
