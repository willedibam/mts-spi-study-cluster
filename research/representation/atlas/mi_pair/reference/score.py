"""Score the bipolar 10-s run against the CAR anchor (preregs/reference-prereg.md).

  python score.py <bipolar dir> <CAR robust dir>

Reads bp10_[AB]_*.npz (robust.py run --shift on the bipolar banks) and rb10_[AB]_*.npz (the CAR sweep, same rows,
seeds and order). Per configuration, as robust.py score: animal means, t(3), pooled drug Δz (raw and in its own
eyes-open SD), per-agent Δz, sleep Δz, and R = bipolar pooled ÷ CAR pooled of the same configuration (CAR kshift:
−.0148 from ksg-bias-result.md; rb10 has no shift column). Verdict on AUTO k4 ψ(T) by the frozen rule.
Descriptive: Pearson r of per-session drug Δ, bipolar against CAR.
"""
import sys, glob, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robust import configs, stats, CSH
from edges import zc
from ksgbias import session_deltas

CAR_KSHIFT, ANCHOR = -.0148, -.0133
SHOW = ('auto k4', 'auto k10', 'kshift', 'w0 k4', '2te k4', '4auto k4')


def residuals(d, prefix):
    L = [np.load(f) for f in sorted(glob.glob(f'{d}/{prefix}_[AB]_*.npz'))]; cat = lambda k: np.concatenate([q[k] for q in L])
    X = np.concatenate([q['X'] for q in L]).astype(float); T = int(L[0]['T'])
    C = configs(X, T, bool(np.isfinite(X[:, :, CSH]).all()))
    Z = {n: np.array([zc(X[w, 0, 0], K[w, 0]) - zc(X[w, 1, 0], K[w, 1]) for w in range(len(X))]) for n, (K, _) in C.items() if n in SHOW}
    return Z, {k: cat(k) for k in ('id', 'unit', 'state', 'agent', 'animal')}, len(X)


def verdict(s, R):
    if R < .25 or s['n_neg'] <= 2: return 'LOST'
    ok = s['n_neg'] == 4 and s['p'] < .05 and all(s['agm'][a][0] < 0 for a in ('KT', 'KTMD', 'MD', 'PF')) and R >= .5
    return 'SURVIVES' if ok else 'PARTIAL'


if __name__ == '__main__':
    B, mB, nB = residuals(sys.argv[1], 'bp10'); C, mC, nC = residuals(sys.argv[2], 'rb10')
    if not (mB['id'] == mC['id']).all(): sys.exit('row order differs between bipolar and CAR')
    m = [mB[k] for k in ('unit', 'state', 'agent', 'animal')]; print(f'bipolar {nB} windows; CAR {nC}')
    for n in SHOW:
        s = stats(B[n], *m); car = CAR_KSHIFT if n == 'kshift' else stats(C[n], *m)['pooled']; R = s['pooled'] / car
        print(f'\n{n}: animal means {np.round(s["A"], 4)}; {s["n_neg"]}/4 < 0; t(3) = {s["t"]:.2f}, p = {s["p"]:.4f}; '
              f'pooled drug Δz {s["pooled"]:+.4f} ({s["pooled"] / s["sd"]:+.1f} own EO SD; EO SD {s["sd"]:.4f}); '
              f'CAR {car:+.4f}; R {R:.2f}; R vs anchor {s["pooled"] / ANCHOR:.2f}; rule: {verdict(s, R)}')
        print('  ' + ' | '.join(f'{a} {v[0]:+.4f} ({v[1]}/{v[2]} down)' for a, v in s['agm'].items()))
        if n != 'kshift':
            _, _, db, _ = session_deltas(B[n], *m); ag, _, dc, _ = session_deltas(C[n], *m); dr = ag != 'SLP'
            print(f'  per-session drug Δ, bipolar vs CAR: r = {np.corrcoef(db[dr], dc[dr])[0, 1]:+.2f} (n = {dr.sum()})')
    s = stats(B['auto k4'], *m); print(f'\nVERDICT (AUTO k4 ψ(T), R vs anchor {ANCHOR}): {verdict(s, s["pooled"] / ANCHOR)}')
