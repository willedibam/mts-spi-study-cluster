"""Windows for monkey B by the frozen rule (preregs/ferraina-prereg.md): bad channels on W1 eligible 100-Hz data, CAR over
good channels (last dropped), 10-s grid windows per state, atlas window_ok with W1's robust centre/scale on all good
channels, exposed-block exclusion. Saves the 32-channel primary set, the all-good set, and per-window burst index and
delta fraction (both on the 32 CAR channels).    python select_fb.py EXTRACT.npz OUT_DIR"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))
from research.representation.atlas.prepare import bad_channels, reference, robust_scale, window_ok, delta_fraction

STATES = dict(W1=(231, 1979), D=(3000, 6000), W2=(8031, 9606))
EXPOSED = [(170.94, 230.94), (4165.94, 4225.94)]
L = 1000
E = np.load(sys.argv[1]); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
Y, FL, t0, ch = E['Y'], E['flat'], float(E['t0']), E['channels']
idx = lambda s: int(round((s - t0) * 100))
grid = {st: [a + 10 * k for k in range(int((b - a) // 10)) if a + 10 * k + 10 <= b] for st, (a, b) in STATES.items()}
w1 = np.concatenate([Y[idx(s):idx(s) + L] for s in grid['W1']]).astype(np.float64)
bad = bad_channels(w1); good = [i for i in range(Y.shape[1]) if i not in bad]
kept = good[:-1]; M = len(kept)
sel32 = np.round(np.arange(32) * (M - 1) / 31).astype(int)
centre, scale = robust_scale(reference(w1[:, good], 'car'))
flat = FL[:, good].any(axis=1)
rows, W32, WALL, WAS = [], [], [], []
bi = lambda w: float(np.median(1.4826 * np.median(np.abs(w - np.median(w, 0)), 0) / w.std(0)))
for st, starts in grid.items():
    for k, s in enumerate(starts):
        excl = any(s < b and s + 10 > a for a, b in EXPOSED)
        i = idx(s); w = reference(Y[i:i + L, good].astype(np.float64), 'car')
        ok = (not excl) and window_ok(w, centre, scale, flat[i:i + L])
        r = dict(state=st, k=k, start_s=s, exposed=excl, ok=bool(ok))
        if ok:
            w32 = w[:, sel32]; r.update(burst_index=bi(w32), delta_fraction=delta_fraction(w32), rms=float(np.median(w32.std(0))))
            W32.append(w32.astype(np.float32)); WALL.append(w.astype(np.float32))
            WAS.append(Y[i:i + L, kept][:, sel32].astype(np.float32))
        rows.append(r)
acc = [r for r in rows if r['ok']]
np.save(f'{out}/W32.npy', np.array(W32)); np.save(f'{out}/WALL.npy', np.array(WALL)); np.save(f'{out}/W32asis.npy', np.array(WAS))
meta = dict(bad_channels=[int(ch[i]) for i in bad], dropped_after_car=int(ch[good[-1]]), n_kept=M,
            sel32_channels=[int(ch[kept[j]]) for j in sel32], n_sel32_le96=int(sum(ch[kept[j]] <= 96 for j in sel32)))
json.dump(dict(meta=meta, rows=rows, accepted=acc), open(f'{out}/sel.json', 'w'), indent=0)
print(json.dumps(meta))
for st in STATES:
    R = [r for r in rows if r['state'] == st]; A = [r for r in R if r['ok']]
    q = lambda key: np.percentile([r[key] for r in A], [5, 50, 95]).round(3).tolist()
    print(st, 'grid', len(R), 'exposed', sum(r['exposed'] for r in R), 'rejected', sum((not r['ok']) and not r['exposed'] for r in R),
          'accepted', len(A), 'BI', q('burst_index'), 'delta', q('delta_fraction'), 'rms', q('rms'))
