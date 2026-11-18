"""Atlas window selection and outcome-blind QC for extracted Claar sessions (no z, no Δ).
Per session: atlas bad-channel screen over all eligible data; per state, robust scale on average-referenced eligible
data, non-overlapping 10-s candidates tiling each eligible range, atlas window_ok, at most 30 spread evenly.
Saves pre-CAR windows of good channels with state, start time, mean running speed and delta fraction.
    python select_claar.py IN_DIR OUT_DIR
"""
import sys, json, glob, os
import numpy as np
sys.path.insert(0, '/Users/wedi0306/Code/mts-spi-study-cluster/.claude/worktrees/real-world-apps-9b53a1')
from research.representation.atlas.prepare import bad_channels, robust_scale, window_ok, spread, reference, delta_fraction
from scipy.signal import welch
L, CAP = 1000, 30
src, dst = sys.argv[1], sys.argv[2]; RULE = sys.argv[3] if len(sys.argv) > 3 else 'awake'   # 'awake' (deviation 1, primary) or 'own' (clarification 4)
burst = lambda c: float(np.median(1.4826 * np.median(np.abs(c - np.median(c, 0)), 0) / c.std(0))); os.makedirs(dst, exist_ok=True); qc = []
for fn in sorted(glob.glob(f'{src}/sub-*.npz')):
    Z = np.load(fn); meta = json.loads(str(Z['meta'])); S = meta['stretches']
    el = []
    for k, s in enumerate(S):
        a = int(round((s['elig'][0] - s['t_first']) * 100)); b = int(round((s['elig'][1] - s['t_first']) * 100))
        el.append((k, s['state'], max(a, 0), min(b, s['n100'] - 300)))      # ≥ 3 s from stretch edges
    bad = bad_channels(np.concatenate([Z[f'X{k}'][a:b] for k, _, a, b in el]).astype(np.float64))
    good = [j for j in range(len(meta['cols'])) if j not in bad]
    rows, W, rec = [], [], dict(session=os.path.basename(fn)[:-4], M_valid=len(meta['cols']), bad=bad, M_used=len(good) - 1,
                                MB=meta['bytes'] / 1e6, timing_ms=meta['timing_err_s'] * 1e3)
    sp = lambda state: [(k, a, b) for k, st, a, b in el if st == state and b - a >= L]
    eligible = lambda state: reference(np.concatenate([Z[f'X{k}'].astype(np.float64)[a:b][:, good] for k, a, b in sp(state)]), 'car')
    if RULE == 'awake': centre, scale = robust_scale(eligible('awake'))
    for state in ('awake', 'iso'):
        spans = sp(state)
        X = {k: Z[f'X{k}'].astype(np.float64)[:, good] for k, _, _ in spans}
        if RULE == 'own': centre, scale = robust_scale(eligible(state))
        cand = [(k, j) for k, a, b in spans for j in range(a, b - L + 1, L)]
        ok = [(k, j) for k, j in cand if window_ok(reference(X[k][j:j + L], 'car'), centre, scale, Z[f'F{k}'][j:j + L])]
        sel = [ok[i] for i in spread(len(ok), CAP)]
        for k, j in sel:
            w = X[k][j:j + L]; c = reference(w, 'car')
            W.append(w.astype(np.float32))
            rows.append(dict(state=state, start_s=S[k]['t_first'] + j / 100, speed=float(Z[f'S{k}'][j:j + L].mean()),
                             delta_fraction=delta_fraction(c), burst=burst(c), rms_uv=float(np.median(c.std(0)))))
        rs = [r for r in rows if r['state'] == state]
        rec[state] = dict(cand=len(cand), passed=len(ok), used=len(sel), rms_uv=float(np.median([r['rms_uv'] for r in rs])) if rs else None,
                          delta=float(np.median([r['delta_fraction'] for r in rs])) if rs else None,
                          speed=float(np.median([r['speed'] for r in rs])) if rs else None,
                          burst=float(np.median([r['burst'] for r in rs])) if rs else None)
    rec['included'] = rec['awake']['used'] >= 10 and rec['iso']['used'] >= 10
    np.savez(f"{dst}/{rec['session']}.npz", W=np.array(W), rows=json.dumps(rows), meta=json.dumps(dict(meta, good=good, bad=bad)))
    qc.append(rec)
    a, i = rec['awake'], rec['iso']
    print(f"{rec['session']:12s} M {rec['M_valid']}→{rec['M_used']} bad {bad} | awake {a['passed']}/{a['cand']}→{a['used']} rms {a['rms_uv']:.0f}µV δ {a['delta']:.2f} run {a['speed']:.1f} | "
          f"iso {i['passed']}/{i['cand']}→{i['used']} rms {i['rms_uv']:.0f}µV δ {i['delta']:.2f} BI {i['burst']:.2f} | {rec['MB']:.0f} MB, {rec['timing_ms']:.2f} ms {'IN' if rec['included'] else 'OUT'}", flush=True)
json.dump(qc, open(f'{dst}/qc.json', 'w'), indent=1)
