"""Addendum A window selection for the induction stretch: same good channels and awake artefact scale as the primary
(claar/ awake stretches + claar_sel/ meta), all passing 10-s candidates (cap 60), with burst index, delta fraction, rms.
    python select_induction.py claar claar_ind claar_sel OUT_DIR"""
import sys, json, glob, os
import numpy as np
sys.path.insert(0, '/Users/wedi0306/Code/mts-spi-study-cluster/.claude/worktrees/real-world-apps-9b53a1')
from research.representation.atlas.prepare import robust_scale, window_ok, spread, reference, delta_fraction
L, CAP = 1000, 60
burst = lambda c: float(np.median(1.4826 * np.median(np.abs(c - np.median(c, 0)), 0) / c.std(0)))
prim, ind, sel, dst = sys.argv[1:5]; os.makedirs(dst, exist_ok=True)
for fn in sorted(glob.glob(f'{ind}/sub-*.npz')):
    s = os.path.basename(fn)[:-4]; P = np.load(f'{prim}/{s}.npz'); I = np.load(fn)
    good = json.loads(str(np.load(f'{sel}/{s}.npz')['meta']))['good']
    mp = json.loads(str(P['meta']))['stretches']; aw = []
    for k, x in enumerate(mp):
        if x['state'] != 'awake': continue
        a = int(round((x['elig'][0] - x['t_first']) * 100)); b = min(int(round((x['elig'][1] - x['t_first']) * 100)), x['n100'] - 300)
        aw.append(P[f'X{k}'].astype(np.float64)[a:b][:, good])
    centre, scale = robust_scale(reference(np.concatenate(aw), 'car'))
    x = json.loads(str(I['meta']))['stretches'][0]; X = I['X0'].astype(np.float64)[:, good]; F = I['F0']
    a = int(round((x['elig'][0] - x['t_first']) * 100)); b = min(int(round((x['elig'][1] - x['t_first']) * 100)), x['n100'] - 300)
    cand = list(range(a, b - L + 1, L)); ok = [j for j in cand if window_ok(reference(X[j:j + L], 'car'), centre, scale, F[j:j + L])]
    use = [ok[i] for i in spread(len(ok), CAP)]; W, rows = [], []
    for j in use:
        c = reference(X[j:j + L], 'car'); W.append(X[j:j + L].astype(np.float32))
        rows.append(dict(state='induction', start_s=x['t_first'] + j / 100, speed=float(I['S0'][j:j + L].mean()),
                         delta_fraction=delta_fraction(c), burst=burst(c), rms_uv=float(np.median(c.std(0)))))
    np.savez(f'{dst}/{s}.npz', W=np.array(W).reshape(len(W), L, len(good)), rows=json.dumps(rows))
    print(f'{s}: induction {x["elig"][1] - x["elig"][0]:.0f} s, {len(ok)}/{len(cand)} pass, {len(use)} used; BI median {np.median([r["burst"] for r in rows]) if rows else np.nan:.2f}, '
          f'δ median {np.median([r["delta_fraction"] for r in rows]) if rows else np.nan:.2f}', flush=True)
