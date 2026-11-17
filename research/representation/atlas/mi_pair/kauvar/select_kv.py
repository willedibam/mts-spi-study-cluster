"""Window selection for the Kauvar sEEG test (kauvar-prereg.md): common bipolar channels per subject, bad channels on the
concatenated sessions, 32 spread in shaft order, reference-session robust scale, 10-s windows over 5–585 s, cap 30.
    python select_kv.py kv OUT_DIR"""
import sys, json, glob, os, collections
import numpy as np
sys.path.insert(0, '/Users/wedi0306/Code/mts-spi-study-cluster/.claude/worktrees/real-world-apps-9b53a1')
from research.representation.atlas.prepare import robust_scale, window_ok, spread, bad_channels, delta_fraction
L, CAP = 1000, 30
burst = lambda c: float(np.median(1.4826 * np.median(np.abs(c - np.median(c, 0)), 0) / c.std(0)))
src, dst = sys.argv[1:3]; os.makedirs(dst, exist_ok=True); qc = []
by = collections.defaultdict(dict)
for f in glob.glob(f'{src}/sub-*_puff*.npz'):
    s, ses = os.path.basename(f)[:-4].split('_'); by[s][ses] = f
for s, d in sorted(by.items()):
    ref = 'puffPreinfusion' if 'puffPreinfusion' in d else 'puffBaseline'; use = [(ref, 'reference'), ('puffInfusion', 'infusion'), ('puffPostinfusion', 'postinfusion')]
    Z = {ses: np.load(d[ses]) for ses, _ in use if ses in d}; meta = {ses: json.loads(str(z['meta'])) for ses, z in Z.items()}
    common = [p for p in meta[ref]['pairs'] if all(p in m['pairs'] for m in meta.values())]
    X = {ses: Z[ses]['X'][:, [meta[ses]['pairs'].index(p) for p in common]].astype(np.float64) for ses in Z}
    bad = bad_channels(np.concatenate(list(X.values()))); good = [k for k in range(len(common)) if k not in bad]
    pick = [good[i] for i in np.round(np.arange(min(32, len(good))) * (len(good) - 1) / max(min(32, len(good)) - 1, 1)).astype(int)]
    centre, scale = robust_scale(X[ref][500:len(X[ref]) - 500][:, pick])
    W, rows, q = [], [], dict(subject=s, reference=ref, n_common=len(common), n_bad=len(bad), channels=[common[k] for k in pick])
    for ses, state in use:
        if ses not in X: q[state] = 'missing'; continue
        x, F = X[ses][:, pick], Z[ses]['F']; end = min(len(x), 58500)
        cand = list(range(500, end - L + 1, L)); ok = [j for j in cand if window_ok(x[j:j + L], centre, scale, F[j:j + L])]
        for j in [ok[i] for i in spread(len(ok), CAP)]:
            w = x[j:j + L]; W.append(w.astype(np.float32))
            rows.append(dict(state=state, ses=ses, start_s=j / 100, delta_fraction=delta_fraction(w), burst=burst(w), rms_uv=float(np.median(w.std(0)))))
        q[state] = f'{len(ok)}/{len(cand)} pass'
    np.savez(f'{dst}/{s}.npz', W=np.array(W), rows=json.dumps(rows)); qc.append(q)
    print(s, {k: v for k, v in q.items() if k != 'channels'}, 'loc sample', q['channels'][:4], flush=True)
json.dump(qc, open(f'{dst}/qc.json', 'w'), indent=1)
