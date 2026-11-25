"""Self-contained interactive explorer of fig:mi panel (b): every window's per-edge (G, K) and its surrogate, the band,
per-edge p-values against the surrogate's residuals, and, for windows with raw samples, a clicked pair's traces, glyph
and position on the brain.

    python explorer.py results/representation/atlas-mi-pair-261117 <out.html>

Reads zns/ (every 10-s window's 496 edges, normal-scored arm, as the figure), figwin/brain_*.npz, and the raw windows
in figwin/windows.npz plus figwin/explorer_windows.npz. The latter holds, for each session of PANEL_B and GLYPH_PAIR,
the window nearest its state's median Δz in that session (plus the figure's own windows); regenerate on Gadi with
`fig_windows.py <banks> explorer_windows.npz <ids>`, the ids printed by this script. The page template is
explorer.html beside this file; the data are inlined so the page works offline.
"""
import base64, io, json, sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import norm, rankdata

import plot_character_draft as P

root, out = Path(sys.argv[1]), Path(sys.argv[2])
df, edges = P.load(root)
iu = np.triu_indices(32, 1)
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
ns = lambda x: norm.ppf((np.apply_along_axis(rankdata, 0, x) - .5) / len(x))   # as edges.ns

# per-window metadata and (G, K, G_s, K_s), int16 at 1e-4
chans, cidx, rows = [], {}, []
chan = {}
for p in sorted((root / "zns").glob("zns_[AB]_*.npz")):
    with np.load(p) as d:
        chan.update(zip(d["id"], d["chan"]))
order = {"awake_eo": 0, "awake_ec": 1, "drug": 2, "sleep": 2, "recovery_ec": 3}
df["num"] = df.id.str.rsplit(":", n=1).str[1].astype(int)
df = df.sort_values(["animal", "unit", "state", "num"], key=lambda s: s.map(order) if s.name == "state" else s)
GK = np.empty((len(df), 4, 496), np.int16)
for i, r in enumerate(df.itertuples()):
    (g, k), (gs, ks) = edges[r.id]
    GK[i] = np.clip(np.round(np.array([g, k, gs, ks]) * 1e4), -32767, 32767)
    c = tuple(int(v) for v in chan[r.id])
    if c not in cidx:
        cidx[c] = len(chans); chans.append(list(c))
    cr, lim, off = P.band(edges, r.id)                   # exact, from float32: int16 storage flips borderline pairs
    cs = np.polyfit(*edges[r.id][1], 1)
    rows.append(dict(id=r.id, unit=r.unit, animal=r.animal, state=r.state, group=r.group, num=int(r.num),
                     sd=round(float(r.sd), 3), z=round(float(r.real), 4), zs=round(float(r.null), 4), c=cidx[c],
                     line=[float(cr[1]), float(cr[0])], sline=[float(cs[1]), float(cs[0])], lim=float(lim),
                     off=b64(np.packbits(off))))
assert np.abs(GK.max()) < 32767, "G or K beyond int16 range"

# raw windows: the 32 zns electrodes, each standardised; checked against banked G
raw = {}
for f in ("windows.npz", "explorer_windows.npz"):
    with np.load(root / "figwin" / f) as W:
        for key in W.files:
            if key.endswith("|chan") or key in raw:
                continue
            lab = list(W[key + "|chan"]); X = W[key].astype(float)[:, [lab.index(e) for e in chan[key]]]
            r = np.corrcoef(ns(X).T)[iu]; g = -.5 * np.log(1 - r ** 2)
            assert np.allclose(g, edges[key][0][0], atol=2e-4), f"G mismatch {key}: {np.abs(g - edges[key][0][0]).max()}"
            X = (X - X.mean(0)) / X.std(0)
            raw[key] = b64(np.clip(np.round(X * 1e3), -32767, 32767).astype(np.int16))
wid, pk = P.GLYPH_PAIR
(g, k), (_, ks) = edges[wid]
print("panel a pair: el.", [int(chan[wid][v[pk]]) for v in iu], "G K K_s", *(round(float(v[pk]), 2) for v in (g, k, ks)))

# brains: the provider's whole photograph (both views: some of the 32 electrodes are medial), greyscale JPEG,
# cropped to the electrodes' extent; electrode i at index i - 1, in crop pixels
brains = {}
for a in ("Chibi", "George"):
    with np.load(root / f"figwin/brain_{a}.npz") as d:
        I, X, Y = d["I"], d["X"], d["Y"]
    top, bot = int(max(Y.min() - 60, 0)), int(min(Y.max() + 60, I.shape[0]))
    left, right = int(max(X.min() - 60, 0)), int(min(X.max() + 60, I.shape[1]))
    im = Image.fromarray(I[top:bot, left:right].mean(2).astype(np.uint8)); buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    brains[a] = dict(img="data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), w=right - left,
                     h=bot - top, x=np.round(X - left, 1).tolist(), y=np.round(Y - top, 1).tolist())

# band check against the figure's own function, and the surrogate scale's stability within session and state
for s_, w_, _ in P.PANEL_B:
    print("band", s_, w_, "off band", int(P.band(edges, w_)[2].sum()))
lims = df.assign(lim=[P.band(edges, w)[1] for w in df.id]).groupby(["unit", "state"]).lim
cv = (lims.std() / lims.mean()).describe()
print("CV of band half-width within session x state: median %.2f, 90%% %.2f" % (cv["50%"], np.nanpercentile(lims.std() / lims.mean(), 90)))

data = dict(Q=P.Q, colors=P.COLOR, panelB=[[s, w, n] for s, w, n in P.PANEL_B], glyph=[wid, int(pk)],
            windows=rows, chans=chans, gk=b64(GK), raw=raw, brains=brains)
html = (Path(__file__).parent / "explorer.html").read_text().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
out.parent.mkdir(parents=True, exist_ok=True); out.write_text(html)
print(out, "%.1f MB" % (len(html) / 1e6), len(raw), "raw windows")
