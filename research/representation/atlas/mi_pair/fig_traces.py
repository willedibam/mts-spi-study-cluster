"""The windows that fig:mi (d) draws: per dataset, the wake and anaesthetised window of one subject chosen for contrast.

    python fig_traces.py <results/representation/atlas-mi-pair-261117> <claar_sel dir> <claar_ind_sel dir> <prios windows dir>
    python fig_traces.py macaque <banks dir>          (on Gadi: prints the two macaque window ids for TRACE_MAC)

Rule, every dataset: each analysed window gets a look score, the sum of two z-scores taken over the dataset's wake and
anaesthetised windows: log slow power (below 2 Hz, as a fraction of 0.5-40 Hz, Welch) and mean absolute correlation
between channels. Per subject, the lowest-scoring wake window and the highest-scoring anaesthetised window; the subject
whose pair differs most is drawn. Selected for contrast, not typical: the slow, synchronous look is also in the joint
IAAFT surrogate (same spectra), so it is context, not the effect. Analysed windows: macaque awake_eo and drug windows
of the banks; mouse awake (claar_z_ns) and qualifying induction (claar_z_ind) windows of the 9 mice of the induction
result; human awake and anaesthetised windows of the 7 kept patients. The mouse and human windows are stored, as scored,
in <root>/figwin/traces_d.npz (local-only, like figwin): mouse CAR with the last channel dropped (prepare.reference),
human CAR over the good non-stimulated channels with the last dropped and the stimulated ones left out (score.py).
Keys `<species>:<wake|anaes>` (T x M, float32) and `<key>|src` (subject and window index, as a string). The macaque
windows go to figwin/windows.npz through fig_windows.py.
"""
import glob
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.signal import welch


def look(x):
    """Slow-power fraction and mean |r| between channels of one window (T x M, 100 Hz)."""
    f, p = welch(x, fs=100, nperseg=min(256, len(x)), axis=0); P = p.sum(1)
    c = np.corrcoef(x.T); iu = np.triu_indices(len(c), 1)
    return P[f < 2].sum() / P[(f >= .5) & (f < 40)].sum(), np.abs(c[iu]).mean()


def contrast(subj, wake, L):
    """subj, wake (bool) and L = (slow, sync) per window -> (wake index, anaesthetised index) of the chosen subject."""
    slow, sync = np.log(L[:, 0]), L[:, 1]
    s = (slow - slow.mean()) / slow.std() + (sync - sync.mean()) / sync.std()
    best = None
    for p in dict.fromkeys(subj):
        w, a = np.flatnonzero((subj == p) & wake), np.flatnonzero((subj == p) & ~wake)
        if len(w) and len(a):
            iw, ia = w[s[w].argmin()], a[s[a].argmax()]
            if best is None or s[ia] - s[iw] > best[0]:
                best = (s[ia] - s[iw], iw, ia)
    return best[1:]


if __name__ == "__main__" and sys.argv[1] == "macaque":
    ids, subj, wake, L = [], [], [], []
    for st in "ab":
        d = f"{sys.argv[2]}/neurotycho-long-stage-{st}"; man = json.load(open(f"{d}/manifest.json"))
        with np.load(f"{d}/windows.npz") as W:
            for r in man["rows"]:
                if r["state"] in ("awake_eo", "drug"):
                    ids.append(r["id"]); subj.append(r["unit"]); wake.append(r["state"] == "awake_eo")
                    L.append(look(W[r["id"]].astype(float)))
    iw, ia = contrast(np.array(subj), np.array(wake), np.array(L))
    print(ids[iw], ids[ia])
elif __name__ == "__main__":
    from plot_character_draft import human_propofol, mouse_induction
    root, sel, ind, prios = (Path(a) for a in sys.argv[1:5])
    out = {}
    _, info, _, _ = mouse_induction(root, detail=True)
    W, subj, wake, src = [], [], [], []
    for d_, k in ((sel, 1), (ind, 2)):
        files = sorted(glob.glob(f"{d_}/sub-*.npz")); j0 = 0
        for f in files:
            Wf = np.load(f)["W"]; s = os.path.basename(f)[:-4]; e = [x for x in info if x[0] == s]
            for j in (np.flatnonzero(e[0][k][j0:j0 + len(Wf)]) if e else []):
                w = Wf[j].astype(float); W.append((w - w.mean(1, keepdims=True))[:, :-1])
                subj.append(s); wake.append(k == 1); src.append(f"{s} window {j0 + j}")
            j0 += len(Wf)
    iw, ia = contrast(np.array(subj), np.array(wake), np.array([look(w) for w in W]))
    for key, i in (("mouse:wake", iw), ("mouse:anaes", ia)):
        out[key] = W[i].astype(np.float32); out[key + "|src"] = src[i]
    D, keep, hs, st, res, _ = human_propofol(root, detail=True)
    W, subj, wake, src = [], [], [], []
    for p in keep:
        f = np.load(prios / f"{p}.npz"); rows = json.loads(str(f["rows"])); assert len(rows) == (hs == p).sum()
        for j, r in enumerate(rows):
            if r["state"] in ("awake", "anaes"):
                w = f["W"][j].astype(float); m = w.shape[1]; kc = np.isin(np.arange(m), r.get("stim", []), invert=True)
                W.append((w - w[:, kc].mean(1, keepdims=True))[:, [c for c in range(m - 1) if kc[c]]])
                subj.append(p); wake.append(r["state"] == "awake"); src.append(f"{p} window {j}")
    iw, ia = contrast(np.array(subj), np.array(wake), np.array([look(w) for w in W]))
    for key, i in (("human:wake", iw), ("human:anaes", ia)):
        out[key] = W[i].astype(np.float32); out[key + "|src"] = src[i]
    for k, v in out.items():
        print(k, v if isinstance(v, str) else v.shape)
    np.savez_compressed(root / "figwin/traces_d.npz", **out)
