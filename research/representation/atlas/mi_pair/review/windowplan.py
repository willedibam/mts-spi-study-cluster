"""Stimulus-free 10-s window budget per Claar session (no EEG samples read).
awake: from EEG start + 30 s to 5 s before the first stimulus of the session.
iso:   inside the isoflurane_anesthesia epoch, from its start + 60 s, at least 5 s after and 1 s before any stimulus."""
import json, numpy as np
S = json.load(open('dandi/scan.json')); rows = []
def nwin(segs): return sum(int((b - a) // 10) for a, b in segs if b > a)
for s in S:
    free = s['free']; ep = {e[0]: (e[1], e[2]) for e in s['epochs']}; a0, a1 = ep['isoflurane_anesthesia']
    first = min(v[0] for v in s['blocks'].values())
    aw = [(s['t0'] + 30, first - 5)]
    iso = [(max(x + 5, a0 + 60), min(y - 1, a1)) for x, y in free]
    rows.append((s['path'].split('/')[1][:24], round(s['gb'], 1), nwin(aw), nwin(iso), [(round(x), round(y)) for x, y in iso if y - x >= 10]))
for r in rows: print(f'{r[0]:26s} {r[1]:5.1f} GB  awake {r[2]:3d}  iso {r[3]:3d}  iso segments {r[4]}')
a = np.array([[r[2], r[3]] for r in rows]); print('mice with ≥10 awake and ≥10 iso windows:', int(((a[:, 0] >= 10) & (a[:, 1] >= 10)).sum()), 'of', len(a), '| ≥6 each:', int(((a >= 6).all(1)).sum()))
