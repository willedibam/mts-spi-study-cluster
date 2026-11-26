"""Post hoc (prereg Clarification 9): per-array common average for the windows already accepted by select_fb.py.
Each array (channels 1-96, 97-192) is referenced to its own good-channel mean and its last good channel dropped. Window
selection is unchanged. Saves W32pa.npy (the same 32 channel numbers as the primary) and WALLpa.npy (all kept channels).
    python select_pa.py EXTRACT.npz SEL_DIR"""
import sys, json, numpy as np
E = np.load(sys.argv[1]); d = sys.argv[2]; S = json.load(open(f'{d}/sel.json'))
Y, t0, ch = E['Y'], float(E['t0']), E['channels']
good = [i for i in range(Y.shape[1]) if int(ch[i]) not in S['meta']['bad_channels']]
arrays = [[i for i in good if ch[i] <= 96], [i for i in good if ch[i] > 96]]
kept = arrays[0][:-1] + arrays[1][:-1]; kch = [int(ch[i]) for i in kept]
sel = [kch.index(c) for c in S['meta']['sel32_channels']]
W32, WALL = [], []
for r in S['accepted']:
    i = int(round((r['start_s'] - t0) * 100)); w = Y[i:i + 1000].astype(np.float64)
    w = np.concatenate([(w[:, a] - w[:, a].mean(1, keepdims=True))[:, :-1] for a in arrays], 1)
    WALL.append(w.astype(np.float32)); W32.append(w[:, sel].astype(np.float32))
np.save(f'{d}/W32pa.npy', np.array(W32)); np.save(f'{d}/WALLpa.npy', np.array(WALL))
json.dump(dict(kept_channels=kch, dropped=[int(ch[a[-1]]) for a in arrays], sel32=S['meta']['sel32_channels']), open(f'{d}/pa.json', 'w'))
print('kept', len(kch), 'dropped', [int(ch[a[-1]]) for a in arrays], 'windows', len(W32))
