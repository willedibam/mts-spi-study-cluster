"""Nearest-neighbour bipolar pairs per animal from the provider electrode maps (geometry only; no signal is read).

The rule of `research/representation/neurotycho/stage_neurotycho_source.py` `montage()`, with two changes:
electrodes unavailable in the atlas banks are excluded before the greedy pass, and 32 pairs are selected.
  - threshold: 1.5 x the median nearest-neighbour distance over all 128 map electrodes;
  - candidates: pairs of available electrodes within the threshold, ascending distance (index ties as sorted());
  - greedy non-overlapping pairs; farthest-point coverage of pair midpoints from the leftmost midpoint (lexsort x, y).
Unavailable = 128 (dropped after the common average in every unit) and the union of the animal's bad channels.
Electrode n of the map (1-based) is taken to be ECoG_ch{n}.mat (assumed; SampleCode.m only plots X, Y in order).
Signal = lower-numbered minus higher-numbered electrode.

    python pairs.py <montage dir with {Chibi,George,Kin2,Su}.zip> pairs.json
"""
import sys, io, json, zipfile, hashlib, numpy as np
from pathlib import Path
from scipy.io import loadmat

COUNT = 32
BAD = {'Chibi': [], 'George': [34, 53], 'Kin2': [10], 'Su': [38]}  # union over units, from the bank manifests
DROPPED = 128


def pairs(path, exclude, count=COUNT):
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith('.mat'))
        data = loadmat(io.BytesIO(z.read(name)), simplify_cells=True)
    xy = np.column_stack([data['X'], data['Y']]).astype(float)
    if xy.shape != (128, 2) or not np.isfinite(xy).all():
        raise ValueError('unexpected electrode map')
    d = np.linalg.norm(xy[:, None] - xy[None], axis=-1); np.fill_diagonal(d, np.inf)
    maximum = 1.5 * np.median(d.min(axis=1))
    ok = [i for i in range(128) if i + 1 not in exclude]
    cand = sorted((d[i, j], i, j) for a, i in enumerate(ok) for j in ok[a + 1:] if d[i, j] <= maximum)
    used, P = set(), []
    for _, i, j in cand:
        if i not in used and j not in used: P.append((i, j)); used.update((i, j))
    if len(P) < count: raise ValueError(f'{path}: only {len(P)} pairs')
    mid = np.array([xy[[i, j]].mean(0) for i, j in P])
    chosen = [int(np.lexsort((mid[:, 1], mid[:, 0]))[0])]
    while len(chosen) < count:
        sep = np.linalg.norm(mid[:, None] - mid[chosen], axis=-1).min(1); sep[chosen] = -np.inf; chosen.append(int(sep.argmax()))
    return dict(pairs=[[P[c][0] + 1, P[c][1] + 1] for c in chosen], n_candidate_pairs=len(P), excluded=sorted(exclude),
                maximum_pair_distance=float(maximum), median_nn=float(np.median(d.min(axis=1))),
                pair_distance=[float(d[P[c]]) for c in chosen], map=name, xy=xy.tolist(),
                map_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


if __name__ == '__main__':
    src, out = Path(sys.argv[1]), sys.argv[2]
    res = {a: pairs(src / f'{a}.zip', set(BAD[a]) | {DROPPED}) for a in BAD}
    json.dump(dict(rule=__doc__.split('\n\n')[0], distance_units='provider image pixels', animals=res), open(out, 'w'), indent=1)
    for a, r in res.items():
        print(a, r['n_candidate_pairs'], 'pairs;', f"threshold {r['maximum_pair_distance']:.1f}; selected distances {min(r['pair_distance']):.1f}-{max(r['pair_distance']):.1f}")
