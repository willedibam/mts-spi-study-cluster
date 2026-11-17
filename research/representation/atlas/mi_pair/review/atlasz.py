import sys, io, json, glob, numpy as np
A = '/scratch/ql44/we2614/mts-spi-study/atlas'; arm = sys.argv[1]
man = {r['id']: r for r in json.load(open(f'{A}/banks/{arm}/manifest.json'))['rows']}
ids, Z, LV, SPIS = [], [], [], None
for p in sorted(glob.glob(f'{A}/pyspi/{arm}/shard-*.npz')):
    b = np.load(p, allow_pickle=False)
    spis = [str(s) for s in b['__spis__']]
    if SPIS is None: SPIS = spis
    assert spis == SPIS
    for k in b.files:
        if k.startswith('__'): continue
        mpi = b[k]; K, M, _ = mpi.shape
        V = mpi[:, ~np.eye(M, dtype=bool)].astype(np.float64)
        lv = np.nanmean(V, 1)
        ok = np.isfinite(V).all(1) & (V.std(1) > 0)
        Vc = (V - V.mean(1, keepdims=True)) / np.where(V.std(1, keepdims=True) > 0, V.std(1, keepdims=True), 1)
        C = Vc @ Vc.T / V.shape[1]; C[~ok, :] = np.nan; C[:, ~ok] = np.nan
        Z.append(C[np.triu_indices(K, 1)].astype(np.float32)); LV.append(lv.astype(np.float32)); ids.append(k)
meta = [man[i] for i in ids]
keys = sorted({k for m in meta for k in m})
out = dict(z=np.array(Z), level=np.array(LV), spi=np.array(SPIS), id=np.array(ids), meta=np.array(json.dumps(meta)))
buf = io.BytesIO(); np.savez(buf, **out); sys.stdout.buffer.write(buf.getvalue())
