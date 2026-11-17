"""Are high-burst-index isoflurane windows continuous EEG or near-isoelectric suppression? Plus 7-mouse subset CI, and the primary without 569070."""
import json, numpy as np
Z = np.load('claar_z_ns.npz'); rows = json.loads(str(Z['rows'])); res = Z['ns'][:, 0] - Z['ns'][:, 1:].mean(1)
sess = np.array([r['session'] for r in rows]); st = np.array([r['state'] for r in rows]); bi = np.array([r['burst'] for r in rows]); rms = np.array([r['rms_uv'] for r in rows])
P = [s for s in dict.fromkeys(sess) if s != 'sub-569073_ses-20210416']; d85 = []
for s in P:
    a, i = (sess == s) & (st == 'awake'), (sess == s) & (st == 'iso'); p5 = np.percentile(bi[a], 5); h = i & (bi >= p5)
    k = i & (bi >= .85)
    print(f'{s[:10]} awake rms {np.median(rms[a]):4.0f} BI p5 {p5:.2f} | iso n {i.sum()} ≥awake-p5 {h.sum():2d} rms {np.median(rms[h]) if h.any() else np.nan:4.0f} | BI≥.85 {k.sum():2d} rms {np.median(rms[k]) if k.any() else np.nan:4.0f} | iso low-BI rms {np.median(rms[i & (bi < .6)]) if (i & (bi < .6)).any() else np.nan:4.0f}')
    if k.sum() >= 5: d85.append(res[k].mean() - res[a].mean())
d85 = np.array(d85); b = np.random.default_rng(0).choice(d85, (10000, len(d85))).mean(1)
print(f'BI≥.85 subset: n={len(d85)} mean {d85.mean():+.4f} [{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}]  (primary −0.0191; macaque KT/KTMD/PF ≈ −0.012 to −0.016)')
d = np.array([res[(sess == s) & (st == 'iso')].mean() - res[(sess == s) & (st == 'awake')].mean() for s in P if s != 'sub-569070_ses-20210401'])
print(f'primary without design session 569070: {np.sum(d < 0)}/{len(d)} negative, mean {d.mean():+.4f} = {d.mean() / .0026:+.1f} null SD')
