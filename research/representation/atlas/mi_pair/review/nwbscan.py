"""All Claar sessions: EEG span/rate, valid channels, epochs, stimulation blocks, and stimulus-free stretches by state."""
import json, h5py, numpy as np
from curlfile import CurlFile
dec = lambda a: [x.decode() if isinstance(x, bytes) else x for x in a]
A = json.load(open('dandi/assets.json'))['results']; out = []
for a in A:
    try:
        f = CurlFile(f"https://api.dandiarchive.org/api/assets/{a['asset_id']}/download/"); h = h5py.File(f, 'r')
        ts = h['acquisition/ElectricalSeriesEEG/timestamps']; n = ts.shape[0]; t0, t1 = float(ts[0]), float(ts[n - 1]); fs = 1 / np.median(np.diff(ts[:2000]))
        E = h['intervals/epochs']; ep = list(zip(dec(E['tags'][:]), E['start_time'][:].tolist(), E['stop_time'][:].tolist()))
        T = h['intervals/trials']; st = T['start_time'][:]; be = np.array(dec(T['behavioral_epoch'][:])); ty = np.array(dec(T['stimulus_type'][:]))
        valid = int(h['general/extracellular_ephys/electrodes/is_data_valid'][:].sum())
        blocks = {b: (float(st[be == b].min()), float(st[be == b].max()), int((be == b).sum()), sorted(set(ty[be == b]))) for b in np.unique(be)}
        s = np.sort(st); gaps = [(float(x), float(y)) for x, y in zip(np.r_[t0, s], np.r_[s, t1]) if y - x >= 30]
        out.append(dict(path=a['path'], gb=a['size'] / 1e9, fs=fs, t0=t0, t1=t1, valid=valid, epochs=ep, blocks=blocks, free=gaps))
        print(a['path'].split('/')[0], f"{a['size']/1e9:5.1f}GB fs {fs:.0f} valid {valid}/30 EEG {t0:.0f}-{t1:.0f}", flush=True)
        print('   epochs', [(e[0], round(e[1]), round(e[2])) for e in ep])
        print('   stim', {b: (round(v[0]), round(v[1]), v[2], v[3]) for b, v in blocks.items()})
        print('   stim-free ≥30 s', [(round(x), round(y)) for x, y in gaps])
    except Exception as e: print(a['path'], 'ERR', e)
json.dump(out, open('dandi/scan.json', 'w'))
