"""Layout of EEG data/timestamps/electrode region and other intervals for one Claar NWB."""
import sys, h5py, numpy as np
from curlfile import CurlFile
f = CurlFile(f'https://api.dandiarchive.org/api/assets/{sys.argv[1]}/download/'); h = h5py.File(f, 'r')
E = h['acquisition/ElectricalSeriesEEG']
for k in ('data', 'timestamps'): d = E[k]; print(k, d.shape, d.dtype, d.chunks, d.compression, dict(d.attrs))
reg = E['electrodes'][:]; print('electrodes region', reg[:5], '...', reg[-3:], len(reg))
el = h['general/extracellular_ephys/electrodes']; v = el['is_data_valid'][:]; print('valid at region', v[reg].astype(int))
print('electrode cols', list(el.keys()))
print('intervals', list(h['intervals'].keys())); print('acquisition', list(h['acquisition'].keys())); print('processing', {k: list(h['processing'][k].keys()) for k in h['processing']})
ts = E['timestamps']; n = ts.shape[0]; idx = [0, n // 3, 2 * n // 3, n - 1]; t = [float(ts[i]) for i in idx]
fs = (t[-1] - t[0]) / (n - 1); print('fs global', fs, 'linear-pred error (s) at thirds', [t[j] - (t[0] + idx[j] / fs) for j in range(4)])
print('requests', f.nreq)
