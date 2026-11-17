"""Epochs, stimulus timing, EEG rate, electrode validity/locations for one remote Claar NWB (small reads only)."""
import sys, h5py, numpy as np, collections
from curlfile import CurlFile
f = CurlFile(f'https://api.dandiarchive.org/api/assets/{sys.argv[1]}/download/'); h = h5py.File(f, 'r')
dec = lambda a: [x.decode() if isinstance(x, bytes) else x for x in a]
ts = h['acquisition/ElectricalSeriesEEG/timestamps']; n = ts.shape[0]; t0 = ts[:2000]
print('EEG samples', n, 'fs ≈', round(1 / np.median(np.diff(t0)), 2), 'span', round(t0[0], 1), '-', round(ts[n - 1], 1), 's')
print('conversion', dict(h['acquisition/ElectricalSeriesEEG/data'].attrs).get('conversion'), 'unit', dict(h['acquisition/ElectricalSeriesEEG/data'].attrs).get('unit'))
E = h['intervals/epochs']; print('epochs', list(zip(dec(E['tags'][:]), E['start_time'][:].round(1), E['stop_time'][:].round(1))))
el = h['general/extracellular_ephys/electrodes']; print('valid', el['is_data_valid'][:].sum(), '/30; ref', set(dec(el['reference'][:])), '; locations', collections.Counter(dec(el['location'][:])).most_common(6))
T = h['intervals/trials']; st = T['start_time'][:]; sp = T['stop_time'][:]; be = np.array(dec(T['behavioral_epoch'][:])); ty = np.array(dec(T['stimulus_type'][:]))
print('trial types', collections.Counter(zip(be, ty)))
for b in np.unique(be):
    m = be == b; s = np.sort(st[m]); gap = np.diff(s)
    print(f'  {b}: n={m.sum()} span {s[0]:.0f}-{s[-1]:.0f} s, ITI median {np.median(gap):.2f} s (min {gap.min():.2f}, max {gap.max():.1f}); trial dur median {np.median(sp[m]-st[m]):.3f}')
print('first trials', list(zip(st[:3].round(2), dec(T['stimulus_description'][:3]))))
print('remote requests', f.nreq)
