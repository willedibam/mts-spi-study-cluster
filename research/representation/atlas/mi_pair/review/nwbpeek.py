"""List the HDF5 tree of one remote NWB file (metadata only, range requests) and print epochs, EEG series shape/rate."""
import sys, h5py, fsspec, numpy as np
url = f'https://api.dandiarchive.org/api/assets/{sys.argv[1]}/download/'
fs = fsspec.filesystem('http'); f = fs.open(url, block_size=2**20, cache_type='blockcache')
h = h5py.File(f, 'r')
def show(name, obj):
    if isinstance(obj, h5py.Dataset) and name.count('/') <= 5: print(f'{name}  {obj.shape} {obj.dtype}')
h.visititems(show)
for k in ('session_description', 'identifier'): print(k, h.attrs.get(k))
