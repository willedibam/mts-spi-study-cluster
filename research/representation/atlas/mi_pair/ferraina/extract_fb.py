"""Monkey B LFP (Zenodo 15466548, MEALFP_MB_PFc&PFc.mat) -> 100 Hz via the atlas `continuous` chain (notch 50, HP 0.5,
LP 40, resample_poly 2:1), applied to the whole recording; per-channel source-rate flat-run mask (atlas rule: run of
>= fs identical samples) reduced to 100 Hz. Rules: preregs/ferraina-prereg.md.
    python extract_fb.py IN.mat OUT.npz"""
import sys, os, numpy as np, h5py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))
from research.representation.atlas.prepare import continuous, flat_runs

FS_SRC, MAINS = 200, 50.0
src, out = sys.argv[1], sys.argv[2]
with h5py.File(src, 'r') as f:
    t = f['MEALFP/time'][:, 0]; ch = f['MEALFP/channels'][:, 0].astype(int)
    dt = float(f['MEALFP/dt'][0, 0]); X = f['MEALFP/values'][()]
assert abs(dt - 1 / FS_SRC) < 1e-9 and np.allclose(np.diff(t), dt, atol=1e-6), 'irregular time'
print('read', X.shape, X.dtype, 't', t[0], t[-1], 'nonfinite', int((~np.isfinite(X)).sum()), flush=True)
n100 = (len(X) + 1) // 2
Y = np.empty((n100, X.shape[1]), np.float32); F = np.zeros((n100, X.shape[1]), bool)
for a in range(0, X.shape[1], 16):
    x = X[:, a:a + 16]
    y = continuous(np.nan_to_num(x), FS_SRC, MAINS); assert len(y) == n100, (len(y), n100)
    Y[:, a:a + 16] = y
    for j in range(x.shape[1]):
        fl = flat_runs(x[:, j:j + 1], FS_SRC) | ~np.isfinite(x[:, j])
        F[:, a + j] = np.pad(fl, (0, 2 * n100 - len(fl))).reshape(n100, 2).any(axis=1)
    print('channels', a + 1, '-', a + x.shape[1], flush=True)
np.savez(out, Y=Y, flat=F, t0=t[0], fs=100.0, channels=ch)
print('saved', out, Y.shape, 'flat samples per channel max', int(F.sum(0).max()))
