import numpy as np, warnings; warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
import decide10b as d
d.B=100   # 1-s blocks at 100 Hz
v=np.load('raw/hprop/views.npz'); keys=[k for k in v.files if v[k].ndim==2 and v[k].dtype.kind=='f']
print(len(keys), [k for k in v.files if k not in keys][:5], flush=True)
res=Parallel(n_jobs=8)(delayed(d.one)(v[k].astype(float),i) for i,k in enumerate(keys))
np.savez('hprop_decide.npz',V=np.array(res),names=d.NAMES,keys=np.array(keys))
