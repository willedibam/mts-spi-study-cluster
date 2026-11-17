"""Network-wide: per edge, kurtosis of the linear-fit residual (intermittent decoupling) and heteroscedasticity, real and the same IAAFT as edgemi_nt2s_32."""
import json, numpy as np, warnings; warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
from surr_pairs import multivariate_iaaft
def f(x):
    x=(x-x.mean(0))/x.std(0); T,M=x.shape; iu=np.triu_indices(M,1); X,Y=x[:,iu[0]],x[:,iu[1]]
    r=(X*Y).mean(0); e=(Y-r*X)/np.sqrt(1-r**2); e2=(X-r*Y)/np.sqrt(1-r**2)
    k=((e**4).mean(0)+(e2**4).mean(0))/2
    h=(np.array([np.corrcoef(np.abs(e[:,m]),np.abs(X[:,m]))[0,1] for m in range(len(r))]))
    return np.stack([k,h])
def one(x,seed): return np.stack([f(x),f(multivariate_iaaft(x,np.random.default_rng(seed),final='spectrum')[0])])
rows=[r for r in json.load(open('raw/nt/manifest.json'))['rows'] if r['state'] in ['awake_ec','awake_eo','drug','sleep','recovery_ec']]
W=np.load('raw/nt/windows.npz'); sub=np.round(np.arange(32)*125/31).astype(int)
res=Parallel(n_jobs=2)(delayed(one)(W[r['id']].astype(float)[:,sub],k) for k,r in enumerate(rows))
np.save('residnet_32.npy',np.array(res))
