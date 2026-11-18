import numpy as np, warnings; warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
from scipy.signal import lfilter
from decide10 import one, NAMES
T,M,NW=1000,16,24
def ar(rng,n,phi=.9): return lfilter([1],[1,-phi],rng.standard_normal(n))
def gen(kind,strength,seed):
    rng=np.random.default_rng(seed)
    L=np.array([ar(rng,T) for _ in range(4)]).T; L=(L-L.mean(0))/L.std(0)
    W=rng.uniform(0,1,(4,M))*(rng.random((4,M))<.5)
    x=L@W+np.array([ar(rng,T) for _ in range(M)]).T
    half=np.arange(M)<M//2
    if kind=='nonlinear': q=L[:,0]**2-1; x[:,half]+=strength*np.outer(q,rng.uniform(.5,1.5,half.sum()))
    if kind=='gain':      # slow (1-3 s) shared amplitude state on half the channels
        u=lfilter([1],[1,-.98],rng.standard_normal(T)); env=np.exp(strength*(u-u.mean())/u.std()); x[:,half]*=env[:,None]
    if kind=='statecoup': # coupling strength (not amplitude) switches with a slow latent state
        u=lfilter([1],[1,-.98],rng.standard_normal(T)); w=1/(1+np.exp(-3*(u-u.mean())/u.std()))
        x[:,half]+=strength*np.outer(w*L[:,1],rng.uniform(.5,1.5,half.sum()))
    return x
out={}
for kind,s in (('linear',0),('nonlinear',.6),('nonlinear',1.2),('gain',.5),('gain',1.0),('statecoup',1.5)):
    r=np.array(Parallel(n_jobs=8)(delayed(one)(gen(kind,s,7919*k+len(kind)),k) for k in range(NW)))
    out[(kind,s)]=(r[:,0]-r[:,1]).mean(0), (r[:,0]-r[:,1]).std(0)/np.sqrt(NW)
b=out[('linear',0)][0]
print('real−IAAFT minus linear baseline; '+'  '.join(NAMES))
for k,(m,se) in out.items(): print(f'{k[0]:10s}{k[1]:4}', '  '.join(f'{v:+.4f}' for v in m-b), '| se', '  '.join(f'{v:.4f}' for v in se))
