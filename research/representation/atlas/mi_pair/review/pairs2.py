"""Per edge, real and the same IAAFT surrogate as edgemi_nt2s_32 (same seeds): Spearman, distance correlation, Hilbert envelope correlation."""
import sys, json, numpy as np, warnings; warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
from scipy.stats import rankdata
from scipy.signal import hilbert
from surr_pairs import multivariate_iaaft
def f(x):
    x=(x-x.mean(0))/x.std(0); T,M=x.shape; iu=np.triu_indices(M,1)
    s=np.corrcoef(np.apply_along_axis(rankdata,0,x).T)[iu]
    D=np.abs(x[:,None,:]-x[None,:,:]); A=D-D.mean(0,keepdims=True)-D.mean(1,keepdims=True)+D.mean((0,1),keepdims=True)
    A=A.reshape(T*T,M); dcov=(A.T@A)/(T*T); dc=np.sqrt(np.clip(dcov[iu],0,None)/np.sqrt(np.outer(np.diag(dcov),np.diag(dcov))[iu]))
    e=np.abs(hilbert(x,axis=0)); ec=np.corrcoef(e.T)[iu]
    return np.stack([s,dc,ec])
def one(x,seed): return np.stack([f(x),f(multivariate_iaaft(x,np.random.default_rng(seed),final='spectrum')[0])])
d,M,out=sys.argv[1],int(sys.argv[2]),sys.argv[3]
states=['awake_ec','awake_eo','drug','sleep','recovery_ec']
W=np.load(f'{d}/windows.npz'); rows=[r for r in json.load(open(f'{d}/manifest.json'))['rows'] if r['state'] in states]
sub=np.round(np.arange(M)*(rows[0]['M']-1)/(M-1)).astype(int)
res=Parallel(n_jobs=2)(delayed(one)(W[r['id']].astype(float)[:,sub],k) for k,r in enumerate(rows))
np.savez(out,F=np.array(res))
