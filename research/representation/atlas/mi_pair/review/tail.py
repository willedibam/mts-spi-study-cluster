"""Per edge: Pearson r, Spearman, and tail co-exceedance, real vs the same IAAFT surrogate as edgemi (same seeds)."""
import sys, json, numpy as np, warnings; warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
from scipy.stats import rankdata
from surr_pairs import multivariate_iaaft
def feats(x):
    x=(x-x.mean(0))/x.std(0); T,M=x.shape; iu=np.triu_indices(M,1)
    r=np.corrcoef(x.T)[iu]; rk=np.apply_along_axis(rankdata,0,x); s=np.corrcoef(rk.T)[iu]
    q=np.abs(x)>np.quantile(np.abs(x),.9,axis=0)          # top-10% |x| per channel
    co=(q[:,iu[0]]&q[:,iu[1]]).mean(0)/.1                  # P(j extreme | i extreme)
    return np.stack([r,s,co])
def one(x,seed): return np.stack([feats(x),feats(multivariate_iaaft(x,np.random.default_rng(seed),final='spectrum')[0])])
if __name__=='__main__':
    d,M,out=sys.argv[1],int(sys.argv[2]),sys.argv[3]
    states=['awake_ec','awake_eo','drug','sleep','recovery_ec']
    W=np.load(f'{d}/windows.npz'); rows=[r for r in json.load(open(f'{d}/manifest.json'))['rows'] if r['state'] in states]
    sub=np.round(np.arange(M)*(rows[0]['M']-1)/(M-1)).astype(int)
    res=Parallel(n_jobs=8)(delayed(one)(W[r['id']].astype(float)[:,sub],k) for k,r in enumerate(rows))
    np.savez(out,F=np.array(res))
