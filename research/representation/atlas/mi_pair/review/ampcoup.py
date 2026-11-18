"""Does coupling strength track amplitude? Per edge, across 1-s blocks: corr(block Fisher-z r, block log RMS of the pair).
Also |Spearman|−|Pearson| at the window level. Real minus IAAFT (same seeds as decide10b)."""
import sys, json, numpy as np, warnings; warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
from scipy.stats import rankdata
from surr_pairs import multivariate_iaaft
B=100
def m(x):
    x=(x-x.mean(0))/x.std(0); T,M=x.shape; iu=np.triu_indices(M,1); nb=T//B; blk=x[:nb*B].reshape(nb,B,M)
    rb=np.arctanh(np.array([np.corrcoef(b.T)[iu] for b in blk]).clip(-.999,.999)); a=np.log(blk.std(1)); amp=(a[:,iu[0]]+a[:,iu[1]])/2
    rc=lambda u,v:((u-u.mean(0))*(v-v.mean(0))).mean(0)/(u.std(0)*v.std(0))
    ca=rc(rb,amp); r=np.corrcoef(x.T)[iu]; s=np.corrcoef(np.apply_along_axis(rankdata,0,x).T)[iu]
    st=np.argsort(np.abs(r))[::-1][:len(r)//4]
    return np.array([np.nanmean(ca),np.nanmean(ca[st]),(np.abs(s)-np.abs(r)).mean(),(np.abs(s)-np.abs(r))[st].mean()])
def one(x,seed): return np.stack([m(x),m(multivariate_iaaft(x,np.random.default_rng(seed),final='spectrum')[0])])
d,M,out=sys.argv[1],int(sys.argv[2]),sys.argv[3]
rows=[r for r in json.load(open(f'{d}/manifest.json'))['rows'] if r['state'] in ('awake_ec','awake_eo','drug','sleep','recovery_ec')]
W=np.load(f'{d}/windows.npz'); sub=np.round(np.arange(M)*(rows[0]['M']-1)/(M-1)).astype(int)
res=Parallel(n_jobs=8)(delayed(one)(W[r['id']].astype(float)[:,sub],k) for k,r in enumerate(rows))
np.savez(out,V=np.array(res),names=['amp_coupling','amp_coupling_s','rank_minus_pearson','rank_minus_pearson_s'],delta_fraction=np.array([r.get('delta_fraction',np.nan) for r in rows],float),**{k:np.array([r.get(k,'') for r in rows]) for k in ('id','unit','state','agent','animal')})
