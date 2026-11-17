import numpy as np, json, sys
from locmi import pair
rows=json.load(open('raw/nt/manifest.json'))['rows']; W=np.load('raw/nt/windows.npz')
def collect(u,i,j,s,nsurr=3):
    R=[];S=[]
    for k,r in enumerate([r for r in rows if r['unit'].startswith(u) and r['state']==s]):
        X=W[r['id']].astype(float)
        for q in range(nsurr):
            d=pair(X,i,j,seed=100*k+q)
            if q==0: R.append((d['x'],d['y'],d['kl'],d['gl'],d['r']))
            S.append((d['sx'],d['sy'],d['ks'],d['gs'],d['rs']))
    cat=lambda L:[np.concatenate([l[m] for l in L]) for m in range(4)]+[np.mean([l[4] for l in L])]
    return cat(R),cat(S)
def strata(x,y,kl,gl,r):
    amp=np.sqrt((x**2+y**2)/2); q=np.quantile(amp,[.5,.9])
    m=[amp<q[0],(amp>=q[0])&(amp<q[1]),amp>=q[1]]
    res=np.abs(y-r*x)
    return [kl[k].mean() for k in m]+[np.corrcoef(res,np.abs(x))[0,1], ((x**4).mean()+(y**4).mean())/2]
if __name__=='__main__':
    for u,i,j in (('20120803',85,97),('20120803',73,85),('20120802',73,85),('20120730',48,52),('20120731',16,28)):
        for s in ('awake_ec','drug'):
            Rr,Ss=collect(u,i,j,s); a=strata(*Rr); b=strata(*Ss)
            print(f'{u} ch{i}-{j} {s:8s} kNN {Rr[2].mean():.2f}/{Ss[2].mean():.2f} | local kNN real−IAAFT: low {a[0]-b[0]:+.2f} mid {a[1]-b[1]:+.2f} top10% {a[2]-b[2]:+.2f} | hetero {a[3]:+.2f}/{b[3]:+.2f} | kurt {a[4]:.1f}/{b[4]:.1f}')
