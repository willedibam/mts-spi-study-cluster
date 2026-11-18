import numpy as np, json, sys
from surr_pairs import multivariate_iaaft
rows=json.load(open('raw/nt/manifest.json'))['rows']; W=np.load('raw/nt/windows.npz')
def stats(x,y):
    x=(x-x.mean())/x.std(); y=(y-y.mean())/y.std(); b=np.dot(x,y)/np.dot(x,x); e=(y-b*x)/np.sqrt(1-b**2+1e-9); e=(e-e.mean())/e.std()
    e2=e**2-1; ac=np.dot(e2[:-5],e2[5:])/np.dot(e2,e2)            # burst clustering of residual power at 50 ms
    big=np.abs(e)>np.quantile(np.abs(e),.9)
    asym=np.mean(np.abs(np.abs(x[big])-np.abs(y[big])))              # one channel large, the other not
    return np.array([(e**4).mean(), ac, asym])
def run(u,i,j,s,ns=3):
    R,S=[],[]
    for k,r in enumerate([r for r in rows if r['unit'].startswith(u) and r['state']==s]):
        X=W[r['id']].astype(float); R.append(stats(X[:,i],X[:,j]))
        for q in range(ns):
            Z=multivariate_iaaft(X[:,[i,j]],np.random.default_rng(100*k+q),final='spectrum')[0]; S.append(stats(Z[:,0],Z[:,1]))
    return np.mean(R,0),np.mean(S,0)
for u,i,j in (('20120803',73,85),('20120803',85,97),('20120802',73,85),('20120730',48,52)):
    for s in ('awake_ec','drug'):
        a,b=run(u,i,j,s); print(f'{u} ch{i}-{j} {s:8s} resid kurtosis {a[0]:.1f}/{b[0]:.1f} | resid-power clustering(50ms) {a[1]:.2f}/{b[1]:.2f} | |x|−|y| asym at big resid {a[2]:.2f}/{b[2]:.2f}')
