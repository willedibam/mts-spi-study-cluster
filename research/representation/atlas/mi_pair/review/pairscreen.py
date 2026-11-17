import numpy as np
B=np.load('edgemi_nt2s_32.npz'); E=B['E']; P=np.load('pairs2_nt2s_32.npz')['F']
unit,state,agent=B['unit'],B['state'],B['agent']
mig=lambda r:-0.5*np.log(1-r**2)
feat={}
for w,o in (('real',0),('surr',1)):
    r=E[:,2*o]; k=E[:,2*o+1]
    feat[w]=dict(r2=r**2, absr=np.abs(r), gmi=mig(r), knn=k, s2=P[:,o,0]**2, dc2=P[:,o,1]**2, env=P[:,o,2])
def z(a,b): 
    a=a-a.mean(1,keepdims=True); b=b-b.mean(1,keepdims=True); return (a*b).sum(1)/np.sqrt((a*a).sum(1)*(b*b).sum(1))
PAIRS=[('gmi','knn','Gaussian MI | kNN MI'),('r2','s2','r² | ρ² (monotone)'),('s2','knn','ρ² | kNN MI (non-monotone)'),('r2','dc2','r² | dCor² (any)'),('absr','env','|r| | envelope r (amplitude)')]
DS=[('KT','drug'),('KTMD','drug'),('MD','drug'),('PF','drug'),('SLP','sleep')]
def con(v,ag,s):
    out=[]
    for u in np.unique(unit[agent==ag]):
        a=(unit==u)&(state==s); c=(unit==u)&(state=='awake_ec')
        if a.any() and c.any(): out.append(v[a].mean()-v[c].mean())
    return np.array(out)
res={}
for a,b,lab in PAIRS:
    zr=z(feat['real'][a],feat['real'][b]); zs=z(feat['surr'][a],feat['surr'][b])
    for nm,v in (('real',zr),('real−IAAFT',zr-zs)):
        eo=np.concatenate([con(v,ag,'awake_eo') for ag in ('KT','KTMD','MD','PF','SLP')]); sd=eo.std(ddof=1)
        print(f'{lab:28s}{nm:11s} wake z {zr[state=="awake_ec"].mean():.2f} EO-SD {sd:.4f} |','  '.join(f'{ag}:{c.mean()/sd:+5.1f}({(c<0).sum()}/{len(c)}↓)' for ag,s in DS for c in [con(v,ag,s)]))
        res[(lab,nm)]=v
np.save('pairscreen_resid.npy',np.array([res[(l,'real−IAAFT')] for *_,l in PAIRS]))
