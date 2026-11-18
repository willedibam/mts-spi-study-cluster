"""Held-out excess kNN MI (real − IAAFT) on the edges with the largest excess, selected on even windows, measured on odd.
Selection bias is matched by doing the same in wake and eyes-open."""
import numpy as np, sys
B=np.load(sys.argv[1]); E=B['E']; unit,state,agent,ids=B['unit'],B['state'],B['agent'],B['id']
ex=E[:,1]-E[:,3]; absr=np.abs(E[:,0]); nE=ex.shape[1]
widx=np.array([int(i.split(':')[-1]) for i in ids]); ev,od=widx%2==0,widx%2==1
ags=[a for a in ('KT','KTMD','MD','PF','SLP') if a in agent]
def held(u,s,frac):
    a=(unit==u)&(state==s)
    if not a.any(): return np.nan,np.nan,np.nan
    k=max(1,int(frac*nE)); top=np.argsort(ex[a&ev].mean(0))[::-1][:k]; bot=np.argsort(ex[a&ev].mean(0))[:k]
    return ex[a&od][:,top].mean(), ex[a&od][:,bot].mean(), ex[a&od].mean()
for frac in (.01,.05):
    print(f'top/bottom {int(frac*100)}% edges by excess (even windows) -> held-out excess on odd windows; mean over sessions')
    for ag in ags:
        s='sleep' if ag=='SLP' else 'drug'; us=np.unique(unit[agent==ag])
        D=np.array([held(u,s,frac) for u in us]); Wk=np.array([held(u,'awake_ec',frac) for u in us]); EO=np.array([held(u,'awake_eo',frac) for u in us])
        dt=D[:,0]-Wk[:,0]
        print(f'  {ag:4s} {s:5s} top {np.nanmean(D[:,0]):+.3f} | wake top {np.nanmean(Wk[:,0]):+.3f} | EO top {np.nanmean(EO[:,0]):+.3f} | Δtop {np.nanmean(dt):+.3f} ({(dt>0).sum()}/{len(dt)}) | bottom {np.nanmean(D[:,1]):+.3f} vs wake {np.nanmean(Wk[:,1]):+.3f} | all {np.nanmean(D[:,2]):+.4f} vs {np.nanmean(Wk[:,2]):+.4f}')
