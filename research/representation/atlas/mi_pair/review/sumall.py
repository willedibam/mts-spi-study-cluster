import numpy as np, glob
fs=sorted(glob.glob('zns_[AB]_*.npz')); L=[np.load(f) for f in fs]; cat=lambda k: np.concatenate([d[k] for d in L]); V=np.concatenate([d['V'] for d in L])
unit,state,agent,animal=cat('unit'),cat('state'),cat('agent'),cat('animal')
def sess(v):
    res=V[:,v,0]-V[:,v,1]; out={}
    for u in np.unique(unit):
        m=lambda s:(unit==u)&(state==s); b=res[m('awake_ec')].mean()
        out[u]=dict(ag=agent[unit==u][0],an=animal[unit==u][0],**{s:res[m(s)].mean()-b for s in ('drug','sleep','awake_eo','recovery_ec') if m(s).any()})
    return out
R={v:sess(v) for v in (0,1)}
eo=np.array([R[1][u]['awake_eo'] for u in R[1]]); sd=eo.std(ddof=1); eo0=np.array([R[0][u]['awake_eo'] for u in R[0]]); sd0=eo0.std(ddof=1)
print(f'session-level EO SD: raw {sd0:.4f}  normal-scored {sd:.4f}   n_sess {len(eo)}')
for ag in ('KT','KTMD','MD','PF','SLP'):
    us=[u for u in R[1] if R[1][u]['ag']==ag]; s='sleep' if ag=='SLP' else 'drug'
    if not us: continue
    a0=np.array([R[0][u][s] for u in us]); a1=np.array([R[1][u][s] for u in us]); rc=np.array([R[1][u].get('recovery_ec',np.nan) for u in us])
    print(f'{ag:5s} n={len(us):2d} raw {a0.mean():+.4f} ({a0.mean()/sd0:+.1f} SD, {np.sum(a0<0)}/{len(us)} down) | normal-scored {a1.mean():+.4f} ({a1.mean()/sd:+.1f} SD, {np.sum(a1<0)}/{len(us)} down) | ns recovery {np.nanmean(rc):+.4f}')
