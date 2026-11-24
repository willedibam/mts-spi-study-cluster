"""Incumbent nonlinearity level against the SPI-SPI coordinate, KSG k=4, normal-scored, 10-s macaque.

Level = mean over edges of (kNN MI - Gaussian MI), real - joint-IAAFT surrogate (Palus 1995; Hlinka et al. 2011);
z = corr over edges of (Gaussian MI, kNN MI), real - surrogate. Session contrast = state mean - awake_ec mean;
each readout in its own eyes-open session SD. Reads zns_[AB]_*.npz (E: window, raw/ns, real/surr, g/k4, edge).

  cd results/representation/atlas-mi-pair-261117/zns && python <this file> > ../level_vs_z.txt
"""
import numpy as np, glob
L=[np.load(f,allow_pickle=True) for f in sorted(glob.glob('zns_[AB]_*.npz'))]; cat=lambda k: np.concatenate([d[k] for d in L])
E=np.concatenate([d['E'] for d in L]).astype(float)[:,1]; V=np.concatenate([d['V'] for d in L])[:,1]  # normal-scored
unit,state,agent,animal=cat('unit'),cat('state'),cat('agent'),cat('animal')
lev=(E[:,:,1]-E[:,:,0]).mean(-1); lres=lev[:,0]-lev[:,1]; zres=V[:,0]-V[:,1]
def sess(res):
    o={}
    for u in np.unique(unit):
        m=lambda s:(unit==u)&(state==s); b=res[m('awake_ec')].mean()
        o[u]=dict(ag=agent[unit==u][0],an=animal[unit==u][0],**{s:res[m(s)].mean()-b for s in ('drug','sleep','awake_eo','recovery_ec') if m(s).any()})
    return o
for name,res in (('z',zres),('level',lres)):
    R=sess(res); sd=np.std([R[u]['awake_eo'] for u in R],ddof=1); print(f'\n{name}: EO session SD {sd:.5f}')
    for ag in ('KT','KTMD','MD','PF','SLP'):
        us=[u for u in R if R[u]['ag']==ag]; s='sleep' if ag=='SLP' else 'drug'
        a=np.array([R[u][s] for u in us]); rc=np.array([R[u].get('recovery_ec',np.nan) for u in us])
        an={x:np.mean([R[u][s] for u in us if R[u]['an']==x])/sd for x in sorted({R[u]['an'] for u in us})}
        print(f'{ag:5s} {a.mean()/sd:+6.1f} SD ({np.sum(a<0)}/{len(a)} down) rec {np.nanmean(rc)/sd:+5.1f} | animals '+' '.join(f'{k} {v:+.1f}' for k,v in an.items()))
