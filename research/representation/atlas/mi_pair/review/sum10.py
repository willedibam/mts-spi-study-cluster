import numpy as np, sys
B=np.load(sys.argv[1]); V=B['V']; N=list(B['names']); unit,state,agent,df=B['unit'],B['state'],B['agent'],B['delta_fraction']
deep=np.zeros(len(unit),bool)
for u in np.unique(unit):
    m=(unit==u)&(state=='sleep')
    if m.any(): deep[m]=df[m]>=np.quantile(df[m],2/3)
def con(v,sel,ag):
    out=[]
    for u in np.unique(unit[agent==ag]):
        a=(unit==u)&sel; c=(unit==u)&(state=='awake_ec')
        if a.any() and c.any(): out.append(np.nanmean(v[a])-np.nanmean(v[c]))
    return np.array(out)
G=[('KT',state=='drug'),('KTMD',state=='drug'),('MD',state=='drug'),('PF',state=='drug'),('SLP',state=='sleep'),('SLP',deep&(state=='sleep'))]
lab=['KT','KTMD','MD','PF','sleep','deep']
for k in N:
    i=N.index(k)
    for nm,v in (('real',V[:,0,i]),('real−IAAFT',V[:,0,i]-V[:,1,i])):
        eo=np.concatenate([con(v,state=='awake_eo',ag) for ag in ('KT','KTMD','MD','PF','SLP')]); sd=np.std(eo,ddof=1)
        print(f'{k:16s}{nm:11s} EO-SD {sd:.4f} |','  '.join(f'{l}:{c.mean()/sd:+.1f}SD({(c>0).sum()}/{len(c)})' for l,(ag,sel) in zip(lab,G) for c in [con(v,sel,ag)]))
