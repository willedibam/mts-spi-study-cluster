"""Post hoc (26 November), descriptive: are the drug changes in G (Gaussian MI) and in k04 z against the matched null broad over
electrodes or carried by a few? Per window, additive node model y_ij = a_i + a_j over the 496 pairs: R^2, share of the top 3 node
terms, spread of G with the pairs touching its 3 strongest nodes dropped. Per drug session: Pearson of node profiles, eyes-closed
mean against drug mean, and the top-3 share of the per-node change.  python dz_nodal.py <zns dir> <ensemble dir>
"""
import glob, sys, numpy as np, pandas as pd
M=32; iu,ju=np.triu_indices(M,1); X=np.zeros((496,M)); X[np.arange(496),iu]=1; X[np.arange(496),ju]=1
P=np.linalg.pinv(X)
def node(y):
    a=P@y; f=X@a; r2=1-((y-f)**2).sum()/((y-y.mean())**2).sum(); s=np.sort(a-a.min())[::-1]
    return a, r2, s[:3].sum()/s.sum()
Z={}
for f in sorted(glob.glob(f'{sys.argv[1]}/zns_[AB]_*.npz')):
    d=np.load(f); Z.update(zip(d['id'], d['E'][:,1].astype(float)))
R=[]; A={}
for f in sorted(glob.glob(f'{sys.argv[2]}/ke_[AB]_*.npz')):
    q=np.load(f); d={k:q[k] for k in ('id','agent','state','animal','unit','real','im_mean','im_sd')}
    for i,w in enumerate(d['id']):
        (g,k),(gs,ks)=Z[w]; z04=(d['real'][i,3]-d['im_mean'][i,3])/d['im_sd'][i,3]; x=(k-g)-(ks-gs)
        grp=d['agent'][i] if d['state'][i]=='drug' else d['state'][i]
        ag,r2g,tg=node(g); az,r2z,tz=node(z04); ax,r2x,tx=node(x)
        # top-node removal: G spread after dropping pairs touching the 3 strongest nodes
        top=np.argsort(ag)[-3:]; keep=~(np.isin(iu,top)|np.isin(ju,top))
        R.append(dict(animal=d['animal'][i],unit=d['unit'][i],grp=grp,sdG=g.std(),sdG_drop3=g[keep].std(),
                      r2G=r2g,top3G=tg,r2z04=r2z,top3z04=tz,r2X=r2x,maxz_node=az.max()))
        A[w]=(d['animal'][i],d['unit'][i],grp,ag,az)
df=pd.DataFrame(R); pd.set_option('display.width',250)
print(df.groupby(['animal','grp']).mean(numeric_only=True).round(3).loc[['Chibi','George']].to_string())
# node-profile stability: correlation of node strength (G) between drug window mean and EC mean per unit
print('\nnode profile of G (and z04): Pearson between unit EC mean and unit drug mean; also drug-EC change concentration')
for u in sorted({v[1] for v in A.values()}):
    ec=[v for v in A.values() if v[1]==u and v[2]=='awake_ec']; dr=[v for v in A.values() if v[1]==u and v[2] not in ('awake_ec','awake_eo','recovery_ec','sleep')]
    if not ec or not dr: continue
    gE,gD=np.mean([v[3] for v in ec],0),np.mean([v[3] for v in dr],0); zE,zD=np.mean([v[4] for v in ec],0),np.mean([v[4] for v in dr],0)
    ch=gD-gE; s=np.sort(np.abs(ch))[::-1]
    print(f'{u[:40]:40s} {dr[0][2]:5s} rG {np.corrcoef(gE,gD)[0,1]:+.2f}  dG mean {ch.mean():+.3f} sd {ch.std():.3f} top3|dG| share {s[:3].sum()/s.sum():.2f} min dG {ch.min():+.3f}  rZ {np.corrcoef(zE,zD)[0,1]:+.2f} dZ mean {(zD-zE).mean():+.2f} top3 share {np.sort(np.abs(zD-zE))[::-1][:3].sum()/np.abs(zD-zE).sum():.2f}')
