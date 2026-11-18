import numpy as np, sys
B=np.load(sys.argv[1]); E=B['E']; unit,state,agent,animal=B['unit'],B['state'],B['agent'],B['animal']
M=len(B['sub']); iu=np.triu_indices(M,1); nE=len(iu[0])
mig=lambda r:-0.5*np.log(1-r**2)
def std(a): return (a-a.mean(1,keepdims=True))/a.std(1,keepdims=True)
A,K,As,Ks=mig(E[:,0]),E[:,1],mig(E[:,2]),E[:,3]
C=std(A)*std(K)/nE; Cs=std(As)*std(Ks)/nE          # per-window per-edge contribution to z (sum = z)
R=C-Cs                                              # residual contribution
ex=K-Ks                                             # edge excess kNN MI beyond IAAFT
DS={'KT':'drug','KTMD':'drug','MD':'drug','PF':'drug','SLP':'sleep'}
def per_unit(v,s,ag,sel=None):
    out=[]
    for u in np.unique(unit[agent==ag]):
        a=(unit==u)&(state==s); c=(unit==u)&(state=='awake_ec')
        if sel is not None: a&=sel; c&=sel
        if a.any() and c.any(): out.append((u,v[a].mean(0)-v[c].mean(0)))
    return out
print('z replicate (sum of contributions): real Δz, residual Δz (units<0/n)')
for ag,s in DS.items():
    d=[x.sum() for _,x in per_unit(C,s,ag)]; r=[x.sum() for _,x in per_unit(R,s,ag)]
    print(f'  {ag:5s} real {np.mean(d):+.3f}  resid {np.mean(r):+.3f} ({sum(x<0 for x in r)}/{len(r)})')
eo=[x.sum() for ag in DS for _,x in per_unit(R,'awake_eo',ag)]; print(f'  EO resid SD {np.std(eo,ddof=1):.3f}')
print('\nLevel: mean edge excess kNN MI (real−IAAFT), Δ vs wake; wake level')
for ag,s in DS.items():
    d=[x.mean() for _,x in per_unit(ex,s,ag)]
    w=np.mean([ex[(agent==ag)&(state=='awake_ec')].mean()])
    print(f'  {ag:5s} wake {w:+.4f}  Δ {np.mean(d):+.4f} ({sum(x>0 for x in d)}/{len(d)} up)')
print('\nConcentration of residual Δz over edges (cross-validated: top edges chosen on even windows, Δz on odd windows without them)')
widx=np.zeros(len(unit),int)
for u in np.unique(unit):
    for s in np.unique(state):
        m=np.where((unit==u)&(state==s))[0]; widx[m]=np.arange(len(m))
ev,od=widx%2==0,widx%2==1
for ag,s in DS.items():
    sel=[(u,x) for u,x in per_unit(R,s,ag,ev)]; tst={u:x for u,x in per_unit(R,s,ag,od)}
    row=[]
    for frac in (0,.01,.05,.10,.25):
        vals=[]
        for u,x in sel:
            k=int(frac*nE); drop=np.argsort(x)[:k]            # most negative contributions = drivers of the fall
            keep=np.setdiff1d(np.arange(nE),drop); vals.append(tst[u][keep].sum())
        row.append(f'{int(frac*100)}%:{np.mean(vals):+.3f}')
    print(f'  {ag:5s} held-out Δz after removing top-k driver edges  '+'  '.join(row))
print('\nNode concentration: share of residual Δz on the top 10% of channels (node = half of each incident edge), vs uniform 10%')
for ag,s in DS.items():
    sh=[]; 
    for u,x in per_unit(R,s,ag):
        node=np.zeros(M); np.add.at(node,iu[0],x/2); np.add.at(node,iu[1],x/2)
        neg=-node; sh.append(np.sort(neg)[::-1][:max(1,M//10)].sum()/neg.sum())
    print(f'  {ag:5s} top-10% channels carry {np.mean(sh):.2f} of the fall (per unit: {" ".join(f"{v:.2f}" for v in sh)})')
print('\nReproducibility of the edge map of Δ excess MI between units: same animal+agent vs other drug units')
maps={}
for ag in ('KT','KTMD','MD','PF'):
    for u,x in per_unit(ex,'drug',ag): maps[u]=(ag,animal[unit==u][0],x)
same,diff=[],[]
us=list(maps)
for i in range(len(us)):
    for j in range(i+1,len(us)):
        a,b=maps[us[i]],maps[us[j]]; c=np.corrcoef(a[2],b[2])[0,1]
        (same if (a[0]==b[0] and a[1]==b[1]) else diff).append(c)
print(f'  same animal+agent r={np.mean(same):.2f} (n={len(same)}), other r={np.mean(diff):.2f} (n={len(diff)})')
