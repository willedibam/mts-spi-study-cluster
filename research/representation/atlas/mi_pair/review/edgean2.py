import numpy as np, sys
exec(open('edgean.py').read().split("print('z replicate")[0])
widx=np.zeros(len(unit),int)
for u in np.unique(unit):
    for s in np.unique(state):
        m=np.where((unit==u)&(state==s))[0]; widx[m]=np.arange(len(m))
ev,od=widx%2==0,widx%2==1
print('Held-out Δz: drop top-5% driver edges vs top-5% strongest edges (wake Gaussian MI, even windows) vs random 5%')
rng=np.random.default_rng(0)
for ag,s in DS.items():
    sel=dict(per_unit(R,s,ag,ev)); tst=dict(per_unit(R,s,ag,od)); k=int(.05*nE); o=[[],[],[],[]]
    for u in sel:
        strong=np.argsort(A[(unit==u)&(state=='awake_ec')&ev].mean(0))[::-1][:k]
        for i,drop in enumerate([[],np.argsort(sel[u])[:k],strong,rng.choice(nE,k,replace=False)]):
            o[i].append(tst[u][np.setdiff1d(np.arange(nE),drop)].sum())
    print(f'  {ag:5s} none {np.mean(o[0]):+.3f}  drivers {np.mean(o[1]):+.3f}  strongest {np.mean(o[2]):+.3f}  random {np.mean(o[3]):+.3f}')
print('\nCross-validated node share: top-3 channels chosen on even windows, share of the held-out fall they carry (uniform = 3/32 = .09)')
def node(x): n=np.zeros(M); np.add.at(n,iu[0],x/2); np.add.at(n,iu[1],x/2); return n
for ag,s in DS.items():
    sel=dict(per_unit(R,s,ag,ev)); tst=dict(per_unit(R,s,ag,od)); sh=[]
    for u in sel:
        top=np.argsort(node(sel[u]))[:3]; nt=node(tst[u]); sh.append(nt[top].sum()/nt.sum())
    print(f'  {ag:5s} mean {np.mean(sh):.2f}  per unit: {" ".join(f"{v:.2f}" for v in sh)}')
print('\nSplit-half reliability of edge maps within unit (even vs odd windows), drug-minus-wake')
for nm,v in (('residual contribution',R),('excess kNN MI',ex),('Gaussian MI',A)):
    out=[]
    for ag,s in DS.items():
        sel=dict(per_unit(v,s,ag,ev)); tst=dict(per_unit(v,s,ag,od))
        out.append(f'{ag}:{np.mean([np.corrcoef(sel[u],tst[u])[0,1] for u in sel]):.2f}')
    print(f'  {nm:22s} '+'  '.join(out))
