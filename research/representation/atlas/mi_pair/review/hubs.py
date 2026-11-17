import numpy as np, json
B=np.load('edgemi_nt2s_126.npz'); E=B['E']; unit,state,agent,animal,ids=B['unit'],B['state'],B['agent'],B['animal'],B['id']
M=len(B['sub']); iu=np.triu_indices(M,1); nE=len(iu[0])
mig=lambda r:-0.5*np.log(1-r**2)
def std(a): return (a-a.mean(1,keepdims=True))/a.std(1,keepdims=True)
G,K,Gs,Ks=mig(E[:,0]),E[:,1],mig(E[:,2]),E[:,3]
R=(std(G)*std(K)-std(Gs)*std(Ks))/nE
widx=np.array([int(i.split(':')[-1]) for i in ids]); ev,od=widx%2==0,widx%2==1
def node(x): n=np.zeros(M); np.add.at(n,iu[0],x/2); np.add.at(n,iu[1],x/2); return n
out={}
print('session: Δz resid (all) | held-out share of fall on top-6 nodes (uniform .05) | on top-5% edges | node-map split-half r')
for u in np.unique(unit):
    w=lambda s,sel=True:(unit==u)&(state==s)&sel
    d_ev=R[w('drug',ev)].mean(0)-R[w('awake_ec',ev)].mean(0); d_od=R[w('drug',od)].mean(0)-R[w('awake_ec',od)].mean(0)
    n_ev,n_od=node(d_ev),node(d_od); top=np.argsort(n_ev)[:6]; te=np.argsort(d_ev)[:int(.05*nE)]
    out[u]=dict(agent=agent[unit==u][0],animal=animal[unit==u][0],node=(node(d_ev)+node(d_od))/2,top=top,edges=te,d=(d_ev+d_od)/2)
    print(f"{out[u]['agent']:3s} {out[u]['animal']:6s} {u[:8]} {d_od.sum():+.3f} | {n_od[top].sum()/n_od.sum():.2f} | {d_od[te].sum()/d_od.sum():.2f} | {np.corrcoef(n_ev,n_od)[0,1]:.2f}  hubs {sorted(top.tolist())}")
print('\nnode-map correlation across sessions (same animal, any drug):')
us=list(out)
for i in range(len(us)):
    for j in range(i+1,len(us)):
        a,b=out[us[i]],out[us[j]]
        if a['animal']==b['animal']: print(f"  {a['agent']}-{us[i][:8]} vs {b['agent']}-{us[j][:8]} ({a['animal']}): r={np.corrcoef(a['node'],b['node'])[0,1]:+.2f}")
np.save('hubs_out.npy',out,allow_pickle=True)
