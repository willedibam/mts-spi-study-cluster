import numpy as np, mech
R=np.load('hubtest_out.npy',allow_pickle=True).item(); N=mech.NAMES
for k,v in R.items():
    nh=len(v['hub']); print(f"\n{k}: hub |r| drug {v['rd'][v['hub']].mean():.2f} wake {v['rw'][v['hub']].mean():.2f}; ctrl drug {v['rd'][v['ctrl']].mean():.2f} wake {v['rw'][v['ctrl']].mean():.2f}")
    print('  hub electrodes', v['hub_elec'][:10])
    print('  ' + ' '*22 + ''.join(f'{n:>8s}' for n in N))
    for st in ('drug','wake'):
        X=v[st]  # win x edge x feat
        print(f'  {st:5s} hub  (z vs surr)  '+''.join(f'{x:+8.2f}' for x in X[:,:nh].mean((0,1))))
        print(f'  {st:5s} ctrl (z vs surr)  '+''.join(f'{x:+8.2f}' for x in X[:,nh:].mean((0,1))))
    D=lambda g: v['drug'][:,g].mean(0)-v['wake'][:,g].mean(0)   # per-edge drug-wake (edge x feat)
    dh,dc=D(slice(0,nh)),D(slice(nh,None))
    # edge-level: hub - ctrl difference; bootstrap over hub edges with their 3 controls
    dcg=dc.reshape(nh,3,-1).mean(1); diff=dh-dcg
    rng=np.random.default_rng(0); bs=np.array([diff[rng.integers(0,nh,nh)].mean(0) for _ in range(4000)])
    lo,hi=np.percentile(bs,[2.5,97.5],0)
    print('  drug−wake, hub − ctrl  '+''.join(f'{x:+8.2f}' for x in diff.mean(0)))
    print('     95% CI (edges)      '+''.join(f'[{a:+.1f},{b:+.1f}]'[:8].rjust(8) for a,b in zip(lo,hi)))
    print('  edges hub>ctrl         '+''.join(f'{(diff[:,i]>0).sum():>5d}/{nh}' for i in range(len(N))))
