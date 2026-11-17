import numpy as np, sys
L=[np.load(f) for f in sys.argv[1:]]; cat=lambda k: np.concatenate([d[k] for d in L]); V=np.concatenate([d['V'] for d in L])
unit,state,animal=cat('unit'),cat('state'),cat('animal')
for v,name in ((0,'raw'),(1,'normal-scored')):
    res=V[:,v,0]-V[:,v,1]; real=V[:,v,0]
    print(name)
    for u in np.unique(unit):
        m=lambda s:(unit==u)&(state==s)
        f=lambda a,s: a[m(s)].mean()-a[m('awake_ec')].mean()
        print(f'  {animal[unit==u][0]:6s} {u[:8]}  real z wake {real[m("awake_ec")].mean():.3f} | Δ real drug {f(real,"drug"):+.4f} | Δ resid drug {f(res,"drug"):+.4f}  EO {f(res,"awake_eo"):+.4f}  recov {f(res,"recovery_ec"):+.4f}')
