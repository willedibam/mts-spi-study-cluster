import json,sys,numpy as np
R=json.load(open(sys.argv[1])); idx={(r['unit'],r['state']):r for r in R}; units={r['unit']:r['agent'] for r in R}
def D(u,s,k): return idx[(u,s)][k]-idx[(u,'awake_ec')][k]
for k in sys.argv[2].split(','):
    f=(lambda u,s:D(u,s,k)) if '-' not in k else (lambda u,s,a=k.split('-')[0],b=k.split('-')[1]:D(u,s,a)-D(u,s,b))
    sd=np.std([f(u,'awake_eo') for u in units if (u,'awake_eo') in idx],ddof=1); out=[]
    for a,s in [('KT','drug'),('KTMD','drug'),('MD','drug'),('PF','drug'),('SLP','sleep')]:
        v=[f(u,s) for u,aa in units.items() if aa==a and (u,s) in idx]; out.append(f"{a}:{np.mean(v):+.3f}({sum(x>0 for x in v)}/{len(v)})")
    print(f'{k:16s} EO-SD {sd:.3f} |','  '.join(out))
