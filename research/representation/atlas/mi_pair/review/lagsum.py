import json,sys,numpy as np
R=json.load(open(sys.argv[1]))
idx={(r['unit'],r['state']):r for r in R}
units={r['unit']:r['agent'] for r in R}
def d(agent,s,k):
    v=[idx[(u,s)][k]-idx[(u,'awake_ec')][k] for u,a in units.items() if a==agent and (u,s) in idx and (u,'awake_ec') in idx]
    return np.mean(v),sum(x>0 for x in v),len(v)
eo={k:np.std([idx[(u,'awake_eo')][k]-idx[(u,'awake_ec')][k] for u in units if (u,'awake_eo') in idx],ddof=1) for k in R[0] if k.startswith(('real','null'))}
for k in ['real_z','null0_z','nullS_z','real_zphase','null0_zphase','nullS_zphase','real_share','null0_share','real_cv','null0_cv']:
    print(f'{k:14s} EO-SD {eo[k]:.3f} |', '  '.join(f"{a}:{m:+.3f}({p}/{n})" for a,s in [('KT','drug'),('KTMD','drug'),('MD','drug'),('PF','drug'),('SLP','sleep')] for m,p,n in [d(a,s,k)]))
