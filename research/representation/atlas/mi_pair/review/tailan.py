import numpy as np,sys
B=np.load(sys.argv[1]); F=np.load(sys.argv[2])['F']   # w x {real,surr} x {r,s,co} x E
unit,state,agent=B['unit'],B['state'],B['agent']
r,rs=F[:,0,0],F[:,1,0]; s,ss=F[:,0,1],F[:,1,1]; co,cos=F[:,0,2],F[:,1,2]
DS={'KT':'drug','KTMD':'drug','MD':'drug','PF':'drug','SLP':'sleep'}
def contrast(v,s_,ag,edges=None):
    out=[]
    for u in np.unique(unit[agent==ag]):
        a=(unit==u)&(state==s_); c=(unit==u)&(state=='awake_ec')
        if not(a.any() and c.any()): continue
        e=edges(u) if edges else slice(None)
        out.append(np.nanmean(v[a][:,e])-np.nanmean(v[c][:,e]))
    return np.array(out)
def strong(u,f=.1):
    g=np.abs(r[(unit==u)&(state=='awake_ec')]).mean(0); return np.argsort(g)[::-1][:int(f*len(g))]
def weak(u,f=.5):
    g=np.abs(r[(unit==u)&(state=='awake_ec')]).mean(0); return np.argsort(g)[:int(f*len(g))]
for nm,v in (('|ρs|−|r|, real minus IAAFT',(np.abs(s)-np.abs(r))-(np.abs(ss)-np.abs(rs))),('tail co-exceedance, real minus IAAFT',co-cos)):
    eo=contrast(v,'awake_eo','KTMD',strong); 
    print(nm, ' EO(KTMD, strong) SD %.4f'%np.std(eo,ddof=1))
    for lab,ed in (('strongest 10%',strong),('all',None),('weakest 50%',weak)):
        print(f'  {lab:14s}','  '.join(f"{ag}:{np.mean(c):+.4f}({(c>0).sum()}/{len(c)})" for ag,s_ in DS.items() for c in [contrast(v,s_,ag,ed)]))
