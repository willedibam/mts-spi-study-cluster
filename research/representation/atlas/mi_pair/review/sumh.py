import numpy as np
D=np.load('hprop_decide.npz'); V=D['V']; N=list(D['names']); keys=D['keys']
subj=np.array([k.split('-')[0] for k in keys]); lev=np.array([k.split('-')[1] for k in keys]); ep=np.array([int(k.split('-e')[1].split('-')[0]) for k in keys])
for name in ('zgk','knn_minus_gauss'):
    i=N.index(name)
    for lab,v in (('real',V[:,0,i]),('real−IAAFT',V[:,0,i]-V[:,1,i])):
        # null: baseline split-half (alternate epochs) difference between subjects
        null=[]; out={l:[] for l in ('level2','level3','level4')}
        for s in np.unique(subj):
            b=(subj==s)&(lev=='level1'); idx=np.where(b)[0]
            if len(idx)<4: continue
            null.append(v[idx[::2]].mean()-v[idx[1::2]].mean())
            for l in out:
                m=(subj==s)&(lev==l)
                if m.any(): out[l].append(v[m].mean()-v[b].mean())
        sd=np.std(null,ddof=1)*np.sqrt(2)/2*np.sqrt(2)  # half-vs-half difference SD ~ scale of a level-vs-baseline difference
        print(f'{name:16s}{lab:11s} baseline mean {np.nanmean(v[lev=="level1"]):+.3f} null SD {sd:.4f} |','  '.join(f'{l}:{np.mean(c)/sd:+.1f}SD({sum(x<0 for x in c)}/{len(c)}↓)' for l,c in out.items()))
