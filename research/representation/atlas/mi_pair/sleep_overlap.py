"""Post hoc (26 November), descriptive: Δz (panel (c) units) of sleep against wake, and of drug against sleep where their delta
fractions overlap. Session means of sleep with a t 90% CI (sessions, not animals: sleep exists in Chibi and George only); per animal,
sleep and drug windows above delta thresholds, and the drug windows inside the sleep delta range by agent.
  python sleep_overlap.py <results root with zns/>
"""
import sys; from pathlib import Path as _P; _h=_P(__file__).resolve().parent; sys.path[:0]=[str(_h), str(_h.parent), str(_h.parents[3] / 'src')]
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from plot_character_draft import load
df,_=load(Path(sys.argv[1]))
S=df.groupby(['unit','animal','group']).agg(sd=('sd','mean'),delta=('delta','mean'),n=('sd','size')).reset_index()
sl=S[S.group=='SLP']; print('sleep sessions (panel units, vs own EC):'); print(sl.round(3).to_string(index=False))
m,se=sl.sd.mean(),sl.sd.std(ddof=1)/np.sqrt(len(sl)); t=stats.t.ppf(.95,len(sl)-1)
print(f'session mean {m:+.2f}, 90% CI [{m-t*se:+.2f}, {m+t*se:+.2f}] (n={len(sl)} sessions, 2 animals)')
dr=S[S.group.isin(['KT','KTMD','MD','PF'])]; print('drug session mean', round(dr.sd.mean(),2), 'n', len(dr))
for lo in (.5,.6,.7):
    for a in ('Chibi','George'):
        w=df[(df.animal==a)&(df.group=='SLP')&(df.delta>lo)]; d=df[(df.animal==a)&(df.state=='drug')&(df.delta>lo)&(df.delta<(w.delta.max() if len(w) else 0))]
        print(f'delta>{lo} {a}: sleep n={len(w)} mean {w.sd.mean():+.2f} | drug in same delta range n={len(d)} mean {d.sd.mean():+.2f}')
for a in ('Chibi','George'):
    e=df[(df.animal==a)&(df.state=='awake_ec')].sd; s=df[(df.animal==a)&(df.group=='SLP')].sd
    print(a,'AUC sleep<EC', round((e.values[:,None]>s.values[None,:]).mean(),2), 'delta sleep/EC', round(df[(df.animal==a)&(df.group=='SLP')].delta.mean(),2), round(df[(df.animal==a)&(df.state=='awake_ec')].delta.mean(),2))
print()
for a in ('Chibi','George'):
    s=df[(df.animal==a)&(df.group=='SLP')]; print(a,'sleep delta range',s.delta.quantile([0,.5,1]).round(2).tolist())
    d=df[(df.animal==a)&(df.state=='drug')]
    print(d.groupby('group').agg(n=('sd','size'),dmin=('delta','min'),dmed=('delta','median'),sd=('sd','mean')).round(2).to_string())
    m=d[d.delta<=s.delta.max()]; print(' drug windows in sleep delta range:', m.groupby('group').agg(n=('sd','size'),sd=('sd','mean'),delta=('delta','mean')).round(2).to_dict('index'))
    # sleep windows in the same delta bins as those drug windows
    if len(m): ss=s[s.delta>=m.delta.min()]; print(' sleep windows with delta >= their min:', len(ss), round(ss.sd.mean(),2))
