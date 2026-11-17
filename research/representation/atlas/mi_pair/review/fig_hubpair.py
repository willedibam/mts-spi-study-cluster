import numpy as np, style, matplotlib.pyplot as plt, sys
from probe import collect
u,i,j=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]); PF='#30638e'
W_,_=collect(u,i,j,'awake_ec',1); D,S=collect(u,i,j,'drug',3)
fig,axs=plt.subplots(1,4,figsize=(8.6,2.3),gridspec_kw=dict(width_ratios=[1,1,1,1.3]))
for ax,(d,lab,c) in zip(axs,((W_,'wake','#555555'),(D,'propofol',PF),(S,'propofol, IAAFT','#9a9a9a'))):
    x,y=d[0],d[1]; n=min(len(x),2000); k=np.random.default_rng(0).choice(len(x),n,replace=False)
    ax.scatter(x[k],y[k],s=1.5,c=c,lw=0,alpha=.6); ax.set_title(lab,fontsize=9,color=c)
    ax.set_xlim(-4,4); ax.set_ylim(-4,4); ax.set_aspect('equal'); ax.set_xticks([-3,0,3]); ax.set_yticks([-3,0,3])
    ax.text(.04,.96,f'r = {d[4]:.2f}',transform=ax.transAxes,va='top',fontsize=7.5)
axs[0].set_ylabel(f'ch{j}'); [a.set_xlabel(f'ch{i}') for a in axs[:3]]
ax=axs[3]
for d,lab,c in ((D,'propofol',PF),(S,'IAAFT','#9a9a9a'),(W_,'wake','#555555')):
    amp=np.sqrt((d[0]**2+d[1]**2)/2); bins=np.quantile(amp,np.linspace(0,1,11)); idx=np.clip(np.digitize(amp,bins)-1,0,9)
    ax.plot(np.arange(10)*10+5,[d[2][idx==b].mean() for b in range(10)],'-o',ms=2.5,color=c,label=lab,lw=1.3)
ax.set_xlabel('joint amplitude percentile'); ax.set_ylabel('local kNN MI (nats)'); ax.legend(fontsize=7)
fig.savefig(f'fig/hubpair_{u}_{i}_{j}.png')
