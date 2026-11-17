import numpy as np, style, matplotlib.pyplot as plt
out=np.load('hubs_out.npy',allow_pickle=True).item(); PF='#30638e'
fig,axs=plt.subplots(1,2,figsize=(6.0,2.9))
for ax,(an,us) in zip(axs,(('George',['20120731','20120803']),('Chibi',['20120730','20120802']))):
    U=sorted([u for u in out if u[:8] in us]); a,b=-out[U[0]]['node']*1e3,-out[U[1]]['node']*1e3
    ax.scatter(a,b,s=8,c=PF,lw=0,alpha=.7); top=np.argsort(a+b)[::-1][:5]
    for c in top: ax.annotate(str(c),(a[c],b[c]),fontsize=7,xytext=(3,2),textcoords='offset points')
    ax.axhline(0,color='#bbbbbb',lw=.6); ax.axvline(0,color='#bbbbbb',lw=.6)
    ax.set_title(f'{an}, propofol  r = {np.corrcoef(a,b)[0,1]:.2f}',fontsize=9); ax.set_xlabel(f'channel share of the fall, {U[0][:8]}'); ax.set_ylabel(f'{U[1][:8]}')
fig.savefig('fig/hub_channels.png')
