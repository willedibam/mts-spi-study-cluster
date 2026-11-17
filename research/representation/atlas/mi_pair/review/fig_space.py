"""SPI–SPI space for one session: each dot an edge (window-averaged), Gaussian MI vs kNN MI.
Hub edges = top 1% by excess kNN MI over IAAFT selected on even windows; plotted values are odd windows (held out)."""
import numpy as np, style, matplotlib.pyplot as plt, sys
B=np.load('edgemi_nt2s_126.npz'); E=B['E']; unit,state,ids=B['unit'],B['state'],B['id']
u=[x for x in np.unique(unit) if x.startswith(sys.argv[1])][0]; col=sys.argv[2] if len(sys.argv)>2 else '#30638e'; lab=sys.argv[3] if len(sys.argv)>3 else 'propofol'
M=len(B['sub']); iu=np.triu_indices(M,1); mig=lambda r:-0.5*np.log(1-r**2)
widx=np.array([int(i.split(':')[-1]) for i in ids]); ev,od=widx%2==0,widx%2==1
ex=E[:,1]-E[:,3]
fig,axs=plt.subplots(1,2,figsize=(6.0,2.9),sharex=True,sharey=True)
for ax,(s,t,c) in zip(axs,(('awake_ec','wake','#555555'),('drug',lab,col))):
    m=(unit==u)&(state==s)
    top=np.argsort(ex[m&ev].mean(0))[::-1][:int(.01*len(iu[0]))]
    g,k,ks=mig(E[m&od,0]).mean(0),E[m&od,1].mean(0),E[m&od,3].mean(0)
    gs=mig(E[m&od,2]).mean(0)
    ax.scatter(gs,ks,s=2,c='#d9d9d9',lw=0,rasterized=True,label='IAAFT')
    ax.scatter(g,k,s=2,c=c,lw=0,alpha=.35,rasterized=True,label='real')
    ax.scatter(g[top],k[top],s=14,facecolors='none',edgecolors='k',lw=.7,label='top 1% (held out)')
    zr=np.mean([np.corrcoef(mig(E[w,0]),E[w,1])[0,1] for w in np.where(m)[0]]); zs=np.mean([np.corrcoef(mig(E[w,2]),E[w,3])[0,1] for w in np.where(m)[0]])
    ax.set_title(t,color=c,fontsize=9); ax.text(.04,.96,f'z {zr:.3f}\nIAAFT {zs:.3f}',transform=ax.transAxes,va='top',fontsize=7.5)
    ax.set_xlabel('Gaussian MI')
axs[0].set_ylabel('kNN MI'); axs[1].legend(fontsize=6.5,loc='lower right',markerscale=1.5)
fig.savefig(f'fig/space_{sys.argv[1]}.png')
