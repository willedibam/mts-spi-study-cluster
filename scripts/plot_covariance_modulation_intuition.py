"""Raw pair geometry for teaching; not a two-channel SPI-SPI experiment."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.covariance_modulation import simulate


def main():
    out=Path('results/covariance_modulation_260909');out.mkdir(parents=True,exist_ok=True)
    with plt.rc_context({'font.size':9,'axes.titlesize':10}):
        fig,axes=plt.subplots(1,3,figsize=(10,3.6),layout='constrained')
        for alpha,ax,color in zip([0.,.9],axes[:2],['#377eb8','#d95f02']):
            x,_=simulate(alpha,.1,np.array([.5,.5]),'iid',260909,t=20000)
            ax.hexbin(x[:,0],x[:,1],gridsize=42,extent=(-3.5,3.5,-3.5,3.5),
                      mincnt=1,cmap='Blues',vmin=0,vmax=300)
            ax.set(xlim=(-3.5,3.5),ylim=(-3.5,3.5),xlabel='Channel A',ylabel='Channel B',
                   title=f'alpha = {alpha:g}\nPopulation correlation = 0.10\nFourth cross-cumulant = {2*alpha**2*.5**2:.3f}')
            ax.set_aspect('equal')
            axes[2].hist(x[:,1],bins=np.linspace(-4,4,61),density=True,histtype='step',
                         color=color,lw=1.5,label=f'alpha = {alpha:g}')
        axes[2].set(xlabel='Channel B value',ylabel='Density',title='Single-channel distribution\nBoth have population law N(0,1)')
        v=np.linspace(-4,4,200)
        axes[2].plot(v,np.exp(-v*v/2)/np.sqrt(2*np.pi),'k--',lw=1,label='N(0,1)')
        axes[2].legend(frameon=False)
        fig.suptitle('Same population covariance and channel marginals; different joint dependence',fontsize=11)
        for suffix in ['png','svg']:fig.savefig(out/f'raw-pair-intuition.{suffix}',dpi=180)


if __name__=='__main__':main()
