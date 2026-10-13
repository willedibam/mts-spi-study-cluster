"""Re-render inherited appendix figures with LaTeX; leave shared source figures/data untouched."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from scripts.benchmark_latex_style import latex_style, prepare_latex_figure
from research.inference.locking.cross_frequency_locking import _physics, STYLE, DELTA, BOUNDARY, GAMMAS, STEP, N_COMMUNITY

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results/order-parameter-inference'
OUT=RESULTS/'locking-composites-261008/latex'


def save(fig, name):
    prepare_latex_figure(fig)
    fig.canvas.draw();fig.canvas.draw();fig.set_layout_engine('none')
    for ext in ['png','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def physics():
    rows=pd.read_csv(RESULTS/'cross-frequency-locking-261006/physics.csv')
    examples=(.04,.09,.14)
    cache=OUT/'physics-display-replay.npz'
    if not cache.exists():
        with ProcessPoolExecutor(max_workers=3) as pool:
            replay=list(pool.map(_physics,[(g,900) for g in examples]))
        arrays={}
        for j,(row,phase,x) in enumerate(replay):
            ref=rows[np.isclose(rows.control,examples[j])&rows.seed.eq(900)].iloc[0]
            for key in ['Q_lock','ratio','coupling']:np.testing.assert_allclose(row[key],ref[key],atol=1e-11,rtol=1e-11)
            arrays[f'phase{j}']=phase;arrays[f'x{j}']=x
        np.savez_compressed(cache,**arrays)
    data=np.load(cache);group=rows.groupby('control');a=GAMMAS*group.coupling.mean().to_numpy()
    fig,axes=plt.subplots(1,4,figsize=(13.6,3.1),constrained_layout=True);colors=['#440154','#31688e','#35b779']
    time=np.arange(600)*STEP
    for j,(gamma,color) in enumerate(zip(examples,colors)):
        phase=data[f'phase{j}'][:600]
        axes[0].plot(time,(phase-phase[0])/(2*np.pi),color=color,label=rf'$\gamma={gamma:.2f}$')
    axes[0].set(xlabel='Time',ylabel=r'Relative phase $(\Psi_B-2\Theta_A)/2\pi$',title='Phase slips stop');axes[0].legend(fontsize=7)
    for j,color,offset,name in [(0,colors[0],1.3,rf'$\gamma={examples[0]:.2f}$, drifting'),(2,colors[2],-1.3,rf'$\gamma={examples[2]:.2f}$, locked')]:
        x=data[f'x{j}'];axes[1].plot(x[0,:700]+offset,x[N_COMMUNITY,:700]+offset,color=color,lw=.5,label=name)
    axes[1].set(xlabel='Slow channel (A)',ylabel='Fast channel (B)',title='Pearson zero in both',xticks=[],yticks=[]);axes[1].legend(fontsize=7,loc='center')
    m=group.ratio.mean();axes[2].plot(m.index,m,'o-',color='#222222');axes[2].axhline(2,color='.6',lw=.8,ls=':')
    axes[2].set(xlabel=r'Cross-coupling $\gamma$',ylabel=r'Observed frequency ratio $\Omega_B/\Omega_A$',title='Ratio pins to 2')
    m=group.Q_lock.mean();axes[3].plot(m.index,m,'o-',color='#222222',label='simulated')
    theory=np.where(a<DELTA,(DELTA-np.sqrt(DELTA**2-np.minimum(a,DELTA)**2))/np.maximum(a,1e-12),1)
    axes[3].plot(GAMMAS,theory,'--',color='#b2182b',lw=1,label='noise-free reduction')
    axes[3].set(xlabel=r'Cross-coupling $\gamma$',ylabel='Locking index $Q$',title='Order parameter');axes[3].legend(fontsize=7)
    critical=float(DELTA/rows[rows.control.between(*BOUNDARY)].coupling.mean())
    for ax in axes[2:]:ax.axvline(critical,color='.6',lw=.8,ls=':')
    save(fig,'physics')


def scatter():
    from research.inference.locking.cross_frequency_locking_snr import leading
    source=RESULTS/'cross-frequency-locking-confirm-261006'
    rows=pd.read_csv(source/'scores.csv');bank=np.load(source/'features.npz');np.testing.assert_array_equal(bank['row_id'],rows.row_id)
    index=np.flatnonzero(rows.system.eq('random-noise').to_numpy());part=rows.iloc[index];fit=part.role.eq('development').to_numpy()
    held=(part.role.eq('evaluation')&part.comparison_eligible).to_numpy()
    fig,axes=plt.subplots(2,2,figsize=(7.4,6),constrained_layout=True,sharex='row',sharey='row')
    for row,(name,key,standard) in enumerate([('Mean of each SPI','mean',True),('SPI-SPI, centered','z',False)]):
        q,model=leading(bank[key][index],fit,standard,2);evr=model.explained_variance_ratio_
        q=q/q[fit].std(0)*np.sign([spearmanr(q[fit,j],part.control[fit]).statistic for j in range(2)])
        for ax,(column,cmap,text) in zip(axes[row],[('Q_lock','viridis','locking index $Q$'),('nuisance','magma',r'sensor-noise SD $\eta$')]):
            points=ax.scatter(q[held,0],q[held,1],c=part[column].to_numpy()[held],cmap=cmap,s=11,alpha=.85,edgecolors='none')
            rho=[abs(spearmanr(q[held,j],part[column].to_numpy()[held]).statistic) for j in range(2)]
            ax.set(xlabel=f'PC1 ({evr[0]:.0%} of variance)',title=f'{name}\n$|\\rho|$ with colour: PC1 {rho[0]:.2f}, PC2 {rho[1]:.2f}')
            fig.colorbar(points,ax=ax,label=text,pad=.02)
        axes[row,0].set_ylabel(f'PC2 ({evr[1]:.0%} of variance)')
    fig.suptitle('First two unsupervised components, random-noise, M=N=48; held recordings',fontsize=10)
    save(fig,'component-scatter-random-noise')


def induction():
    rows=pd.read_csv(RESULTS/'neurotycho-induction-261006/scores.csv')
    animals=sorted(rows.animal.unique());color=dict(zip(animals,['#0072B2','#D55E00','#009E73','#CC79A7']))
    shown=[('mean_abs_r','Mean absolute Pearson'),('mean_PC1','Mean-SPI PC1 (fit-animal SD)'),('z_PC1','SPI-SPI PC1, centered (fit-animal SD)')]
    fig,axes=plt.subplots(3,2,figsize=(10.5,8.2),constrained_layout=True,sharex='col',sharey='row')
    for column,agent in enumerate(['ktmd','propofol']):
        for row,(label,name) in enumerate(shown):
            ax=axes[row,column]
            for archive,part in rows[rows.agent.eq(agent)].groupby('archive'):
                c=color[part.animal.iloc[0]];course=part[part.time.notna()].sort_values('time');awake=part[part.state.eq('awake')][label]
                smooth=course[label].rolling(5,center=True,min_periods=1).median()
                ax.plot(course.time/60,smooth,color=c,lw=.9,alpha=.85)
                ax.errorbar(-4-.25*animals.index(part.animal.iloc[0]),awake.median(),yerr=[[awake.median()-awake.quantile(.1)],[awake.quantile(.9)-awake.median()]],fmt='o',color=c,ms=3,lw=.8)
                if np.isfinite(part.onset.iloc[0]):ax.plot(part.onset.iloc[0]/60,ax.get_ylim()[0],'|',color=c,ms=7,clip_on=False)
            ax.axvline(0,color='.6',lw=.8,ls=':');ax.set(title=f'{agent}: {name}' if row==0 else None,ylabel=name if column==0 else None,
                xlabel='Minutes since first injection (awake eyes-closed reference at left)' if row==2 else None,xlim=(-6,45 if agent=='ktmd' else 62))
    axes[0,0].legend(handles=[plt.Line2D([],[],color=color[a],label=a) for a in animals],fontsize=7,ncol=4,loc='upper right')
    fig.suptitle('NeuroTycho induction, M=16 bipolar ECoG, 8 s windows; coordinates fitted without the plotted animal; ticks mark the anaesthetized label',fontsize=10)
    save(fig,'induction-trajectories')


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update(STYLE);latex_style()
    with threadpool_limits(limits=4):
        scatter();induction();physics()
