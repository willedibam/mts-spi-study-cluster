"""Compact scientific figures for the CCEP pilot."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.serif':['Computer Modern Roman','DejaVu Serif'],
        'mathtext.fontset':'cm','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
        'legend.fontsize':7,'xtick.labelsize':8,'ytick.labelsize':8,'legend.frameon':False,
        'figure.constrained_layout.use':True,'figure.dpi':180,'lines.linewidth':1.3,
        'lines.markersize':3,'xtick.direction':'out','ytick.direction':'out',
        'figure.facecolor':'white','axes.facecolor':'white'})
    return plt


def save(fig, folder, name):
    import matplotlib.pyplot as plt
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    for ext in ('png','svg','pdf'):
        fig.savefig(folder/f'{name}.{ext}',dpi=180 if ext=='png' else 600,bbox_inches='tight')
    plt.close(fig)


def scope_figures(root):
    import seaborn as sns
    root=Path(root);plt=style();out=root/'figures'
    participants=json.loads((root/'scope/participants.json').read_text())
    subjects=[r['subject'] for r in participants if r['status']=='audited']
    colors=dict(zip(subjects,plt.cm.viridis(np.linspace(.05,.88,len(subjects)))))
    sampling=pd.read_csv(root/'scope/sampling.csv');duration=pd.read_csv(root/'scope/duration.csv')
    reference=pd.read_csv(root/'scope/reference.csv')
    fig,axes=plt.subplots(1,3,figsize=(9.4,3.2))
    for subject in subjects:
        with np.load(root/'prepared'/f'{subject}.npz') as b:
            f=b['w0_psd_f'];p=b['w0_psd_median']*1e12
            mask=(f>=.5)&(f<=200)
            axes[0].loglog(f[mask],p[mask],color=colors[subject],label=subject[-2:])
        group=sampling.query("subject == @subject and comparison == 'bandwidth_vs_200Hz'").groupby('fs').z_error_p95.max()
        axes[1].plot(np.arange(3),group.reindex([128,256,512]),'o-',color=colors[subject])
        d=duration.query('subject == @subject and fs == 128').z_error_p95.median()
        r=reference.query('subject == @subject and fs == 128').z_error_p95.iloc[0]
        axes[2].plot([0,1],[d,r],'o-',color=colors[subject],alpha=.8)
    axes[0].axvline(45,color='.5',ls=':',lw=.8)
    axes[0].set(xlabel='Frequency (Hz)',ylabel=r'Median PSD ($\mu$V$^2$/Hz)',title='A  Native baseline spectra')
    axes[0].legend(title='Participant',ncol=2,fontsize=6,title_fontsize=7)
    axes[1].set(xticks=[0,1,2],xticklabels=['128 / 45','256 / 100','512 / 200'],xlabel='Rate / bandwidth (Hz)',
                ylabel=r'95th percentile $|\Delta z|$',title='B  Compared with native 0.5–200 Hz')
    axes[2].set(xticks=[0,1],xticklabels=['16 vs 32 s','CAR vs acquisition'],xlim=(-.2,1.2),
                ylabel=r'95th percentile $|\Delta z|$',title='C  Larger sensitivity sources')
    for ax in axes:ax.set_box_aspect(1)
    save(fig,out,'sampling-overview')
    fig,axes=plt.subplots(3,2,figsize=(8.4,7.2));axes=axes.ravel()
    panels=[]
    for ax,subject in zip(axes,subjects):
        with np.load(root/'prepared'/f'{subject}.npz') as bank:
            x=bank['w0_fs128_bw45'][:,:512]
        q25,q75=np.quantile(x,[.25,.75],axis=1)
        x=(x-np.median(x,axis=1,keepdims=True))/np.maximum(q75-q25,1e-12)[:,None]
        im=ax.imshow(x,origin='lower',aspect='auto',extent=(0,4,.5,len(x)+.5),
                     cmap=sns.color_palette('icefire',as_cmap=True),vmin=-3,vmax=3,interpolation='nearest')
        ax.set(title=f'Participant {subject[-2:]} · M={len(x)}',xlabel='Window time (s)',ylabel='Retained ECoG channel')
        panels.append(ax)
    for ax in axes[len(subjects):]:ax.set_visible(False)
    fig.colorbar(im,ax=panels,shrink=.5,pad=.015,label='Channel median/IQR-scaled voltage')
    fig.suptitle('Baseline previews · common average reference · 0.5–45 Hz, 128 Hz',fontsize=11)
    save(fig,out,'baseline-heatmaps')
    return out
