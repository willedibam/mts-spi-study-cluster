"""Exploratory target-blind PC1 comparison for a literature-defined GS scout."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from scripts.scout_dependence_transition import OUT


def coordinate(x,fit,standard):
    keep=np.isfinite(x[fit]).all(axis=0)&(np.nanstd(x[fit],axis=0)>1e-10)
    if not keep.any():raise ValueError('No nonconstant training-complete features')
    selected=x[:,keep];fill=np.median(selected[fit],axis=0)
    missing=np.mean(~np.isfinite(selected),axis=1)
    selected=np.where(np.isfinite(selected),selected,fill)
    center=selected[fit].mean(0)
    scale=selected[fit].std(0) if standard else np.ones(selected.shape[1])
    x=(selected-center)/scale
    model=PCA(n_components=1,svd_solver='full').fit(x[fit])
    return model.transform(x)[:,0],float(model.explained_variance_ratio_[0]),missing


def analyze(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.plot_large_m_pair_sampling import figure_style
    rows=pd.read_csv(folder/'rows.csv');features=np.load(folder/'probes.npz')
    metrics=[];scores=[]
    figure_style();fig,axes=plt.subplots(4,3,figsize=(11,10),constrained_layout=True)
    for ri,(system,part) in enumerate(rows.groupby('system',sort=False)):
        idx=part.index.to_numpy();fit=part.seed.to_numpy()<4;held=~fit
        target=part.CLE.to_numpy();out=part.copy();pc_fields=[]
        for label,key,standard in [('mean_PC1','mean',True),('z_PC1','z',False),('z_standard_PC1','z',True)]:
            q,evr,missing=coordinate(features[key][idx],fit,standard)
            sign=1 if spearmanr(q[fit],target[fit]).statistic>=0 else -1
            out[label]=sign*q;pc_fields.append(label)
            metrics.append(dict(system=system,method=label,held_abs_rho=abs(spearmanr(q[held],target[held]).statistic),
                evr=evr,held_max_missing=float(missing[held].max())))
        means=features['mean'][idx]
        strength=[abs(spearmanr(means[fit,j],target[fit]).statistic) for j in range(means.shape[1])]
        selected=int(np.nanargmax(strength))
        for field,values in [('mean_r',part.mean_r.to_numpy()),('mean_abs_r',part.mean_abs_r.to_numpy()),
                             ('training_selected_mean',means[:,selected])]:
            metrics.append(dict(system=system,method=field,held_abs_rho=abs(spearmanr(values[held],target[held]).statistic),
                                selected_mean=str(features['names'][selected]) if field=='training_selected_mean' else ''))
        mean=part.groupby('control')[['CLE','CLE_first','CLE_second','aux_error','mean_r','mean_abs_r']].mean()
        ax=axes[ri,0];ax.plot(mean.index,mean.CLE,'o-',color='#0072B2');ax.axhline(0,color='.5',lw=.7,ls=':')
        ax.fill_between(mean.index,mean[['CLE_first','CLE_second']].min(axis=1),mean[['CLE_first','CLE_second']].max(axis=1),alpha=.15)
        ax.set(title=system,ylabel='Conditional Lyapunov exponent')
        right=ax.twinx();right.plot(mean.index,mean.aux_error,'s--',color='#D55E00',alpha=.7);right.set_ylabel('Auxiliary error',color='#D55E00')
        for field in ['mean_r','mean_abs_r']:
            axes[ri,1].plot(mean.index,mean[field],'o-',label=field)
        axes[ri,1].legend(fontsize=7);axes[ri,1].set_ylabel('Mean Pearson')
        for field,color in zip(pc_fields,['#D55E00','#0072B2','#009E73']):
            q=out[field].to_numpy();q=(q-q[fit].mean())/q[fit].std()
            curve=out.assign(display_q=q).loc[held].groupby('control').display_q.mean()
            axes[ri,2].plot(curve.index,curve,'o-',color=color,label=field)
        axes[ri,2].legend(fontsize=7);axes[ri,2].set_ylabel('PC1 (training SD units)')
        for ax in axes[ri]:ax.set_xlabel('Coupling');ax.grid(axis='y',alpha=.12)
        scores.append(out)
    fig.suptitle('Generalized-synchronization scope: physics and 11 cheap probes (not p90)\nEight seeds; PC1 fit on four, shown on four held scout seeds',fontsize=11)
    for ext in ['png','svg']:fig.savefig(folder/f'physics-and-probes.{ext}',dpi=180)
    plt.close(fig)
    pd.DataFrame(metrics).to_csv(folder/'probe-metrics.csv',index=False)
    pd.concat(scores).to_csv(folder/'probe-scores.csv',index=False)
    print(pd.DataFrame(metrics).round(3).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,default=OUT/'physics');a=p.parse_args();analyze(a.folder)
