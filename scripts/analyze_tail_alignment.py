"""Full-p90 exploratory readouts for the analytically matched tail toy."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from scripts.scout_tail_alignment import ROOT,OUT
from scripts.report_dependence_transition import coordinate
from scripts.dependence_transition_pipeline import predict_means
from scripts.spi_baseline_exploration import sha

DATA=ROOT/'data/order-parameter-inference/tail-alignment-261005'


def analyze(data,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.plot_large_m_pair_sampling import figure_style
    manifest=json.loads((data/'manifest.json').read_text())
    rows=pd.DataFrame(manifest['rows'])
    bank=np.load(out/'features.npz');np.testing.assert_array_equal(bank['row_id'],rows.row_id)
    fit=rows.role.eq('development').to_numpy();held=~fit;y=rows.Q_tail.to_numpy()
    metrics=[];scores=rows.copy();settings={};missing_by_method={}
    # rows.mean_r is a population identity, not the empirical scalar baseline.
    with np.load(data/'observations.npz') as raw:
        correlations=[np.corrcoef(raw[key]) for key in rows.row_id]
    offdiag=~np.eye(correlations[0].shape[0],dtype=bool)
    scores['empirical_mean_r']=[c[offdiag].mean() for c in correlations]
    scores['empirical_mean_abs_r']=[np.abs(c[offdiag]).mean() for c in correlations]
    for label,key,standard in [('z_PC1','z',False),('z_standard_PC1','z',True),
                              ('mean_PC1','mean',True),('distribution_PC1','distribution',True)]:
        q,evr,missing=coordinate(bank[key],fit,standard)
        sign=1 if spearmanr(q[fit],y[fit]).statistic>=0 else -1
        scores[label]=sign*q
        missing_by_method[label]=missing
        eligible=held&(missing<=.05)
        metrics.append(dict(method=label,held_abs_rho=abs(spearmanr(q[eligible],y[eligible]).statistic),
                            n=int(eligible.sum()),max_missing=float(missing[held].max()),evr=evr))
    mean=bank['mean'];valid=np.isfinite(mean[fit]).all(0)&(np.nanstd(mean[fit],axis=0)>1e-10)
    ranks=[abs(spearmanr(mean[fit,j],y[fit]).statistic) if valid[j] else -np.inf for j in range(mean.shape[1])]
    best=int(np.argmax(ranks));scores['selected_mean']=mean[:,best]
    settings['selected_mean']=str(bank['spi_order'][best])
    for nonlinear,label in [(False,'mean_ridge'),(True,'mean_RBF')]:
        scores[label],settings[label]=predict_means(mean,y,rows.seed.to_numpy(),fit,nonlinear)
    for label in ['selected_mean','mean_ridge','mean_RBF','empirical_mean_r','empirical_mean_abs_r']:
        metrics.append(dict(method=label,held_abs_rho=abs(spearmanr(scores.loc[held,label],y[held]).statistic),n=int(held.sum())))
    common=held.copy()
    for missing in missing_by_method.values():common &= missing<=.05
    for metric in metrics:common &= np.isfinite(scores[metric['method']])
    if common.sum()<4:raise ValueError('Too few commonly eligible evaluation records')
    for metric in metrics:
        metric['own_eligible_abs_rho']=metric['held_abs_rho'];metric['own_eligible_n']=metric['n']
        metric['held_abs_rho']=abs(spearmanr(scores.loc[common,metric['method']],y[common]).statistic)
        metric['n']=int(common.sum())
    scores['comparison_eligible']=common|fit
    figure_style();fig,axes=plt.subplots(1,4,figsize=(13,3.5),constrained_layout=True)
    physical=rows.groupby('control').Q_tail.first()
    axes[0].plot(physical.index,physical,'ko-');axes[0].set(ylabel='Population mean tail dependence',title='Known construction truth')
    for label,color in [('z_PC1','#0072B2'),('z_standard_PC1','#CC79A7'),('mean_PC1','#D55E00'),('distribution_PC1','#009E73')]:
        q=scores[label].to_numpy();q=(q-q[fit].mean())/q[fit].std()
        curve=scores.assign(display_q=q)[common].groupby('control').display_q
        axes[1].plot(curve.mean().index,curve.mean(),'o-',color=color,label=label)
        axes[1].fill_between(curve.mean().index,curve.quantile(.1),curve.quantile(.9),color=color,alpha=.10)
    axes[1].set(ylabel='PC1, training SD units',title=r'Held seeds; 10--90\% instance spread');axes[1].legend(fontsize=6)
    for label in ['mean_ridge','mean_RBF']:
        curve=scores[common].groupby('control')[label].mean()
        axes[2].plot(curve.index,curve,'o-',label=label)
    axes[2].plot(physical.index,physical,'k:',label='truth');axes[2].legend(fontsize=7)
    axes[2].set(ylabel='Predicted tail dependence',title='Supervised full-mean controls')
    for label,color in [('empirical_mean_r','#0072B2'),('empirical_mean_abs_r','#D55E00')]:
        curve=scores[common].groupby('control')[label]
        axes[3].plot(curve.mean().index,curve.mean(),'o-',color=color,label=label.replace('empirical_',''))
        axes[3].fill_between(curve.mean().index,curve.quantile(.1),curve.quantile(.9),color=color,alpha=.10)
    population_r=rows.groupby('control').mean_r.first()
    axes[3].plot(population_r.index,population_r,color='.5',ls=':',lw=.8,label='Population signed mean')
    axes[3].set(ylabel='Observed mean Pearson',title='Primary scalar baseline');axes[3].legend(fontsize=6)
    for ax in axes:ax.set_xlabel('Fraction of assignments swapped');ax.grid(axis='y',alpha=.12)
    fig.suptitle(manifest.get('figure_title','Student-t block alignment: M=N=16, T=1000; 32 training / 32 held seeds')+'\nCompositional stochastic control, not a bifurcation',fontsize=10)
    for ext in ['png','svg']:fig.savefig(out/f'p90-tail-comparison.{ext}',dpi=180)
    plt.close(fig);scores.to_csv(out/'scores.csv',index=False);pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    (out/'analysis.json').write_text(json.dumps(dict(settings=settings,features_sha256=sha(out/'features.npz'),
        source_sha256=sha(__file__),qualification=manifest.get('analysis_scope','Exploratory p90 extension on the same seeds as focused probes. Selected Pearson/Kendall/MI population means are matched, not all289 means.')),indent=2)+'\n')
    print(pd.DataFrame(metrics).round(4).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['extract','analyze']);p.add_argument('--data',type=Path,default=DATA)
    p.add_argument('--output',type=Path,default=OUT/'p90');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.stage=='extract':
        from scripts.analyze_native_coupling import extract
        manifest=json.loads((a.data/'manifest.json').read_text())
        extract(a.data,a.output,corpus=manifest.get('corpus','tail-alignment-261005'))
    else:
        with threadpool_limits(limits=4):analyze(a.data,a.output)
