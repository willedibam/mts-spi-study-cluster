"""Bounded p90 comparison after GS physics scope; all candidates/outcomes retained."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.scout_dependence_transition import ROOT,OUT,CONFIG,case,normalize
from scripts.report_dependence_transition import coordinate
from scripts.spi_baseline_exploration import sha

RUN='dependence-transition-261005'
DATA=ROOT/'data/order-parameter-inference'/RUN
REMOTE=f'/scratch/ql44/we2614/mts-spi-study/order-parameter-inference/{RUN}'


def prepare(workers):
    if (DATA/'manifest.json').exists():raise FileExistsError(DATA)
    config=yaml.safe_load(CONFIG.read_text());scope=config['p90_scope'];tasks=[]
    for name in scope['systems']:
        cfg=config['candidates'][name].copy()
        cfg['burn_time']*=5;cfg['reference_time']*=5
        for k in scope['controls'][name]:
            for seed in scope['seeds']:tasks.append((name,cfg,k,seed,config['seed'],2))
    rows=[];arrays={}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,(r,_,_,raw,_) in enumerate(pool.map(case,tasks)):
            key=f"{r['system']}-k{r['control']:.9f}-s{r['seed']:03d}"
            arrays[key]=normalize(raw)
            rows.append(dict(row_id=key,corpus_index=i,instance=r['seed'],block=r['seed'],
                label=r['system'],role='development' if r['seed'] in scope['training_seeds'] else 'evaluation',**r))
            if i%32==0:print('prepared',i+1,'/',len(tasks),flush=True)
    DATA.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(DATA/'observations.npz',**arrays,__dataset_names__=np.array(list(arrays)),
        __labels_json__=np.array([json.dumps([r['label']]) for r in rows]),
        __shapes__=np.array([[r['M'],r['T']] for r in rows]),__axis_order__=np.array(['process','observation']))
    manifest=dict(rows=rows,archive_sha256=sha(DATA/'observations.npz'),protocol_sha256=sha(CONFIG),
        generator_sha256=sha(ROOT/'scripts/scout_dependence_transition.py'),builder_sha256=sha(__file__))
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    config=dict(name=RUN,source=dict(format='named-npz-v1',archive=REMOTE+'/observations.npz',
        sha256=manifest['archive_sha256'],axis_order=['process','observation']),base_output_dir=REMOTE+'/mpis',
        pyspi_config='configs/pyspi/benchmarked_p90.yaml',normalise=False,random_seed=261052)
    (ROOT/f'configs/external/{RUN}.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    print('Prepared',len(rows),'records',manifest['archive_sha256'])


def predict_means(x,y,groups,fit,nonlinear=False):
    keep=np.isfinite(x[fit]).all(0)&(np.nanstd(x[fit],axis=0)>1e-10)
    x=x[:,keep];indices=np.flatnonzero(fit)
    params=[(a,g) for a in [0.1,1.,10.,100.] for g in ([.1,1.,10.] if nonlinear else [None])]
    candidates=[]
    for alpha,gamma in params:
        errors=[]
        for train,test in GroupKFold(4).split(indices,groups=groups[fit]):
            model=make_pipeline(SimpleImputer(),StandardScaler(),
                KernelRidge(alpha=alpha,kernel='rbf',gamma=gamma/max(1,x.shape[1])) if nonlinear else Ridge(alpha=alpha))
            model.fit(x[indices[train]],y[indices[train]])
            errors.extend((model.predict(x[indices[test]])-y[indices[test]])**2)
        candidates.append(np.mean(errors))
    alpha,gamma=params[int(np.argmin(candidates))]
    model=make_pipeline(SimpleImputer(),StandardScaler(),
        KernelRidge(alpha=alpha,kernel='rbf',gamma=gamma/max(1,x.shape[1])) if nonlinear else Ridge(alpha=alpha))
    model.fit(x[fit],y[fit])
    return model.predict(x),dict(alpha=alpha,gamma=gamma,features=int(keep.sum()))


def analyze(data,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.plot_large_m_pair_sampling import figure_style
    rows=pd.DataFrame(json.loads((data/'manifest.json').read_text())['rows'])
    arrays=np.load(out/'features.npz');np.testing.assert_array_equal(arrays['row_id'],rows.row_id)
    metrics=[];all_scores=[];settings={}
    figure_style();fig,axes=plt.subplots(2,3,figsize=(11,6.5),constrained_layout=True)
    for ri,(system,part) in enumerate(rows.groupby('system',sort=False)):
        ix=part.index.to_numpy();fit=part.role.eq('development').to_numpy();held=~fit
        y=part.CLE.to_numpy();out_rows=part.copy();predictions={};missing={}
        for label,key,standard in [('z_PC1','z',False),('z_standard_PC1','z',True),
                                  ('mean_PC1','mean',True),('distribution_PC1','distribution',True)]:
            q,evr,invalid=coordinate(arrays[key][ix],fit,standard)
            sign=1 if spearmanr(q[fit],y[fit]).statistic>=0 else -1
            predictions[label]=q*sign;missing[label]=invalid
            settings[system+'/'+label]=dict(training_variance_explained=evr,display_sign=sign)
        mean=arrays['mean'][ix];valid=np.isfinite(mean[fit]).all(0)&(np.nanstd(mean[fit],axis=0)>1e-10)
        ranks=np.array([abs(spearmanr(mean[fit,j],y[fit]).statistic) if valid[j] else -np.inf for j in range(mean.shape[1])])
        best=int(np.argmax(ranks));q=mean[:,best];sign=1 if spearmanr(q[fit],y[fit]).statistic>=0 else -1
        predictions['selected_mean']=sign*q
        settings[system+'/selected_mean']=str(arrays['spi_order'][best])
        for nonlinear,label in [(False,'mean_ridge'),(True,'mean_RBF')]:
            predictions[label],settings[system+'/'+label]=predict_means(mean,y,part.seed.to_numpy(),fit,nonlinear)
        predictions.update(mean_r=part.mean_r.to_numpy(),mean_abs_r=part.mean_abs_r.to_numpy())
        common=np.isfinite(y)
        for label,q in predictions.items():
            common &= np.isfinite(q)&(missing.get(label,np.zeros(len(q)))<=.05)
        out_rows['comparison_eligible']=common
        for label,q in predictions.items():
            finite=np.isfinite(q)&np.isfinite(y);eligible=finite&(missing.get(label,np.zeros(len(q)))<=.05)
            own=held&eligible;use=held&common
            metrics.append(dict(system=system,method=label,n=int(use.sum()),total_held=int(held.sum()),
                held_abs_rho=float(abs(spearmanr(q[use],y[use]).statistic)),
                own_eligible_n=int(own.sum()),own_eligible_abs_rho=float(abs(spearmanr(q[own],y[own]).statistic)),
                within_control_rho=float(spearmanr(pd.Series(q[use]).groupby(part.control.to_numpy()[use]).transform(lambda a:a-a.mean()),
                   pd.Series(y[use]).groupby(part.control.to_numpy()[use]).transform(lambda a:a-a.mean())).statistic),
                max_missing=float(missing.get(label,np.zeros(len(q)))[held].max())))
            out_rows[label]=q
        use=out_rows[held&common];curve=use.groupby('control').mean(numeric_only=True)
        ax=axes[ri,0];ax.plot(curve.index,curve.CLE,'o-',color='#222222');ax.axhline(0,color='.5',ls=':',lw=.8)
        ax.set(title=system,ylabel='Future conditional Lyapunov exponent')
        ax2=ax.twinx();ax2.plot(curve.index,curve.aux_error,'s--',color='#D55E00');ax2.set_ylabel('Auxiliary error',color='#D55E00')
        for label,color in [('z_PC1','#0072B2'),('mean_PC1','#D55E00'),('distribution_PC1','#009E73'),('z_standard_PC1','#CC79A7')]:
            q=out_rows[label].to_numpy();q=(q-q[fit].mean())/q[fit].std()
            c=out_rows.assign(q_scaled=q)[held&common].groupby('control').q_scaled.mean()
            axes[ri,1].plot(c.index,c,'o-',color=color,label=label)
        axes[ri,1].legend(fontsize=6);axes[ri,1].set_ylabel('PC1, training SD units')
        for label in ['mean_r','mean_abs_r']:axes[ri,2].plot(curve.index,curve[label],'o-',label=label)
        axes[ri,2].legend(fontsize=7);axes[ri,2].set_ylabel('Mean Pearson')
        for ax in axes[ri]:ax.set_xlabel('Coupling');ax.grid(axis='y',alpha=.12)
        all_scores.append(out_rows)
    fig.suptitle('Local generalized-synchronization sweeps: full p90\nM=N=6, T=1000; eight training / eight held seeds; lines: eligible held-instance means',fontsize=11)
    for ext in ['png','svg']:fig.savefig(out/f'p90-comparison.{ext}',dpi=180)
    plt.close(fig)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    all_scores=pd.concat(all_scores)
    all_scores.to_csv(out/'scores.csv',index=False)
    from scripts.transition_sharpness import summarize
    summarize(all_scores[all_scores.comparison_eligible],['CLE','aux_error','mean_r','mean_abs_r','mean_PC1','z_PC1','z_standard_PC1',
                          'distribution_PC1','selected_mean','mean_ridge','mean_RBF'],out/'sharpness.csv')
    (out/'analysis.json').write_text(json.dumps(dict(settings=settings,features_sha256=sha(out/'features.npz'),
        source_sha256=sha(__file__),scope='Exploratory held-seed comparison; no inference of marginal information absence from PC1.'),indent=2)+'\n')
    print(pd.DataFrame(metrics).round(4).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','extract','analyze'])
    p.add_argument('--data',type=Path,default=DATA);p.add_argument('--output',type=Path,default=OUT/'p90')
    p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if a.stage=='prepare':prepare(a.workers)
    elif a.stage=='extract':
        from scripts.analyze_native_coupling import extract
        extract(a.data,a.output,corpus=RUN)
    else:
        with threadpool_limits(limits=4):analyze(a.data,a.output)
