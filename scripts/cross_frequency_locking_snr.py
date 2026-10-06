"""2:1 locking sweep observed through sensor noise: homogeneous versus recording-specific SNR.

Same dynamics, parameters and truth as scripts/cross_frequency_locking.py. Each z-scored sin(phase)
channel receives independent white observation noise of SD eta (in signal-SD units) before the final
z-score. Arm 'fixed-noise' uses eta=.5 in every recording. Arm 'random-noise' draws eta~U[.3,.9]
once per recording, independently of the control: recordings differ in quality, as sessions and
subjects do. The nuisance moves every dependence estimate together; the locking index does not
depend on it. Declared before p90: centered z-PC1 is primary, and held |Spearman| with Q and with
eta are the headline scores. Supervised mean readouts remain information ceilings.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scripts.cross_frequency_locking import ROOT,GAMMAS,N_COMMUNITY,T,DELTA,BOUNDARY,STYLE,Z_CURVES,MEAN_CURVES,simulate,partitions,step_scores,tracking_panel,share_limits

RUN='cross-frequency-locking-snr-261006'
DATA=ROOT/'data/order-parameter-inference'/RUN
OUT=ROOT/'results/order-parameter-inference'/RUN
ARMS=('fixed-noise','random-noise')
FIRST_SEED={'fixed-noise':100,'random-noise':200}
SEEDS,FIT_SEEDS=16,8
NOISE_SEED,PYSPI_SEED=261091,261092
READOUTS=['mean_abs_r','mean_r','mean_PC1','distribution_PC1','z_PC1','z_standard_PC1','selected_mean','mean_ridge','mean_RBF']


def observe(raw,truth,rng,eta,common=0.):
    """Add white sensor noise of SD eta, and optionally a shared white signal of SD common, in signal-SD units."""
    x=raw/raw.std(1,keepdims=True)+eta*rng.normal(size=raw.shape)
    if common:x=x+common*rng.normal(size=raw.shape[1])
    x=(x-x.mean(1,keepdims=True))/x.std(1,keepdims=True)
    r=np.corrcoef(x);off=~np.eye(len(x),dtype=bool);n=len(x)//3;block=np.arange(len(x))//n
    truth.update(eta=eta,mean_r=float(r[off].mean()),mean_abs_r=float(abs(r[off]).mean()),abs_r_AB=float(abs(r[:n,n:2*n]).mean()),
                 abs_r_within=float(abs(r[off&(block[:,None]==block[None,:])]).mean()),condition=float(np.linalg.cond(r)))
    return x,truth


def record(task):
    arm,gamma,seed=task;raw,truth=simulate(gamma,seed)
    rng=np.random.default_rng(np.random.SeedSequence([NOISE_SEED,ARMS.index(arm),int(round(gamma*1000)),seed]))
    return observe(raw,truth,rng,.5 if arm==ARMS[0] else float(rng.uniform(.3,.9)))


def prepare(workers,run=RUN,arms=ARMS,first=FIRST_SEED,make=record,n=N_COMMUNITY,pyspi_seed=PYSPI_SEED,source=__file__):
    import yaml
    from scripts.spi_baseline_exploration import sha
    data=ROOT/'data/order-parameter-inference'/run;remote='/scratch/ql44/we2614/mts-spi-study/order-parameter-inference/'+run
    if (data/'manifest.json').exists():raise FileExistsError(data)
    data.mkdir(parents=True,exist_ok=True)
    tasks=[(arm,float(g),first[arm]+s) for arm in arms for g in GAMMAS for s in range(SEEDS)];rows=[];raw={}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for (arm,gamma,seed),(x,truth) in zip(tasks,pool.map(make,tasks,chunksize=4)):
            name=f'xfreq21-{arm}-g{round(gamma*1000):03d}-s{seed:03d}'
            assert x.shape==(3*n,T) and np.isfinite(x).all()
            raw[name]=x
            rows.append(dict(row_id=name,corpus_index=len(rows),M=3*n,N=3*n,T=T,seed=seed,instance=seed,block=seed,
                label=arm,system=arm,control=gamma,role='development' if seed-first[arm]<FIT_SEEDS else 'evaluation',**truth))
    np.savez_compressed(data/'observations.npz',**raw,__dataset_names__=np.array(list(raw)),
        __labels_json__=np.array([json.dumps([r['label']]) for r in rows]),__shapes__=np.array([[3*n,T]]*len(rows)),
        __axis_order__=np.array(['process','observation']))
    manifest=dict(rows=rows,corpus=run,archive_sha256=sha(data/'observations.npz'),generator_sha256=sha(source),
        dynamics_sha256=sha(ROOT/'scripts/cross_frequency_locking.py'),
        analysis_scope='Fixed before p90 outcomes; see docs/research/order-parameter-benchmarks/cross-frequency-locking-261006.md. '
            'Independent realization per arm, control and seed; first eight seeds per control fit, last eight evaluate; arms analysed separately.')
    (data/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    config=dict(name=run,source=dict(format='named-npz-v1',archive=remote+'/observations.npz',sha256=manifest['archive_sha256'],
        axis_order=['process','observation']),base_output_dir=remote+'/mpis',pyspi_config='configs/pyspi/benchmarked_p90.yaml',
        normalise=False,random_seed=pyspi_seed)
    (data/'corpus.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    for name,indices in partitions(len(rows)).items():
        (data/f'{name}-indices.txt').write_text(''.join(f'{i}\n' for i in indices))
    frame=pd.DataFrame(rows)
    print(frame.groupby(['system','control'])[['Q_lock','X1','eta','mean_abs_r','abs_r_AB','abs_r_within','condition']].mean().iloc[::4].round(3).to_string())
    print(len(rows),manifest['archive_sha256'])


def figure(scores,out,stem='snr-comparison',heading='2:1 locking through sensor noise'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update(STYLE);arms=list(dict.fromkeys(scores.system))
    fig,axes=plt.subplots(len(arms),3,figsize=(11.6,3.1*len(arms)),constrained_layout=True,squeeze=False);right=[]
    for row,arm in enumerate(arms):
        part=scores[scores.system.eq(arm)];near=part[part.control.between(*BOUNDARY)];critical=float(DELTA/(near.X2.mean()+2*near.Y.mean()))
        right+=[tracking_panel(axes[row,0],part,Z_CURVES,critical,arm),tracking_panel(axes[row,1],part,MEAN_CURVES,critical,arm)]
        group=part[part.role.eq('evaluation')&part.comparison_eligible].groupby('control').mean_abs_r;m=group.mean();ax=axes[row,2]
        ax.plot(m.index,m,'o-',color='#009E73',label='mean $|r|$, all pairs');ax.fill_between(m.index,group.quantile(.1),group.quantile(.9),color='#009E73',alpha=.12,lw=0)
        ax.axvline(critical,color='.6',lw=.8,ls=':');ax.set(xlabel=r'Cross-coupling $\gamma$',ylabel='Mean absolute Pearson',ylim=(0,.45),title=arm);ax.legend(fontsize=7)
    share_limits(right)
    fig.suptitle(f"{heading}, M=N={scores['M'].iloc[0]}, T={scores['T'].iloc[0]}; held seeds, bands 10-90 per cent of instances",fontsize=10)
    for ext in ['png','svg']:fig.savefig(out/f'{stem}.{ext}',dpi=180)
    plt.close(fig)


def leading(x,fit,standard,k=3):
    """Leading PCs with coordinate()'s preprocessing; scores for every row and explained-variance ratios."""
    from sklearn.decomposition import PCA
    keep=np.isfinite(x[fit]).all(0)&(np.nanstd(x[fit],axis=0)>1e-10);x=x[:,keep];x=np.where(np.isfinite(x),x,np.median(x[fit],axis=0))
    x=(x-x[fit].mean(0))/(x[fit].std(0) if standard else 1);model=PCA(k,svd_solver='full').fit(x[fit])
    return model.transform(x),model


def components(x,fit,held,standard,targets,k=3):
    """Explained variance and held |Spearman| of each leading PC with each target."""
    q,model=leading(x,fit,standard,k);q=q[held]
    return dict(evr=[round(float(v),4) for v in model.explained_variance_ratio_],
        **{name:[round(float(abs(spearmanr(q[:,j],t).statistic)),4) for j in range(k)] for name,t in targets.items() if np.unique(t).size>1})


def scatter(out,arm=ARMS[1],nuisance='eta',label=r'sensor-noise SD $\eta$',stem='component-scatter'):
    """Held records in the plane of the first two target-blind components, coloured by Q and by the nuisance."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update(STYLE);rows=pd.read_csv(out/'scores.csv');bank=np.load(out/'features.npz');np.testing.assert_array_equal(bank['row_id'],rows.row_id)
    index=np.flatnonzero(rows.system.eq(arm).to_numpy());part=rows.iloc[index];fit=part.role.eq('development').to_numpy()
    held=(part.role.eq('evaluation')&part.comparison_eligible).to_numpy()
    fig,axes=plt.subplots(2,2,figsize=(7.4,6),constrained_layout=True,sharex='row',sharey='row')
    for row,(name,key,standard) in enumerate([('Mean of each SPI','mean',True),('SPI-SPI, centered','z',False)]):
        q,model=leading(bank[key][index],fit,standard,2);evr=model.explained_variance_ratio_
        q=q/q[fit].std(0)*np.sign([spearmanr(q[fit,j],part.control[fit]).statistic for j in range(2)])  # display orientation only
        for ax,(column,cmap,text) in zip(axes[row],[('Q_lock','viridis','locking index $Q$'),(nuisance,'magma',label)]):
            points=ax.scatter(q[held,0],q[held,1],c=part[column].to_numpy()[held],cmap=cmap,s=11,alpha=.85,edgecolors='none')
            rho=[abs(spearmanr(q[held,j],part[column].to_numpy()[held]).statistic) for j in range(2)]
            ax.set(xlabel=f'PC1 ({evr[0]:.0%} of variance)',title=f'{name}\n$|\\rho|$ with colour: PC1 {rho[0]:.2f}, PC2 {rho[1]:.2f}')
            fig.colorbar(points,ax=ax,label=text,pad=.02)
        axes[row,0].set_ylabel(f'PC2 ({evr[1]:.0%} of variance)')
    fig.suptitle(f"First two unsupervised components, {arm}, M=N={part['M'].iloc[0]}; held recordings",fontsize=10)
    for ext in ['png','svg']:fig.savefig(out/f'{stem}.{ext}',dpi=180)
    plt.close(fig)


def analyze(data,out,nuisance='eta',stem='snr-comparison',heading='2:1 locking through sensor noise',
            qualification='Exploratory first p90 pass of the sensor-noise variant. Primary readout and scores were fixed before outcomes; supervised mean readouts are information ceilings.'):
    from scripts.report_dependence_transition import coordinate
    from scripts.dependence_transition_pipeline import predict_means
    from scripts.spi_baseline_exploration import sha
    rows=pd.DataFrame(json.loads((data/'manifest.json').read_text())['rows'])
    bank=np.load(out/'features.npz');np.testing.assert_array_equal(bank['row_id'],rows.row_id)
    metrics=[];scores=[];settings={}
    for arm in dict.fromkeys(rows.system):
        index=np.flatnonzero(rows.system.eq(arm).to_numpy());part=rows.iloc[index].reset_index(drop=True)
        fit=part.role.eq('development').to_numpy();held=~fit;y=part.Q_lock.to_numpy();common=held.copy();pcs={}
        for label,key,standard in [('z_PC1','z',False),('z_standard_PC1','z',True),('mean_PC1','mean',True),('distribution_PC1','distribution',True)]:
            q,evr,missing=coordinate(bank[key][index],fit,standard)
            # Orientation uses the control on fit rows only; fitting stays blind to control, Q and the nuisance.
            part[label]=q*(1 if spearmanr(q[fit],part.control[fit]).statistic>=0 else -1)
            pcs[label]=dict(evr=evr,max_missing=float(missing[held].max()));common&=missing<=.05
        mean=bank['mean'][index];valid=np.isfinite(mean[fit]).all(0)&(np.nanstd(mean[fit],axis=0)>1e-10)
        rank=[abs(spearmanr(mean[fit,j],y[fit]).statistic) if valid[j] else -np.inf for j in range(mean.shape[1])]
        best=int(np.argmax(rank));part['selected_mean']=mean[:,best];settings[arm]=dict(selected_mean=str(bank['spi_order'][best]))
        for nonlinear,label in [(False,'mean_ridge'),(True,'mean_RBF')]:
            part[label],settings[arm][label]=predict_means(mean,y,part.seed.to_numpy(),fit,nonlinear)
        for label in READOUTS:common&=np.isfinite(part[label]).to_numpy()
        counts=part[common].groupby('control').size()
        if len(counts)!=len(GAMMAS) or counts.min()<4:raise ValueError(f'{arm}: too few commonly eligible evaluation records per control')
        part['comparison_eligible']=common|fit;evaluation=part[common];scores.append(part)
        # Diagnostic: which factor each representation ranks first, second and third.
        targets={'abs_rho_Q':y[common],f'abs_rho_{nuisance}':part[nuisance].to_numpy()[common]}
        settings[arm]['components']={name:components(bank[key][index],fit,common,standard,targets) for name,key,standard in
            [('mean',  'mean',True),('z','z',False),('z_standard','z',True)]}
        for label in READOUTS:
            metrics.append(dict(arm=arm,method=label,kind='supervised ceiling' if label in ('selected_mean','mean_ridge','mean_RBF') else 'unsupervised',
                n=int(common.sum()),abs_rho_Q=float(abs(spearmanr(evaluation[label],evaluation.Q_lock).statistic)),
                **{f'abs_rho_{nuisance}':float(abs(spearmanr(evaluation[label],evaluation[nuisance]).statistic)) if evaluation[nuisance].nunique()>1 else np.nan},
                **{k:v for k,v in step_scores(evaluation,label).items() if k!='abs_rho_Q'},**pcs.get(label,{})))
    scores=pd.concat(scores);figure(scores,out,stem,heading);scores.to_csv(out/'scores.csv',index=False);metrics=pd.DataFrame(metrics);metrics.to_csv(out/'metrics.csv',index=False)
    (out/'analysis.json').write_text(json.dumps(dict(settings=settings,features_sha256=sha(out/'features.npz'),source_sha256=sha(__file__),
        qualification=qualification),indent=2)+'\n')
    print(metrics[['arm','method','kind','n','abs_rho_Q',f'abs_rho_{nuisance}','step_contrast','steepest_interval','evr','max_missing']].round(3).to_string(index=False));print(settings)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','extract','analyze','figure','scatter'])
    p.add_argument('--data',type=Path,default=DATA);p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    if a.stage=='prepare':prepare(a.workers)
    else:
        a.output.mkdir(parents=True,exist_ok=True)
        if a.stage=='extract':
            from scripts.analyze_native_coupling import extract
            extract(a.data,a.output,corpus=RUN)
        elif a.stage=='figure':figure(pd.read_csv(a.output/'scores.csv'),a.output)
        elif a.stage=='scatter':scatter(a.output)
        else:
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=4):analyze(a.data,a.output)
