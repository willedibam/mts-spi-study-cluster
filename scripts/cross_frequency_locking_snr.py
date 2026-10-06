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
from scripts.cross_frequency_locking import ROOT,GAMMAS,N_COMMUNITY,T,DELTA,simulate,partitions,step_scores

RUN='cross-frequency-locking-snr-261006'
DATA=ROOT/'data/order-parameter-inference'/RUN
OUT=ROOT/'results/order-parameter-inference'/RUN
REMOTE='/scratch/ql44/we2614/mts-spi-study/order-parameter-inference/'+RUN
ARMS=('fixed-noise','random-noise')
FIRST_SEED={'fixed-noise':100,'random-noise':200}
SEEDS,FIT_SEEDS=16,8
NOISE_SEED,PYSPI_SEED=261091,261092
READOUTS=['mean_abs_r','mean_r','mean_PC1','distribution_PC1','z_PC1','z_standard_PC1','selected_mean','mean_ridge','mean_RBF']


def record(task):
    arm,gamma,seed=task;raw,truth=simulate(gamma,seed)
    rng=np.random.default_rng(np.random.SeedSequence([NOISE_SEED,ARMS.index(arm),int(round(gamma*1000)),seed]))
    eta=.5 if arm==ARMS[0] else float(rng.uniform(.3,.9))
    x=raw/raw.std(1,keepdims=True)+eta*rng.normal(size=raw.shape)
    x=(x-x.mean(1,keepdims=True))/x.std(1,keepdims=True)
    r=np.corrcoef(x);off=~np.eye(len(x),dtype=bool);n=N_COMMUNITY;block=np.arange(len(x))//n
    truth.update(eta=eta,mean_r=float(r[off].mean()),mean_abs_r=float(abs(r[off]).mean()),abs_r_AB=float(abs(r[:n,n:2*n]).mean()),
                 abs_r_within=float(abs(r[off&(block[:,None]==block[None,:])]).mean()),condition=float(np.linalg.cond(r)))
    return x,truth


def prepare(workers):
    import yaml
    from scripts.spi_baseline_exploration import sha
    if (DATA/'manifest.json').exists():raise FileExistsError(DATA)
    DATA.mkdir(parents=True,exist_ok=True)
    tasks=[(arm,float(g),FIRST_SEED[arm]+s) for arm in ARMS for g in GAMMAS for s in range(SEEDS)];rows=[];raw={}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for (arm,gamma,seed),(x,truth) in zip(tasks,pool.map(record,tasks,chunksize=4)):
            name=f'xfreq21-{arm}-g{round(gamma*1000):03d}-s{seed:03d}'
            assert x.shape==(3*N_COMMUNITY,T) and np.isfinite(x).all()
            raw[name]=x
            rows.append(dict(row_id=name,corpus_index=len(rows),M=3*N_COMMUNITY,N=3*N_COMMUNITY,T=T,seed=seed,instance=seed,block=seed,
                label=arm,system=arm,control=gamma,role='development' if seed-FIRST_SEED[arm]<FIT_SEEDS else 'evaluation',**truth))
    np.savez_compressed(DATA/'observations.npz',**raw,__dataset_names__=np.array(list(raw)),
        __labels_json__=np.array([json.dumps([r['label']]) for r in rows]),__shapes__=np.array([[3*N_COMMUNITY,T]]*len(rows)),
        __axis_order__=np.array(['process','observation']))
    manifest=dict(rows=rows,corpus=RUN,archive_sha256=sha(DATA/'observations.npz'),generator_sha256=sha(__file__),
        dynamics_sha256=sha(ROOT/'scripts/cross_frequency_locking.py'),
        analysis_scope='Fixed before p90 outcomes; see docs/research/order-parameter-benchmarks/cross-frequency-locking-261006.md. '
            'Independent realization per arm, control and seed; first eight seeds per control fit, last eight evaluate; arms analysed separately.')
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    config=dict(name=RUN,source=dict(format='named-npz-v1',archive=REMOTE+'/observations.npz',sha256=manifest['archive_sha256'],
        axis_order=['process','observation']),base_output_dir=REMOTE+'/mpis',pyspi_config='configs/pyspi/benchmarked_p90.yaml',
        normalise=False,random_seed=PYSPI_SEED)
    (DATA/'corpus.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    for name,indices in partitions(len(rows)).items():
        (DATA/f'{name}-indices.txt').write_text(''.join(f'{i}\n' for i in indices))
    frame=pd.DataFrame(rows)
    print(frame.groupby(['system','control'])[['Q_lock','eta','mean_abs_r','abs_r_AB','abs_r_within','condition']].mean().iloc[::4].round(3).to_string())
    print(len(rows),manifest['archive_sha256'])


def analyze(data,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.report_dependence_transition import coordinate
    from scripts.dependence_transition_pipeline import predict_means
    from scripts.spi_baseline_exploration import sha
    rows=pd.DataFrame(json.loads((data/'manifest.json').read_text())['rows'])
    bank=np.load(out/'features.npz');np.testing.assert_array_equal(bank['row_id'],rows.row_id)
    metrics=[];scores=[];settings={}
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
        'legend.frameon':False,'lines.linewidth':1.7,'lines.markersize':2.7})
    fig,axes=plt.subplots(2,3,figsize=(11,6.2),constrained_layout=True)
    for row,arm in enumerate(ARMS):
        index=np.flatnonzero(rows.system.eq(arm).to_numpy());part=rows.iloc[index].reset_index(drop=True)
        fit=part.role.eq('development').to_numpy();held=~fit;y=part.Q_lock.to_numpy();common=held.copy();pcs={}
        for label,key,standard in [('z_PC1','z',False),('z_standard_PC1','z',True),('mean_PC1','mean',True),('distribution_PC1','distribution',True)]:
            q,evr,missing=coordinate(bank[key][index],fit,standard)
            # Orientation uses the control on fit rows only; fitting stays blind to control, Q and eta.
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
        for label in READOUTS:
            metrics.append(dict(arm=arm,method=label,kind='supervised ceiling' if label in ('selected_mean','mean_ridge','mean_RBF') else 'unsupervised',
                n=int(common.sum()),abs_rho_Q=float(abs(spearmanr(evaluation[label],evaluation.Q_lock).statistic)),
                abs_rho_eta=float(abs(spearmanr(evaluation[label],evaluation.eta).statistic)) if arm==ARMS[1] else np.nan,
                **{k:v for k,v in step_scores(evaluation,label).items() if k!='abs_rho_Q'},**pcs.get(label,{})))
        group=evaluation.groupby('control')
        def scaled(label):
            m=group[label].mean();lo,hi=m.iloc[0],m.iloc[-1]
            return [(v-lo)/(hi-lo) for v in (m,group[label].quantile(.1),group[label].quantile(.9))]
        critical=float(DELTA/(part.X2[part.control.between(.08,.12)].mean()+2*part.Y[part.control.between(.08,.12)].mean()))
        for ax,labels in [(axes[row,0],[('z_PC1','#E69F00','SPI-SPI PC1, centered'),('z_standard_PC1','#D55E00','SPI-SPI PC1, standardized')]),
                          (axes[row,1],[('mean_PC1','#0072B2','mean-SPI PC1'),('distribution_PC1','#56B4E9','distribution PC1')])]:
            m,_,_=scaled('Q_lock');ax.plot(m.index,m,'ko-',label='$Q$: 2:1 locking index')
            for label,color,name in labels:
                m,lo,hi=scaled(label);ax.plot(m.index,m,'o-',color=color,label=name);ax.fill_between(m.index,lo,hi,color=color,alpha=.12,lw=0)
            ax.axvline(critical,color='.6',lw=.8,ls=':');ax.set(xlabel=r'Cross-coupling $\gamma$',ylabel='Rescaled to end-point means',title=arm);ax.legend(fontsize=7)
        ax=axes[row,2];m=group.mean_abs_r.mean();ax.plot(m.index,m,'o-',color='#009E73',label='mean $|r|$, all pairs')
        ax.fill_between(m.index,group.mean_abs_r.quantile(.1),group.mean_abs_r.quantile(.9),color='#009E73',alpha=.12,lw=0)
        ax.axvline(critical,color='.6',lw=.8,ls=':');ax.set(xlabel=r'Cross-coupling $\gamma$',ylabel='Mean absolute Pearson (raw)',ylim=(0,.45),title=arm);ax.legend(fontsize=7)
    fig.suptitle(f'2:1 locking through sensor noise, M=N={3*N_COMMUNITY}, T={T}; held seeds, bands 10-90 per cent of instances',fontsize=10)
    for ext in ['png','svg']:fig.savefig(out/f'snr-comparison.{ext}',dpi=180)
    plt.close(fig);pd.concat(scores).to_csv(out/'scores.csv',index=False);metrics=pd.DataFrame(metrics);metrics.to_csv(out/'metrics.csv',index=False)
    (out/'analysis.json').write_text(json.dumps(dict(settings=settings,features_sha256=sha(out/'features.npz'),source_sha256=sha(__file__),
        qualification='Exploratory first p90 pass of the sensor-noise variant. Primary readout and scores were fixed before outcomes; supervised mean readouts are information ceilings.'),indent=2)+'\n')
    print(metrics[['arm','method','kind','n','abs_rho_Q','abs_rho_eta','step_contrast','steepest_interval','evr','max_missing']].round(3).to_string(index=False));print(settings)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','extract','analyze'])
    p.add_argument('--data',type=Path,default=DATA);p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    if a.stage=='prepare':prepare(a.workers)
    else:
        a.output.mkdir(parents=True,exist_ok=True)
        if a.stage=='extract':
            from scripts.analyze_native_coupling import extract
            extract(a.data,a.output,corpus=RUN)
        else:
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=4):analyze(a.data,a.output)
