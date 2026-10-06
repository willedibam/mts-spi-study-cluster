"""Exploratory: time-resolved NeuroTycho anaesthetic induction as a real regime change for SPI-SPI.

Uses the staged KTMD and propofol sessions and the pilot's montage and preprocessing unchanged
(16 bipolar channels, 8 s at 250 Hz, 28 s filter context). Unlike the pilot's stable-state windows,
windows here tile the injection session: before injection, induction, the labelled anaesthetized
interval and, where recorded, emergence and recovery; plus the separate awake-eyes-closed session.
There is no measured control or per-window order parameter: truth is the event markers. Fitting is
target-blind and leaves the evaluated animal out.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import spearmanr
from src.neurotycho_pilot import preprocess

ROOT=Path(__file__).resolve().parents[1]
RUN='neurotycho-induction-261006'
DATA=ROOT/'data/order-parameter-inference'/RUN
OUT=ROOT/'results/order-parameter-inference'/RUN
REMOTE='/scratch/ql44/we2614/mts-spi-study/order-parameter-inference/'+RUN
RAW=ROOT/'data/representation/neurotycho/inputs'
PILOT=ROOT/'results/representation/neurotycho'
SOURCES=[(RAW/'neurotycho_source_260910',PILOT/'neurotycho_source_pilot_260910/float64/all-source','ktmd'),
         (RAW/'neurotycho_target_260910',PILOT/'neurotycho_target_pilot_260910','propofol')]
STRIDE=dict(awake=40000,injection=20000,after=60000)
PYSPI_SEED=261098


def events(session):
    c=loadmat(session/'Condition.mat',simplify_cells=True)
    labels=[str(x) for x in np.atleast_1d(c['ConditionLabel'])];index=np.atleast_1d(c['ConditionIndex']).astype(int)-1
    return list(zip(labels,index.tolist()))


def windows(marks,n):
    """Window starts (1 kHz samples) and phases for one session of n samples."""
    first=dict();[first.setdefault(k,v) for k,v in marks];out=[];feasible=lambda s:s>=10000 and s+18000<=n
    if 'AnestheticInjection' in first:
        t0=first['AnestheticInjection'];a0=first.get('Anesthetized-Start',n);a1=first.get('Anesthetized-End',n)
        r0=first.get('RecoveryEyesClosed-Start',n);start=10000
        while feasible(start):
            mid=start+4000
            phase='pre' if start+8000<=t0 else 'induction' if mid<a0 else 'anaesthetized' if mid<a1 else 'emergence' if mid<r0 else 'recovery'
            out.append((start,phase,(mid-t0)/1000));start+=STRIDE['injection'] if mid<a1 else STRIDE['after']
    elif 'AwakeEyesClosed-Start' in first:
        for start in range(first['AwakeEyesClosed-Start']+30000,first['AwakeEyesClosed-End']-38000,STRIDE['awake']):
            if feasible(start):out.append((start,'awake',np.nan))
    return out


def prepare():
    import yaml
    from scripts.spi_baseline_exploration import sha
    if (DATA/'manifest.json').exists():raise FileExistsError(DATA)
    DATA.mkdir(parents=True,exist_ok=True);rows=[];raw={};rejected=0
    for root,meta_root,agent in SOURCES:
        for archive in sorted(p for p in root.iterdir() if p.is_dir()):
            meta=json.loads((meta_root/f'{archive.name}.json').read_text());pairs=np.array(meta['montage']['pairs'])
            channels=sorted(set(pairs.ravel().tolist()));local=np.vectorize(channels.index)(pairs)
            for session in sorted(archive.glob('Session*')):
                if not (session/'Condition.mat').exists() or not (session/f'ECoG_ch{channels[0]}.mat').exists():continue
                marks=events(session)
                if not windows(marks,10**12):continue
                signal=np.stack([np.ravel(next(v for k,v in loadmat(session/f'ECoG_ch{c}.mat').items() if not k.startswith('__'))).astype(float)
                                 for c in channels])
                for start,phase,time in windows(marks,signal.shape[1]):
                    x,quality=preprocess(signal[:,start-10000:start+18000],local)
                    if x is None:rejected+=1;continue
                    x=(x-x.mean(1,keepdims=True))/x.std(1,keepdims=True)
                    name=f'{archive.name[:8]}-{agent}-{meta["animal"]}-{session.name}-{start:08d}';raw[name]=x
                    r=np.corrcoef(x);off=~np.eye(16,dtype=bool)
                    rows.append(dict(row_id=name,corpus_index=len(rows),M=16,N=16,T=2000,seed=0,instance=len(rows),block=archive.name,label=agent,
                        system=agent,control=None if np.isnan(time) else float(time),role='evaluation',animal=meta['animal'],archive=archive.name,
                        session=session.name,agent=agent,phase=phase,time=None if np.isnan(time) else float(time),start=int(start),
                        mean_abs_r=float(abs(r[off]).mean()),mean_r=float(r[off].mean())))
                print(archive.name,session.name,[m for m in marks],len(rows),flush=True)
    np.savez_compressed(DATA/'observations.npz',**raw,__dataset_names__=np.array(list(raw)),
        __labels_json__=np.array([json.dumps([r['agent'],r['animal'],r['phase']]) for r in rows]),__shapes__=np.array([[16,2000]]*len(rows)),
        __axis_order__=np.array(['process','observation']))
    manifest=dict(rows=rows,corpus=RUN,archive_sha256=sha(DATA/'observations.npz'),generator_sha256=sha(__file__),rejected_windows=rejected,
        analysis_scope='Exploratory real-data attempt; truth is event markers only. Leave-one-animal-out target-blind fitting.')
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    config=dict(name=RUN,source=dict(format='named-npz-v1',archive=REMOTE+'/observations.npz',sha256=manifest['archive_sha256'],
        axis_order=['process','observation']),base_output_dir=REMOTE+'/mpis',pyspi_config='configs/pyspi/benchmarked_p90.yaml',
        normalise=False,random_seed=PYSPI_SEED)
    (DATA/'corpus.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    total=len(rows);smoke=[1,total];rest=[i for i in range(1,total+1) if i not in smoke]
    for name,indices in dict(smoke=smoke,rest=rest).items():(DATA/f'{name}-indices.txt').write_text(''.join(f'{i}\n' for i in indices))
    frame=pd.DataFrame(rows);print(frame.groupby(['agent','animal','archive']).phase.value_counts().unstack(fill_value=0).to_string())
    print(total,'rejected',rejected,manifest['archive_sha256'])


def states(rows):
    """Refine phases with the event intervals: single-session dates hold the awake intervals before injection."""
    marks={};out=[]
    for r in rows.itertuples():
        key=(r.archive,r.session)
        if key not in marks:
            root=next(root for root,_,_ in SOURCES if (root/r.archive).exists());first={}
            [first.setdefault(k,v) for k,v in events(root/r.archive/r.session)];marks[key]=first
        m=marks[key];inside=lambda name:name+'-Start' in m and m[name+'-Start']<=r.start and r.start+8000<=m[name+'-End']
        out.append('awake' if inside('AwakeEyesClosed') else 'eyes-open' if inside('AwakeEyesOpened') else 'recovery' if inside('RecoveryEyesClosed')
                   else r.phase if r.phase!='pre' else 'pre')
    onset=[(marks[(r.archive,r.session)].get('Anesthetized-Start',np.nan)-marks[(r.archive,r.session)].get('AnestheticInjection',np.nan))/1000 for r in rows.itertuples()]
    return np.array(out),np.array(onset)


def project(x,fit,standard,k=3):
    from sklearn.decomposition import PCA
    keep=np.isfinite(x[fit]).all(0)&(np.nanstd(x[fit],axis=0)>1e-10);x=x[:,keep];x=np.where(np.isfinite(x),x,np.median(x[fit],axis=0))
    x=(x-x[fit].mean(0))/(x[fit].std(0) if standard else 1);model=PCA(k,svd_solver='full').fit(x[fit]);q=model.transform(x)
    return q/q[fit].std(0),model.explained_variance_ratio_


def eta2(values,groups):
    values=np.asarray(values,float);total=((values-values.mean())**2).sum()
    return float(sum(len(v)*(v.mean()-values.mean())**2 for v in (values[groups==g] for g in np.unique(groups)))/total)


def analyze(data,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score
    from scripts.cross_frequency_locking import STYLE
    rows=pd.DataFrame(json.loads((data/'manifest.json').read_text())['rows']);bank=np.load(out/'features.npz')
    np.testing.assert_array_equal(bank['row_id'],rows.row_id);rows['state'],rows['onset']=states(rows)
    plateau=rows.state.isin(['awake','anaesthetized']).to_numpy();y=rows.state.eq('anaesthetized').to_numpy().astype(float)
    reps=[('mean_PC','mean',True),('z_PC','z',False),('z_standard_PC','z',True)];metrics=[];components=[]
    for name,key,standard in reps:
        for j in range(3):rows[f'{name}{j+1}']=np.nan
    for animal in sorted(rows.animal.unique()):
        held=rows.animal.eq(animal).to_numpy();fit=~held
        for name,key,standard in reps:
            q,evr=project(bank[key],fit,standard)
            # Sign from the fitting animals' plateau labels only; the held animal stays untouched.
            sign=np.sign([spearmanr(q[fit&plateau,j],y[fit&plateau]).statistic for j in range(3)]);q=q*sign
            for j in range(3):
                rows.loc[held,f'{name}{j+1}']=q[held,j]
                components.append(dict(held=animal,representation=name,component=j+1,evr=float(evr[j]),
                    fit_state_eta2=eta2(q[fit&plateau,j],y[fit&plateau]),fit_date_eta2=eta2(q[fit&plateau,j],rows.archive.to_numpy()[fit&plateau]),
                    fit_animal_eta2=eta2(q[fit&plateau,j],rows.animal.to_numpy()[fit&plateau])))
    readouts=['mean_abs_r','mean_PC1','z_PC1','z_standard_PC1','mean_PC2','z_PC2']
    for archive,part in rows.groupby('archive'):
        flat=part[part.state.isin(['awake','anaesthetized'])];course=part[part.time.notna()&(part.time>=0)&part.state.isin(['induction','anaesthetized'])]
        for label in readouts:
            awake,deep=flat[label][flat.state.eq('awake')],flat[label][flat.state.eq('anaesthetized')]
            auc=roc_auc_score(flat.state.eq('anaesthetized'),flat[label]) if len(awake) and len(deep) else np.nan
            contrast=(deep.mean()-awake.mean())/np.sqrt((deep.var()+awake.var())/2) if len(awake) and len(deep) else np.nan
            metrics.append(dict(archive=archive,animal=part.animal.iloc[0],agent=part.agent.iloc[0],readout=label,auroc=auc,contrast=contrast,
                rho_time=spearmanr(course[label],course.time).statistic,
                induction_rho_time=spearmanr(part[label][part.state.eq('induction')],part.time[part.state.eq('induction')]).statistic))
    metrics=pd.DataFrame(metrics);components=pd.DataFrame(components)
    rows.to_csv(out/'scores.csv',index=False);metrics.to_csv(out/'metrics.csv',index=False);components.to_csv(out/'components.csv',index=False)
    summary=metrics.assign(oriented_auroc=metrics.auroc,abs_contrast=metrics.contrast).groupby(['agent','readout'])[['auroc','contrast','rho_time','induction_rho_time']].agg(['mean','min','max']).round(2)
    print(summary.to_string());print(components.groupby(['representation','component'])[['evr','fit_state_eta2','fit_date_eta2','fit_animal_eta2']].mean().round(2).to_string())
    plt.rcParams.update(STYLE);animals=sorted(rows.animal.unique());color=dict(zip(animals,['#0072B2','#D55E00','#009E73','#CC79A7']))
    shown=[('mean_abs_r','Mean absolute Pearson'),('mean_PC1','Mean-SPI PC1 (fit-animal SD)'),('z_PC1','SPI-SPI PC1, centered (fit-animal SD)')]
    fig,axes=plt.subplots(3,2,figsize=(10.5,8.2),constrained_layout=True,sharex='col',sharey='row')
    for column,agent in enumerate(['ktmd','propofol']):
        for row,(label,name) in enumerate(shown):
            ax=axes[row,column]
            for archive,part in rows[rows.agent.eq(agent)].groupby('archive'):
                c=color[part.animal.iloc[0]];course=part[part.time.notna()].sort_values('time');awake=part[part.state.eq('awake')][label]
                smooth=course[label].rolling(5,center=True,min_periods=1).median()
                ax.plot(course.time/60,smooth,color=c,lw=.9,alpha=.85)
                ax.errorbar(-4-.25*animals.index(part.animal.iloc[0]),awake.median(),yerr=[[awake.median()-awake.quantile(.1)],[awake.quantile(.9)-awake.median()]],
                            fmt='o',color=c,ms=3,lw=.8)
                if np.isfinite(part.onset.iloc[0]):ax.plot(part.onset.iloc[0]/60,ax.get_ylim()[0],'|',color=c,ms=7,clip_on=False)
            ax.axvline(0,color='.6',lw=.8,ls=':');ax.set(title=f'{agent}: {name}' if row==0 else None,ylabel=name if column==0 else None,
                xlabel='Minutes since first injection (awake eyes-closed reference at left)' if row==2 else None,xlim=(-6,45 if agent=='ktmd' else 62))
    axes[0,0].legend(handles=[plt.Line2D([],[],color=color[a],label=a) for a in animals],fontsize=7,ncol=4,loc='upper right')
    fig.suptitle('NeuroTycho induction, M=16 bipolar ECoG, 8 s windows; coordinates fitted without the plotted animal; ticks mark the anaesthetized label',fontsize=10)
    for ext in ['png','svg']:fig.savefig(out/f'induction-trajectories.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(11.4,6.4),constrained_layout=True);order=['eyes-open','awake','pre','induction','anaesthetized','emergence','recovery']
    everything=np.ones(len(rows),bool);code=rows.state.map({k:i for i,k in enumerate(order)}).to_numpy()
    for row,(name,key,standard) in enumerate([('Mean of each SPI','mean',True),('SPI-SPI, centered','z',False)]):
        q,evr=project(bank[key],everything,standard,2)
        for ax,(values,cmap,title) in zip(axes[row],[(code,'viridis','state'),(rows.animal.map({a:i for i,a in enumerate(animals)}).to_numpy(),'tab10','animal'),
                                                     (rows.agent.eq('propofol').to_numpy().astype(int),'coolwarm','agent')]):
            ax.scatter(q[:,0],q[:,1],c=values,cmap=cmap,s=5,alpha=.6,edgecolors='none')
            groups=rows.state.to_numpy() if title=='state' else rows.animal.to_numpy() if title=='animal' else rows.agent.to_numpy()
            ax.set(xlabel=f'PC1 ({evr[0]:.0%})',ylabel=f'PC2 ({evr[1]:.0%})',title=f'{name}, by {title}\n$\\eta^2$: PC1 {eta2(q[:,0],groups):.2f}, PC2 {eta2(q[:,1],groups):.2f}')
    fig.suptitle('All windows pooled: first two unsupervised components',fontsize=10)
    for ext in ['png','svg']:fig.savefig(out/f'pooled-components.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','extract','analyze'])
    p.add_argument('--data',type=Path,default=DATA);p.add_argument('--output',type=Path,default=OUT);a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='extract':
        a.output.mkdir(parents=True,exist_ok=True)
        from scripts.analyze_native_coupling import extract
        extract(a.data,a.output,corpus=RUN)
    else:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=4):analyze(a.data,a.output)
