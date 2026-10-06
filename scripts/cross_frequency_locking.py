"""2:1 cross-frequency locking between oscillator communities: generator, p90 bundle and readouts.

Microscopic model: Komarov & Pikovsky, arXiv:1502.06193 (PRE 92, 012906), Eq. (10) with
alpha1=alpha2=beta=0, eps1=eps2=eps and gamma1=gamma2=gamma. Slow community A (phases phi,
frequency 1) and fast community B (psi, frequency 2+delta) share the resonant cross-coupling.
Choices made here, not taken from the paper: narrow Gaussian-quantile frequencies instead of
unit-width Lorentzians, independent phase noise, small finite communities, and a third community
C (frequency 1.37) with internal coupling only. C supplies channel pairs that stay unrelated, so
that linear and nonlinear SPI profiles across pairs can stop being collinear when A-B lock.
The control is gamma alone; internal coupling is fixed above the synchrony threshold.

Reduction used for the reference boundary (derived here): for coherent communities the slow phase
Phi=Psi_B-2*Theta_A obeys dPhi/dt=delta-gamma*(X2+2*Y)*sin(Phi), so locking sets in at
gamma_c=delta/(X2+2*Y). Q is the 2:1 locking index |<exp(i*Phi)>_t| on a disjoint future window.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import erfinv
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
RUN='cross-frequency-locking-261006'
DATA=ROOT/'data/order-parameter-inference'/RUN
OUT=ROOT/'results/order-parameter-inference'/RUN
REMOTE='/scratch/ql44/we2614/mts-spi-study/order-parameter-inference/'+RUN
N_COMMUNITY,T,STEP=8,1000,.5
DELTA,EPS,WIDTH,SIGMA,FREQ_C=.3,.5,.05,.1,1.37
GAMMAS=np.round(np.linspace(0,.2,21),3)
SEEDS,FIT_SEEDS=16,8
SEED,PYSPI_SEED=261085,261086
BOUNDARY=(.08,.12)


def velocity(a,b,c,gamma,wa,wb,wc,eps=EPS):
    """Eq. (10) in mean-field form: sin(psi_k-2phi_n) on A, sin(2phi_k-psi_m) on B."""
    x1=np.exp(1j*a).mean();x2=np.exp(2j*a).mean();y=np.exp(1j*b).mean();zc=np.exp(1j*c).mean()
    da=wa+eps*(x1*np.exp(-1j*a)).imag+gamma*(y*np.exp(-2j*a)).imag
    db=wb+eps*(y*np.exp(-1j*b)).imag+gamma*(x2*np.exp(-1j*b)).imag
    dc=wc+eps*(zc*np.exp(-1j*c)).imag
    return da,db,dc,(x1,x2,y,zc)


def simulate(gamma,seed,n=N_COMMUNITY,delta=DELTA,dt=.02,burn=500.,reference=1000.,samples=T,eps=EPS):
    """Euler-Maruyama. Returns raw sin(phase) channels ordered A|B|C and future-window truth."""
    index=int(np.argmin(np.abs(GAMMAS-gamma))) if np.min(np.abs(GAMMAS-gamma))<1e-9 else int(round(gamma*1e6))+1000
    rng=np.random.default_rng(np.random.SeedSequence([SEED,n,index,seed]))
    q=np.sqrt(2)*WIDTH*erfinv(2*(np.arange(n)+.5)/n-1)
    wa=1+rng.permutation(q);wb=2+delta+rng.permutation(q);wc=FREQ_C+rng.permutation(q)
    a,b,c=rng.uniform(0,2*np.pi,(3,n));every=round(STEP/dt);x=[];fields=[]
    for k in range(round((burn+samples*STEP+reference)/dt)):
        da,db,dc,z=velocity(a,b,c,gamma,wa,wb,wc,eps)
        e=SIGMA*np.sqrt(dt)*rng.normal(size=(3,n))
        a=a+dt*da+e[0];b=b+dt*db+e[1];c=c+dt*dc+e[2]
        if k>=round(burn/dt) and (k+1)%every==0:
            if len(x)<samples:x.append(np.r_[np.sin(a),np.sin(b),np.sin(c)])
            else:fields.append(z)
    fields=np.array(fields);phase=np.unwrap(np.angle(fields[:,2]*np.conj(fields[:,1])))
    halves=[abs(np.exp(1j*h).mean()) for h in np.array_split(phase,2)]
    truth=dict(Q_lock=float(abs(np.exp(1j*phase).mean())),Q_lock_first=float(halves[0]),Q_lock_second=float(halves[1]),
               slip=float(abs(phase[-1]-phase[0])/(len(phase)*STEP)),X1=float(abs(fields[:,0]).mean()),
               X2=float(abs(fields[:,1]).mean()),Y=float(abs(fields[:,2]).mean()),RC=float(abs(fields[:,3]).mean()))
    return np.array(x).T,truth


def record(task):
    gamma,seed=task;raw,truth=simulate(gamma,seed);n=N_COMMUNITY
    x=(raw-raw.mean(1,keepdims=True))/raw.std(1,keepdims=True)
    r=np.corrcoef(x);off=~np.eye(len(x),dtype=bool);block=np.arange(len(x))//n
    within=off&(block[:,None]==block[None,:])
    truth.update(mean_r=float(r[off].mean()),mean_abs_r=float(abs(r[off]).mean()),abs_r_AB=float(abs(r[:n,n:2*n]).mean()),
                 abs_r_within=float(abs(r[within]).mean()),max_r=float(r[off].max()),condition=float(np.linalg.cond(r)))
    return x,truth


def partitions(total):
    smoke={1,total};node=set(np.linspace(2,total-1,48,dtype=int).tolist());rest=set(range(1,total+1))-smoke-node
    return dict(smoke=sorted(smoke),node=sorted(node),rest=sorted(rest))


def prepare(workers):
    import yaml
    from scripts.spi_baseline_exploration import sha
    if (DATA/'manifest.json').exists():raise FileExistsError(DATA)
    DATA.mkdir(parents=True,exist_ok=True)
    tasks=[(float(g),s) for g in GAMMAS for s in range(SEEDS)];rows=[];raw={}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for (gamma,seed),(x,truth) in zip(tasks,pool.map(record,tasks,chunksize=4)):
            name=f'xfreq21-g{round(gamma*1000):03d}-s{seed:02d}'
            assert x.shape==(3*N_COMMUNITY,T) and np.isfinite(x).all() and truth['max_r']<1-1e-6
            raw[name]=x
            rows.append(dict(row_id=name,corpus_index=len(rows),M=3*N_COMMUNITY,N=3*N_COMMUNITY,T=T,seed=seed,instance=seed,
                block=seed,label=RUN,system=RUN,control=gamma,role='development' if seed<FIT_SEEDS else 'evaluation',**truth))
    np.savez_compressed(DATA/'observations.npz',**raw,__dataset_names__=np.array(list(raw)),
        __labels_json__=np.array([json.dumps([RUN])]*len(rows)),__shapes__=np.array([[3*N_COMMUNITY,T]]*len(rows)),
        __axis_order__=np.array(['process','observation']))
    manifest=dict(rows=rows,corpus=RUN,archive_sha256=sha(DATA/'observations.npz'),generator_sha256=sha(__file__),
        parameters=dict(n_per_community=N_COMMUNITY,delta=DELTA,eps=EPS,frequency_sd=WIDTH,phase_noise=SIGMA,frequency_C=FREQ_C,
                        sample_interval=STEP,dt=.02,burn=500,reference=1000,channels='A(0-7)|B(8-15)|C(16-23), sin(phase), z-scored'),
        analysis_scope='Fixed before p90 outcomes; see docs/research/order-parameter-benchmarks/cross-frequency-locking-261006.md. '
            'Independent realization per control and seed. Seeds0-7 fit target-blind PC1s and supervised ceilings; seeds8-15 evaluate.')
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    config=dict(name=RUN,source=dict(format='named-npz-v1',archive=REMOTE+'/observations.npz',sha256=manifest['archive_sha256'],
        axis_order=['process','observation']),base_output_dir=REMOTE+'/mpis',pyspi_config='configs/pyspi/benchmarked_p90.yaml',
        normalise=False,random_seed=PYSPI_SEED)
    (DATA/'corpus.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    for name,indices in partitions(len(rows)).items():
        (DATA/f'{name}-indices.txt').write_text(''.join(f'{i}\n' for i in indices))
    frame=pd.DataFrame(rows)
    print(frame.groupby('control')[['Q_lock','slip','X1','X2','Y','mean_abs_r','abs_r_AB','abs_r_within','condition']].mean().round(4).to_string())
    print(len(rows),manifest['archive_sha256'])


def step_scores(frame,column,target='Q_lock'):
    """Held-record transition scores; frame holds evaluation rows only."""
    group=frame.groupby('control')[column];mean=group.mean();sd=group.std()
    low,high=BOUNDARY;first,last=mean.index.min(),mean.index.max()
    total=mean.loc[last]-mean.loc[first];step=mean.loc[high]-mean.loc[low]
    noise=np.sqrt((sd.loc[low]**2+sd.loc[high]**2)/2);jumps=mean.diff().abs().iloc[1:]
    return dict(step_contrast=float(abs(step)/noise),step_share=float(step/total),
        post_locking_drift=float(abs(mean.loc[last]-mean.loc[high])/abs(total)),
        steepest_interval=f'{mean.index[jumps.values.argmax()]:.3f}-{jumps.idxmax():.3f}',
        abs_rho_Q=float(abs(spearmanr(frame[column],frame[target]).statistic)))


Q_COLOR='#222222'
Z_CURVES=[('z_PC1','#E69F00','SPI-SPI PC1, centered'),('z_standard_PC1','#D55E00','SPI-SPI PC1, standardized')]
MEAN_CURVES=[('mean_PC1','#0072B2','mean-SPI PC1'),('distribution_PC1','#56B4E9','distribution PC1')]
STYLE={'font.family':'serif','mathtext.fontset':'cm','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
       'legend.frameon':False,'lines.linewidth':1.7,'lines.markersize':2.7}


def tracking_panel(ax,scores,curves,critical,title=None):
    """Q in physical units on the left axis; target-blind q in fit-record SD units on the right. Returns the q axis."""
    fit=scores[scores.role.eq('development')];held=scores[scores.role.eq('evaluation')&scores.comparison_eligible]
    group=held.groupby('control').Q_lock;m=group.mean();right=ax.twinx();right.spines['right'].set_visible(True)
    lines=ax.plot(m.index,m,'o-',color=Q_COLOR,label='$Q$: 2:1 locking index')
    ax.fill_between(m.index,group.quantile(.1),group.quantile(.9),color=Q_COLOR,alpha=.12,lw=0)
    for label,color,name in curves:
        group=((held[label]-fit[label].mean())/fit[label].std()).groupby(held.control);m=group.mean()
        lines+=right.plot(m.index,m,'s-',color=color,label=name)
        right.fill_between(m.index,group.quantile(.1),group.quantile(.9),color=color,alpha=.12,lw=0)
    ax.axvline(critical,color='.6',lw=.8,ls=':');ax.set(xlabel=r'Cross-coupling $\gamma$',ylabel='Locking index $Q$',ylim=(-.05,1.05),title=title)
    right.set_ylabel('$q$ (fit-record SD units)');ax.legend(lines,[l.get_label() for l in lines],fontsize=7,loc='upper left')
    return right


def share_limits(axes):
    low=min(a.get_ylim()[0] for a in axes);high=max(a.get_ylim()[1] for a in axes)
    for a in axes:a.set_ylim(low,high)


def figure(scores,out,critical):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update(STYLE)
    fig,axes=plt.subplots(1,3,figsize=(11.6,3.3),constrained_layout=True)
    share_limits([tracking_panel(axes[0],scores,Z_CURVES,critical),tracking_panel(axes[1],scores,MEAN_CURVES,critical)])
    group=scores[scores.role.eq('evaluation')&scores.comparison_eligible].groupby('control')
    for label,color,name in [('abs_r_within','.6','within community'),('mean_abs_r','#009E73','all pairs'),('abs_r_AB','#CC79A7','A-B pairs')]:
        axes[2].plot(group[label].mean().index,group[label].mean(),'o-',color=color,label=name)
    axes[2].axvline(critical,color='.6',lw=.8,ls=':');axes[2].set(xlabel=r'Cross-coupling $\gamma$',ylabel='Mean absolute Pearson',ylim=(-.03,1.05));axes[2].legend(fontsize=7)
    fig.suptitle(f"2:1 cross-frequency locking, M=N={scores['M'].iloc[0]}, T={scores['T'].iloc[0]}; held seeds, bands 10-90 per cent of instances",fontsize=10)
    for ext in ['png','svg']:fig.savefig(out/f'cross-frequency-comparison.{ext}',dpi=180)
    plt.close(fig)


def analyze(data,out):
    from scripts.report_dependence_transition import coordinate
    from scripts.dependence_transition_pipeline import predict_means
    from scripts.spi_baseline_exploration import sha
    rows=pd.DataFrame(json.loads((data/'manifest.json').read_text())['rows'])
    bank=np.load(out/'features.npz');np.testing.assert_array_equal(bank['row_id'],rows.row_id)
    fit=rows.role.eq('development').to_numpy();held=~fit;y=rows.Q_lock.to_numpy()
    scores=rows.copy();settings={};common=held.copy();pcs={}
    for label,key,standard in [('z_PC1','z',False),('z_standard_PC1','z',True),('mean_PC1','mean',True),('distribution_PC1','distribution',True)]:
        q,evr,missing=coordinate(bank[key],fit,standard)
        # Orientation uses the control on fit rows only; fitting stays blind to control and Q.
        scores[label]=q*(1 if spearmanr(q[fit],rows.control[fit]).statistic>=0 else -1)
        pcs[label]=dict(evr=evr,max_missing=float(missing[held].max()));common&=missing<=.05
    mean=bank['mean'];valid=np.isfinite(mean[fit]).all(0)&(np.nanstd(mean[fit],axis=0)>1e-10)
    rank=[abs(spearmanr(mean[fit,j],y[fit]).statistic) if valid[j] else -np.inf for j in range(mean.shape[1])]
    best=int(np.argmax(rank));scores['selected_mean']=mean[:,best];settings['selected_mean']=str(bank['spi_order'][best])
    for nonlinear,label in [(False,'mean_ridge'),(True,'mean_RBF')]:
        scores[label],settings[label]=predict_means(mean,y,rows.seed.to_numpy(),fit,nonlinear)
    readouts=['mean_abs_r','mean_r','mean_PC1','distribution_PC1','z_PC1','z_standard_PC1','selected_mean','mean_ridge','mean_RBF']
    for label in readouts:common&=np.isfinite(scores[label]).to_numpy()
    counts=scores[common].groupby('control').size()
    if len(counts)!=len(GAMMAS) or counts.min()<4:raise ValueError('Too few commonly eligible evaluation records per control')
    scores['comparison_eligible']=common|fit;evaluation=scores[common]
    kind={'Q_lock':'truth','slip':'truth','selected_mean':'supervised ceiling','mean_ridge':'supervised ceiling','mean_RBF':'supervised ceiling'}
    metrics=[dict(method=label,kind=kind.get(label,'unsupervised'),n=int(common.sum()),**step_scores(evaluation,label),**pcs.get(label,{}))
             for label in ['Q_lock','slip']+readouts]
    near=rows[rows.control.between(*BOUNDARY)];critical=float(DELTA/(near.X2.mean()+2*near.Y.mean()))
    figure(scores,out,critical)
    scores.to_csv(out/'scores.csv',index=False);pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    (out/'analysis.json').write_text(json.dumps(dict(settings=settings,reference_gamma_c=critical,boundary_interval=BOUNDARY,
        features_sha256=sha(out/'features.npz'),source_sha256=sha(__file__),eligible_per_control=counts.to_dict(),
        qualification='Exploratory first p90 pass. Step scores and intervals were fixed before outcomes; supervised mean readouts are information ceilings, not like-for-like competitors.'),indent=2)+'\n')
    print(pd.DataFrame(metrics).round(3).to_string(index=False));print(settings,'gamma_c',round(critical,4))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','extract','analyze','figure'])
    p.add_argument('--data',type=Path,default=DATA);p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    if a.stage=='prepare':prepare(a.workers)
    else:
        a.output.mkdir(parents=True,exist_ok=True)
        if a.stage=='extract':
            from scripts.analyze_native_coupling import extract
            extract(a.data,a.output,corpus=RUN)
        elif a.stage=='figure':
            figure(pd.read_csv(a.output/'scores.csv'),a.output,json.loads((a.output/'analysis.json').read_text())['reference_gamma_c'])
        else:
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=4):analyze(a.data,a.output)
