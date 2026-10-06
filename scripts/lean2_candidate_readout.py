"""Bundle physics-gated observations and report frozen, matched baseline PC1s."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,pearsonr

ROOT=Path(__file__).resolve().parents[1]


def bundle(physics,out,corpus,remote,fit_seed_stop):
    import yaml
    if (out/'manifest.json').exists():raise FileExistsError(out)
    rows=[];raw={}
    for p in sorted(physics.glob('case-*.json')):
        row=json.loads(p.read_text());a=np.load(p.with_suffix('.npz'))['observations']
        name=f"{row['system']}-{p.stem}"
        if np.linalg.matrix_rank(a)<len(a):raise ValueError('Rank-deficient observations')
        r=np.corrcoef(a);off=~np.eye(len(a),dtype=bool)
        for key,value in [('mean_r',r[off].mean()),('mean_abs_r',abs(r[off]).mean())]:
            if key in row:np.testing.assert_allclose(row[key],value,rtol=0,atol=1e-12)
        raw[name]=(a-a.mean(1,keepdims=True))/a.std(1,keepdims=True)
        row.update(row_id=name,corpus_index=len(rows),instance=row['seed'],block=row['seed'],label=corpus,
            role='development' if row['seed']<fit_seed_stop else 'evaluation')
        rows.append(row)
    assert rows and len(set(r['system'] for r in rows))==1
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'observations.npz',**raw,__dataset_names__=np.array(list(raw)),
        __labels_json__=np.array([json.dumps([corpus])]*len(rows)),__shapes__=np.array([x.shape for x in raw.values()]),
        __axis_order__=np.array(['process','observation']))
    digest=hashlib.sha256((out/'observations.npz').read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(dict(rows=rows,corpus=corpus,archive_sha256=digest,
        protocol='Physics-selected interval. Fit and evaluation use disjoint seeds. Centered SPI-SPI PC1 primary; standardized sensitivity. Mean-SPI PC1 standardized. Q never fits PC1.'),indent=2)+'\n')
    (out/'corpus.yaml').write_text(yaml.safe_dump(dict(name=corpus,source=dict(format='named-npz-v1',
        archive=remote+'/observations.npz',sha256=digest,axis_order=['process','observation']),
        base_output_dir=remote+'/mpis',pyspi_config='configs/pyspi/benchmarked_p90.yaml',normalise=False,random_seed=261109),sort_keys=False))
    smoke={1,len(rows)};node=set(np.linspace(2,len(rows)-1,min(48,len(rows)-2),dtype=int));rest=set(range(1,len(rows)+1))-smoke-node
    for part,indices in [('smoke',smoke),('node',node),('rest',rest)]:
        (out/f'{part}-indices.txt').write_text(''.join(f'{i}\n' for i in sorted(indices)))


def analyze(data,out):
    from scripts.report_dependence_transition import coordinate
    from scripts.dependence_transition_pipeline import predict_means
    rows=pd.DataFrame(json.loads((data/'manifest.json').read_text())['rows'])
    bank=np.load(out/'features.npz');np.testing.assert_array_equal(bank['row_id'],rows.row_id)
    fit=rows.role.eq('development').to_numpy();held=~fit;y=rows.Q.to_numpy();eligible=held.copy();details={}
    scores=rows.copy()
    for label,key,standard in [('z_PC1','z',False),('z_standard_PC1','z',True),('mean_PC1','mean',True)]:
        q,evr,missing=coordinate(bank[key],fit,standard)
        sign=1 if spearmanr(q[fit],rows.control[fit]).statistic>=0 else -1
        scores[label]=sign*(q-q[fit].mean())/q[fit].std()
        scores[label+'_missing']=missing;eligible&=missing<=.05
        details[label]=dict(evr=evr,max_held_missing=float(missing[held].max()))
    means=bank['mean'];valid=np.isfinite(means[fit]).all(0)&(np.nanstd(means[fit],axis=0)>1e-10)
    ranks=[abs(spearmanr(means[fit,j],y[fit]).statistic) if valid[j] else -np.inf for j in range(means.shape[1])]
    best=int(np.argmax(ranks));scores['selected_mean']=means[:,best]
    scores['mean_ridge'],details['mean_ridge']=predict_means(means,y,rows.seed.to_numpy(),fit,False)
    methods=['z_PC1','z_standard_PC1','mean_abs_r','mean_r','mean_PC1','selected_mean','mean_ridge']
    for label in methods:eligible&=np.isfinite(scores[label]).to_numpy()
    scores['comparison_eligible']=eligible|fit;metrics=[]
    counts=scores[eligible].groupby('control').size()
    if len(counts)!=rows.control.nunique() or counts.min()<4:
        raise ValueError('Insufficient common eligibility: retain the failure, do not silently compare different cohorts')
    for label in methods:
        rho=abs(spearmanr(scores.loc[eligible,label],y[eligible]).statistic)
        kind='supervised information check' if label in ('selected_mean','mean_ridge') else 'unsupervised comparison'
        metrics.append(dict(method=label,kind=kind,held_abs_rho=float(rho),
            held_abs_pearson=float(abs(pearsonr(scores.loc[eligible,label],y[eligible]).statistic)),n=int(eligible.sum())))
    # Resample entire held seed trajectories; g values within a quenched network
    # are not treated as independent replicates. These are exploratory intervals.
    rng=np.random.default_rng(261110);seeds=np.unique(rows.seed[eligible]);boot=[]
    for _ in range(1000):
        ix=np.concatenate([np.flatnonzero(eligible&(rows.seed.to_numpy()==s)) for s in rng.choice(seeds,len(seeds),replace=True)])
        boot.append([abs(spearmanr(scores[label].to_numpy()[ix],y[ix]).statistic) for label in methods])
    boot=np.array(boot)
    for j,item in enumerate(metrics):
        item['bootstrap_low'],item['bootstrap_high']=np.nanquantile(boot[:,j],[.025,.975]).tolist()
    details['z_minus_baseline_rho_intervals']={label:np.nanquantile(boot[:,0]-boot[:,methods.index(label)],[.025,.975]).tolist()
        for label in ['mean_abs_r','mean_PC1']}
    details['selected_mean']=str(bank['spi_order'][best]);details['eligible_per_control']=scores[eligible].groupby('control').size().to_dict()
    scores.to_csv(out/'scores.csv',index=False);pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    (out/'analysis.json').write_text(json.dumps(details,indent=2)+'\n')
    figure(scores,out)
    if rows.system.iloc[0]=='cgle':secondary_amplitude(scores,out)
    print(pd.DataFrame(metrics).to_string(index=False))


def secondary_amplitude(scores,out):
    """Reuse primary target-blind scores; only display signs change for a falling Q."""
    held=scores.role.eq('evaluation')&scores.comparison_eligible
    methods=['z_PC1','z_standard_PC1','mean_abs_r','mean_PC1']
    metrics=[]
    for col in methods:
        metrics.append(dict(method=col,target='finite-window minimum amplitude',n=int(held.sum()),
            held_abs_rho=float(abs(spearmanr(scores.loc[held,col],scores.loc[held,'min_amplitude']).statistic)),
            held_abs_pearson=float(abs(pearsonr(scores.loc[held,col],scores.loc[held,'min_amplitude']).statistic))))
    pd.DataFrame(metrics).to_csv(out/'minimum-amplitude-metrics.csv',index=False)
    display=scores.copy();display['Q']=display.min_amplitude
    for col in ['z_PC1','mean_PC1']:display[col]=-display[col]
    figure(display,out,target_label=r'Finite-window $\min |A|$',filename='minimum-amplitude-comparison')


def figure(scores,out,target_label=None,filename='baseline-comparison'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'legend.frameon':False,'lines.linewidth':1.7,'lines.markersize':2.7})
    held=scores[scores.role.eq('evaluation')&scores.comparison_eligible]
    system=scores.system.iloc[0]
    label={'cgle':'$c_3$','rate':'$g$','crossfreq':r'$\gamma$'}[system]
    truth_label={'cgle':'Defect density $Q$','rate':r'$Q=\lambda_{\max}$','crossfreq':'2:1 locking index $Q$'}[system]
    if target_label is not None:truth_label=target_label
    fig,axes=plt.subplots(1,3,figsize=(11,3.3),layout='constrained')
    for ax,col,title in zip(axes,['z_PC1','mean_abs_r','mean_PC1'],['SPI–SPI PC1','Mean absolute Pearson','Mean-SPI PC1']):
        group=held.groupby('control');q=group.Q.mean();a=group[col].mean();right=ax.twinx();right.spines['right'].set_visible(True)
        lines=ax.plot(q.index,q,'o-',color='#222222',label='$Q$')
        ax.fill_between(q.index,group.Q.quantile(.1),group.Q.quantile(.9),color='#222222',alpha=.12,lw=0)
        lines+=right.plot(a.index,a,'s-',color='#31688e',label='$q$' if col!='mean_abs_r' else r'$\overline{|r|}$')
        right.fill_between(a.index,group[col].quantile(.1),group[col].quantile(.9),color='#31688e',alpha=.12,lw=0)
        rho=abs(spearmanr(held[col],held.Q).statistic)
        ax.set(xlabel=label,ylabel=truth_label,title=title+rf' · $|\rho|={rho:.2f}$')
        right.set_ylabel('Fit-record SD units' if col!='mean_abs_r' else 'Mean absolute Pearson')
        ax.legend(lines,[v.get_label() for v in lines],loc='upper left',fontsize=7)
    name={'cgle':'Complex Ginzburg–Landau','rate':'Driven rate network','crossfreq':'Cross-frequency locking'}[system]
    fig.suptitle(f"{name} · M={scores['M'].iloc[0]}, N={scores['N'].iloc[0]}, T={scores['T'].iloc[0]} · held seeds; bands 10–90%",fontsize=10)
    for ext in ('png','svg'):fig.savefig(out/f'{filename}.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['bundle','extract','analyze'])
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--corpus');p.add_argument('--remote');p.add_argument('--fit-seed-stop',type=int)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.stage=='bundle':bundle(a.data,a.out,a.corpus,a.remote,a.fit_seed_stop)
    elif a.stage=='extract':
        from scripts.analyze_native_coupling import extract
        extract(a.data,a.out,corpus=a.corpus)
    else:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):analyze(a.data,a.out)
