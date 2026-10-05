"""Analytic Student-t block alignment control, not a dynamical bifurcation."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import gammaln,digamma
from scipy.stats import t as student_t,rankdata,kendalltau,spearmanr
from sklearn.feature_selection import mutual_info_regression
from scripts.scout_dependence_transition import ROOT,normalize
from scripts.report_dependence_transition import coordinate
from scripts.spi_baseline_exploration import sha

OUT=ROOT/'results/order-parameter-inference/dependence-transition-261005/tail-alignment'
M,T=16,1000


def parameters(swaps):
    nu=np.repeat([3.,30.],4);rho=np.repeat([.1,.2],4)
    for j in range(swaps):rho[j],rho[j+4]=rho[j+4],rho[j]
    return rho,nu


def entropy(d,nu):
    return (gammaln(nu/2)-gammaln((nu+d)/2)+d/2*np.log(nu*np.pi)
            +(nu+d)/2*(digamma((nu+d)/2)-digamma(nu/2)))


def population(swaps):
    rho,nu=parameters(swaps)
    mi=-.5*np.log1p(-rho*rho)+2*entropy(1,nu)-entropy(2,nu)
    tau=2/np.pi*np.arcsin(rho)
    tail=2*student_t.cdf(-np.sqrt((nu+1)*(1-rho)/(1+rho)),nu+1)
    edge_r=np.r_[rho,np.zeros(112)];edge_mi=np.r_[mi,np.zeros(112)]
    return dict(swaps=swaps,Q_tail=tail.sum()/120,mean_r=rho.sum()/120,
                mean_tau=tau.sum()/120,mean_MI=mi.sum()/120,z_r_MI=np.corrcoef(edge_r,edge_mi)[0,1])


def recording(swaps,seed):
    rng=np.random.default_rng(np.random.SeedSequence([261053,seed]))
    rho,nu=parameters(swaps);noise=rng.normal(size=(8,2,T))
    scales=np.sqrt(rng.chisquare(nu[:,None],size=(8,T))/(nu[:,None]-2))
    x=np.empty((16,T))
    x[::2]=noise[:,0]/scales
    x[1::2]=(rho[:,None]*noise[:,0]+np.sqrt(1-rho[:,None]**2)*noise[:,1])/scales
    return normalize(x[rng.permutation(16)])


def estimate(task):
    swaps,seed=task;x=recording(swaps,seed);ii,jj=np.triu_indices(M,1)
    pearson=np.corrcoef(x)[ii,jj];spearman=np.corrcoef(rankdata(x,axis=1))[ii,jj]
    kendall=np.array([kendalltau(x[i],x[j]).statistic for i,j in zip(ii,jj)])
    mi=np.zeros((M,M))
    for i in range(M-1):
        mi[i,i+1:]=mutual_info_regression(x[i+1:].T,x[i],n_neighbors=4,random_state=261053)
    profiles=np.array([pearson,spearman,kendall,mi[ii,jj]])
    z=np.corrcoef(profiles)[np.triu_indices(4,1)]
    return dict(swaps=swaps,seed=seed,Q_tail=population(swaps)['Q_tail']),profiles.mean(1),z


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'rows.csv').exists():raise FileExistsError(OUT)
    pop=pd.DataFrame([population(j) for j in range(5)])
    assert np.max(np.ptp(pop[['mean_r','mean_tau','mean_MI']].to_numpy(),axis=0))<1e-14
    pop.to_csv(OUT/'population.csv',index=False)
    tasks=[(j,seed) for j in range(5) for seed in range(64)]
    rows=[];mean=[];z=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,(r,m,v) in enumerate(pool.map(estimate,tasks)):
            rows.append(r);mean.append(m);z.append(v)
            if i%32==0:print('tail control',i+1,'/',len(tasks),flush=True)
    rows=pd.DataFrame(rows);mean=np.array(mean);z=np.array(z)
    fit=rows.seed.to_numpy()<32;held=~fit;y=rows.Q_tail.to_numpy()
    metrics=[];scores=rows.copy()
    for label,x,standard in [('mean_PC1',mean,True),('z_PC1',z,False),('z_standard_PC1',z,True)]:
        q,evr,_=coordinate(x,fit,standard);scores[label]=q
        metrics.append(dict(method=label,held_abs_rho=abs(spearmanr(q[held],y[held]).statistic),evr=evr))
    for j,name in enumerate(['Pearson','Spearman','Kendall','KSG_MI']):
        metrics.append(dict(method='mean_'+name,held_abs_rho=abs(spearmanr(mean[held,j],y[held]).statistic)))
    metrics.append(dict(method='prespecified_z_Pearson_MI',held_abs_rho=abs(spearmanr(z[held,2],y[held]).statistic)))
    rows.to_csv(OUT/'rows.csv',index=False);scores.to_csv(OUT/'scores.csv',index=False)
    np.savez_compressed(OUT/'features.npz',mean=mean,z=z)
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
    (OUT/'report.json').write_text(json.dumps(dict(records=320,M=M,T=T,source_sha256=sha(__file__),
        scope='Four focused probes, not p90. Analytically equal population means are not exactly equal finite-sample estimates. Tail dependence is a known statistical quantity; the assignment sweep is engineered, not a bifurcation.',
        seed=261053,training='seeds0-31',evaluation='seeds32-63'),indent=2)+'\n')
    print(pop.to_string(index=False));print(pd.DataFrame(metrics).to_string(index=False))


def prepare_p90():
    import yaml
    run_id='tail-alignment-261005'
    data=ROOT/'data/order-parameter-inference'/run_id
    if (data/'manifest.json').exists():raise FileExistsError(data)
    data.mkdir(parents=True,exist_ok=True)
    arrays={};rows=[]
    for swaps in range(5):
        for seed in range(64):
            name=f'tail-swaps{swaps}-seed{seed:03d}'
            arrays[name]=recording(swaps,seed)
            rows.append(dict(row_id=name,corpus_index=len(rows),M=M,T=T,label='tail-alignment',
                system='tail-alignment',control=swaps/4,seed=seed,instance=seed,block=seed,
                role='development' if seed<32 else 'evaluation',**population(swaps)))
    np.savez_compressed(data/'observations.npz',**arrays,__dataset_names__=np.array(list(arrays)),
        __labels_json__=np.array([json.dumps(['tail-alignment'])]*len(rows)),
        __shapes__=np.array([[M,T]]*len(rows)),__axis_order__=np.array(['process','observation']))
    manifest=dict(rows=rows,archive_sha256=sha(data/'observations.npz'),generator_sha256=sha(__file__),
        qualification='Exploratory p90 extension after seeing focused-probe results on these seeds; not fresh confirmation. All320 records retained.')
    (data/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    remote=f'/scratch/ql44/we2614/mts-spi-study/order-parameter-inference/{run_id}'
    cfg=dict(name=run_id,source=dict(format='named-npz-v1',archive=remote+'/observations.npz',
        sha256=manifest['archive_sha256'],axis_order=['process','observation']),base_output_dir=remote+'/mpis',
        pyspi_config='configs/pyspi/benchmarked_p90.yaml',normalise=False,random_seed=261054)
    (ROOT/f'configs/external/{run_id}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    print(len(rows),manifest['archive_sha256'])


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--prepare-p90',action='store_true');a=p.parse_args()
    if a.prepare_p90:prepare_p90()
    else:run()
