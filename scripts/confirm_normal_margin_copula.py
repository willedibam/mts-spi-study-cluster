"""Frozen-readout replication and enlarged-training comparison; no held-target fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from scripts.report_dependence_transition import coordinate
from scripts.analyze_tail_alignment import analyze
from scripts.spi_baseline_exploration import sha


def combined_rows(old,new):
    retained=[dict(r,original_development=True) for r in old if r['role']=='development']
    retained.extend(dict(r,original_development=False) for r in new)
    assert len({r['row_id'] for r in retained})==len(retained)
    return retained


def confirm(old_data,old_output,new_data,new_output,out):
    out.mkdir(parents=True,exist_ok=True)
    om=json.loads((old_data/'manifest.json').read_text());nm=json.loads((new_data/'manifest.json').read_text())
    rows=combined_rows(om['rows'],nm['rows']);frame=pd.DataFrame(rows)
    ob=np.load(old_output/'features.npz');nb=np.load(new_output/'features.npz')
    np.testing.assert_array_equal(ob['spi_order'],nb['spi_order'])
    np.testing.assert_array_equal(ob['row_id'],[r['row_id'] for r in om['rows']])
    np.testing.assert_array_equal(nb['row_id'],[r['row_id'] for r in nm['rows']])
    old_fit=np.array([r['role']=='development' for r in om['rows']])
    bank={key:np.concatenate([ob[key][old_fit],nb[key]]) for key in ['z','mean','distribution']}
    bank.update(row_id=frame.row_id.to_numpy(),spi_order=ob['spi_order'])
    np.savez_compressed(out/'features.npz',**bank)
    raw={}
    for data,records in [(old_data,[r for r in om['rows'] if r['role']=='development']),(new_data,nm['rows'])]:
        assert sha(data/'observations.npz')==json.loads((data/'manifest.json').read_text())['archive_sha256']
        with np.load(data/'observations.npz') as archive:
            raw.update({r['row_id']:archive[r['row_id']] for r in records})
    np.savez_compressed(out/'observations.npz',**raw)
    manifest=dict(rows=rows,analysis_scope=nm['analysis_scope'],
        figure_title='Fresh Gaussian-margin replication: M=N=16, T=1000; 240 training / 160 held records')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    analyze(out,out)
    scores=pd.read_csv(out/'scores.csv');fit=frame.original_development.to_numpy();held=frame.role.eq('evaluation').to_numpy()
    old_scores=pd.read_csv(old_output/'scores.csv')
    y=frame.Q_tail.to_numpy();common=held&scores.comparison_eligible.to_numpy()
    frozen=[]
    for name,key,standard in [('frozen_z_PC1','z',False),('frozen_z_standard_PC1','z',True),('frozen_mean_PC1','mean',True)]:
        q,evr,missing=coordinate(bank[key],fit,standard)
        sign=1 if spearmanr(q[fit],y[fit]).statistic>=0 else -1
        np.testing.assert_allclose(sign*q[fit],old_scores.loc[old_fit,name.removeprefix('frozen_')],atol=1e-8)
        scores[name]=sign*q;common&=missing<=.05
        frozen.append(dict(method=name,evr=evr,max_missing=float(missing[held].max())))
    # All comparisons use the same fresh held rows; previously examined held80 never enter.
    methods=['frozen_z_PC1','frozen_z_standard_PC1','frozen_mean_PC1','z_PC1','z_standard_PC1',
        'mean_PC1','distribution_PC1','selected_mean','mean_ridge','mean_RBF','empirical_mean_r','empirical_mean_abs_r']
    for name in methods:common &= np.isfinite(scores[name])
    use=np.flatnonzero(common);assert len(use)>0
    # Independent records across levels; stratified record bootstrap preserves the fixed sweep design.
    rng=np.random.default_rng(261077)
    groups=[use[frame.control.to_numpy()[use]==c] for c in sorted(frame.control.unique())]
    draws=np.concatenate([g[rng.integers(len(g),size=(2000,len(g)))] for g in groups],axis=1)
    metrics=[];boots={}
    for name in methods:
        values=scores[name].to_numpy();rho=abs(spearmanr(values[use],y[use]).statistic)
        boot=np.array([abs(spearmanr(values[d],y[d]).statistic) for d in draws]);boots[name]=boot
        lo,hi=np.quantile(boot,[.025,.975]);metrics.append(dict(method=name,n=len(use),held_abs_rho=rho,low=lo,high=hi))
    paired=[]
    for name in methods:
        if name=='frozen_z_PC1':continue
        delta=boots['frozen_z_PC1']-boots[name];lo,hi=np.quantile(delta,[.025,.975])
        paired.append(dict(comparison='frozen_z_PC1 - '+name,low=lo,high=hi))
    pd.DataFrame(metrics).to_csv(out/'replication-metrics.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'replication-paired.csv',index=False)
    scores['fresh_comparison_eligible']=common;scores.to_csv(out/'replication-scores.csv',index=False)
    # Independent raw scalar check on all combined recordings, against saved analysis output.
    expected=[np.corrcoef(raw[k])[~np.eye(16,dtype=bool)].mean() for k in frame.row_id]
    np.testing.assert_allclose(scores.empirical_mean_r,expected,atol=1e-14)
    provenance=dict(old_features_sha256=sha(old_output/'features.npz'),new_features_sha256=sha(new_output/'features.npz'),
        source_sha256=sha(__file__),old_training=int(fit.sum()),enlarged_training=int((~held).sum()),
        fresh_held=int(held.sum()),common_held=len(use),frozen=frozen,original_evaluation_used=False,
        qualification='Fixed readouts re-fitted deterministically on original development only; fresh held never fits or selects. Bootstrap intervals conditional, exploratory, unadjusted; no absence-of-information theorem.')
    (out/'replication-audit.json').write_text(json.dumps(provenance,indent=2)+'\n')
    plot(scores,frame,fit,common,out)
    print(pd.DataFrame(metrics).round(4).to_string(index=False))


def plot(scores,rows,old_fit,held,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.plot_large_m_pair_sampling import figure_style
    figure_style();fig,axes=plt.subplots(1,2,figsize=(9,3.5),constrained_layout=True)
    methods=[('frozen_z_PC1','Frozen SPI--SPI PC1','#0072B2'),('frozen_mean_PC1','Frozen mean PC1','#E69F00'),
        ('z_PC1','Enlarged-training SPI--SPI PC1','#56B4E9'),('mean_PC1','Enlarged-training mean PC1','#D55E00')]
    for field,label,color in methods:
        fit=old_fit if field.startswith('frozen') else rows.role.eq('development').to_numpy()
        q=scores[field];q=(q-q[fit].mean())/q[fit].std(ddof=0)
        curve=rows.assign(q=q)[held].groupby('control').q
        axes[0].plot(curve.mean().index,curve.mean(),'o-',label=label,color=color)
        axes[0].fill_between(curve.mean().index,curve.quantile(.1),curve.quantile(.9),alpha=.08,color=color)
    axes[0].set(ylabel='PC1, respective training SD units',title=r'Fresh held means; 10--90\% instance spread');axes[0].legend(fontsize=6)
    metrics=pd.read_csv(out/'replication-metrics.csv')
    axes[1].hlines(np.arange(len(metrics)),metrics.low,metrics.high,lw=1)
    axes[1].plot(metrics.held_abs_rho,np.arange(len(metrics)),'o',ms=3)
    axes[1].set_yticks(np.arange(len(metrics)),metrics.method,fontsize=6);axes[1].invert_yaxis()
    axes[1].set(xlabel=r'Held $|\rho(q,Q)|$',title=r'Conditional 95\% bootstrap intervals')
    axes[0].set_xlabel('Fraction of assignments swapped');axes[0].grid(axis='y',alpha=.12)
    fig.suptitle('Fixed-generator replication, M=N=16, T=1000; 160 fresh held records\nCompositional control, not a physical bifurcation',fontsize=10)
    for ext in ['png','svg']:fig.savefig(out/f'replication-comparison.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['old-data','old-output','new-data','new-output','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(limits=4):confirm(a.old_data,a.old_output,a.new_data,a.new_output,a.output)
