"""Development-ranked SPI–SPI galleries from existing feature banks; no SPI family filtering."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import textwrap
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research.inference.locking.locking_feature_sensitivity_261007 import project
from scripts.benchmark_latex_style import latex_style, prepare_latex_figure

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_SOURCE=ROOT/'results/order-parameter-inference/cross-frequency-locking-snr-261006'
DEFAULT_OUT=ROOT/'results/order-parameter-inference/locking-features-m24-261009'


def column_rho(x,y):
    """Spearman per coordinate; handles missing/constant held coordinates explicitly."""
    x=np.asarray(x,float);y=np.asarray(y,float)
    result=np.full(x.shape[1],np.nan)
    complete=np.isfinite(x).all(0)
    if complete.any():
        r=rankdata(x[:,complete],axis=0);r-=r.mean(0)
        v=rankdata(y);v-=v.mean()
        denominator=np.sqrt((r*r).sum(0)*(v*v).sum())
        result[complete]=np.divide(v@r,denominator,out=np.full(denominator.shape,np.nan),where=denominator>0)
    for j in np.flatnonzero(~complete):
        valid=np.isfinite(x[:,j])&np.isfinite(y)
        if valid.sum()>=4 and np.unique(x[valid,j]).size>1:
            result[j]=spearmanr(x[valid,j],y[valid]).statistic
    return result


def ranking_indices(weights, rho):
    # Stable feature-order tie breaking; neither ordering uses held outcomes.
    return {'weight':np.argsort(-np.abs(weights),kind='stable'),
            'rho':np.argsort(-np.nan_to_num(np.abs(rho),nan=-1.),kind='stable')}


def short_name(name):
    aliases={'dcorr_biased':'Distance correlation (biased)', 'tlmi_gaussian':'Time-lagged MI (Gaussian)',
             'xme_gaussian_k1':'Cross-map entropy (Gaussian, k1)'}
    if name in aliases:return aliases[name]
    match=re.fullmatch(r'coint_johansen_(max_eig_stat|trace_stat)_order-(\d+)_ardiff-(\d+)',name)
    if match:
        statistic='max-eigen' if match[1]=='max_eig_stat' else 'trace'
        return f'Johansen {statistic} (d={match[2]}, l={match[3]})'
    aliases={'xme_kernel_W-0.5_k1':'Conditional entropy (kernel, k1)',
             'te_kraskov_NN-4_DCE-AUTO_k-1_kt-1_l-1_lt-1':'Transfer entropy (KSG, DCE-auto)',
             'lcss_constraint-itakura':'LCSS (Itakura)',
             'lcss_constraint-sakoe-chiba':'LCSS (Sakoe–Chiba)'}
    if name in aliases:return aliases[name]
    match=re.fullmatch(r'corr_(pearson|spearman|kendall)_tau-(\d+)(-sq)?',name)
    if match:
        return f'{match[1].title()} lag {match[2]}'+(' squared' if match[3] else '')
    return name.replace('cov_EmpiricalCovariance','Empirical covariance').replace('coint_johansen_','Johansen ').replace('max_eig_stat','max-eigen').replace('trace_stat','trace').replace('GraphicalLassoCV','Graphical Lasso CV').replace('_',' ')


def correlation_endpoint(name):
    """Classical correlation family; empirical covariance is Pearson on these z-scored channels.

    Excludes distance/envelope correlation, coherence, precision and other covariance estimators.
    These boundaries are semantic labels, not a proof of statistical independence.
    """
    return name.startswith(('corr_','xcorr')) or name in {
        'spearmanr','spearmanr-sq','kendalltau','kendalltau-sq',
        'cov_EmpiricalCovariance','cov-sq_EmpiricalCovariance'}


def correlation_composition(out=DEFAULT_OUT):
    """Describe the unchanged PC1; no refitting or target-based selection."""
    table=pd.read_csv(out/'all-feature-rankings.csv')
    a=table.spi_a.map(correlation_endpoint);b=table.spi_b.map(correlation_endpoint)
    both=a&b;table['correlation_endpoints']=a.astype(int)+b.astype(int)
    table[both].sort_values('rank_weight').to_csv(out/'correlation-pair-rankings.csv',index=False)
    top=table[~both].sort_values('rank_weight').head(20)
    top.to_csv(out/'top20-at-most-one-correlation.csv',index=False)
    # One neutral cloud/box for ALL weights; only the raw points carry family colour.
    weights=table.weight.to_numpy();mask=both.to_numpy()
    fig,ax=plt.subplots(figsize=(5.8,4.8),constrained_layout=True)
    bodies=ax.violinplot([weights],positions=[0],widths=.68,showextrema=False,points=200)['bodies']
    for body in bodies:
        vertices=body.get_paths()[0].vertices;vertices[:,0]=np.maximum(vertices[:,0],0)
        body.set_facecolor('.7');body.set_edgecolor('none');body.set_alpha(.45)
    ax.boxplot([weights],positions=[0],widths=.04,showfliers=False,patch_artist=True,
               boxprops=dict(facecolor='white',edgecolor='.25',lw=.8),
               medianprops=dict(color='.2',lw=1.2),whiskerprops=dict(color='.3'),capprops=dict(color='.3'))
    jitter=np.random.default_rng(261009).uniform(-.46,-.055,len(weights))
    ax.scatter(jitter[~mask],weights[~mask],s=2.3,color='.55',alpha=.12,linewidths=0,
               rasterized=True,label=f'Other coordinates (n={int((~mask).sum()):,})')
    ax.scatter(jitter[mask],weights[mask],s=4,color='#D55E00',alpha=.55,linewidths=0,
               rasterized=True,label=f'Correlation × correlation (n={int(mask.sum()):,})')
    ax.axhline(0,color='.5',lw=.6,ls=':');ax.set_xlim(-.51,.39)
    ax.set_xticks([0],['All eligible SPI–SPI coordinates'])
    ax.set_ylabel(r'Signed PC1 coefficient $w_{z_{ab}}$')
    ax.set_title('PC1 weight distribution · M=24, T=1000\nVarying independent sensor noise; every eligible coordinate shown',fontsize=9)
    ax.legend(loc='upper right',fontsize=7,frameon=False)
    for spine in ax.spines.values():spine.set_visible(True)
    prepare_latex_figure(fig)
    for ext in ['png','svg']:fig.savefig(out/f'weight-raincloud.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    lag_rows=[]
    for lag in [1,2,3,4,5,10,20]:
        pattern=rf'_tau-{lag}(?:$|-)'
        m=table.spi_a.str.contains(pattern)|table.spi_b.str.contains(pattern)
        lag_rows.append(dict(lag=lag,coordinates=int(m.sum()),squared_weight_share=float((table.weight[m]**2).sum()),
                             best_rank=int(table.rank_weight[m].min()),top100=int((m&(table.rank_weight<=100)).sum())))
    pd.DataFrame(lag_rows).to_csv(out/'lag-weight-summary.csv',index=False)
    broad_a=a|table.spi_a.str.startswith(('cov_','cov-sq_'))
    broad_b=b|table.spi_b.str.startswith(('cov_','cov-sq_'))
    summary=dict(eligible=len(table),both_correlation=int(both.sum()),fraction=float(both.mean()),
        squared_weight_share=float((table.weight[both]**2).sum()),
        rank_quantiles=table.loc[both,'rank_weight'].quantile([0,.25,.5,.75,1]).to_dict(),
        counts_in_top={str(n):int((both&(table.rank_weight<=n)).sum()) for n in [20,50,100,500,1000]},
        correlation_spi_names=sorted(set(table.spi_a[a])|set(table.spi_b[b])),
        definition='Classical zero-lag/lagged/cross-correlation including squared variants and empirical covariance on standardized channels. Not distance/envelope correlation, coherence, other covariance or precision.',
        broader_all_covariance_count=int((broad_a&broad_b).sum()),
        broader_all_covariance_squared_weight_share=float((table.weight[broad_a&broad_b]**2).sum()),
        lag_summary=lag_rows,source_ranking_sha256=hashlib.sha256((out/'all-feature-rankings.csv').read_bytes()).hexdigest())
    (out/'correlation-composition.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    print(top[['rank_weight','spi_a','spi_b','weight','rho_held']].to_string(index=False))
    return summary


def mechanistic_lag_audit(source=DEFAULT_SOURCE,out=DEFAULT_OUT):
    """Post-hoc community decomposition of the rank-7 feature, noisy M24 only.

    Uses existing held observations; this is explanation, not fresh confirmation.
    No fitting, parameter tuning, or changes to the original PC1.
    """
    scores=pd.read_csv(source/'scores.csv');bank=np.load(source/'features.npz')
    np.testing.assert_array_equal(scores.row_id,bank['row_id'])
    order=list(bank['spi_order']);aa,bb=np.triu_indices(len(order),1)
    a,b=sorted([order.index('corr_pearson_tau-2'),order.index('corr_pearson_tau-20-sq')])
    feature=int(np.flatnonzero((aa==a)&(bb==b))[0]);reference=np.asarray(bank['z'][:,feature],dtype=float)
    archive=ROOT/'data/order-parameter-inference'/source.name/'observations.npz'
    observations=np.load(archive);i,j=np.triu_indices(24,1);gi,gj=i//8,j//8
    masks=[(gi==g)&(gj==h) for g,h in [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]]
    records=[]
    for index,row in scores[scores.system.eq('random-noise')&scores.role.eq('evaluation')&scores.comparison_eligible].iterrows():
        x=observations[row.row_id];assert x.shape==(24,1000)
        def lag(t):
            if t==0:return np.corrcoef(x)
            matrix=np.corrcoef(np.vstack([x[:,t:],x[:,:-t]]))[:24,24:]
            return (matrix+matrix.T)/2
        r0,r2,r20=lag(0),lag(2),lag(20);u,v=r2[i,j],r20[i,j]**2
        ub,vb=u.copy(),v.copy()
        for mask in masks:ub[mask]=u[mask].mean();vb[mask]=v[mask].mean()
        z=float(np.corrcoef(u,v)[0,1]);z_block=float(np.corrcoef(ub,vb)[0,1])
        covariance=float(np.mean((u-u.mean())*(v-v.mean())))
        between=float(np.mean((ub-ub.mean())*(vb-vb.mean())))
        record=dict(row_id=row.row_id,control=row.control,seed=row.seed,eta=row.eta,Q=row.Q_lock,q=row.z_PC1,
                    z=z,z_block=z_block,replay_error=abs(z-reference[index]),between_cov=between,within_cov=covariance-between)
        for group,mask in zip('ABC',masks[:3]):
            record[group+'_r2']=float(u[mask].mean());record[group+'_r20sq']=float(v[mask].mean())
            record[group+'_r20_over_r0']=float(r20[i[mask],j[mask]].mean()/r0[i[mask],j[mask]].mean())
        record['contrast']=(record['B_r20sq']-record['A_r20sq'])/(record['B_r20sq']+record['A_r20sq'])
        records.append(record)
    frame=pd.DataFrame(records);assert frame.replay_error.max()<1e-6
    frame.to_csv(out/'mechanistic-lag-audit.csv',index=False)
    summary=dict(scope='Post-hoc explanation on 168 existing held M24 random-noise records; not independent confirmation.',
        feature='corr_pearson_tau-2 × corr_pearson_tau-20-sq',global_weight_rank=7,
        source_sha256=hashlib.sha256((source/'features.npz').read_bytes()).hexdigest(),
        observations_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        max_replay_error=float(frame.replay_error.max()),
        block_rmse=float(np.sqrt(np.mean((frame.z-frame.z_block)**2))),
        block_rho=float(spearmanr(frame.z,frame.z_block).statistic),
        between_absolute_covariance_share=float(abs(frame.between_cov).sum()/(abs(frame.between_cov).sum()+abs(frame.within_cov).sum())),
        associations={key:{target:float(spearmanr(frame[key],frame[target]).statistic) for target in ['Q','q','eta']} for key in ['z','z_block','contrast']},
        endpoint_means=frame[frame.control.isin([0,.14,.2])].groupby('control').mean(numeric_only=True).to_dict('index'))
    (out/'mechanistic-lag-audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2));return summary


def mechanistic_johansen_audit(source=DEFAULT_SOURCE,out=DEFAULT_OUT):
    """Three fixed-control, mid-noise examples; exact rank-26 feature replay, not population validation."""
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    scores=pd.read_csv(source/'scores.csv');bank=np.load(source/'features.npz')
    np.testing.assert_array_equal(scores.row_id,bank['row_id'])
    names=('coint_johansen_max_eig_stat_order-0_ardiff-10','coint_johansen_trace_stat_order-1_ardiff-1')
    order=list(bank['spi_order']);aa,bb=np.triu_indices(len(order),1)
    a,b=sorted(order.index(name) for name in names);k=int(np.flatnonzero((aa==a)&(bb==b))[0])
    reference=np.asarray(bank['z'][:,k],float)
    archive=ROOT/'data/order-parameter-inference'/source.name/'observations.npz';observations=np.load(archive)
    part=scores[scores.system.eq('random-noise')&scores.role.eq('evaluation')&scores.comparison_eligible]
    i,j=np.triu_indices(24,1);edges=[];blocks=[];records=[]
    with threadpool_limits(limits=1):
        for gamma in [0.,.1,.14]:
            subset=part[np.isclose(part.control,gamma)];index=(subset.eta-.6).abs().idxmin();row=scores.loc[index]
            x=observations[row.row_id];assert x.shape==(24,1000)
            u=np.array([coint_johansen(x[[a,b]].T,0,10).max_eig_stat[0] for a,b in zip(i,j)])
            v=np.array([coint_johansen(x[[a,b]].T,1,1).trace_stat[0] for a,b in zip(i,j)])
            ub,vb=u.copy(),v.copy();z=float(np.corrcoef(u,v)[0,1])
            assert abs(z-reference[index])<1e-6
            for ga,gb in [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]:
                mask=(i//8==ga)&(j//8==gb);ub[mask]=u[mask].mean();vb[mask]=v[mask].mean()
                blocks.append(dict(row_id=row.row_id,control=gamma,eta=row.eta,category=chr(65+ga)+chr(65+gb),
                                   long_history_mean=float(u[mask].mean()),short_history_mean=float(v[mask].mean())))
            records.append(dict(row_id=row.row_id,control=gamma,eta=row.eta,Q=row.Q_lock,z=z,
                                block_z=float(np.corrcoef(ub,vb)[0,1]),replay_error=abs(z-reference[index])))
            edges.extend(dict(row_id=row.row_id,i=int(a),j=int(b),long_history=float(l),short_history=float(s)) for a,b,l,s in zip(i,j,u,v))
    pd.DataFrame(blocks).to_csv(out/'johansen-block-means.csv',index=False)
    pd.DataFrame(edges).to_csv(out/'johansen-example-edges.csv',index=False)
    audit=dict(scope='Illustration only: three held M24 random-noise records; fixed controls, eta nearest .6, no Q selection.',
               spi_pair=names,records=records,source_sha256=hashlib.sha256((source/'features.npz').read_bytes()).hexdigest(),
               observations_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (out/'johansen-mechanism-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2));return audit


def draw_grid(part,x,table,order,kind,target,out,*,at_most_one_correlation=False):
    held=part.role.eq('evaluation').to_numpy()&part.comparison_eligible.to_numpy()
    truth=part[held].groupby('control')[target].mean()
    fig,axes=plt.subplots(4,4,figsize=(8.0,9.8),constrained_layout=True)
    for number,(ax,j) in enumerate(zip(axes.flat,order[:16]),1):
        record=table.iloc[j]
        group=pd.Series(x[held,j]).groupby(part.control[held].to_numpy())
        mean=group.mean();lo=group.quantile(.1);hi=group.quantile(.9)
        ax.plot(mean.index,mean,'s-',color='#0072B2',ms=2.0,lw=1.3)
        ax.fill_between(mean.index,lo,hi,color='#0072B2',alpha=.16,lw=0)
        ax.set_box_aspect(1)
        for spine in ax.spines.values():spine.set_visible(True)
        right=ax.twinx();right.set_box_aspect(1);right.spines['right'].set_visible(True)
        right.plot(truth.index,truth,color='#222222',lw=1.1,alpha=.75)
        truth_span=max(float(truth.max()-truth.min()),1e-12)
        if target=='Q_lock':
            right.set_ylim(-.05,1.05);right.set_yticks([0,1])
        else:
            right.set_ylim(truth.min()-.05*truth_span,truth.max()+.05*truth_span)
            right.set_yticks([truth.min(),truth.max()]);right.set_yticklabels([f'{truth.min():.2g}',f'{truth.max():.2g}'])
        if (number-1)%4!=3:right.set_yticklabels([])
        else:right.set_ylabel('$Q$',fontsize=7)
        # Keep signed z values. Negative associations must visibly decline rather than be flipped.
        prefix=f'Global rank {int(record.rank_weight)}\n' if at_most_one_correlation else f'{number}. '
        title=prefix+ '\n'.join(textwrap.wrap(short_name(record.spi_a),36))+'\n× '+'\n'.join(textwrap.wrap(short_name(record.spi_b),36))
        ax.set_title(title,fontsize=6)
        ax.text(.03,.97,rf'$w={record.weight:+.3f}$'+'\n'+rf'$\rho_{{dev}}={record.rho_dev:+.2f},\ \rho_{{held}}={record.rho_held:+.2f}$',
                transform=ax.transAxes,ha='left',va='top',fontsize=5.7,bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=1))
        ax.set_xticks([0,.1,.2]);ax.tick_params(labelsize=6);right.tick_params(labelsize=6)
        if number>12:ax.set_xlabel(r'Cross-coupling $\gamma$',fontsize=7)
        if (number-1)%4==0:ax.set_ylabel('$z_{ab}$',fontsize=7)
        ax.locator_params(axis='y',nbins=4)
    criterion=r'$|w_j|$ of centered PC1' if kind=='weight' else r'$|\rho_s(z_{ab},Q)|$ on development records'
    subset='At most one classical-correlation endpoint' if at_most_one_correlation else 'no family filtering'
    fig.suptitle('Top 16 by '+criterion+'\n'+
                 f'{part.system.iloc[0]} · M={int(part.M.iloc[0])}, T={int(part["T"].iloc[0])} · '+subset+'\n'+
                 'Held means: blue feature (left), black Q (right); bands 10–90% of instances; feature scales differ',fontsize=9)
    prepare_latex_figure(fig);fig.canvas.draw();fig.canvas.draw();fig.set_layout_engine('none')
    suffix='weight-non-double-correlation' if at_most_one_correlation else kind
    for ext in ['png','svg']:fig.savefig(out/f'top16-{suffix}.{ext}',dpi=180,bbox_inches='tight')
    for ax in axes.flat:
        bbox=ax.get_window_extent();assert abs(bbox.width/bbox.height-1)<.01
    plt.close(fig)


def weight_diagnostics(part,x,w,fit,held,table,order,target):
    """Fixed-mask leave-seed-out loading stability; descriptive held checks only."""
    chosen=order[:20];folds=[];cosines=[]
    for seed in sorted(part.loc[fit,'seed'].unique()):
        train=fit & part.seed.ne(seed).to_numpy()
        other=PCA(1,svd_solver='full').fit(x[train]).components_[0]
        other*=1 if other@w>=0 else -1
        folds.append(other[chosen]);cosines.append(float(other@w))
    folds=np.asarray(folds);top=table.iloc[chosen].copy()
    top['weight_fold_min']=folds.min(0);top['weight_fold_max']=folds.max(0)
    top['same_sign_folds']=(folds*w[chosen]>0).sum(0)
    seed_rho=[]
    for seed in sorted(part.loc[held,'seed'].unique()):
        mask=held & part.seed.eq(seed).to_numpy()
        seed_rho.append(column_rho(x[mask][:,chosen],part.loc[mask,target]))
    aligned=np.asarray(seed_rho)*np.sign(top.rho_dev.to_numpy())
    top['min_held_seed_aligned_rho']=aligned.min(0)
    top['held_rho_eta']=column_rho(x[held][:,chosen],part.loc[held,'eta'])
    # Median control curves resist a few extreme recordings; not a new predictor.
    med=pd.DataFrame(x[held][:,chosen]).assign(control=part.control[held].to_numpy()).groupby('control').median()
    truth=part[held].groupby('control')[target].median().reindex(med.index)
    top['control_median_rho']=column_rho(med.to_numpy(),truth)
    return top,dict(folds=len(folds),min_loading_cosine=min(cosines),
                    top20_squared_weight_share=float((w[chosen]**2).sum()),
                    min_top20_held_seed_aligned_rho=float(aligned.min()))


def draw_weights(top,out):
    """Sorted horizontal bars: coefficient magnitude, with signed labels."""
    fig,ax=plt.subplots(figsize=(8.1,6.3),constrained_layout=True)
    y=np.arange(len(top));values=top.weight.abs().to_numpy()
    colors=np.where(top.weight.to_numpy()>0,'#0072B2','#D55E00')
    ax.barh(y,values,color=colors,height=.7)
    lo=np.minimum(abs(top.weight_fold_min),abs(top.weight_fold_max)).to_numpy()
    hi=np.maximum(abs(top.weight_fold_min),abs(top.weight_fold_max)).to_numpy()
    ax.hlines(y,lo,hi,color='#222222',lw=1)
    labels=[f'{i+1}. '+short_name(r.spi_a)+' × '+short_name(r.spi_b) for i,(_,r) in enumerate(top.iterrows())]
    ax.set_yticks(y,labels,fontsize=7);ax.invert_yaxis()
    for yi,v,w in zip(y,values,top.weight):ax.text(max(v,hi[yi])+.0005,yi,f'{w:+.4f}',va='center',fontsize=7)
    ax.set_xlim(0,max(values.max(),hi.max())*1.2)
    ax.set_xlabel(r'Absolute PC1 coefficient $|w_j|$ (unit-norm loading vector)')
    for spine in ax.spines.values():spine.set_visible(True)
    ax.set_title('Top 20 SPI–SPI PC1 weights · M=24, T=1000 · varying independent sensor noise\n'
                 'Blue: positive; orange: negative; black: leave-development-seed-out range (not CI)',fontsize=9)
    prepare_latex_figure(fig)
    for ext in ['png','svg']:fig.savefig(out/f'top20-weights.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def run(source=DEFAULT_SOURCE,out=DEFAULT_OUT,arm='random-noise',target='Q_lock'):
    out.mkdir(parents=True,exist_ok=True)
    scores=pd.read_csv(source/'scores.csv');bank=np.load(source/'features.npz')
    np.testing.assert_array_equal(scores.row_id,bank['row_id'])
    selection=scores.system.eq(arm).to_numpy() if arm else np.ones(len(scores),bool)
    part=scores[selection].copy().reset_index(drop=True);raw=bank['z'][selection]
    fit=part.role.eq('development').to_numpy();held=part.role.eq('evaluation').to_numpy()&part.comparison_eligible.to_numpy()
    q,missing,keep,w,transformed,evr=project(raw,fit,False)
    sign=1 if spearmanr(q[fit],part.control[fit]).statistic>=0 else -1
    q*=sign;w*=sign
    error=float(abs(q-part.z_PC1.to_numpy()).max());assert error<1e-4
    assert (missing[held]<=.05).all()
    x=raw[:,keep].astype(float)
    dev_rho=column_rho(x[fit],part.loc[fit,target]);held_rho=column_rho(x[held],part.loc[held,target])
    a,b=np.triu_indices(len(bank['spi_order']),1);ids=np.flatnonzero(keep)
    table=pd.DataFrame(dict(feature_index=ids,spi_a=bank['spi_order'][a[keep]],spi_b=bank['spi_order'][b[keep]],
                            weight=w,rho_dev=dev_rho,rho_held=held_rho,
                            dev_sd=np.nanstd(x[fit],axis=0),held_sd=np.nanstd(x[held],axis=0)))
    order=ranking_indices(w,dev_rho)
    for name,rank in order.items():table[f'rank_{name}']=np.argsort(rank)+1
    top,stability=weight_diagnostics(part,x,w,fit,held,table,order['weight'],target)
    top.to_csv(out/'top20-weights.csv',index=False);draw_weights(top,out)
    for name,rank in order.items():
        table.iloc[rank[:16]].to_csv(out/f'top16-{name}.csv',index=False)
        draw_grid(part,x,table,rank,name,target,out)
    both=table.spi_a.map(correlation_endpoint)&table.spi_b.map(correlation_endpoint)
    filtered=order['weight'][~both.to_numpy()[order['weight']]]
    draw_grid(part,x,table,filtered,'weight',target,out,at_most_one_correlation=True)
    table.to_csv(out/'all-feature-rankings.csv',index=False)
    correlation_composition(out)
    summary=dict(source=str(source.relative_to(ROOT)),arm=arm,target=target,M=int(part.M.iloc[0]),T=int(part['T'].iloc[0]),
                 fit=int(fit.sum()),held=int(held.sum()),eligible_coordinates=int(keep.sum()),evr=evr,
                 q_replay_error=error,original_q_held_abs_rho=float(abs(spearmanr(q[held],part.loc[held,target]).statistic)),
                 top16_overlap=int(len(set(order['weight'][:16])&set(order['rho'][:16]))),stability=stability,
                 source_features_sha256=hashlib.sha256((source/'features.npz').read_bytes()).hexdigest(),
                 source_scores_sha256=hashlib.sha256((source/'scores.csv').read_bytes()).hexdigest(),
                 selection='Only original train-complete/nonconstant eligibility; rankings on development only; no SPI family or duplicate removal.',
                 top_weight=table.iloc[order['weight'][:4]].to_dict('records'),top_rho=table.iloc[order['rho'][:4]].to_dict('records'))
    (out/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE);parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--arm',default='random-noise');parser.add_argument('--target',default='Q_lock');args=parser.parse_args()
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    latex_style()
    with threadpool_limits(limits=4):run(args.source,args.output,args.arm,args.target)
