"""Frozen-coordinate galleries and original-edge explanations; no model fitting."""
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mtsspi.spi_spi_contract import build_unified_feature_values
from research.inference.benchmarks.three_systems import ROOT, OUT, check, sha, save, curve, heatmaps, dual, Z_COLOR
from research.inference.benchmarks.spi_feature_gallery import correlation_endpoint, short_name as catalogue_name


def short_name(name):
    aliases={'pec_orth':'Envelope corr. (orth.)','pec_orth_abs':'Absolute envelope corr. (orth.)',
             'pec_orth_log':'Log-envelope corr. (orth.)','pec_orth_log_abs':'Absolute log-envelope corr. (orth.)',
             'pec':'Envelope correlation','dcorr':'Distance correlation','lcss':'LCSS similarity',
             'cov-sq_EmpiricalCovariance':'Squared empirical covariance',
             'si_kernel_W-0.5_k-1':'Stochastic interaction (kernel)',
             'ids':'Interdependence score'}
    return aliases.get(name,catalogue_name(name))


def ranking(order, indices, weights):
    a,b=np.triu_indices(len(order),1)
    table=pd.DataFrame(dict(feature_index=indices,spi_a=np.asarray(order)[a[indices]],
                            spi_b=np.asarray(order)[b[indices]],weight=weights))
    table=table.sort_values('weight',key=abs,ascending=False,kind='stable').reset_index(drop=True)
    table.insert(0,'rank',np.arange(1,len(table)+1))
    table['both_correlation']=[correlation_endpoint(a) and correlation_endpoint(b) for a,b in zip(table.spi_a,table.spi_b)]
    return table


def load_quadratic():
    source=ROOT/'results/baseline-comparison_260921/inference-KanekoBand.csv'
    frame=pd.read_csv(source).query('evaluation').reset_index(drop=True)
    check(len(frame)==164 and frame.M.eq(32).all() and frame.common_eligible.sum()==140,'Quadratic scope changed')
    inventory=json.loads((ROOT/'data/baseline-comparison_260921/legacy-feature-inventory.json').read_text())
    order=inventory['quadratic_cml_development_full_p90']['spi_order']
    records=json.loads((ROOT/'data/baseline-comparison_260921/extension-inputs.json').read_text())['records']
    records={r['dataset_path']:r for r in records if r.get('bank')=='quadratic_cml_development_full_p90'}
    paths=[];zs=[]
    for row in frame.itertuples():
        record=records[row.row_id]
        path=ROOT/record['mpi_path'].replace('spi_baseline_exploration_260921','baseline-comparison_260921')
        check(sha(path)==record['mpi_sha256'],f'Quadratic MPI changed: {path}')
        with np.load(path) as a:
            zs.append(build_unified_feature_values({n:a[n] for n in order},order)[0])
        paths.append(path)
    z=np.asarray(zs)
    modelpath=ROOT/'results/order-parameter-inference/quadratic-cml/quadratic_cml_development_analysis/fitted_target_blind_model.npz'
    with np.load(modelpath) as m:
        selected=z[:,m['keep_indices']]
        replay=((np.where(np.isfinite(selected),selected,m['impute_values'])-m['center'])/m['block_scale'])@m['q_components'][0]
        error=float(np.max(abs(replay-frame.q)))
        check(error<1e-5,f'Quadratic frozen score replay failed: {error}')
        ranks=ranking(order,m['keep_indices'],m['q_components'][0])
    examples=[]
    for value in (1.6,1.75,2.):
        with np.load(ROOT/f'data/order-parameter-inference/figure-examples_260916/quadratic-cml-{value:g}.npz') as a:
            examples.append(dict(control=value,x=a['X'],field=a['field']))
    return dict(frame=frame,z=z,ranks=ranks,paths=paths,examples=examples,control='control',
                label='Quadratic CML',slug='quadratic-cml',xlabel=r'Local nonlinearity $\alpha$',
                score_sha256=sha(source),model_sha256=sha(modelpath),replay_error=error,
                rho=float(spearmanr(frame.q,frame.Q_reference).statistic))


def draw_quadratic(data):
    fig,axes=plt.subplots(3,3,figsize=(10,7.2),layout='constrained',gridspec_kw={'height_ratios':[1,1,1.15]})
    for ax,e in zip(axes[0],data['examples']):
        image=ax.imshow(e['field'],origin='lower',aspect='auto',cmap='viridis',vmin=-1,vmax=1,extent=(0,100,1,512),rasterized=True)
        ax.set(title=rf"$\alpha={e['control']:g}$",xlabel='Consecutive step',ylabel='All lattice sites')
    fig.colorbar(image,ax=list(axes[0]),shrink=.7,label='Map state $x$')
    heatmaps(fig,axes[1],data['examples'],count=100,dt=1)
    for ax,key,label in zip(axes[2],['q','mean_PC1','mean_abs_correlation'],['SPI–SPI PC1','Mean-MPI PC1',r'Mean $|r_{ij}|$']):
        dual(ax,data['frame'].query('common_eligible'),'control','Q_reference',key,xlabel=data['xlabel'],truthlabel='Spatial band power',scorelabel=label)
        ax.set_title(label)
    return save(fig,'quadratic-cml')


def feature_banks(cml,locking,hardware,quadratic):
    cpath=ROOT/'data/order-parameter-inference/cml2d/cml2d_confirmation_260911/primary-analysis'
    with np.load(cpath/'features.npz') as f,np.load(cpath/'model.npz') as m:
        c=dict(frame=cml['frame'].copy(),z=f['z'],ranks=ranking(f['spi_order'],m['keep'],m['component']),
               control='r',label='Collective period-two CML',slug='cml2d',xlabel='Local nonlinearity $r$')
    lpath=ROOT/'results/order-parameter-inference/cross-frequency-locking-snr-261006'
    allrows=pd.read_csv(lpath/'scores.csv');mask=allrows.system.eq('random-noise')&allrows.role.eq('evaluation')
    with np.load(lpath/'features.npz') as f:
        np.testing.assert_array_equal(f['row_id'],allrows.row_id)
        l=dict(frame=allrows[mask].reset_index(drop=True),z=f['z'][mask],control='control',label='Noisy 2:1 locking',
               slug='noisy-locking',xlabel=r'Cross-coupling $\gamma$')
    ranks=pd.read_csv(ROOT/'results/order-parameter-inference/locking-features-m24-261009/all-feature-rankings.csv')
    l['ranks']=ranks.rename(columns={'rank_weight':'rank'}).sort_values('rank').copy()
    l['ranks']['both_correlation']=l['ranks'].apply(lambda r:correlation_endpoint(r.spi_a) and correlation_endpoint(r.spi_b),axis=1)
    # Verify the archived loading coefficients reproduce the original coordinate.
    with np.load(lpath/'features.npz') as f:
        arm=allrows.system.eq('random-noise').to_numpy();fit=allrows.loc[arm,'role'].eq('development').to_numpy()
        selected=f['z'][arm][:,l['ranks'].feature_index.to_numpy()].astype(float)
        selected=np.where(np.isfinite(selected),selected,np.nanmedian(selected[fit],axis=0))
        replay=(selected-selected[fit].mean(0))@l['ranks'].weight.to_numpy()
        np.testing.assert_allclose(replay,allrows.loc[arm,'z_PC1'],atol=1e-6)
    hpath=ROOT/'results/hardware-pilot-261012/full-p90-r9-261014/analysis'
    with np.load(hpath/'features.npz') as f:
        h=dict(frame=hardware['frame'].copy(),z=f['z'],ranks=ranking(f['spi_order'],f['er_features'],f['er_components'][0]),
               control='voltage',label='Hardware (ER-fitted)',slug='hardware',xlabel='Coupling voltage (V)')
    return dict(cml2d=c,quadratic=quadratic,locking=l,hardware=h)


def selections(data):
    """Second display excludes double correlations and numerically duplicate curves.

    Duplicate test is display-only, on the shown rows, without Q. It does not
    establish global equivalence or modify the original feature basis/PCA.
    """
    ranks=data['ranks'];full=ranks.head(16);kept=[]
    for r in ranks.itertuples():
        if r.both_correlation:continue
        if any(np.allclose(data['z'][:,r.feature_index],data['z'][:,j],atol=1e-7,rtol=0,equal_nan=True) for j in kept):continue
        kept.append(r.feature_index)
        if len(kept)==16:break
    return full,ranks.set_index('feature_index').loc[kept].reset_index()


def gallery(data,filtered=False):
    full,reduced=selections(data);table=reduced if filtered else full
    OUT.mkdir(parents=True,exist_ok=True)
    data['ranks'].to_csv(OUT/f"{data['slug']}-loadings.csv",index=False)
    if filtered and table.feature_index.tolist()==full.feature_index.tolist():
        print(f"{data['label']}: filtering leaves the top 16 unchanged; duplicate panel omitted.")
        return None
    fig,axes=plt.subplots(4,4,figsize=(9,10.5),layout='constrained')
    for ax,row in zip(axes.flat,table.itertuples()):
        f=data['frame'].copy();f['value']=data['z'][:,row.feature_index]
        groups=[('ER',Z_COLOR),('SW','#d95f02')] if 'topology' in f else [(None,Z_COLOR)]
        for top,color in groups:
            part=f[f.topology.eq(top)] if top else f
            mean,lo,hi=curve(part,data['control'],'value')
            ax.plot(mean.index,mean,color=color,marker='o',ms=2,lw=1)
            if not top:ax.fill_between(mean.index,lo,hi,color=color,alpha=.14,lw=0)
        label='\n× '.join(textwrap.fill(short_name(s),27) for s in (row.spi_a,row.spi_b))
        ax.set_title(f'#{row.rank}  w={row.weight:+.4f}\n{label}',fontsize=7)
        ax.set(xlabel=data['xlabel'],ylabel=r'$z_{ab}$',ylim=(-1.05,1.05));ax.set_box_aspect(1)
        ax.tick_params(labelsize=6);ax.xaxis.label.set_size(7);ax.yaxis.label.set_size(7)
        ax.locator_params(axis='both',nbins=3)
        if 'topology' in f:
            ax.set_xscale('symlog',linthresh=.2,linscale=.75)
            ax.set_xticks([0,.1,.2,1,3.5],['0','.1','.2','1','3.5'])
        for spine in ax.spines.values():spine.set_visible(True)
    title='Top 16 by absolute PC1 loading' if not filtered else 'Top 16 after display filters; original ranks and weights'
    fig.suptitle(data['label']+' · '+title,fontsize=10)
    name=data['slug']+('-loadings-filtered' if filtered else '-loadings-top16')
    table.to_csv(OUT/f'{name}.csv',index=False)
    return save(fig,name)


def edge_sources(data,key,hardware=None):
    if key=='cml2d':
        root=ROOT/'data/order-parameter-inference/cml2d/cml2d_confirmation_260911'
        source={r['row_id']:r for r in json.loads((root/'primary-analysis/eligibility.json').read_text())['sources']}
        frame=data['frame'];seed=frame.seed.min();rows=frame[frame.seed.eq(seed)].sort_values('r').iloc[[0,8,-1]]
        result=[]
        for index,row in rows.iterrows():
            slug=row.row_id.replace('.','-');matches=list((root/'primary/mpi/primary').glob(f'*-{slug}/spi_mpis.npz'))
            check(len(matches)==1,'CML edge source ambiguous');path=matches[0]
            check(sha(path)==source[row.row_id]['mpi_sha256'],'CML edge archive changed')
            result.append((index,path,rf'$r={row.r:g}$',row.row_id))
        return result
    if key=='quadratic':
        frame=data['frame'];rows=frame[frame.instance.eq(4)&frame.control.isin([1.6,1.75,2.])].sort_values('control')
        return [(i,data['paths'][i],rf'$\alpha={r.control:g}$',r.row_id) for i,r in rows.iterrows()]
    if key=='locking':
        from research.inference.locking.plot_locking_m24_edge_profiles import choose_records
        rows=choose_records(pd.read_csv(ROOT/'results/order-parameter-inference/cross-frequency-locking-snr-261006/scores.csv'))
        sources={r['row_id']:r for r in json.loads((ROOT/'results/order-parameter-inference/cross-frequency-locking-snr-261006/sources.json').read_text())['sources']}
        result=[]
        for row in rows:
            path=ROOT/'results/order-parameter-inference/locking-features-m24-261009/mpi-examples'/f"{row.row_id.removeprefix('xfreq21-random-noise-')}.npz"
            check(sha(path)==sources[row.row_id]['mpi_sha256'],'Locking MPI changed')
            index=int(np.flatnonzero(data['frame'].row_id.eq(row.row_id))[0])
            result.append((index,path,rf'$\gamma={row.control:.2f}$',row.row_id))
        return result
    bank=hardware['bank'];keys=[('ER',23),('ER',24),('SW',31),('SW',34)]
    lookup={(r['topology'],r['step']):i for i,r in enumerate(bank.rows)}
    return [(lookup[k],bank.paths[lookup[k]],f'{k[0]} setting {k[1]}',bank.rows[lookup[k]]['name']) for k in keys]


PAIRS={
    'cml2d':[('pec_orth','pec_orth_abs'),('corr_pearson_tau-5','pec_orth_abs')],
    'quadratic':[('corr_spearman_tau-2-sq','corr_spearman_tau-4-sq'),('cov-sq_EmpiricalCovariance','dcorr')],
    'locking':[('corr_pearson_tau-2','corr_pearson_tau-20-sq'),('lcss','te_kraskov_NN-4_DCE-AUTO_k-1_kt-1_l-1_lt-1')],
    'hardware':[('si_kernel_W-0.5_k-1','ids'),('corr_pearson_tau-1','corr_pearson_tau-10')],
}


def edges(data,key,hardware=None):
    sources=edge_sources(data,key,hardware);fig,axes=plt.subplots(2,len(sources),figsize=(9,5.2 if key=='hardware' else 6.4),layout='constrained',squeeze=False)
    receipts=[];groups=None
    if key=='hardware':
        from research.empirical.hardware_pilot.full_p90_analysis import carrier_groups
        groups=carrier_groups(hardware['bank'])
    for ir,(a,b) in enumerate(PAIRS[key]):
        table=data['ranks'];match=table[(table.spi_a.eq(a)&table.spi_b.eq(b))|(table.spi_a.eq(b)&table.spi_b.eq(a))]
        check(len(match)==1,'Edge pair missing from frozen PC1');row=match.iloc[0];profiles=[]
        for ax,(index,path,title,rowid) in zip(axes[ir],sources):
            with np.load(path) as m:
                mask=~np.eye(m[a].shape[0],dtype=bool);x,y=m[a][mask],m[b][mask]
            check(np.isfinite(x).all() and np.isfinite(y).all(),'Nonfinite edge profile')
            z=float(np.corrcoef(x,y)[0,1]);expected=float(data['z'][index,int(row.feature_index)])
            check(abs(z-expected)<1e-6,'Original edge replay failed')
            profiles.append((x,y))
            if key=='locking':
                from research.inference.locking.plot_locking_m24_edge_profiles import community_masks,GROUPS
                ii,jj=np.where(mask)
                for selected,(label,color,marker) in zip(community_masks(ii,jj),GROUPS):
                    ax.scatter(x[selected],y[selected],s=5,color=color,marker=marker,alpha=.3,lw=0,rasterized=True)
                    ax.scatter(x[selected].mean(),y[selected].mean(),s=24,color=color,marker=marker,edgecolors='k',lw=.4)
            elif groups is not None:
                coherent=groups[index]
                for selected,color in [(~coherent,Z_COLOR),(coherent,'#222222')]:
                    ax.scatter(x[selected],y[selected],s=5,color=color,alpha=.4,lw=0,rasterized=True)
            else:ax.scatter(x,y,s=5,color=Z_COLOR,alpha=.3,lw=0,rasterized=True)
            ax.set(title=title+f' · #{int(row["rank"])}\n'+rf'$z={z:+.3f}$',xlabel=textwrap.fill(short_name(a),29),ylabel=textwrap.fill(short_name(b),29) if ax is axes[ir,0] else '')
            ax.set_box_aspect(1);ax.locator_params(axis='both',nbins=3)
            ax.tick_params(labelsize=7);ax.title.set_size(8);ax.xaxis.label.set_size(8);ax.yaxis.label.set_size(8)
            for spine in ax.spines.values():spine.set_visible(True)
            receipts.append(dict(system=key,row_id=rowid,spi_a=a,spi_b=b,rank=int(row['rank']),z=z,stored_z=expected,error=abs(z-expected),sd_a=float(x.std()),sd_b=float(y.std()),mpi_sha256=sha(path)))
        for dim,setter in [(0,'set_xlim'),(1,'set_ylim')]:
            vals=np.concatenate([p[dim] for p in profiles]);pad=max(float(np.ptp(vals))*.05,1e-9)
            for ax in axes[ir]:getattr(ax,setter)(vals.min()-pad,vals.max()+pad)
    if key=='locking':
        from matplotlib.lines import Line2D
        handles=[Line2D([],[],marker=m,color=c,ls='',label=g,ms=4) for g,c,m in GROUPS]
        fig.legend(handles=handles,loc='outside lower center',ncol=6,fontsize=7)
    fig.suptitle(data['label']+' · same ordered channel pairs; shared limits within each row',fontsize=10)
    pd.DataFrame(receipts).to_csv(OUT/f"{data['slug']}-edge-replay.csv",index=False)
    return save(fig,data['slug']+'-edge-profiles')
