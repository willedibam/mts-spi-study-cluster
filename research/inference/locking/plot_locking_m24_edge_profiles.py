"""Matched M24 noisy-record edge profiles from hash-verified archived MPI matrices."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scripts.benchmark_latex_style import latex_style,prepare_latex_figure
from research.inference.benchmarks.spi_feature_gallery import DEFAULT_SOURCE,DEFAULT_OUT

GROUPS=[('A–A','#0072B2','o'),('B–B','#D55E00','^'),('C–C','#009E73','s'),
        ('A–B','#CC79A7','D'),('A–C','#7B3294','P'),('B–C','#666666','X')]
PAIRS=[
    ('mechanisms','coint_johansen_max_eig_stat_order-0_ardiff-10','coint_johansen_trace_stat_order-1_ardiff-1',
     'Johansen max-eigen (d=0, l=10)','Johansen trace (d=1, l=1)'),
    ('mechanisms','lcss','te_kraskov_NN-4_DCE-AUTO_k-1_kt-1_l-1_lt-1',
     'LCSS similarity','Transfer entropy (KSG, DCE-auto)'),
    ('mechanisms','corr_pearson_tau-2','corr_pearson_tau-20-sq','Lag-2 Pearson','Squared lag-20 Pearson'),
    ('information','corr_kendall_tau-20-sq','xme_kernel_W-0.5_k1','Squared lag-20 Kendall','One-step conditional entropy (kernel)'),
    ('information','corr_kendall_tau-20-sq','tlmi_gaussian','Squared lag-20 Kendall','One-step mutual information (Gaussian)'),
]


def choose_records(scores):
    held=scores[scores.system.eq('random-noise')&scores.role.eq('evaluation')&scores.comparison_eligible]
    return [held[np.isclose(held.control,g)].sort_values('row_id').iloc[
        np.argmin(abs(held[np.isclose(held.control,g)].sort_values('row_id').eta.to_numpy()-.6))]
        for g in [0.,.10,.14]]


def community_masks(i,j,n=8):
    a,b=np.minimum(i//n,j//n),np.maximum(i//n,j//n)
    return [(a==g)&(b==h) for g,h in [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]]


def full_limits(arrays):
    values=np.concatenate(arrays);low,high=float(values.min()),float(values.max())
    pad=.06*max(high-low,1e-6)
    return low-pad,high+pad


def run():
    scores=pd.read_csv(DEFAULT_SOURCE/'scores.csv');bank=np.load(DEFAULT_SOURCE/'features.npz')
    np.testing.assert_array_equal(scores.row_id,bank['row_id']);zbank=bank['z']
    names=list(bank['spi_order']);aa,bb=np.triu_indices(len(names),1)
    ranks=pd.read_csv(DEFAULT_OUT/'all-feature-rankings.csv').set_index('feature_index')
    sources={r['row_id']:r for r in json.loads((DEFAULT_SOURCE/'sources.json').read_text())['sources']}
    chosen=choose_records(scores);matrices=[];receipts=[]
    for row in chosen:
        suffix=row.row_id.removeprefix('xfreq21-random-noise-')
        path=DEFAULT_OUT/'mpi-examples'/f'{suffix}.npz';digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==sources[row.row_id]['mpi_sha256'],path
        matrices.append(np.load(path))
        receipts.append(dict(row_id=row.row_id,control=row.control,eta=row.eta,Q=row.Q_lock,mpi_sha256=digest))
    i,j=np.where(~np.eye(24,dtype=bool));masks=community_masks(i,j)
    assert np.array_equal(np.sum(masks,axis=0),np.ones(len(i)))
    records=[];edges=[];blocks=[];selected={}
    for family,a,b,xlabel,ylabel in PAIRS:
        ia,ib=sorted([names.index(a),names.index(b)])
        k=int(np.flatnonzero((aa==ia)&(bb==ib))[0]);global_rank=int(ranks.loc[k,'rank_weight'])
        views=[]
        for row,mpis in zip(chosen,matrices):
            x,y=mpis[a][i,j],mpis[b][i,j]
            assert np.isfinite(x).all() and np.isfinite(y).all()
            z=float(np.corrcoef(x,y)[0,1]);expected=float(zbank[int(row.name),k])
            assert abs(z-expected)<1e-6,(a,b,row.row_id,z,expected)
            records.append(dict(row_id=row.row_id,feature_index=k,spi_a=a,spi_b=b,global_rank=global_rank,
                                z=z,stored_z=expected,error=abs(z-expected)))
            views.append((x,y,z))
            for mask,(label,_,_) in zip(masks,GROUPS):
                blocks.append(dict(row_id=row.row_id,control=row.control,feature_index=k,category=label,
                                   mean_a=float(x[mask].mean()),mean_b=float(y[mask].mean())))
                edges.extend(dict(row_id=row.row_id,feature_index=k,i=int(u),j=int(v),category=label,
                                  spi_a_value=float(xv),spi_b_value=float(yv)) for u,v,xv,yv in zip(i[mask],j[mask],x[mask],y[mask]))
        selected.setdefault(family,[]).append((global_rank,xlabel,ylabel,views))
    plt.rcParams.update({'font.size':7,'axes.titlesize':7,'axes.labelsize':7,'xtick.labelsize':6,
                         'ytick.labelsize':6,'legend.frameon':False})
    latex_style()
    for family,panels in selected.items():
        fig,axes=plt.subplots(len(panels),3,figsize=(8.2,3.05*len(panels)+.65),sharex='row',sharey='row',layout='constrained')
        for r,(rank,xlabel,ylabel,views) in enumerate(panels):
            xlim=full_limits([v[0] for v in views]);ylim=full_limits([v[1] for v in views])
            for c,((x,y,z),row) in enumerate(zip(views,chosen)):
                ax=axes[r,c];ax.set_box_aspect(1)
                for mask,(label,color,marker) in zip(masks,GROUPS):
                    ax.scatter(x[mask],y[mask],s=6,c=color,marker=marker,alpha=.32,linewidths=0,rasterized=True)
                    ax.scatter(x[mask].mean(),y[mask].mean(),s=25,c=color,marker=marker,edgecolors='black',linewidths=.4,zorder=4)
                phase=['Before','Near boundary','After'][c]
                ax.set(title=rf'{phase}: $\gamma={row.control:.2f}$; rank {rank}'+'\n'+rf'$\eta={row.eta:.2f},\ z_{{ab}}={z:.2f}$',
                       xlabel=xlabel,ylabel=ylabel if c==0 else '',xlim=xlim,ylim=ylim)
                ax.locator_params(axis='both',nbins=4)
                for spine in ax.spines.values():spine.set_visible(True)
        handles=[Line2D([],[],marker=m,ls='',color=c,ms=4,label=g) for g,c,m in GROUPS]
        fig.legend(handles=handles,loc='outside lower center',ncol=6,fontsize=7)
        fig.suptitle('Varying independent sensor noise · M=24, T=1000 · same records in every row\n'
                     'Points: 552 directed edges (symmetric entries overlap); outlined markers: community-pair means',fontsize=8)
        prepare_latex_figure(fig);fig.canvas.draw();fig.set_layout_engine('none')
        for ax in axes.flat:
            box=ax.get_window_extent();assert abs(box.width/box.height-1)<.01
        for ext in ['png','svg']:fig.savefig(DEFAULT_OUT/f'edge-profiles-{family}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    pd.DataFrame(records).to_csv(DEFAULT_OUT/'edge-profile-replay.csv',index=False)
    pd.DataFrame(edges).to_csv(DEFAULT_OUT/'edge-profile-values.csv',index=False)
    pd.DataFrame(blocks).to_csv(DEFAULT_OUT/'edge-profile-blocks.csv',index=False)
    audit=dict(selection='Fixed gamma 0/.10/.14; held record nearest eta=.6, ties by row ID; not selected by Q or z.',
               records=receipts,pairs=[dict(spi_a=a,spi_b=b) for _,a,b,_,_ in PAIRS],
               ordered_edges_per_panel=len(i),panels=len(records),max_replay_error=max(r['error'] for r in records),
               features_sha256=hashlib.sha256((DEFAULT_SOURCE/'features.npz').read_bytes()).hexdigest(),
               note='Illustrative three-record profiles, not population uncertainty or causal inference; per-row common limits include every point.')
    (DEFAULT_OUT/'edge-profile-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':run()
