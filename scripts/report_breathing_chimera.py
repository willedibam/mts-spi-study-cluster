"""Finite-size physics plots and expressly exploratory eleven-probe readouts."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from scripts.scout_breathing_chimera import OUT
from scripts.scout_dependence_transition import probes
from scripts.report_dependence_transition import coordinate
from scripts.spi_baseline_exploration import sha


def report(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.plot_large_m_pair_sampling import figure_style
    rows=pd.read_csv(folder/'rows.csv');oa=pd.read_csv(folder/'continuum.csv')
    boundary=json.loads((folder/'report.json').read_text())['hopf'][0]
    figure_style();fig,axes=plt.subplots(2,3,figsize=(11,6),constrained_layout=True)
    colors=dict(zip([16,32,64,256],['#440154','#31688e','#35b779','#c5bd26']))
    for ri,kind in enumerate(['stratified','random']):
        for M,color in colors.items():
            part=rows[(rows.M==M)&(rows.initialization==kind)]
            for ci,field in enumerate(['Q_breathing','mean_r2','mean_Pearson']):
                g=part.groupby('A')[field];mid=g.mean()
                axes[ri,ci].plot(mid.index,mid,'o-',color=color,label=f'N={M}')
                axes[ri,ci].fill_between(mid.index,g.quantile(.1),g.quantile(.9),color=color,alpha=.12)
        for ci,field in enumerate(['Q_breathing','mean_r2']):
            axes[ri,ci].plot(oa.A,oa[field],color='.2',ls='--',label='Continuum OA')
        for ci,label in enumerate(['Future breathing amplitude: SD(r2)','Future mean coherence r2','Observed mean signed Pearson']):
            ax=axes[ri,ci];ax.set(ylabel=label,xlabel='Coupling disparity A',title=kind+' initial phases')
            ax.axvline(boundary,color='.5',ls=':',lw=.8);ax.grid(axis='y',alpha=.12)
        axes[ri,0].legend(fontsize=6,ncol=2)
    fig.suptitle(r'Breathing-chimera physics only; four seeds per curve, 10--90\% instance spread'+'\nContinuum Hopf line is not an exact finite-N threshold',fontsize=10)
    for ext in ['png','svg']:fig.savefig(folder/f'physics.{ext}',dpi=180)
    plt.close(fig)
    subset=rows[rows.M<=32].copy();mean=[];z=[]
    with np.load(folder/'observations.npz') as raw:
        for key in subset.row_id:
            m,v,names=probes(raw[key]);mean.append(m);z.append(v)
    mean=np.array(mean);z=np.array(z);metrics=[];scores=[]
    for (M,kind),part in subset.reset_index(drop=True).groupby(['M','initialization']):
        ix=part.index.to_numpy();fit=part.seed.to_numpy()<2;held=~fit;y=part.Q_breathing.to_numpy();s=part.copy()
        for label,x,standard in [('z_PC1',z[ix],False),('z_standard_PC1',z[ix],True),('mean_PC1',mean[ix],True)]:
            q,evr,missing=coordinate(x,fit,standard);s[label]=q
            metrics.append(dict(M=M,initialization=kind,method=label,
                held_abs_rho=abs(spearmanr(q[held],y[held]).statistic),evr=evr,max_missing=missing[held].max()))
        ranked=[abs(spearmanr(mean[ix][fit,j],y[fit]).statistic) for j in range(len(names))]
        best=int(np.nanargmax(ranked))
        for label,v in [('mean_Pearson',part.mean_Pearson.to_numpy()),('selected_mean',mean[ix,best])]:
            metrics.append(dict(M=M,initialization=kind,method=label,
                held_abs_rho=abs(spearmanr(v[held],y[held]).statistic),selected=names[best] if label=='selected_mean' else ''))
        scores.append(s)
    pd.DataFrame(metrics).to_csv(folder/'probe-metrics.csv',index=False)
    pd.concat(scores).to_csv(folder/'probe-scores.csv',index=False)
    np.savez_compressed(folder/'probes.npz',mean=mean,z=z,names=names,row_id=subset.row_id.to_numpy())
    (folder/'probe-analysis.json').write_text(json.dumps(dict(source_sha256=sha(__file__),
        rows_sha256=sha(folder/'rows.csv'),scope='Eleven probes only, not p90. Two training/two exploratory validation seeds, separate size/start families. No confirmation or all-mean-information absence.'),indent=2)+'\n')
    print(pd.DataFrame(metrics).round(3).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,default=OUT);a=p.parse_args()
    with threadpool_limits(limits=4):report(a.folder)
