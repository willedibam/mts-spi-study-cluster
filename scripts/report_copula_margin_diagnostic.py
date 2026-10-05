"""Codex review/figure for Claude's completed, separately owned probe scout."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scripts import diagnose_tail_copula_margins as d
from scripts.spi_baseline_exploration import sha


def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.plot_large_m_pair_sampling import figure_style
    out=d.OUT.parent;rows=pd.read_csv(d.OUT/'rows.csv');bank=np.load(d.OUT/'features.npz')
    metadata=json.loads((d.OUT/'report.json').read_text())
    assert metadata['source_sha256']==sha(d.__file__)
    replay=[]
    for arm in d.ARMS:
        i=rows.index[(rows.arm==arm)&(rows.swaps==2)&(rows.seed==20)][0]
        expected,z=d.estimate((arm,2,20))
        for key,value in expected.items():
            if isinstance(value,(float,np.floating)):np.testing.assert_allclose(rows.loc[i,key],value,atol=1e-12)
            else:assert rows.loc[i,key]==value
        np.testing.assert_allclose(z,bank['z'][i],atol=1e-12)
        replay.append(dict(arm=arm,swaps=2,seed=20))
    figure_style();fig,axes=plt.subplots(1,3,figsize=(11,3.7),constrained_layout=True)
    labels=['t copula / t margins','t copula / Gaussian margins','Gaussian copula / t margins']
    methods=[('z_PC1','SPI--SPI PC1','#0072B2'),('mean_PC1','Four-mean PC1','#D55E00'),
             ('mean_Pearson','Mean Pearson','#777777'),('robust_cov_sq_mean','Mean robust covariance squared','#CC79A7')]
    rng=np.random.default_rng(261076)
    for ax,arm,title in zip(axes,d.ARMS,labels):
        p=rows[rows.arm==arm];fit=p.role.eq('fit');held=~fit
        for field,label,color in methods:
            q=p[field];sign=1 if spearmanr(q[fit],p.loc[fit,'swaps']).statistic>=0 else -1
            scaled=sign*(q-q[fit].mean())/q[fit].std(ddof=0)
            mean=[];low=[];high=[]
            for level in range(5):
                v=scaled[held&(p.swaps==level)].to_numpy()
                boot=v[rng.integers(len(v),size=(2000,len(v)))].mean(1)
                mean.append(v.mean());lo,hi=np.quantile(boot,[.025,.975]);low.append(lo);high.append(hi)
            ax.plot(np.arange(5)/4,mean,'o-',color=color,label=label)
            ax.fill_between(np.arange(5)/4,low,high,color=color,alpha=.12)
        ax.set(title=title,xlabel='Assignment fraction',ylabel='Readout, training SD units');ax.grid(axis='y',alpha=.12)
    axes[0].legend(fontsize=6)
    fig.suptitle(r'Copula/margin diagnostic: M=N=16, T=1000; focused probes, not p90'+'\n'+r'Independent records; bands: 95\% bootstrap intervals of held means; signs from training only',fontsize=10)
    for ext in ['png','svg']:fig.savefig(out/f'copula-margin-diagnostic.{ext}',dpi=180)
    plt.close(fig)
    (out/'copula-margin-review.json').write_text(json.dumps(dict(records=len(rows),replayed_records=replay,
        generator_sha256=sha(d.__file__),review_source_sha256=sha(__file__),
        qualification='Source/tests inspected; three exact feature replays pass. Intervals are exploratory, unadjusted. Four-probe means are not full-p90 means. Population Pearson matching is approximate after PIT.'),indent=2)+'\n')


if __name__=='__main__':run()
