"""Three matched, frozen-readout plots against analytic copula truth; no refitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mtsspi.figures import figure_style
from research.proof.p90.spi_baseline_exploration import sha

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/order-parameter-inference/dependence-transition-261005/normal-margin-replication'


def plot():
    rows=pd.read_csv(OUT/'replication-scores.csv')
    fit=rows.original_development.to_numpy(bool)
    held=rows.fresh_comparison_eligible.to_numpy(bool)
    assert fit.sum()==80 and held.sum()==160 and not (fit&held).any()
    assert rows.loc[held].groupby('control').size().eq(32).all()
    # Independent reconstruction of the analytic all-dyad truth, without using stored Q.
    for control,group in rows.groupby('control'):
        rho=np.repeat([.1,.2],4);nu=np.repeat([3.,30.],4)
        for j in range(round(control*4)):rho[j],rho[j+4]=rho[j+4],rho[j]
        expected=(2*t.cdf(-np.sqrt((nu+1)*(1-rho)/(1+rho)),nu+1)).sum()/120
        np.testing.assert_allclose(group.Q_tail,expected,rtol=0,atol=1e-15)
    def standard(values):
        return (values-values[fit].mean())/values[fit].std(ddof=0)
    qtruth=standard(rows.Q_tail)
    methods=[('frozen_z_PC1',r'SPI--SPI: centered PC1 $q$','#31688e'),
             ('empirical_mean_r',r'Mean signed Pearson $\bar r$','#D55E00'),
             ('frozen_mean_PC1',r'Mean-SPIs: standardized PC1 $q_\mu$','#009E73')]
    figure_style();fig,axes=plt.subplots(1,3,figsize=(11.3,3.8),sharex=True,sharey=True,layout='constrained')
    rng=np.random.default_rng(261078);records=[]
    for ax,(field,title,color) in zip(axes,methods):
        scaled=standard(rows[field]);rho=abs(spearmanr(rows.loc[held,field],rows.loc[held,'Q_tail']).statistic)
        truth=[];mean=[];low=[];high=[]
        controls=sorted(rows.control.unique())
        for control in controls:
            selected=held&rows.control.eq(control).to_numpy();v=scaled[selected].to_numpy()
            boot=v[rng.integers(len(v),size=(5000,len(v)))].mean(1)
            lo,hi=np.quantile(boot,[.025,.975]);mean.append(v.mean());low.append(lo);high.append(hi)
            truth.append(qtruth[selected].iloc[0])
            records.append(dict(method=field,control=control,n=len(v),mean=v.mean(),low=lo,high=hi,
                Q_raw=rows.loc[selected,'Q_tail'].iloc[0],Q_display=truth[-1],held_abs_rho=rho))
        ax.plot(controls,truth,'o-',color='#222222',label=r'Analytic global truth $Q$',zorder=3)
        ax.plot(controls,mean,'s-',color=color,label='Observed readout',zorder=4)
        ax.fill_between(controls,low,high,color=color,alpha=.16)
        ax.set_title(title+'\n'+rf'Held $|\rho(\mathrm{{readout}},Q)|={rho:.3f}$')
        ax.set_xticks(controls);ax.set_xlabel(r'Assignment swap fraction $c$');ax.grid(axis='y',alpha=.12)
        ax.legend(fontsize=7,loc='upper left')
    axes[0].set_ylabel('Value (original-training SD units)')
    fig.suptitle(r'Gaussian-margin Student-$t$ copula $\cdot\ M=N=16,\ T=1000$'+'\n'+
        r'Frozen PCs: 80 training records; 160 fresh held records; bands: 95\% bootstrap intervals of held means',fontsize=10)
    for ext in ['png','svg']:fig.savefig(OUT/f'control-Q-readouts.{ext}',dpi=180)
    plt.close(fig)
    pd.DataFrame(records).to_csv(OUT/'control-Q-readouts.csv',index=False)
    (OUT/'control-Q-readouts.json').write_text(json.dumps(dict(source_sha256=sha(__file__),
        scores_sha256=sha(OUT/'replication-scores.csv'),training=80,held=160,
        qualification='Presentation only, no model refit. Each displayed quantity independently centered/scaled using original training80; not calibrated predictions of Q. PC signs fixed from training in original analysis; Pearson sign unchanged. Analytic Q independently reconstructed. Means baseline is target-blind PC1 of the per-SPI mean vector, not every possible readout.'),indent=2)+'\n')


if __name__=='__main__':plot()
