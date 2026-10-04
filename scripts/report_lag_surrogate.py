"""Lean report for the prospective circular-second-order matched experiment."""
import argparse,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import nbformat as nbf
from sklearn.decomposition import PCA
from umap import UMAP
from threadpoolctl import threadpool_limits
from scripts.build_lag_surrogate import DATA,OUT,ROOT,RUN
from scripts.refresh_case_figures import style,save

COLORS=['#0072B2','#D55E00']
LABELS=['Alternating lag','Shared-phase surrogate']


def mechanism():
    style();t=100;lag=5;a=.7
    perm=np.roll(np.arange(t).reshape(-1,2*lag),lag,axis=1).ravel()
    K=a*np.eye(t)[:,perm]
    rng=np.random.default_rng(261017);phase=np.exp(1j*rng.uniform(-np.pi,np.pi,t//2+1));phase[0]=phase[-1]=1
    Q=np.fft.irfft(np.fft.rfft(np.eye(t))*phase,n=t)
    H=Q.T@K@Q
    np.testing.assert_allclose(Q.T@Q,np.eye(t),atol=1e-12)
    np.testing.assert_allclose(np.linalg.svd(K,compute_uv=False),np.linalg.svd(H,compute_uv=False),atol=1e-12)
    lags=np.arange(-12,13)
    first=np.array([np.diag(np.roll(K,k,axis=1)).mean() for k in lags]);second=np.array([np.diag(np.roll(H,k,axis=1)).mean() for k in lags])
    np.testing.assert_allclose(first,second,atol=1e-12)
    fig,axes=plt.subplots(1,3,figsize=(10.2,3.8),layout='constrained')
    for ax,matrix,title in zip(axes[:2],[K,H],LABELS):
        im=ax.imshow(matrix,cmap='RdBu_r',vmin=-a,vmax=a,origin='lower',interpolation='nearest')
        ax.set(title=title,xlabel='Time in channel y',ylabel='Time in channel x')
    fig.colorbar(im,ax=axes[:2],shrink=.75,label='Cross-time covariance')
    axes[2].plot(lags,first,color=COLORS[0],label=LABELS[0]);axes[2].plot(lags,second,'--',color=COLORS[1],label=LABELS[1]);axes[2].set(title='Same time-averaged lag covariance',xlabel='Circular lag',ylabel='Covariance');axes[2].set_box_aspect(1);axes[2].legend(fontsize=7)
    save(fig,OUT/'figures','mechanism')
    (OUT/'mechanism-audit.json').write_text(json.dumps(dict(T=t,loading=a,lag=lag,max_lag_profile_difference=float(abs(first-second).max()),operator_singular_value_error=float(abs(np.linalg.svd(K,compute_uv=False)-np.linalg.svd(H,compute_uv=False)).max()),qualification='Population illustration at T100; p90 experiment uses T1000 and eight pairs. Temporal localization changes, singular strengths and circular lag averages do not.'),indent=2)+'\n')


def plots(stage):
    target=OUT/stage;style()
    manifest=pd.DataFrame(json.loads((DATA/'manifest.json').read_text())['rows'])
    test=manifest[(manifest.panel=='matched') & (manifest.development_part.eq('validation') if stage=='development' else manifest.role.eq('evaluation'))].reset_index(drop=True)
    labels=test.label.eq('shared-phase').to_numpy().astype(int)
    fig,axes=plt.subplots(1,3,figsize=(10.5,4.1),layout='constrained')
    xy=np.c_[test.mean_covariance,np.random.default_rng(261018).uniform(-.4,.4,len(test))]
    with np.load(target/'projections.npz') as p:
        train=p['z_complete_train'];held=p['z_complete_test']
        coords=[xy,PCA(n_components=2).fit(train).transform(held),UMAP(n_neighbors=30,min_dist=.1,random_state=261003,n_jobs=1).fit(train).transform(held)]
        for ax,v,title,xlabel,ylabel in zip(axes,coords,[r'Mean correlation $b$',r'SPI–SPI $z$: PCA',r'SPI–SPI $z$: UMAP'],[r'Mean off-diagonal correlation $b$','PCA 1','UMAP 1'],['Display jitter (no data)','PCA 2','UMAP 2']):
            for c,label in enumerate(LABELS):ax.scatter(*v[labels==c].T,color=COLORS[c],s=18*np.sqrt(2),alpha=.8,label=label)
            ax.set(title=title,xlabel=xlabel,ylabel=ylabel);ax.set_box_aspect(1)
            for spine in ax.spines.values():spine.set_visible(True)
        axes[0].set_yticks([]);axes[0].xaxis.set_major_locator(MaxNLocator(4));axes[2].set(xticks=[],yticks=[])
        handles,names=axes[0].get_legend_handles_labels();fig.legend(handles,names,loc='outside lower center',ncol=2)
        save(fig,target/'figures','scalar-vs-z')
        fig,axes=plt.subplots(1,3,figsize=(10.5,4.1),layout='constrained')
        for ax,key,title in zip(axes,['mean','distribution','z_complete'],[r'Per-SPI means $m$','Per-SPI distributions',r'SPI–SPI $z$']):
            v=PCA(n_components=2).fit(p[key+'_train']).transform(p[key+'_test'])
            for c,label in enumerate(LABELS):ax.scatter(*v[labels==c].T,color=COLORS[c],s=18*np.sqrt(2),alpha=.8,label=label)
            ax.set(title=title,xlabel='PCA 1',ylabel='PCA 2');ax.set_box_aspect(1)
            for spine in ax.spines.values():spine.set_visible(True)
        fig.legend(handles,names,loc='outside lower center',ncol=2);save(fig,target/'figures','marginal-comparison')



def selected_mean_plot(stage):
    target=OUT/stage;frame=pd.DataFrame(json.loads((DATA/'manifest.json').read_text())['rows'][:96])
    test=frame.development_part.eq('validation').to_numpy();f=frame[test].reset_index(drop=True)
    index=int(pd.read_csv(target/'selected-mean-control.csv').iloc[0].feature_index)
    jitter=np.random.default_rng(261018).uniform(-.4,.4,len(f));style()
    with np.load(target/'features.npz') as bank,np.load(target/'projections.npz') as p:
        coordinates=[np.c_[f.mean_covariance,jitter],np.c_[bank['mean'][:96][test,index],jitter],PCA(n_components=2).fit(p['z_complete_train']).transform(p['z_complete_test'])]
    fig,axes=plt.subplots(1,3,figsize=(10.5,4.1),layout='constrained')
    for ax,xy,title,xlabel in zip(axes,coordinates,[r'Mean correlation $b$','Training-selected MPI mean',r'SPI–SPI $z$: PCA'],[r'Mean correlation $b$',r'Mean squared Kendall correlation, lag 5','PCA 1']):
        for label,color,name in zip(['alternating-lag','shared-phase'],COLORS,LABELS):
            keep=f.label.eq(label);ax.scatter(*xy[keep].T,color=color,s=18*np.sqrt(2),alpha=.8,label=name)
        ax.set(title=title,xlabel=xlabel);ax.set_box_aspect(1)
        for spine in ax.spines.values():spine.set_visible(True)
    for ax in axes[:2]:
        ax.set(ylabel='Display jitter (no data)',yticks=[]);ax.xaxis.set_major_locator(MaxNLocator(4))
    axes[2].set_ylabel('PCA 2');handles,names=axes[0].get_legend_handles_labels();fig.legend(handles,names,loc='outside lower center',ncol=2)
    save(fig,target/'figures','selected-mean-vs-z')


def notebook(stage):
    target=OUT/stage;score=pd.read_csv(target/'metrics.csv').set_index('method').BA;analysis=json.loads((target/'analysis.json').read_text())
    selected=pd.read_csv(target/'selected-mean-control.csv')
    text=f'**{stage.title()} result:** covariance mean {score.b:.1%}; p90 means {score["mean"]:.1%} (strongest mean readout {score[["mean","mean_full","mean_RBF","mean_trees"]].max():.1%}); distributions {score.distribution:.1%}; primary SPI–SPI {score.z_complete:.1%}. A subsequently inspected training-ranked single mean reaches **{selected.BA.max():.1%}**, so the apparent advantage over marginal readouts needs qualification. Chance is 50%. Held-release gate: **{analysis["development_gate"]["all_pass"]}**.'
    cells=[nbf.v4.new_markdown_cell('# Temporal organization beyond matched circular covariance\n\n'+text),nbf.v4.new_markdown_cell(r'''This paired Gaussian experiment changes the temporal arrangement of dependence while preserving each realized covariance matrix and every circular cross-correlation function. It tests a specific dependence property, rather than differences between unrelated MTS families. The full p90 mean/distribution baselines are retained: no assumption says they must fail.

For each of eight independent pairs, $x_t=\epsilon_t$ and $y_t=a\epsilon_{\pi(t)}+\sqrt{1-a^2}\eta_t$, with independent standard Gaussian drivers. The permutation $\pi$ swaps adjacent five-sample blocks, producing alternating $+5/-5$ lags. The eight coefficients span .35–.85. Channel labels are randomly permuted and channels empirically standardized. Individual channels before standardization are Gaussian iid; the cross-channel dependence is time-varying. This is not a nonlinear or causal dynamical model.

The paired surrogate multiplies every channel's Fourier coefficient at frequency $f$ by the same random unit phase $e^{i\phi_f}$. Thus $\tilde X_i(f)\tilde X_j(f)^*=X_i(f)X_j(f)^*$ exactly. Covariance and circular auto/cross-correlation follow unchanged. This applies to the global circular definition, not necessarily windowed spectral estimators or correlations that trim the ends. Realized amplitude histograms are not preserved; third/fourth-moment checks are therefore included in the cheap scout.

In the population illustration below, the temporal cross-covariance operator changes from $K=aP$ to $Q^\top KQ$, where $P$ is the lag permutation and $Q$ the shared orthogonal Fourier rotation. Singular values—and hence operator strength—remain unchanged. The location of dependence across time changes; the surrogate is not asserted to be stationary. This follows the principle of constrained [surrogate-data methods](https://arxiv.org/abs/chao-dyn/9909037); the figure illustrates T100, whereas the p90 bank uses M16/T1000.'''),nbf.v4.new_code_cell(f'''from pathlib import Path
import json,pandas as pd
from IPython.display import display,Image
ROOT=Path.cwd().resolve()
while not (ROOT/'pyproject.toml').exists() and ROOT!=ROOT.parent: ROOT=ROOT.parent
OUT=ROOT/'results/representation/{RUN}'; TARGET=OUT/{stage!r}
display(Image(filename=str(OUT/'figures/mechanism.png')))
display(pd.read_csv(OUT/'probe-metrics.csv'))'''),nbf.v4.new_markdown_cell('Development uses 32 paired training realizations and 16 paired validation realizations, all at M16/T1000. An additional 16 zero-coupling pairs test whether the procedure creates a class signature without dependence. A separate 32-pair held bank remains uncomputed unless every frozen gate passes. The six-probe scout is development evidence, not confirmation. Splits, classifier settings and gates were committed before p90 outcomes.'),nbf.v4.new_code_cell("display(pd.read_csv(TARGET/'metrics.csv').round(3))\ndisplay(pd.read_csv(TARGET/'paired.csv').round(3))\na=json.loads((TARGET/'analysis.json').read_text()); display(a['development_gate']); display(a['null_accuracy']); display(pd.read_csv(TARGET/'null-projection-validity.csv'))"),nbf.v4.new_markdown_cell(r'''Every feature filter, imputation, scaling and PCA fit uses training data only. Primary $z$ uses training-complete features and centered PCA20/logistic C1; means/distributions use SD scaling, clipping at five SD and PCA20. Full linear/RBF/tree mean controls, standardized $z$, validity-only and independently edge-shuffled $z$ remain visible. Paired bootstrap intervals condition on the training set and chosen construction. Plots show every validation/held point, projected with training-fitted PCA/UMAP (30 neighbors, min_dist .1, seed261003); no embedding setting was selected from the outcome. Marker area follows $\sqrt M$, constant here because every recording has M16. Scalar vertical jitter is display-only.'''),nbf.v4.new_code_cell("display(Image(filename=str(TARGET/'figures/scalar-vs-z.png')))\ndisplay(Image(filename=str(TARGET/'figures/marginal-comparison.png')))"),nbf.v4.new_markdown_cell('## Stronger mean diagnostic\n\nAfter the primary development scores, training-ranked mean features revealed a strong lag-5 Kendall signal. Ranking uses only the 64 training recordings (largest absolute standardized class difference); the selected mean is then classified with a one-dimensional logistic model and a decision stump, also fitted only on training. These are exploratory diagnostics inspected after validation outcomes, not independent confirmation. The logistic model scores 93.8% against SPI–SPI 100%, and the paired difference is unresolved. The fixed PCA/UMAP displays are mixed despite perfect classification using 20 components, so this does not establish the desired visual contrast either. The original all-means≤65% gate failed. A proposed separate confirmation was cancelled before submission after this diagnostic; both the original held bank and the subsequently prepared fresh bank remain uncomputed.'),nbf.v4.new_code_cell("display(pd.read_csv(TARGET/'selected-mean-control.csv').round(4))\ndisplay(Image(filename=str(TARGET/'figures/selected-mean-vs-z.png')))"),nbf.v4.new_markdown_cell('A successful scalar comparison shows information beyond complete global circular second-order summaries in this constructed example. It does not establish that p90 means or distributions are blind to temporal organization, nor that every z coordinate has a unique physical interpretation. Stationary-model and causal interpretations of individual SPIs are not warranted for this time-varying construction. Weakness of the tested mean classifiers would still not establish absence of information. Any failed development gate leaves held SPI computation unreleased; do not tune parameters or classifiers on held outcomes.'),nbf.v4.new_code_cell("display(a['diagnostics'])\ndisplay(json.loads((OUT/'execution.json').read_text()))")]
    nb=nbf.v4.new_notebook(cells=cells);nb.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
    path=ROOT/'notebooks/embeddings/spi_lag_surrogate_261004.ipynb';nbf.write(nb,path);print(path)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['development','final'],default='development');a=p.parse_args()
    with threadpool_limits(limits=4):mechanism();plots(a.stage);selected_mean_plot(a.stage);notebook(a.stage)
