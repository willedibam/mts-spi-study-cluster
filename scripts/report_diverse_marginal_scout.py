"""Display the frozen six-family local scout without selecting embeddings."""
import json

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, silhouette_score
from umap import UMAP

from scripts.refresh_case_figures import save, style
from scripts.scout_diverse_marginal_match import FAMILIES, OUT, ROOT, RUN, SEED
from scripts.spi_baseline_exploration import project_features, sha
from src.corpus_geometry import fit_geometry_transform

COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00', '#56B4E9']


def report():
    rows = pd.read_csv(OUT/'fresh/rows.csv')
    scores = pd.read_csv(OUT/'metrics.csv')
    with np.load(OUT/'fresh/features.npz') as archive:
        bank = {k:archive[k] for k in archive.files}
    train = rows.block.to_numpy()<32
    labels = rows.loc[~train, 'family'].to_numpy()
    coordinates, projection_metrics = {}, []
    for key in ['mean', 'distribution', 'z']:
        _, d, q = project_features(bank[key][train],bank[key][~train],dimensions=20,standard=key!='z')
        coordinates[key+'_PCA'] = q[:,:2]
        model = LogisticRegression(C=1,max_iter=5000).fit(d[:,:2],rows.loc[train,'family'])
        projection_metrics.append(dict(representation=key,
            BA=balanced_accuracy_score(labels,model.predict(q[:,:2])),
            silhouette=silhouette_score(q[:,:2],labels)))
        reducer = UMAP(n_components=2,n_neighbors=15,min_dist=.1,metric='euclidean',random_state=SEED,transform_seed=SEED,n_jobs=1).fit(d)
        coordinates[key+'_UMAP'] = reducer.transform(q)
    np.savez_compressed(OUT/'coordinates.npz',**coordinates)
    pd.DataFrame(projection_metrics).to_csv(OUT/'pca2-diagnostic.csv',index=False)
    transform = fit_geometry_transform(bank['mean'][train],scaling='standard',minimum_valid_fraction=1.)
    values = np.clip(transform.transform(bank['mean']),-5,5)
    fvalues, _ = f_classif(values[train],rows.loc[train,'family'])
    ranked = np.argsort(-fvalues)[:5]
    pd.DataFrame(dict(feature=bank['names'][transform.keep_indices[ranked]],training_F=fvalues[ranked])).to_csv(OUT/'training-selected-means.csv',index=False)
    style()
    fig, axes = plt.subplots(2,4,figsize=(12.4,6.8),layout='constrained')
    jitter = np.random.default_rng(SEED).uniform(-1,1,size=(2,len(labels)))
    for row, algorithm in enumerate(['PCA','UMAP']):
        for col, key in enumerate(['scalar','z','mean','distribution']):
            ax = axes[row,col]
            for f, color in zip(FAMILIES,COLORS):
                use = labels==f
                if key == 'scalar':
                    x,y = bank['mean'][~train,0],jitter[row]
                else:
                    x,y = coordinates[key+'_'+algorithm].T
                ax.scatter(x[use],y[use],s=16*np.sqrt(16/8),alpha=.65,color=color,label=f,linewidths=0)
            ax.set_box_aspect(1)
            for spine in ax.spines.values():
                spine.set_visible(True)
            if key == 'scalar':
                ax.set(title=r'Mean correlation $\bar r$',xlabel=r'$\bar r$',ylabel='Display jitter only',yticks=[])
            else:
                title = {'mean':r'Probe means $m$','distribution':'Probe distributions','z':r'Probe SPI–SPI $z$'}[key]
                ax.set(title=title+' — '+algorithm,xlabel=algorithm+' 1',ylabel=algorithm+' 2',xticks=[],yticks=[])
    handles, legend_labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='outside lower center',ncol=6)
    fig.suptitle('Six families; fresh development validation — 24 probes, not p90\n'+r'$M=16,\ T=1000$; settings chosen using separate scout means')
    save(fig,OUT/'figures','six-family-embeddings')
    plt.close(fig)
    predictions = pd.read_csv(OUT/'predictions.csv')
    methods = ['mean_linear','mean_RBF','mean_trees','z_center']
    fig, axes = plt.subplots(1,4,figsize=(12,3.6),layout='constrained')
    for ax, method in zip(axes,methods):
        p = predictions[predictions.method==method]
        matrix = confusion_matrix(p.family,p.prediction,labels=FAMILIES,normalize='true')
        im=ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues')
        ax.set(title=method.replace('_',' '),xticks=range(6),yticks=range(6),
               xticklabels=FAMILIES,yticklabels=FAMILIES,xlabel='Predicted',ylabel='True')
        plt.setp(ax.get_xticklabels(),rotation=45,ha='right')
    fig.colorbar(im,ax=axes,shrink=.7,label='Fraction within true family')
    save(fig,OUT/'figures','confusion')
    plt.close(fig)
    # Conditional uncertainty over 32 independent validation blocks; families paired.
    block_scores = predictions.assign(correct=predictions.family==predictions.prediction).groupby(['method','block']).correct.mean().unstack(0)
    resamples=np.random.default_rng(SEED).integers(0,len(block_scores),size=(2000,len(block_scores)))
    intervals=[]
    for method in block_scores:
        samples=block_scores[method].to_numpy()[resamples].mean(1)
        intervals.append(dict(method=method,lower=np.quantile(samples,.025),upper=np.quantile(samples,.975)))
    pd.DataFrame(intervals).to_csv(OUT/'intervals.csv',index=False)
    selected = json.loads((OUT/'selection.json').read_text())
    settings_table = pd.DataFrame([dict(family=f,**c[0]) for f,c in selected['configurations'].items()])
    settings_table.to_csv(OUT/'selected-settings.csv',index=False)
    summary=rows.groupby('family').agg(native_variance_fraction_median=('retained_native_variance','median'),
                                     common_boost_median=('common_boost','median'),warnings=('graphical_warnings','sum'))
    summary.to_csv(OUT/'observation-summary.csv')
    decision=json.loads((OUT/'decision.json').read_text())
    outcome=('The feasibility gate passes; this still requires p90 validation.' if decision['pass_gate'] else
             'The feasibility gate fails. The requested weak-marginal/strong-z contrast is not established by this grid; no p90 expansion is justified.')
    cells=[nbf.v4.new_markdown_cell('# Diverse families with mean-selected settings\n\n'+outcome+'\n\nAll six families—VAR, Wave, CML, Kuramoto, correlated Gaussian and correlated Cauchy—are retained. These are **24 local probes**, not the 289-SPI p90 catalogue. This experiment addresses the user’s diverse-system requirement; earlier factor-only examples are supporting evidence at most.'),
        nbf.v4.new_markdown_cell('## Design\n\nNine settings per family and four scout realizations per setting are compared using only the standardized mean-probe vectors. A deterministic multi-start alternating search selects one setting per family to minimize their between-family mean differences. This is deliberate development design, not unsupervised discovery or a guaranteed global optimum. Settings were committed at `e1c39ae` before the fresh bank was generated. The selected generators then supply 32 fresh training and 32 fresh validation realizations per family, all at M16/T1000. Original held banks are untouched.\n\nM16 provides 240 ordered channel pairs at moderate cost; these are not independent samples. Holding M and T fixed isolates this generator feasibility question before revisiting the requested nine-size grid. The independent-noise controls from the earlier six-family reports remain separate; this scout evaluates only the six correlated families. No families, parameters, seeds or embeddings are chosen using fresh validation outcomes.'),
        nbf.v4.new_code_cell(f"from pathlib import Path\nimport pandas as pd,json\nfrom IPython.display import display,Image\nROOT=Path.cwd().resolve()\nwhile not (ROOT/'pyproject.toml').exists() and ROOT!=ROOT.parent: ROOT=ROOT.parent\nOUT=ROOT/'results/representation/{RUN}'\ndisplay(pd.read_csv(OUT/'selected-settings.csv'))"),
        nbf.v4.new_markdown_cell(r'''## What “strength matched” means here

Channels are centered and scaled to unit sample variance. Each block has a shared target $\overline{|r|}\in[.065,.075]$ and $\bar r\in[.003,.007]$. Observation-noise amplitude, a noise rotation and channel polarities match these two summaries numerically. A weak native recording may need an added shared Gaussian observation input before attenuation. This is explicitly an observation model, not equal physical coupling or equal native Jacobian gain. Cauchy covariance is finite-sample only.

For the four dynamical families, the reported native variance fraction is the nominal $1/(1+b^2+\sigma^2)$, ignoring empirical cross terms, where $b$ is the common-input loading and $\sigma$ the added noise amplitude. It is a diagnostic of how much calibration changes the recording, not an exact variance decomposition. Neither added noise nor sign calibration is evidence of dependence-character isolation.

The probes cover contemporaneous, rank, lagged, precision/regularized covariance, spectral coherence and phase-locking summaries. Gaussian plug-in MI is explicitly not KSG MI; the focused catalogue is not an exact p90 subset. Means, seven distribution summaries per probe and correlations across aligned ordered off-diagonal entries are extracted from the same records.'''),
        nbf.v4.new_code_cell("display(pd.read_csv(OUT/'observation-summary.csv').round(4))\ndisplay(pd.read_csv(OUT/'metrics.csv').merge(pd.read_csv(OUT/'intervals.csv'),on='method').round(4))\ndisplay(pd.read_csv(OUT/'training-selected-means.csv').round(4))"),
        nbf.v4.new_markdown_cell('Mean and distribution controls include full linear, RBF and tree readouts, plus a training-ranked single-summary tree. The primary focused z readout is centered PCA20 plus logistic C1; standardized z is a sensitivity. Preprocessing and embeddings fit training only. Chance balanced accuracy is 1/6. Intervals resample the 32 validation blocks, preserving the six-family pairing; they do not account for the earlier choice to pursue this experimental direction. A prespecified development gate requires every mean readout ≤.30 and centered z ≥.70. Finite-readout weakness would not prove information absence.\n\nThe leftmost display is the actual one-dimensional signed mean correlation with explicitly meaningless vertical jitter, not a two-dimensional embedding. The remaining panels use fixed PCA/UMAP settings. UMAP uses the first 20 training-fitted PCs, 15 neighbors, min_dist .1 and seed261025; validation is transformed without refitting. Marker area scales as sqrt(M); all records here have M16. Visually mixed two-dimensional projections do not override successful classification.'),
        nbf.v4.new_code_cell("display(Image(filename=str(OUT/'figures/six-family-embeddings.png')))\ndisplay(Image(filename=str(OUT/'figures/confusion.png')))\ndisplay(json.loads((OUT/'decision.json').read_text()))"),
        nbf.v4.new_markdown_cell('## What the visual contrast does establish\n\nThe fixed z PCA2 view is clearer than the means PCA2 view. An exploratory diagnostic added after inspecting the fixed figures fits the same logistic C1 readout to their first two PCs: z gives 89.1%, means 67.7%, distributions 93.8%. Validation silhouette is .447 for z versus .044 for means. This is a low-dimensional geometry benefit in this focused catalogue, not evidence that the marginal representations lack class information; their full-vector accuracy is higher. No embedding or parameter was changed for this diagnostic.\n\nThe strongest single mean ranked on training is squared lag-1 Spearman correlation. Matching contemporaneous correlation leaves lagged rank dependence informative. Calibration also dilutes the nominal native variance fraction to about 7% for Kuramoto and 9% for Wave (versus roughly 62–64% for CML/VAR). This limits the physical interpretation of the calibrated systems, independently of their classification scores.'),
        nbf.v4.new_code_cell("display(pd.read_csv(OUT/'pca2-diagnostic.csv').round(4))"),
        nbf.v4.new_markdown_cell('## Scope\n\nThis tests a finite, declared parameter grid and observation model. Failure does not prove that a diverse-system counterexample is impossible. It does show that matching two scalar strengths and choosing native settings by marginal similarity are insufficient for this construction. Raw means across many dependence measures can retain information about dependence character: a distinction between “strength” and “character” is not a general theorem separating m from z. Cliff et al. assemble measures with diverse mathematical sensitivities; this experiment tests an additional representation, not a criticism of that work. [Cliff et al., Unifying pairwise interactions in complex dynamics](https://www.nature.com/articles/s43588-023-00519-x).\n\nFor missing ignored feature caches, run `python -m scripts.audit_diverse_marginal_scout --rebuild-missing-caches`, then `python -m scripts.scout_diverse_marginal_match evaluate` and `python -m scripts.report_diverse_marginal_scout`. The original scout/fresh stages are immutable; cache restoration does not repeat parameter selection. Protocol, selected settings, all failures, source hashes and numerical results live alongside this report’s figures.'),
        nbf.v4.new_code_cell("display(json.loads((OUT/'protocol.json').read_text()))\ndisplay(json.loads((OUT/'audit.json').read_text()))")]
    nb=nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
    path=ROOT/'notebooks/embeddings/spi_diverse_marginal_scout_261005.ipynb'
    nbf.write(nb,path)
    print(path)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        report()
