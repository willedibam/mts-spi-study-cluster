"""Fixed plots and an append-only extension to the user's strength notebook."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient
from sklearn.manifold import TSNE
from umap import UMAP

from research.proof.baselines.strength_factorial import ROOT,DATA,OUT,CLASSES,ARMS
from research.proof.baselines.analyze_strength_factorial import split
from research.proof.baselines.proof_strength_nuisance import style,save

NOTEBOOK=ROOT/'research/proof/baselines/notebooks/proof_strength_nuisance_261006.ipynb'
TAG='strength-factorial-261015'
COLORS=dict(zip(CLASSES,['#0072B2','#332288','#882255','#E69F00','#009E73','#555555','#999933']))
DISPLAY={'VAR':'VAR','Kuramoto':'Kuramoto','Wave':'Wave 1D','CML-intermittency':'CML: intermittency',
         'CML-defect':'CML: defect turbulence','Gaussian-shared':'Gaussian: shared input','Cauchy-shared':'Cauchy: shared input'}
ARM_LABELS={'fixed_clean':'Fixed coupling · no sensor noise','fixed_sensor':'Fixed coupling · variable sensor noise',
            'varying_clean':'Variable coupling · no sensor noise','varying_sensor':'Variable coupling · variable sensor noise'}


def plots():
    rows=pd.DataFrame(json.loads((DATA/'manifest.json').read_text())['rows']);train,evaluation,_=split(rows)
    reference=rows.role.eq('reference').to_numpy() & rows.instance.ge(20).to_numpy()
    with np.load(OUT/'coordinates.npz') as archive:spaces={k:archive[k] for k in archive.files}
    cached=OUT/'embedding-coordinates.npz'
    coordinates={}
    if cached.exists():
        with np.load(cached,allow_pickle=True) as archive:
            np.testing.assert_array_equal(archive['row_id'],rows.row_id)
            coordinates={k:archive[k] for k in archive.files if k!='row_id'}
    for name in ['m','z']:
        values=spaces[name]
        coordinates[name+'_PCA']=values[:,:2]
        if name+'_UMAP' not in coordinates:
            model=UMAP(n_neighbors=25,min_dist=.25,metric='euclidean',random_state=261015,transform_seed=261015,n_jobs=1).fit(values[train])
            result=np.full((len(rows),2),np.nan);result[train]=model.embedding_;result[evaluation|reference]=model.transform(values[evaluation|reference])
            coordinates[name+'_UMAP']=result
        if name+'_t-SNE' not in coordinates:
            # sklearn t-SNE has no held-out transform. Fit one joint descriptive
            # display, including all conditions; never use it for reported scores.
            use=train|evaluation|reference
            result=np.full((len(rows),2),np.nan)
            result[use]=TSNE(n_components=2,perplexity=30,init='pca',learning_rate='auto',
                            max_iter=1000,random_state=261015,n_jobs=1).fit_transform(values[use])
            coordinates[name+'_t-SNE']=result
    np.savez_compressed(OUT/'embedding-coordinates.npz',row_id=rows.row_id.to_numpy(dtype=str),**coordinates)
    style()
    for coloring in ['class','strength']:
        arms=ARMS if coloring=='class' else ('fixed_sensor','varying_sensor')
        fig,axes=plt.subplots(len(arms),6,figsize=(18,3*len(arms)+.4),layout='constrained',squeeze=False,sharex='col',sharey='col')
        scatter=None
        for r,arm in enumerate(arms):
            use=evaluation & rows.arm.eq(arm).to_numpy()
            for c,(name,method) in enumerate([(name,method) for method in ['PCA','UMAP','t-SNE'] for name in ['m','z']]):
                ax=axes[r,c];xy=coordinates[name+'_'+method]
                if coloring=='class':
                    for label,marker in [('Gaussian-independent','^'),('Cauchy-independent','s')]:
                        subset=reference & rows.label.eq(label).to_numpy()
                        ax.scatter(*xy[subset].T,s=14,alpha=.45,facecolors='none',edgecolors='.5',marker=marker,label=label,linewidths=.6)
                    for label in CLASSES:
                        subset=use & rows.label.eq(label).to_numpy()
                        ax.scatter(*xy[subset].T,s=18,alpha=.8,c=COLORS[label],label=DISPLAY[label],linewidths=0)
                else:
                    scatter=ax.scatter(*xy[use].T,s=18,alpha=.8,c=rows.loc[use,'mean_abs_r'],cmap='viridis',vmin=0,vmax=1,linewidths=0)
                ax.set_box_aspect(1)
                for spine in ax.spines.values():spine.set_visible(True)
                ax.set(xticks=[],yticks=[],xlabel=method+' 1',ylabel=(ARM_LABELS[arm]+'\n' if c==0 else '')+method+' 2')
                if r==0:ax.set_title((r'Mean-MPI $m$' if name=='m' else r'SPI–SPI $z$')+' — '+method)
        if coloring=='class':
            handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=8)
        else:fig.colorbar(scatter,ax=axes.ravel().tolist(),shrink=.8,label=r'Observed mean $|r|$')
        fig.suptitle('Within-class coupling and measurement variation · M=16, T=1000\nEvaluation points; PCA/UMAP trained on fixed conditions; t-SNE jointly fitted across conditions')
        save(fig,OUT/'figures','factorial-'+coloring);plt.close(fig)
    # Requested display-only omission: retain exactly the all-class coordinates.
    fig,axes=plt.subplots(2,3,figsize=(10,7),layout='constrained')
    shown=[c for c in CLASSES if not c.endswith('-shared')]
    for r,name in enumerate(['m','z']):
        for c,method in enumerate(['PCA','UMAP','t-SNE']):
            ax=axes[r,c];xy=coordinates[name+'_'+method]
            for label,marker in [('Gaussian-independent','^'),('Cauchy-independent','s')]:
                subset=reference & rows.label.eq(label).to_numpy()
                ax.scatter(*xy[subset].T,s=14,alpha=.45,facecolors='none',edgecolors='.5',marker=marker,label=label,linewidths=.6)
            for label in shown:
                subset=evaluation & rows.arm.eq('varying_sensor').to_numpy() & rows.label.eq(label).to_numpy()
                ax.scatter(*xy[subset].T,s=18,alpha=.8,c=COLORS[label],label=DISPLAY[label],linewidths=0)
            # Preserve full-population limits too; the only change is visibility.
            all_xy=xy[evaluation|reference];low=np.nanmin(all_xy,axis=0);high=np.nanmax(all_xy,axis=0);pad=.05*(high-low)
            ax.set(xlim=(low[0]-pad[0],high[0]+pad[0]),ylim=(low[1]-pad[1],high[1]+pad[1]),
                   xticks=[],yticks=[],xlabel=method+' 1',ylabel=method+' 2',
                   title=(r'Mean-MPI $m$' if name=='m' else r'SPI–SPI $z$')+' — '+method)
            ax.set_box_aspect(1)
            for spine in ax.spines.values():spine.set_visible(True)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=8)
    fig.suptitle('Variable coupling + variable sensor noise · shared-input classes hidden\nSame coordinates; independent references retained; t-SNE is a joint descriptive fit')
    save(fig,OUT/'figures','factorial-no-shared');plt.close(fig)
    metrics=pd.read_csv(OUT/'metrics.csv')
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.3),layout='constrained')
    for name,color in [('m','#D55E00'),('z','#0072B2'),('m+z','#009E73')]:
        group=metrics[metrics.representation.eq(name)].set_index('arm').loc[list(ARMS)]
        for ax,key,label in zip(axes,['spread_to_separation','mAP'],['Within-class spread / between-class separation\nLower is better','Retrieval to clean, different-seed recordings\nmAP; higher is better']):
            ax.plot(range(4),group[key],marker='o',color=color,label=name)
            ax.set(xticks=range(4),xticklabels=['Fixed / clean','Fixed / noise','Varying / clean','Varying / noise'],ylabel=label)
            ax.tick_params(axis='x',rotation=25)
    axes[1].set_ylim(0,1.03);axes[0].legend(title='Representation')
    save(fig,OUT/'figures','factorial-summary');plt.close(fig)


def cells():
    md=nbformat.v4.new_markdown_cell;code=nbformat.v4.new_code_cell
    result=[md(r'''## Added 15 October: physical coupling versus measurement noise

The nuisance varies **between recordings within each class**. We cross fixed/variable physical coupling with absent/variable independent sensor noise, keeping initial conditions or innovations paired across the four conditions. Classes are VAR, Kuramoto, wave 1D, nominal CML intermittency/defect turbulence, and shared-input Gaussian/Cauchy processes; independent Gaussian/Cauchy references are outside the primary comparison. M=16, T=1000; 40 instances/class, four conditions, plus 80 references: 1,200 fresh recordings, the repository p90 catalogue under one fresh computation version. Historical features above are not pooled with this run. Every channel is sample-centred and standardized before and after adding sensor noise; covariance therefore represents sample correlation up to the estimator denominator.

For standardized finite-variance signals and independent unit-variance sensor errors, $y_i=x_i+\eta\epsilon_i$ gives $\rho^y_{ij}=\rho^x_{ij}/(1+\eta^2)$ at population level. Here each recording gets its own noise level, with nominal attenuation uniform on [.25,.95]. Finite-record estimates and non-correlation SPIs need not follow this identity; Cauchy population covariance is undefined. Sensor noise leaves latent dynamics fixed, while changing measurement quality ([classical attenuation](https://psychclassics.yorku.ca/Spearman/chap1-4.htm)).

Physical coupling parameters independently get multipliers uniform on [.5,1.5]: VAR neighbour coefficient, Kuramoto K, wave c² at fixed cadence, CML epsilon, or common-input loading. Multiplication preserves the coupling rule's form, **not necessarily its emergent dependence pattern or regime**. For example, changing Kuramoto coupling can change synchronization ([finite-network theory](https://arxiv.org/abs/1011.3878)). Shared-input loading is functional dependence, not causal coupling. The ranges and seven classes were fixed before SPI results; no severe-noise tail was selected to make a baseline fail.'''),
        md('Instances 0–19, fixed clean/sensor conditions only, fit all preprocessing and PCA50/UMAP. Instances 20–39 evaluate all four conditions in those same coordinates. Added t-SNE is a joint descriptive fit of all conditions on the same training-derived PCA50 inputs (perplexity 30, PCA initialization, seed 261015, 1,000 iterations); it is not an out-of-sample projection and does not enter any scores. The additional five-class view hides shared-input Gaussian/Cauchy points without refitting or changing limits; independent references remain. Hollow grey symbols project independent Gaussian/Cauchy references; they do not enter fitting or the seven-class scores. The main baseline is the raw MPI mean vector, with training-fitted feature scaling/clipping for differing SPI units; no per-recording mean-vector normalization. z uses the historical symmetrized profile definition and centering. Training validity/variance filtering retains 247 of 289 means and 25,971 z coordinates (12,071 in the complete-training control). Raw-mean full-vector distances/linear/RBF classifiers, standardized/complete-training z, and validity-only classification are recorded as controls. Retrieval uses the fixed-clean training gallery, so query and gallery never share a simulation parent. This tests system/regime recognition and nuisance robustness; it does not independently identify a nonlinear or lag mechanism.'),
        code("from pathlib import Path\nimport json,pandas as pd\nfrom IPython.display import display,Image\nFACTOR_ROOT=Path.cwd().resolve()\nwhile not (FACTOR_ROOT/'pyproject.toml').exists() and FACTOR_ROOT!=FACTOR_ROOT.parent: FACTOR_ROOT=FACTOR_ROOT.parent\nFACTOR_OUT=FACTOR_ROOT/'results/proof/strength-factorial-261015'\ndisplay(Image(filename=str(FACTOR_OUT/'figures/factorial-class.png')))"),
        code("display(Image(filename=str(FACTOR_OUT/'figures/factorial-no-shared.png')))\ndisplay(Image(filename=str(FACTOR_OUT/'figures/factorial-strength.png')))\ndisplay(Image(filename=str(FACTOR_OUT/'figures/factorial-summary.png')))"),
        code("factor_scores=pd.read_csv(FACTOR_OUT/'metrics.csv')\ndisplay(factor_scores[factor_scores.representation.isin(['m','z','m+z'])].pivot(index='arm',columns='representation',values=['spread_to_separation','mAP','accuracy']).round(3))\ndisplay(pd.read_csv(FACTOR_OUT/'intervention-intervals.csv').query(\"representation == 'z minus m'\").round(3))"),
        md(r'''**Result: relative geometry and clean-reference retrieval favour z; classification does not.** With both variations, spread/separation is 1.130 for m versus .380 for z; silhouette .090 versus .355; retrieval mAP .405 versus .665 (paired difference .260, 95% interval [.216,.303]). More importantly, deterioration from fixed/clean is .841 for m versus .197 for z; the paired difference in deterioration is −.645 [−.926,−.271]. These intervals resample evaluation instances, keeping classes/arms paired, conditional on the training fit. Sensor noise alone compresses m's between-class separation more than its within-class spread: “more diffuse” means worse relative separation, not literally larger clouds.

The means remain informative: full-vector linear accuracy is .921 versus z's .893 under both variations. z's retrieval gain is broad but not universal: Wave, shared Gaussian/Cauchy and intermittency improve; nominal defect turbulence worsens (.323 versus m's .482), and Kuramoto remains poor (.217). Combining m+z improves m's retrieval (.524), but does not beat z. The physical-only condition also perturbs z, consistent with altered emergent dynamics; nominal CML regime preservation was not established.

**Checks and limits.** Standardizing z retains a smaller retrieval advantage (.549); restricting to features complete in training strengthens it (.768). Missingness alone predicts classes (accuracy .37–.71 across conditions), so estimator validity is a real confound; complete-training z has only .073% missing evaluation entries in the combined condition. The primary m convention clips training-standardized values at ±5 SD; an explicitly post-result *unclipped* audit gives combined-condition full-mean mAP .401 and accuracy .921, so clipping does not explain that contrast. Extreme clean-condition mean values make unclipped spread particularly unstable. All 1,200 outputs passed provenance checks; 4,624 means and 1,600 z values were independently checked. This supports robustness of this representation/metric pipeline across these systems, not absent information in means or a uniquely identified dependence-character axis.'''),
        code("display(pd.read_csv(FACTOR_OUT/'per-class.csv').query(\"arm == 'varying_sensor' and representation in ['m','z']\").pivot(index='label',columns='representation',values=['spread_to_separation','mAP']).round(3))\nfactor_audit=json.loads((FACTOR_OUT/'analysis-audit.json').read_text())\nprint('SPIs:',factor_audit['catalogue_size'],'; computation:',factor_audit['computation'],'; validity-only accuracy:',factor_audit['validity_only_accuracy'])")]
    for cell in result:cell.metadata['tags']=[TAG]
    return result


def append():
    original=nbformat.read(NOTEBOOK,as_version=4)
    preserved=[c for c in original.cells if TAG not in c.metadata.get('tags',[])]
    before=json.dumps(preserved,sort_keys=True)
    addition=nbformat.v4.new_notebook(cells=cells(),metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    NotebookClient(addition,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    original.cells=preserved+addition.cells
    assert json.dumps(original.cells[:len(preserved)],sort_keys=True)==before
    nbformat.write(original,NOTEBOOK)
    (OUT/'notebook-append-audit.json').write_text(json.dumps(dict(preserved_cells=len(preserved),appended_cells=len(addition.cells),existing_cells_and_outputs_unchanged=True),indent=2)+'\n')
    print('Appended',len(addition.cells),'executed cells; preserved',len(preserved),'existing cells and outputs.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['plots','append']);a=p.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):plots() if a.stage=='plots' else append()
