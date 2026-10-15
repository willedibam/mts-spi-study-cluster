"""Size-scaled displays and append-only grid results."""
import argparse
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from umap import UMAP
from threadpoolctl import threadpool_limits
from research.proof.baselines.strength_grid import ROOT,DATA,OUT,CLASSES,ARMS,MS,TS
from research.proof.baselines.analyze_strength_grid import split
from research.proof.baselines.report_strength_factorial import NOTEBOOK,ARM_LABELS
from research.proof.baselines.proof_strength_nuisance import style,save

TAG='strength-grid-261015'
COLORS=dict(zip(CLASSES,['#0072B2','#332288','#882255','#E69F00','#CC79A7','#555555','#999933']))
COLORS.update({'Gaussian-independent':'#999999','Cauchy-independent':'#333333'})
DISPLAY={'VAR':'VAR','Kuramoto':'Kuramoto','Wave':'Wave 1D','CML-intermittency':'CML: intermittency','CML-frozen':'CML: frozen chaos','Gaussian-shared':'Gaussian: shared input','Cauchy-shared':'Cauchy: shared input','Gaussian-independent':'Gaussian: independent','Cauchy-independent':'Cauchy: independent'}
MARKERS={500:'o',1000:'s',2000:'^'}
VIEWS=[(name,method) for method in ['PCA','UMAP','t-SNE'] for name in ['m','z']]


def scatter(ax,xy,rows,use,labels,coloring):
    artist=None
    for label in labels:
        for t,marker in MARKERS.items():
            choose=use & rows.label.eq(label).to_numpy() & rows['T'].eq(t).to_numpy()
            if not choose.any():continue
            sizes=16*np.sqrt(rows.loc[choose,'M'].to_numpy()/8)
            if coloring=='strength':
                artist=ax.scatter(*xy[choose].T,s=sizes,marker=marker,alpha=.8,c=rows.loc[choose,'mean_abs_r'],cmap='viridis',vmin=0,vmax=1,linewidths=0)
            elif label.endswith('independent'):
                ax.scatter(*xy[choose].T,s=sizes,marker=marker,alpha=.5,facecolors='none',edgecolors=COLORS[label],linewidths=.6)
            else:ax.scatter(*xy[choose].T,s=sizes,marker=marker,alpha=.8,c=COLORS[label],linewidths=0)
    return artist


def legend(fig,labels):
    handles=[Line2D([],[],ls='',marker='o',color=COLORS[c],markerfacecolor='none' if c.endswith('independent') else COLORS[c],markersize=4,label=DISPLAY[c]) for c in labels]
    handles += [Line2D([],[],ls='',marker='o',color='.3',markersize=np.sqrt(16*np.sqrt(m/8)),label=f'M={m}') for m in MS]
    handles += [Line2D([],[],ls='',marker=MARKERS[t],color='.3',markersize=4,label=f'T={t}') for t in TS]
    fig.legend(handles=handles,loc='outside lower center',ncol=5 if labels else 6,fontsize=8)


def plots():
    rows=pd.DataFrame(json.loads((DATA/'manifest.json').read_text())['rows']);train,evaluation,_=split(rows)
    reference=rows.role.eq('reference').to_numpy() & rows.instance.ge(3).to_numpy()
    with np.load(OUT/'coordinates.npz') as a:spaces={k:a[k] for k in a.files}
    coords={}
    for name in ['m','z']:
        values=spaces[name];coords[name+'_PCA']=values[:,:2]
        model=UMAP(n_neighbors=25,min_dist=.25,random_state=261015,transform_seed=261015,n_jobs=1).fit(values[train])
        xy=np.full((len(rows),2),np.nan);xy[train]=model.embedding_;xy[evaluation|reference]=model.transform(values[evaluation|reference]);coords[name+'_UMAP']=xy
        xy=np.full((len(rows),2),np.nan);use=train|evaluation|reference
        xy[use]=TSNE(perplexity=30,init='pca',learning_rate='auto',max_iter=1000,random_state=261015,n_jobs=1).fit_transform(values[use]);coords[name+'_t-SNE']=xy
    np.savez_compressed(OUT/'embedding-coordinates.npz',row_id=rows.row_id.to_numpy(dtype=str),**coords)
    style();all_labels=list(CLASSES)+['Gaussian-independent','Cauchy-independent']
    for coloring in ['class','strength']:
        arms=ARMS if coloring=='class' else ['varying_sensor']
        fig,axes=plt.subplots(len(arms),6,figsize=(18,3*len(arms)+1),layout='constrained',squeeze=False,sharex='col',sharey='col')
        for r,arm in enumerate(arms):
            use=evaluation & rows.arm.eq(arm).to_numpy()
            if coloring=='class':use=use|reference
            for c,(name,method) in enumerate(VIEWS):
                ax=axes[r,c];artist=scatter(ax,coords[name+'_'+method],rows,use,all_labels,coloring)
                ax.set_box_aspect(1)
                for spine in ax.spines.values():spine.set_visible(True)
                ax.set(xticks=[],yticks=[],xlabel=method+' 1',ylabel=(ARM_LABELS[arm]+'\n' if c==0 else '')+method+' 2')
                if r==0:ax.set_title((r'Mean-MPI $m$' if name=='m' else r'SPI–SPI $z$')+' — '+method)
        if coloring=='class':legend(fig,all_labels)
        else:
            fig.colorbar(artist,ax=axes.ravel().tolist(),shrink=.8,label=r'Observed mean $|r|$')
            legend(fig,[])
        fig.suptitle('Size grid: M=8/16/32, T=500/1000/2000 · five instances per class/cell\nEvaluation points; marker area ∝ √M, shape indicates T; t-SNE is a joint descriptive fit')
        save(fig,OUT/'figures','grid-'+coloring);plt.close(fig)
    labels=[c for c in all_labels if not c.endswith('shared')]
    fig,axes=plt.subplots(2,3,figsize=(10,7.5),layout='constrained')
    use=(evaluation & rows.arm.eq('varying_sensor').to_numpy())|reference
    for r,name in enumerate(['m','z']):
        for c,method in enumerate(['PCA','UMAP','t-SNE']):
            ax=axes[r,c];xy=coords[name+'_'+method];scatter(ax,xy,rows,use,labels,'class')
            full=xy[evaluation|reference];low=np.nanmin(full,0);high=np.nanmax(full,0);pad=.05*(high-low)
            ax.set(xlim=(low[0]-pad[0],high[0]+pad[0]),ylim=(low[1]-pad[1],high[1]+pad[1]),xticks=[],yticks=[],xlabel=method+' 1',ylabel=method+' 2',title=(r'Mean-MPI $m$' if name=='m' else r'SPI–SPI $z$')+' — '+method)
            ax.set_box_aspect(1)
            for spine in ax.spines.values():spine.set_visible(True)
    legend(fig,labels);fig.suptitle('Both variations · shared-input classes hidden without refitting\nMarker area ∝ √M; shape indicates T; independent references retained')
    save(fig,OUT/'figures','grid-no-shared');plt.close(fig)


def cells():
    md=nbformat.v4.new_markdown_cell;code=nbformat.v4.new_code_cell
    cells=[md(r'''## Size-grid rerun: intermittency and frozen chaos

Five independent realizations per class and $(M,T)$ cell, $M\in\{8,16,32\}$ and $T\in\{500,1000,2000\}$. Four paired conditions per realization and seven classes plus two independent references give 1,350 recordings. Nominal CML frozen chaos ($\alpha=1.45$, $\epsilon=.2$) replaces defect turbulence; the earlier result remains above. Coupling multipliers $U[.5,1.5]$ and nominal noise attenuation $U[.25,.95]$ are unchanged. Increasing T changes recording duration at fixed cadence; the grid avoids T=100, where estimator validity/precision could dominate.

M counts observed sensors: all hosts have 32 channels, with a uniform M-channel subset (CML evolves 100 sites and supplies a central 32-site window). Parents are independent across size cells; the four arms share a parent's initial conditions/innovations and sensor-noise array. This is observation-size robustness, not physical finite-size scaling. Frozen chaos/intermittency name the base settings; varying coupling does not certify preservation of either regime.

Instances 0–2, fixed clean/noisy only, fit preprocessing/PCA50 and UMAP; instances 3–4 evaluate all conditions. Retrieval uses a clean training gallery with different parents. Means use the same training scaling/clipping as before, with full/unclipped controls. t-SNE jointly fits all display conditions on the training-derived PCA50 inputs and is descriptive only. Marker **area scales as $\sqrt{M}$**; shape denotes T. No class/point/embedding selection based on outcomes. Cross-size retrieval restricts the gallery to recordings with both a different M and a different T. Descriptive M/T effects are fractions of within-class squared spread explained by each size factor, allowing a different shift per class; they include finite-sample variation and do not imply causal size effects. Only two evaluation instances per class/cell means cell-specific scores are exploratory.'''),
        code("from pathlib import Path\nimport json,pandas as pd\nfrom IPython.display import display,Image\nGRID_ROOT=Path.cwd().resolve()\nwhile not (GRID_ROOT/'pyproject.toml').exists() and GRID_ROOT!=GRID_ROOT.parent: GRID_ROOT=GRID_ROOT.parent\nGRID_OUT=GRID_ROOT/'results/proof/strength-grid-261015'\ndisplay(Image(filename=str(GRID_OUT/'figures/grid-class.png')))"),
        code("display(Image(filename=str(GRID_OUT/'figures/grid-no-shared.png')))\ndisplay(Image(filename=str(GRID_OUT/'figures/grid-strength.png')))"),
        code("grid_scores=pd.read_csv(GRID_OUT/'metrics.csv')\ndisplay(grid_scores.query(\"cell == 'all' and representation in ['m','z','m+z']\").pivot(index='arm',columns='representation',values=['spread_to_separation','mAP','accuracy']).round(3))\ndisplay(grid_scores.query(\"cell != 'all' and arm == 'varying_sensor' and representation in ['m','z']\").pivot(index='cell',columns='representation',values=['spread_to_separation','mAP']).round(3))\ndisplay(pd.read_csv(GRID_OUT/'cross-size-retrieval.csv').query(\"representation in ['m','z']\").groupby(['arm','representation']).mAP.mean().unstack().round(3))\ndisplay(pd.read_csv(GRID_OUT/'size-effects.csv').query(\"representation in ['m','z']\").round(3))"),
        md('Class labels, observation size, duration, estimator validity and noise can all contribute to geometry. Compact-looking groups alone do not establish a dependence-character mechanism. Keep pooled and cellwise results separate; no claim that marginal information is absent. Complete-training/standardized z, full/unclipped means and validity-only controls are retained in the result files.'),
        code("display(pd.read_csv(GRID_OUT/'preprocessing.csv').round(3))\ndisplay(grid_scores.query(\"cell == 'all' and arm == 'varying_sensor'\")[[\"representation\",\"mAP\",\"accuracy\",\"RBF_accuracy\"]].round(3))\nprint('Validity-only accuracy:',json.loads((GRID_OUT/'analysis-audit.json').read_text())['validity_only_accuracy'])")]
    for c in cells:c.metadata['tags']=[TAG]
    return cells


def append():
    notebook=nbformat.read(NOTEBOOK,4);preserved=[c for c in notebook.cells if TAG not in c.metadata.get('tags',[])];before=json.dumps(preserved,sort_keys=True)
    addition=nbformat.v4.new_notebook(cells=cells(),metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    NotebookClient(addition,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    notebook.cells=preserved+addition.cells;nbformat.write(notebook,NOTEBOOK)
    reloaded=nbformat.read(NOTEBOOK,4);assert json.dumps(reloaded.cells[:len(preserved)],sort_keys=True)==before
    (OUT/'notebook-append-audit.json').write_text(json.dumps(dict(preserved_cells=len(preserved),appended_cells=len(addition.cells),original_cells_outputs_unchanged=True),indent=2)+'\n')
    print('Appended',len(addition.cells),'cells; preserved',len(preserved))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['plots','append']);a=parser.parse_args()
    with threadpool_limits(limits=4):plots() if a.stage=='plots' else append()
