"""Fixed PCA/UMAP views, every-dose curves, and a lean append-only report."""
import argparse,json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd
from umap import UMAP
from threadpoolctl import threadpool_limits
from research.proof.baselines.noise_experiments import ROOT,DATA,OUT,CLASSES,ETAS,sha
from research.proof.baselines.analyze_noise_experiments import split,CONDITIONS
from research.proof.baselines.proof_strength_nuisance import style,save
from research.proof.baselines.report_strength_factorial import NOTEBOOK

TAG='noise-experiments-261016'
COLORS=dict(zip(CLASSES,['#0072B2','#332288','#882255','#E69F00','#CC79A7','#555555','#999933']))
DISPLAY={'VAR':'VAR','Kuramoto':'Kuramoto','Wave':'Wave 1D','CML-intermittency':'CML: intermittency','CML-frozen':'CML: frozen chaos','Gaussian-shared':'Gaussian: shared input','Cauchy-shared':'Cauchy: shared input'}
VIEWS=[(name,method) for method in ['PCA','UMAP'] for name in ['m','z']]

def arm_label(arm):
    if arm=='clean':return 'Clean observations'
    if arm=='noise-limit':return 'Pure Gaussian noise\nSource-parent labels only'
    if arm.startswith('random'):return 'Previous random noise' if arm.endswith('previous') else 'Stronger random noise'
    profile,eta=arm.split('-eta');return ('Equal sensor SD' if profile=='equal' else 'Exponential sensor SD')+'\n'+r'$\eta='+eta+'$'

def plots():
    rows=pd.DataFrame(json.loads((DATA/'manifest.json').read_text())['rows']);train,evaluation,_=split(rows)
    with np.load(OUT/'coordinates.npz') as a:spaces={k:a[k] for k in a.files}
    xy={}
    for name in ['m','z']:
        xy[name+'_PCA']=spaces[name][:,:2]
        model=UMAP(n_neighbors=25,min_dist=.25,random_state=261016,transform_seed=261016,n_jobs=1).fit(spaces[name][train])
        v=np.full((len(rows),2),np.nan);v[train]=model.embedding_;v[evaluation]=model.transform(spaces[name][evaluation]);xy[name+'_UMAP']=v
    np.savez_compressed(OUT/'embedding-coordinates.npz',row_id=rows.row_id.to_numpy(dtype=str),**xy)
    limits={}
    for key,v in xy.items():
        low=v[evaluation].min(0);high=v[evaluation].max(0);pad=.06*(high-low);limits[key]=[(low[i]-pad[i],high[i]+pad[i]) for i in range(2)]
    style()
    sets={'dose-main':['clean','equal-eta2','exponential-eta2','equal-eta8','exponential-eta8','noise-limit'],
          'random':['clean','random-previous','random-stronger'],
          'dose-equal-full':['clean']+[f'equal-eta{e:g}' for e in ETAS[1:]]+['noise-limit'],
          'dose-exponential-full':['clean']+[f'exponential-eta{e:g}' for e in ETAS[1:]]+['noise-limit']}
    for figure,arms in sets.items():
        for coloring in (['class','strength'] if figure in ['dose-main','random'] else ['class']):
            fig,axes=plt.subplots(len(arms),4,figsize=(12,2.8*len(arms)+.8),layout='constrained',squeeze=False)
            artist=None
            for i,arm in enumerate(arms):
                use=evaluation & rows.arm.eq(arm).to_numpy()
                for j,(name,method) in enumerate(VIEWS):
                    ax=axes[i,j];key=name+'_'+method;v=xy[key]
                    for label in CLASSES:
                        ix=use & rows.label.eq(label).to_numpy()
                        kwargs=dict(c=rows.loc[ix,'mean_abs_r'],cmap='viridis',vmin=0,vmax=1) if coloring=='strength' else dict(c=COLORS[label])
                        artist=ax.scatter(*v[ix].T,s=20,alpha=.8,linewidths=0,**kwargs)
                    ax.set(xlim=limits[key][0],ylim=limits[key][1],xticks=[],yticks=[],xlabel=method+' 1',ylabel=(arm_label(arm)+'\n' if j==0 else '')+method+' 2')
                    ax.set_box_aspect(1)
                    for spine in ax.spines.values():spine.set_visible(True)
                    if i==0:ax.set_title((r'Mean-MPI $m$' if name=='m' else r'SPI–SPI $z$')+' — '+method)
            if coloring=='class':
                handles=[Line2D([],[],marker='o',ls='',color=COLORS[c],markersize=4,label=DISPLAY[c]) for c in CLASSES]
                fig.legend(handles=handles,loc='outside lower center',ncol=4,fontsize=8)
            else:fig.colorbar(artist,ax=axes.ravel().tolist(),shrink=.6,label=r'Observed mean $|r|$')
            fig.suptitle('Fixed latent dynamics · M=16, T=1000 · 16 evaluation instances/class\nOne training-fitted coordinate system per representation; all severe-noise cases retained')
            save(fig,OUT/'figures',figure+'-'+coloring);plt.close(fig)
    metrics=pd.read_csv(OUT/'metrics.csv').set_index(['representation','arm'])
    fig,axes=plt.subplots(1,4,figsize=(14,3.6),layout='constrained')
    for rep,color in [('m','#D55E00'),('z','#0072B2')]:
        for profile,ls in [('equal','-'),('exponential','--')]:
            arms=['clean']+[f'{profile}-eta{e:g}' for e in ETAS[1:]]+['noise-limit']
            d=metrics.loc[[(rep,a) for a in arms]]
            for ax,key in zip(axes,['mAP','spread_to_separation','accuracy','paired_movement']):
                ax.plot(range(7),d[key].to_numpy(),color=color,ls=ls,marker='o',ms=3,label=f'{rep}: {profile}')
                ax.set(xticks=range(7),xticklabels=['0','.5','1','2','4','8','Noise\nonly'],xlabel=r'Noise SD $\eta$; final point is the pure-noise limit')
    axes[0].set(ylabel='Clean-reference retrieval mAP',ylim=(0,1.03));axes[1].set(ylabel='Within-class spread / separation',yscale='log');axes[2].set(ylabel='Fixed-readout classification accuracy',ylim=(0,1.03));axes[2].axhline(1/7,color='.6',ls=':',lw=1);axes[3].set(ylabel='Paired movement from clean\nRelative to training class separation')
    axes[0].legend(fontsize=8);fig.suptitle('All doses · equal and exponential profiles matched in expected total noise power')
    save(fig,OUT/'figures','dose-curves');plt.close(fig)
    (OUT/'plot-audit.json').write_text(json.dumps(dict(source_sha256=sha(__file__),coordinate_sha256=sha(OUT/'embedding-coordinates.npz'),training=336,evaluation=1568,UMAP=dict(neighbors=25,min_dist=.25,seed=261016),selected_main_arms=sets['dose-main'],all_doses_retained=True),indent=2)+'\n')

def append():
    md=nbformat.v4.new_markdown_cell;code=nbformat.v4.new_code_cell
    cells=[md(r'''## Paired noise-dose and stronger random-noise experiments — 16 October

Fresh M=16/T=1000 recordings, 32 realizations/class, seven classes including CML intermittency/frozen chaos. Physical coupling is fixed. Each parent supplies identical latent data and independent Gaussian error samples to all its conditions. The 2,688-record dose bank tests $\eta=0,.5,1,2,4,8$, equal versus exponential relative channel SDs, plus the error array alone as a pure-noise limit. The 448-record random bank compares matched-quantile attenuations $U[.25,.95]$ and $U[.05,.50]$. No failed seed, class or severe dose is removed.

For the unequal-SD condition, $w_i\sim\mathrm{Exp}(1)$ and $\eta_i=\eta w_i/\sqrt{M^{-1}\sum_j w_j^2}$. The mean channel noise variance is exactly $\eta^2$ before observation standardization; realized noise energy is audited separately. Equal power is **not equal average attenuation**: uneven allocation leaves some sensors cleaner. At eta8, theoretical mean pair attenuation averages .015 for equal noise versus .129 for these exponential profiles (finite-variance model, not a Cauchy population identity).

Instances 0–15 at clean/equal-eta1/exponential-eta1 fit all transforms and readouts; instances 16–31 evaluate every condition. Clean-gallery retrieval uses different parents. Primary means retain training feature scaling/clipping but no per-recording normalization. Full/unclipped means, standardized/complete-training z and validity-only prediction remain controls. A secondary direction-normalized mean vector subtracts the training pure-noise centroid and normalizes length; it is not the primary baseline. Common PCA50/UMAP coordinates and limits are fixed across conditions. The main montage shows prospectively selected clean/eta2/eta8/limit; all doses appear in the curves and full-profile figure files. Intervals resample instance blocks, retaining classes/conditions, conditional on training.'''),
    code("from pathlib import Path\nimport json,pandas as pd\nfrom IPython.display import display,Image\nNE_ROOT=Path.cwd().resolve()\nwhile not (NE_ROOT/'pyproject.toml').exists() and NE_ROOT!=NE_ROOT.parent:NE_ROOT=NE_ROOT.parent\nNE_OUT=NE_ROOT/'results/proof/noise-experiments-261016'\ndisplay(Image(filename=str(NE_OUT/'figures/dose-curves.png')))\ndisplay(Image(filename=str(NE_OUT/'figures/dose-main-class.png')))"),
    code("display(Image(filename=str(NE_OUT/'figures/random-class.png')))\nne_scores=pd.read_csv(NE_OUT/'metrics.csv')\ndisplay(ne_scores.query(\"representation in ['m','z'] and arm in ['clean','equal-eta8','exponential-eta8','noise-limit','random-previous','random-stronger']\").pivot(index='arm',columns='representation',values=['mAP','spread_to_separation','accuracy']).round(3))"),
    md(r'''**Stronger noise preserves a raw-mean retrieval contrast, but does not improve it monotonically.** For previous → stronger random noise, m/z clean-reference mAP changes .411/.810 → .278/.669; the stronger-arm difference is .391 [95% paired interval .319,.455]. Spread/separation changes .607/.201 → .523/.287: increasing noise actually narrows this geometry advantage. Means contract towards the noise reference rather than becoming literally more spread out. The PCA plots show this contraction; UMAP retains some mean clusters and mixes others. Both use fixed training coordinates, so collapse is not evidence that all residual information is absent. Full-mean linear accuracy remains .821 (z .911).

**Noise allocation matters, and severe failures remain.** At equal eta2, m/z retrieval is .277/.586; at eta8 it is .275/.246 and accuracy .143/.161: both fixed readouts fail. Equal-eta8 spread/separation is also worse for z (2.307 versus 1.203). Exponential eta8 instead gives .277/.566 retrieval and .393/.973 accuracy, with spread/separation .623/.265; its cleaner channels preserve usable signal despite identical expected total noise power. This is not a comparison at matched observed dependence. The pure-noise endpoint has no source-family signal by construction; both primary classifiers score 1/7. Its raw AP differs (.275/.221) because gallery rankings have different class structure: the exact randomized-query-label AP expectation is reported as `mAP_label_null`, and excess over it is approximately zero for both. Small UMAP islands at this endpoint are projection artifacts, not recovered families.

**A simple normalized-mean control remains competitive.** Subtracting the training noise centroid and normalizing mean-vector length gives stronger-random mAP .665 (z .669), exponential-eta8 .777 (z .566), and equal-eta8 .373 with accuracy .580 (z .161). Thus raw means lose useful distance geometry before their information disappears, and the advantage is not unique to SPI–SPI. The main baseline remains raw m. Unclipped/full means give the same retrieval conclusion; standardized z is weaker (stronger-random .452), whereas complete-training z improves to .740. Validity alone classifies .295 under stronger random noise; complete-training z still has .138% missing held entries there, so estimator validity is not wholly eliminated as a confound. Kuramoto remains difficult for clean-reference z retrieval (.166); benefits are not uniform across families.

All 3,136 records completed with unchanged p90/r9 provenance. Independent checks of 28,322 means and 9,800 z values agree within $3\times10^{-8}$. These experiments support robustness of the specified representation/readout to observation noise, not identified causal/nonlinear character or intrinsic marginal-information absence. All-dose figures and mean-$|r|$ colour analogues are saved under `results/proof/noise-experiments-261016/figures/`; no held-result choice of seeds, doses, classes or embedding settings was made.'''),
    code("ne_arms=['equal-eta8','exponential-eta8','noise-limit','random-stronger']\ndisplay(ne_scores.query('arm in @ne_arms').pivot(index='representation',columns='arm',values=['mAP','mAP_excess','accuracy']).round(3))\ndisplay(pd.read_csv(NE_OUT/'paired-intervals.csv').query('arm in @ne_arms').round(3))\nprint('Validity-only accuracy:',json.loads((NE_OUT/'analysis-audit.json').read_text())['validity_only_accuracy'])")]
    for c in cells:c.metadata['tags']=[TAG]
    assert 'RESULT_INTERPRETATION_PENDING' not in ''.join(c.source for c in cells),'Write the result interpretation before appending'
    original=NOTEBOOK.read_bytes();nb=nbformat.reads(original.decode(),4);preserved=[c for c in nb.cells if TAG not in c.metadata.get('tags',[])]
    addition=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    NotebookClient(addition,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    assert NOTEBOOK.read_bytes()==original,'Notebook edited during execution; rerun with saved version'
    nb.cells=preserved+addition.cells;nbformat.write(nb,NOTEBOOK);assert nbformat.read(NOTEBOOK,4).cells[:len(preserved)]==preserved
    (OUT/'notebook-append-audit.json').write_text(json.dumps(dict(preserved_cells=len(preserved),new_cells=len(cells),all_previous_cells_outputs_unchanged=True),indent=2)+'\n')
    print('Preserved',len(preserved),'cells; appended',len(cells))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['plots','append']);a=p.parse_args()
    with threadpool_limits(limits=4):plots() if a.stage=='plots' else append()
