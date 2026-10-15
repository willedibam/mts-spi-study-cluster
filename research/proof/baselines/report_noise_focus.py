"""Focused noise-only presentation of frozen factorial coordinates; no new fits."""
import json
import hashlib
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd
from research.proof.baselines.strength_factorial import ROOT,DATA,OUT,CLASSES
from research.proof.baselines.report_strength_factorial import NOTEBOOK,COLORS,DISPLAY
from research.proof.baselines.proof_strength_nuisance import style,save

TAG='noise-focus-261015'

def main(strength_only=False):
    source=OUT/'embedding-coordinates.npz';before=hashlib.sha256(source.read_bytes()).hexdigest()
    rows=pd.DataFrame(json.loads((DATA/'manifest.json').read_text())['rows'])
    with np.load(source) as a:
        np.testing.assert_array_equal(a['row_id'],rows.row_id)
        xy={k:a[k] for k in ['m_PCA','z_PCA','m_UMAP','z_UMAP']}
    use=rows.label.isin(CLASSES) & rows.instance.ge(20) & rows.arm.isin(['fixed_clean','fixed_sensor'])
    assert use.sum()==280
    # Pair identity is established before plotting; all evaluation instances remain.
    clean=rows[use & rows.arm.eq('fixed_clean')].set_index(['label','instance'])
    noisy=rows[use & rows.arm.eq('fixed_sensor')].set_index(['label','instance'])
    assert clean.index.equals(noisy.index)
    style()
    limits={}
    for key,v in xy.items():
        low=np.min(v[use],0);high=np.max(v[use],0);pad=.08*(high-low)
        limits[key]=[(low[i]-pad[i],high[i]+pad[i]) for i in range(2)]
    for suffix,labels in [('all',CLASSES),('dynamics',tuple(c for c in CLASSES if not c.endswith('shared')))]:
        fig,axes=plt.subplots(2,4,figsize=(12,6.8),layout='constrained')
        for col,key in enumerate(xy):
            rep,method=key.split('_');v=xy[key]
            for row,arm in enumerate(['fixed_clean','fixed_sensor']):
                ax=axes[row,col]
                for label in labels:
                    chosen=use & rows.label.eq(label) & rows.arm.eq(arm)
                    color_args=dict(c=rows.loc[chosen,'mean_abs_r'],cmap='viridis',vmin=0,vmax=1) if strength_only else dict(c=COLORS[label])
                    artist=ax.scatter(*v[chosen].T,s=22,alpha=.85,linewidths=0,**color_args)
                ax.set(xlim=limits[key][0],ylim=limits[key][1],xticks=[],yticks=[],xlabel=method+' 1',ylabel=('Clean observations\n' if row==0 else 'Varying sensor noise\n')+method+' 2' if col==0 else method+' 2')
                ax.set_box_aspect(1)
                for spine in ax.spines.values():spine.set_visible(True)
                if row==0:ax.set_title((r'Mean-MPI $m$' if rep=='m' else r'SPI–SPI $z$')+' — '+method)
        if strength_only:
            fig.colorbar(artist,ax=axes.ravel().tolist(),shrink=.8,label=r'Observed mean $|r|$')
        else:
            handles=[Line2D([],[],ls='',marker='o',color=COLORS[c],markersize=4,label=DISPLAY[c]) for c in labels]
            fig.legend(handles=handles,loc='outside lower center',ncol=4,fontsize=8)
        fig.suptitle('Same latent dynamics and physical coupling; different recording quality\nM=16, T=1000 · 20 evaluation instances/class · fixed coordinates and axes'+('\nShared-input classes hidden without refitting' if suffix=='dynamics' else ''))
        save(fig,OUT/'figures','noise-focus-'+suffix+('-strength' if strength_only else ''));plt.close(fig)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    audit=dict(source='strength-factorial-261015',embedding_sha256=before,refitted=False,instances=list(range(20,40)),main_classes=list(CLASSES),subset_rule='Secondary display omits shared-input controls categorically; no refit or axis change.',arms=['fixed_clean','fixed_sensor'],reference_controls='Independent references omitted from this focused display; retained in original factorial figures.',selection='Post-result presentation of the already-completed noise-only arm, not a new confirmation.')
    if strength_only:audit['coloring']='Observed mean absolute Pearson correlation; shared viridis [0,1]. Coordinates, limits and selection unchanged.'
    (OUT/('noise-focus-strength-audit.json' if strength_only else 'noise-focus-audit.json')).write_text(json.dumps(audit,indent=2)+'\n')
    md=nbformat.v4.new_markdown_cell;code=nbformat.v4.new_code_cell
    additions=[md(r'''## Focused interpretation: fixed coupling, variable recording quality

This is a **post-result presentation of cached data**, with no new embedding fit or parameter search. The earlier fixed-size factorial has 20 evaluation instances/class at M=16, T=1000, including nominal CML **defect turbulence**; the newer size-grid appendix instead uses frozen chaos. Every clean/noisy pair has identical latent data and physical coupling. Main figure: all seven classes; the second figure hides shared-input controls categorically, with identical coordinates/limits. Independent references remain in the original figures above.

For unit-variance, finite-variance latent channels and independent unit-variance errors, $y_i=(x_i+\eta_d\epsilon_i)/\sqrt{1+\eta_d^2}$ gives $\mathrm{corr}(y_i,y_j)=\alpha_d\mathrm{corr}(x_i,x_j)$, where $\alpha_d=(1+\eta_d^2)^{-1}$ varies between recordings. Importantly, adding independent noise **without restandardizing does not attenuate the population off-diagonal covariance**; attenuation here concerns correlation and the covariance of standardized observations. The Cauchy control has no population covariance; the identity is not asserted for it.

If each noisy SPI profile were a positive affine transform of its clean profile, $v'_k=a_kv_k+b_k\mathbf{1}$, then $m'_k=a_km_k+b_k$ while $z'_{kl}=z_{kl}$ exactly. Actual sensor noise only approximates this situation for some statistics, and sufficiently heavy noise eventually destroys both representations. The present result supports partial robustness, not universal strength invariance.

Noise-only clean-reference retrieval changes from .912 to .409 for means, and .977 to .797 for z; spread/separation changes .288→.483 versus .184→.212. Means’ **absolute within-class variance decreases** (77.8→26.6), but between-class separation decreases more (269.7→55.1): the geometry loses relative separation rather than simply expanding. Full-mean classification remains high. This isolates an observation nuisance more cleanly than varying physical parameters, which can also alter synchronization, timescales or CML regimes.'''),code("from pathlib import Path\nfrom IPython.display import display,Image\nNOISE_ROOT=Path.cwd().resolve()\nwhile not (NOISE_ROOT/'pyproject.toml').exists() and NOISE_ROOT!=NOISE_ROOT.parent:NOISE_ROOT=NOISE_ROOT.parent\nNOISE_FIG=NOISE_ROOT/'results/proof/strength-factorial-261015/figures'\ndisplay(Image(filename=str(NOISE_FIG/'noise-focus-all.png')))\ndisplay(Image(filename=str(NOISE_FIG/'noise-focus-dynamics.png')))")]
    tag=TAG
    if strength_only:
        tag='noise-focus-strength-261016'
        additions=[md(r"""### The same views coloured by observed mean $|r|$

Only point colour changes; the all-class/five-dynamical-class selections, coordinates and axis limits match the preceding figures. Both rows share the viridis scale [0,1]. Noise samples already follow independent $N(0,1)$ draws across channels/time; the uniform draw is the recording-level nominal attenuation $\alpha_d\sim U[.25,.95]$, converted to a shared channel noise SD $\eta_d=\sqrt{1/\alpha_d-1}$. Unequal sensor noise SDs and exponential noise samples have not been tested here.

The combined intervention gives a stronger **relative compactness** contrast than noise alone: spread/separation rises from .288/.184 (m/z) to 1.130/.380, factors 3.92/2.07. Noise-only factors are 1.67/1.16. Conversely, the **retrieval** advantage is larger under noise alone (mAP difference .387) than both (.260). These are different properties of the same geometry; neither negates the other. Noise-only remains the cleaner intervention for holding latent dynamics fixed."""),code("from pathlib import Path\nfrom IPython.display import display,Image\nNS_ROOT=Path.cwd().resolve()\nwhile not (NS_ROOT/'pyproject.toml').exists() and NS_ROOT!=NS_ROOT.parent:NS_ROOT=NS_ROOT.parent\nNS_FIG=NS_ROOT/'results/proof/strength-factorial-261015/figures'\ndisplay(Image(filename=str(NS_FIG/'noise-focus-all-strength.png')))\ndisplay(Image(filename=str(NS_FIG/'noise-focus-dynamics-strength.png')))")]
    for c in additions:c.metadata['tags']=[tag]
    original=NOTEBOOK.read_bytes();nb=nbformat.reads(original.decode(),4)
    preserved=[c for c in nb.cells if tag not in c.metadata.get('tags',[])]
    extra=nbformat.v4.new_notebook(cells=additions,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    NotebookClient(extra,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    assert NOTEBOOK.read_bytes()==original,'Notebook changed; rerun against saved version'
    nb.cells=preserved+extra.cells;nbformat.write(nb,NOTEBOOK)
    assert nbformat.read(NOTEBOOK,4).cells[:len(preserved)]==preserved
    print('Preserved',len(preserved),'cells; appended',len(extra.cells),'cells. No coordinates changed.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--strength-only',action='store_true')
    main(parser.parse_args().strength_only)
