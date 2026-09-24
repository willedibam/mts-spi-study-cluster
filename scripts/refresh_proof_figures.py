"""Lean proof embeddings and train-selected SPI-pair rainclouds for the 260924 rerun."""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from umap import UMAP
import yaml

from scripts.refresh_case_figures import style, save, raincloud

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'configs/analysis/proof-cases-refresh-260924.yaml'
CML=['frozen-chaos','sti-i','defect-turbulence','fdstc']
DISPLAY={'frozen-chaos':'Frozen chaos','sti-i':'Spatiotemporal intermittency',
         'defect-turbulence':'Defect turbulence','fdstc':'Fully developed turbulence',
         'cauchy-noise':'Cauchy noise','gaussian-noise':'Gaussian noise',
         'kuramoto_omega-fast':'Kuramoto: fast frequencies','kuramoto_omega-slow':'Kuramoto: slow frequencies',
         'wave-1d':'Wave equation'}
CML_COLORS=dict(zip(CML,['#CC79A7','#E69F00','#009E73','#D55E00']))


def display_name(name):
    return DISPLAY.get(name,name.replace('var-self-', 'VAR: self ').replace('_total-', ', total coupling ').replace('var-phi-', 'VAR: requested self ').replace('_cpl-', ', neighbour '))


def load_bank(path):
    # Repository-generated v2 archives store feature-name arrays as objects.
    with np.load(path,allow_pickle=True) as d:
        bank={k:d[k] for k in ['X_sym','y','M','T','instance','spi_order','feature_spi_a','feature_spi_b','feature_block']}
        bank['provenance']=json.loads(str(d['pyspi_provenance_json']))
    bank['y']=bank['y'].astype(str);bank['instance']=bank['instance'].astype(int)
    sym=bank['feature_block'].astype(str)=='sym'
    bank['pairs']=np.column_stack([bank['feature_spi_a'][sym],bank['feature_spi_b'][sym]]).astype(str)
    assert bank['X_sym'].shape==(1080,41616),bank['X_sym'].shape
    assert len(bank['spi_order'])==289
    keys=list(zip(bank['y'],bank['M'],bank['T'],bank['instance']))
    assert len(set(keys))==1080
    for cls in np.unique(bank['y']):
        for m in [8,16,32]:
            for t in [500,1000,2000]:
                got=bank['instance'][(bank['y']==cls)&(bank['M']==m)&(bank['T']==t)]
                assert sorted(got)==list(range(10)),(cls,m,t,got)
    colors=dict(CML_COLORS)
    others=[x for x in sorted(set(bank['y'])) if x not in colors]
    for name,color in zip(others,['#0072B2','#56B4E9','#332288','#88CCEE','#882255','#AA4499','#999933','#555555']):colors[name]=color
    bank['colors']=colors
    return bank


def preprocess(X):
    finite=np.isfinite(X)
    keep=finite.mean(0)>=.95
    Y=X[:,keep].astype(float)
    med=np.nanmedian(np.where(np.isfinite(Y),Y,np.nan),axis=0)
    Y=np.where(np.isfinite(Y),Y,med)
    variable=Y.var(0)>1e-8;Y=Y[:,variable]
    keep[np.flatnonzero(keep)[~variable]]=False
    return Y,keep


def embeddings(bank, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    protocol=yaml.safe_load(PROTOCOL.read_text())['proof']
    selections={'inter':~np.isin(bank['y'],protocol['inter_class_exclude']), 'cml':np.isin(bank['y'],CML)}
    coords={};audit={}
    for name,mask in selections.items():
        Y,keep=preprocess(bank['X_sym'][mask]);pca=PCA(n_components=min(50,*Y.shape),svd_solver='randomized',random_state=260924)
        scores=pca.fit_transform(Y)
        umap=UMAP(**protocol['umap'],n_jobs=1).fit_transform(scores)
        coords.update({name+'_indices':np.flatnonzero(mask),name+'_pca':scores[:,:2],name+'_umap':umap,name+'_variance':pca.explained_variance_ratio_[:2]})
        audit[name]={'datasets':int(mask.sum()),'features':int(keep.sum()),'pca50_variance':float(pca.explained_variance_ratio_.sum()),'imputed_values':int((~np.isfinite(bank['X_sym'][mask][:,keep])).sum())}
    np.savez_compressed(output/'embedding-coordinates.npz',**coords)
    (output/'embedding-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    return coords


def plot_embedding(bank, coords, panel, output):
    style();fig,axes=plt.subplots(1,2,figsize=(9.5,4.8));fig.subplots_adjust(bottom=.27,wspace=.3)
    inds=coords[panel+'_indices'];labels=bank['y'][inds];order=[c for c in sorted(set(labels)) if c not in CML]+[c for c in CML if c in labels]
    sizes=np.array([{8:7,16:13,32:23}[int(m)] for m in bank['M'][inds]])
    for ax,method in zip(axes,['pca','umap']):
        xy=coords[panel+'_'+method]
        for cls in order:
            take=labels==cls;ax.scatter(xy[take,0],xy[take,1],s=sizes[take],c=bank['colors'][cls],alpha=.42,linewidths=0,rasterized=True)
        ax.set_box_aspect(1)
        if method=='pca':
            v=coords[panel+'_variance'];ax.set(xlabel=f'PC1 ({v[0]:.1%})',ylabel=f'PC2 ({v[1]:.1%})',title='PCA')
        else:ax.set(xlabel='UMAP 1',ylabel='UMAP 2',xticks=[],yticks=[],title='UMAP')
    handles=[Line2D([],[],marker='o',ls='',color=bank['colors'][c],markersize=5,label=display_name(c)) for c in order]
    fig.legend(handles=handles,loc='lower center',ncols=2 if len(order)>4 else 2,fontsize=7.5,bbox_to_anchor=(.5,.01))
    return save(fig,Path(output),panel+'-embeddings')


def auc(a,b):
    a=np.asarray(a);b=np.asarray(b);a=a[np.isfinite(a)];b=b[np.isfinite(b)]
    if not len(a) or not len(b):return np.nan
    ranks=rankdata(np.r_[a,b]);return float((ranks[len(a):].sum()-len(b)*(len(b)+1)/2)/(len(a)*len(b)))


def rank_pair(bank,class_a,class_b):
    train=bank['instance']<6
    a=bank['X_sym'][train&(bank['y']==class_a)];b=bank['X_sym'][train&(bank['y']==class_b)]
    valid=np.isfinite(a).all(0)&np.isfinite(b).all(0)
    both=np.vstack([a[:,valid],b[:,valid]])
    ranks=rankdata(both,axis=0);aucs=(ranks[len(a):].sum(0)-len(b)*(len(b)+1)/2)/(len(a)*len(b))
    separation=np.abs(aucs-.5);winner=int(np.argmax(separation));col=int(np.flatnonzero(valid)[winner])
    test=bank['instance']>=6
    aa=bank['X_sym'][test&(bank['y']==class_a),col];bb=bank['X_sym'][test&(bank['y']==class_b),col]
    orientation=1 if aucs[winner]>=.5 else -1
    held=auc(aa,bb);held=held if orientation>0 else 1-held
    return {'class_a':class_a,'class_b':class_b,'feature_index':col,'spi_a':bank['pairs'][col,0], 'spi_b':bank['pairs'][col,1],
            'training_auc_oriented':float(.5+separation[winner]),'display_auc_fixed_orientation':float(held),
            'training_orientation':orientation,'display_n_a':int(np.isfinite(aa).sum()),'display_n_b':int(np.isfinite(bb).sum())}


def rank_features(bank,output):
    pairs=list(combinations(CML,2))
    pairs.append(('gaussian-noise','cauchy-noise'))
    var=sorted(c for c in set(bank['y']) if c.startswith('var-'))
    # Compare all VAR pairs so naming changes cannot silently select the wrong contrast.
    pairs.extend(combinations(var,2))
    rows=[rank_pair(bank,a,b) for a,b in pairs]
    frame=pd.DataFrame(rows);frame.to_csv(Path(output)/'selected-features.csv',index=False);return frame


def plot_discriminative(bank,ranked,scope,output):
    style();subset=ranked[ranked.class_a.isin(CML)&ranked.class_b.isin(CML)] if scope=='cml' else ranked[~ranked.class_a.isin(CML)]
    n=len(subset);fig,axes=plt.subplots(int(np.ceil(n/2)),2,figsize=(10,2.8*int(np.ceil(n/2))),layout='constrained',squeeze=False)
    for ax,(_,row) in zip(axes.flat,subset.iterrows()):
        col=int(row.feature_index)
        for pos,cls in enumerate([row.class_a,row.class_b]):
            mask=(bank['y']==cls)&(bank['instance']>=6)
            raincloud(ax,bank['X_sym'][mask,col],pos,bank['colors'][cls],np.random.default_rng(260924+col+pos),width=.65)
        ax.set(xticks=[0,1],xticklabels=[display_name(row.class_a),display_name(row.class_b)],ylim=(-1.05,1.05),xlim=(-.6,1.6),ylabel='SPI–SPI Pearson correlation')
        ax.tick_params(axis='x',labelsize=7)
        # One SPI per line keeps identifiers legible without arbitrary renaming.
        ax.set_title(f'{row.spi_a}\n× {row.spi_b}',fontsize=8)
        ax.text(.98,.04,f'Display AUC {row.display_auc_fixed_orientation:.2f}',transform=ax.transAxes,ha='right',fontsize=7)
    for ax in list(axes.flat)[n:]:ax.set_visible(False)
    return save(fig,Path(output),scope+'-discriminative')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('bank',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
    bank=load_bank(args.bank);coords=embeddings(bank,args.output);ranked=rank_features(bank,args.output)
    for panel in ['inter','cml']:plot_embedding(bank,coords,panel,args.output)
    for scope in ['cml','coarse']:plot_discriminative(bank,ranked,scope,args.output)
    plt.close('all')


if __name__=='__main__':main()
