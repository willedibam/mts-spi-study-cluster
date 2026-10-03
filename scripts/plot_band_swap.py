"""Fixed-setting views of the band-swap marginal comparison."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from umap import UMAP
from threadpoolctl import threadpool_limits
from scripts.analyze_band_swap import get_rows
from scripts.build_band_swap import OUT
from scripts.band_organization_experiment import covariance_layers
from scripts.refresh_case_figures import style,save


def population():
    """Population covariances: common colour scale and explicit off-diagonal scope."""
    style();fig,axes=plt.subplots(2,3,figsize=(8.4,5.7),layout='constrained')
    layers=covariance_layers(.55)/3
    for row,label in enumerate(['BCA','CBA']):
        for col,(band,pattern) in enumerate(zip(['Low','Middle','High'],label)):
            matrix=layers['ABC'.index(pattern)].copy();np.fill_diagonal(matrix,np.nan)
            ax=axes[row,col];im=ax.imshow(matrix,vmin=0,vmax=.55/3,cmap='viridis',interpolation='none')
            ax.set(title=f'{band}: pattern {pattern}',xticks=[],yticks=[],xlabel='Channel')
            if col==0:ax.set_ylabel(label+'\nChannel')
    fig.colorbar(im,ax=axes,shrink=.8,label='Population band covariance (diagonal omitted)')
    return save(fig,OUT/'figures','population-coupling')


def plot(stage='development',focused=True):
    rows=get_rows(stage);test=rows.block.ge(24 if stage=='development' else 32).to_numpy()
    prefix='direct-' if focused else '';path=OUT/stage/(prefix+'projections.npz')
    methods=[('band_mean' if focused else 'mean',r'Per-SPI means $m$'),('band_z' if focused else 'z',r'SPI–SPI $z$')]
    names={'BCA':'BCA: mid–high overlap','CBA':'CBA: low–high overlap'};colors={'BCA':'#0072B2','CBA':'#D55E00'}
    style();fig,axes=plt.subplots(2,2,figsize=(8.6,8.6),layout='constrained')
    with np.load(path) as a:
        for col,(method,title) in enumerate(methods):
            d,h=a[method+'_train'],a[method+'_test']
            for row,kind in enumerate(['PCA','UMAP']):
                mapper=PCA(n_components=2) if kind=='PCA' else UMAP(n_neighbors=30,min_dist=.1,random_state=261003,transform_seed=261003,n_jobs=1)
                mapper.fit(d);xy=mapper.transform(h);ax=axes[row,col]
                for label in names:
                    take=rows.loc[test,'label'].eq(label).to_numpy()
                    ax.scatter(*xy[take].T,s=32,color=colors[label],label=names[label],alpha=.8,edgecolors='white',linewidths=.3)
                ax.set(title=title+' — 2 classes',xlabel=kind+' 1',ylabel=kind+' 2');ax.set_box_aspect(1)
                for spine in ax.spines.values():spine.set_visible(True)
                if kind=='UMAP':ax.set(xticks=[],yticks=[])
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2)
    return save(fig,OUT/stage/'figures',('focused' if focused else 'p90')+'-embeddings')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['development','final'],default='development');p.add_argument('--p90',action='store_true');p.add_argument('--population',action='store_true');a=p.parse_args()
    with threadpool_limits(limits=4):population() if a.population else plot(a.stage,not a.p90)
