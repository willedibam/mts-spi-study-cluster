import os
import dill

import matplotlib.pyplot as plt
from matplotlib import colors, cm
import seaborn as sns
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, linkage, cophenet, dendrogram, set_link_color_palette

import numpy as np
from scipy.stats import zscore
import random

random.seed(1)

basedir = os.path.dirname(os.path.abspath(__file__))
savedir = os.path.join(basedir,'plots')

# Load pre-computed spearman adjacency matrix
path = os.path.join('results', 'dd_adj.csv')
dd_adj = pd.read_csv(path,index_col=0)
mvtsnames = dd_adj.columns
np.fill_diagonal(dd_adj.values,np.nan)

with open('data/database.pkl','rb') as f:
    database = dill.load(f)

with open('data/database-archive.pkl','rb') as f:
    database_archive = dill.load(f)

database.update(database_archive)

# Normalize all entries
for name in database:
    database[name]['data'] = zscore(database[name]['data'])

print(f'Average: {dd_adj.mean().mean()}')

method = 'average'

y = dd_adj.fillna(0).values[np.triu_indices(dd_adj.shape[0],1)]
Z = linkage(y,metric='euclidean',method=method,optimal_ordering=True)

threshold = 100

C = cophenet(Z,y)
print(f'Cophenetic distance: {C[0]}')

try:
    os.mkdir(savedir)
except FileExistsError:
    pass

fig, ax = plt.subplots(figsize=(100, 15))

clusters = fcluster(Z,threshold,criterion='distance')
uniq_clusters = np.unique([c for c in clusters if list(clusters).count(c) > 1])
print(f'There are {uniq_clusters.size} unique (non-singleton) clusters.')

cmap = cm.get_cmap('turbo')
cols = [colors.to_hex(cmap(c)) for c in np.linspace(0,1,uniq_clusters.size)]

set_link_color_palette(cols)
dn = dendrogram(Z,labels=mvtsnames,orientation='top',
                    color_threshold=threshold,leaf_font_size=4,
                    count_sort='ascending',above_threshold_color='k')
plt.axvline(x=threshold, c='grey', lw=1, linestyle='dashed')

min_csize = 20
uniq_cols = np.unique(dn['color_list'])
large_clusters = uniq_cols[[list(dn['color_list']).count(c) > min_csize for c in uniq_cols]]
large_clusters = np.setdiff1d(large_clusters,['k'])

large_cluster_ids = [np.where([c1 == c0 for c1 in cols])[0][0] for c0 in large_clusters]
large_clusters_sorted = [c for _, c in sorted(zip(large_cluster_ids,large_clusters))]

print(f'There are {len(large_clusters)} clusters with >{min_csize} datasets in them.')

if savedir is not None:
    fig.savefig(savedir+'/dendrogram.jpg',dpi=300,bbox_inches='tight',pad_inches=0)
    fig.savefig(savedir+'/dendrogram.pdf',dpi=300,bbox_inches='tight',pad_inches=0)
    plt.close(fig)
else:
    plt.show()

# How many MTS are we going to sample from the dendrogram?
nsamples = 4

for r, col in enumerate(large_clusters_sorted):

    fig, axs = plt.subplots(nrows=1,ncols=nsamples,figsize=(15,2))
    idxs = np.where([c == col for c in dn['leaves_color_list']])[0]

    fig.patch.set_facecolor(colors.to_rgb(col))

    samples = random.sample(sorted(idxs),nsamples)
    for c, sample in enumerate(samples):
        mvts = dn['ivl'][sample]
        try:
            data = database[mvts]['data'].T
            axs[c].pcolormesh(data,cmap=sns.color_palette('icefire_r', as_cmap=True))
        except KeyError as err:
            print(err)
            
        axs[c].set_title(mvts,fontdict={'fontsize': 6})
        axs[c].tick_params(labelsize=6)

    plt.tight_layout(h_pad=1.5,w_pad=1.05)
    if savedir is not None:
        fig.savefig(savedir+f'/cluster-{r}.jpg',dpi=300,bbox_inches='tight',pad_inches=0)
        fig.savefig(savedir+f'/cluster-{r}.pdf',dpi=300,bbox_inches='tight',pad_inches=0)
        plt.close(fig)
plt.show()