import os
import _pickle as cPickle
import dill

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import seaborn as sns

import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import robust_scale
import random
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

random.seed(1)

basedir = os.path.dirname(os.path.abspath(__file__))
savedir = os.path.join(basedir,'plots/fig4/')

# resdir = './results'
# datatypes = ['CML','VAR','kuramoto','noise']
# for d in datatypes:
#     with open(os.path.join(resdir,d)+'.pkl','rb') as f:
#         cf1 = cPickle.load(f)
#         try:
#             cf.merge(cf1)
#         except NameError:
#             cf = cf1

path = 'results/library_df.pkl'
print(f'Loading CorrelationFrame from {path}...')
with open(path,'rb') as f:
    cf = cPickle.load(f)
print('Done.')

print('Getting feature matrix...')
feature_matrix = cf.get_feature_matrix().fillna(0)
print('Done.')

with open('database/database.pkl','rb') as f:
    database = dill.load(f)

with open('database/database-archive.pkl','rb') as f:
    database.update(dill.load(f))

with open('database/bkp-database.pkl','rb') as f:
    database.update(dill.load(f))

# Normalize all entries
for name in database:
    database[name]['data'] = robust_scale(database[name]['data'],unit_variance=True)

approach = 'hdbscan'
cluster_what = 'tsne'

if cluster_what == 'umap':
    reducer = UMAP(random_state=42)
    X_ld = reducer.fit_transform(feature_matrix.T)
elif cluster_what == 'tsne':
    reducer = TSNE(init='pca',learning_rate='auto',random_state=42,perplexity=10)
    X_ld = reducer.fit_transform(feature_matrix.T)

clusterer = None
if approach == 'kmedoids':
    nclusters = 200
    from sklearn_extra.cluster import KMedoids
    clusterer = KMedoids(n_clusters=nclusters, random_state=42)
elif approach == 'affinity':
    from sklearn.cluster import AffinityPropagation
    clusterer = AffinityPropagation(random_state=42)
elif approach == 'hdbscan':
    from hdbscan import HDBSCAN
    clusterer = HDBSCAN(min_cluster_size=5,gen_min_span_tree=True)

savedir += cluster_what + '/'

if cluster_what == 'umap' or cluster_what == 'tsne':
    labels = clusterer.fit_predict(X_ld)
else:
    labels = clusterer.fit_predict(feature_matrix.T)
clusters = np.unique(labels)
print(f'Number of clusters: {len(clusters)}')

cmap = sns.color_palette("hls", len(clusters)-1)
# Randomize the order of the colors to make it easier to see
cmap = cmap.as_hex()
random.seed(3)
random.shuffle(cmap)
cmap = sns.color_palette(cmap)

if approach == 'hdbscan':
    plt.subplots()
    clusterer.condensed_tree_.plot(select_clusters=True,selection_palette=cmap)
    _, ax = plt.subplots()
    clusterer.minimum_spanning_tree_.plot(axis=ax,edge_cmap='Spectral',
                                      edge_alpha=1.,
                                      node_size=5,
                                      node_alpha=0.5,
                                      edge_linewidth=.5)

bw_adjust = 1.2
lab = 'TSNE'
if cluster_what == 'umap':
    bw_adjust = 2.6
    lab = 'UMAP'

plt.subplots(ncols=1,nrows=1,figsize=(7,7))
df = pd.DataFrame(X_ld,index=feature_matrix.columns,columns=[f'{lab}-1',f'{lab}-2'])
df['n_processes'] = cf.shapes.loc[df.index]['n_processes']
df['cluster'] = labels
df = df[df['cluster'] != -1]

# focal_clusters = [0,8,19,21,22,26,17,29,41,42,44,48,51,55,57,60]
# sub_clusters = {17: [24,25,35,36],26: [52,53], 60: [61,62,63], 48: [66]}

# focal_clusters = [20,26,27,34,35,37,45,48,55,56,57,63]
focal_clusters = [16,17,38,41,45]
sub_clusters = {38: [48]}

palette = sns.color_palette("husl", len(focal_clusters))

cmap = {c: np.random.uniform(low=0,high=0.6)*np.array([1.,1.,1.]) for c in clusters}
for i, label in enumerate(focal_clusters):
    cmap[label] = palette[i]

for label in sub_clusters:
    for key in sub_clusters[label]:
        cmap[key] = cmap[label]

all_clusters = focal_clusters + [y for x in sub_clusters.values() for y in x]

ids = [c in all_clusters for c in df['cluster']]
ax = sns.kdeplot(data=df[[not i for i in ids]],x=f'{lab}-1',y=f'{lab}-2',hue='cluster',shade=True,bw_adjust=bw_adjust,levels=5,palette=cmap)
ax = sns.kdeplot(data=df[ids],x=f'{lab}-1',y=f'{lab}-2',hue='cluster',shade=True,bw_adjust=bw_adjust,levels=3,palette=cmap)
for c in ax.collections[:]:
    c.set_alpha(0.3)

sns.scatterplot(data=df,x=f'{lab}-1',y=f'{lab}-2',hue='cluster',size='n_processes',sizes=(5,40),palette=cmap)
# plt.legend([],[], frameon=False)
sns.move_legend(
    ax, "lower center",
    bbox_to_anchor=(.5, 1), ncol=10, title=None,
)
plt.tight_layout()
plt.show()

savedir += approach
# If savedir doesn't exist, make it
if not os.path.exists(savedir):
    os.makedirs(savedir)

pd.Series(index=feature_matrix.columns,data=labels).to_csv(f'results/{approach}_{cluster_what}_clusters.csv')

figbase = (5,2)
figbase_dims = (750,10)
for c, cluster in enumerate([c for c in clusters if c != -1]):
    print(f'Plotting cluster {cluster}')

    idxs = np.where([l == cluster for l in labels])[0]

    if idxs.size < 5:
        continue

    for idx in idxs:
        mvts = feature_matrix.columns[idx]
        try:
            data = database[mvts]['data'].T
            vlims = np.percentile(data,(5,95))
            vlim = max(vlims)
            vmin, vmax = -vlim, vlim

            figsize = (figbase[0]*data.shape[1]/figbase_dims[0],figbase[1]*data.shape[0]/figbase_dims[1])
            fig, ax = plt.subplots(figsize=figsize)
            ax.pcolormesh(data,cmap=sns.color_palette('icefire', as_cmap=True),vmin=vmin,vmax=vmax)
        except KeyError as err:
            print(f'{mvts} not found in database.')
            continue
            
        ax.set_aspect(10)
        plt.axis('off')
        
        if savedir is not None:
            clusterdir = savedir+f'/cluster-{cluster}'
            if not os.path.exists(clusterdir):
                os.mkdir(clusterdir)
            fig.savefig(clusterdir+f'/{mvts}.jpg',dpi=100,bbox_inches='tight',pad_inches=0)
        plt.close(fig)