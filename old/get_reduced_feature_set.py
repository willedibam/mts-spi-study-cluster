import os, random
import pandas as pd, numpy as np
import _pickle as cPickle
from scipy.spatial.distance import pdist, squareform
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

random.seed(1)

basedir = os.path.dirname(os.path.abspath(__file__))
savedir = os.path.join(basedir,'plots')

path = os.path.join('results', 'library_df.pkl')
print(f'Loading CorrelationFrame from {path}...')
with open(path,'rb') as f:
    cf = cPickle.load(f)
print('Done.')

print('Getting feature matrix...')
feature_matrix = cf.get_feature_matrix()

# Load pre-computed Euclidean distance matrix
dd_adj = pd.read_csv('results/dd_adj.csv',index_col=0)

# Pick subset that passed our tests
non_variant = pd.read_csv('variant_test/features.csv',index_col=0)
nonvariant_idx = [(s0,s1) for s0, s1 in non_variant.values if (s0,s1,'spearman') in feature_matrix.index]
feature_matrix = feature_matrix.droplevel(2).T[nonvariant_idx].T
feature_matrix = feature_matrix[dd_adj.columns]

final_D = dd_adj.fillna(0).values[np.triu_indices(dd_adj.shape[0],1)]

score = 0
n_features = 1
X = np.nan_to_num(feature_matrix.values).T
X = StandardScaler().fit_transform(X)

pca = PCA(n_components=1000)
pca.fit(X)

plt.subplots()
explained_variance = np.cumsum(pca.explained_variance_ratio_)
plt.plot(explained_variance)
plt.show()

n_clusters = 1
while score < 0.95:
    kmedoids = KMedoids(n_clusters=n_clusters).fit(X)
    reduced_feature_set = feature_matrix.index[kmedoids.medoid_indices_]
    reduced_feature_matrix = feature_matrix.loc[reduced_feature_set]
    
    reduced_D = pdist(np.nan_to_num(reduced_feature_matrix.T.values))
    new_score = np.corrcoef(final_D,reduced_D)[0,1] ** 2

    score = new_score
    n_clusters += 1
    print(f'Score with {len(reduced_feature_set)} features: {score}')

print(f'Done with final score: {score}.')
reduced_feature_matrix.to_csv('results/reduced_feature_set.csv')

print('Computing Euclidean distance between datasets...')
data = squareform(pdist(reduced_feature_matrix.T.values))
dd_adj = pd.DataFrame(data=data,columns=reduced_feature_matrix.columns,index=reduced_feature_matrix.columns)
fname = os.path.join('results','reduced_dd_adj.csv')
print(f'Done, saving to {fname}.')
dd_adj.to_csv(fname)