import os
import _pickle as cPickle
import pandas as pd, numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore

basedir = os.path.dirname(os.path.abspath(__file__))
savedir = os.path.join(basedir,'plots')

path = os.path.join('results', 'library_df.pkl')
print(f'Loading CorrelationFrame from {path}...')
with open(path,'rb') as f:
    cf = cPickle.load(f)
print('Done.')

# Load the low-dimensional data list
lowdim = pd.read_csv('results/lowdim.csv',index_col=0).values

print('Getting feature matrix...')
feature_matrix = cf.get_feature_matrix()

feature_matrix = feature_matrix[[idx for idx in feature_matrix.columns if idx not in lowdim]]

# Pick subset that passed our tests
non_variant = pd.read_csv('variant_test/features.csv',index_col=0)
nonvariant_idx = [(s0,s1) for s0, s1 in non_variant.values if (s0,s1,'spearman') in feature_matrix.index]
feature_matrix = feature_matrix.droplevel(2).T[nonvariant_idx].T

# Get rid of features that didn't compute
feature_matrix = feature_matrix.apply(lambda x : zscore(x,nan_policy='omit'),axis=1)
feature_matrix = feature_matrix.dropna(axis=0,thresh=0.9*feature_matrix.shape[1]).dropna(axis=1,thresh=0.8*feature_matrix.shape[0])
feature_matrix = feature_matrix.fillna(0)

print(f'Size: {feature_matrix.shape}.')

print('Computing Euclidean distance between datasets...')
data = squareform(pdist(feature_matrix.T.values))
dd_adj = pd.DataFrame(data=data,columns=feature_matrix.columns,index=feature_matrix.columns)
fname = os.path.join('results','dd_adj.csv')
print(f'Done, saving to {fname}.')
dd_adj.to_csv(fname)