from pynats.calculator import CorrelationFrame
import os
import _pickle as cPickle
import dill

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import fcluster, linkage, cophenet
from sklearn_extra.cluster import KMedoids
import matplotlib as mpl

import numpy as np
from scipy.stats import zscore
import random

from utils import draw_network

random.seed(1)

basedir = os.path.dirname(os.path.abspath(__file__))
savedir = os.path.join(basedir,'plots')

# Load pre-computed spearman adjacency matrix
path = 'results/dd_adj.csv'
# path = os.path.join('results', 'reduced_dd_adj.csv')
dd_adj = pd.read_csv(path,index_col=0)
mvtsnames = dd_adj.columns
np.fill_diagonal(dd_adj.values,np.nan)

with open('database/database.pkl','rb') as f:
    database = dill.load(f)

with open('database/database-archive.pkl','rb') as f:
    database_archive = dill.load(f)

database.update(database_archive)

# Normalize all entries
for name in database:
    database[name]['data'] = zscore(database[name]['data'])

print(f'Average: {dd_adj.mean().mean()}')

# method = 'weighted'
method = 'weighted'

D = dd_adj.fillna(0).values[np.triu_indices(dd_adj.shape[0],1)]
Z = linkage(D,metric='euclidean',method=method,optimal_ordering=True)

C = cophenet(Z,D)
print(f'Cophenetic distance: {C[0]}')

focal_mvts = [
                ('oscillator_hysteresis_ow--4_nw--1_is-rand_io-neg-M10-T1000',120),
                ('spatiotemporal_intermittency_i_alpha-1-7522_epsilon-0-00115_M20_T500',110),
                ('chaotic_brownian_motion_of_defect_alpha-1-85_epsilon-0-1_M10_T100',120),
                ('wave-1D_M-9_T-1000',155),
                ('brownian_arithmetic_M-25_T-1000',120),
                ('oscillator_sync_k--20_conn_bidir-M10-T100',110),
                ('mousefMRI_S-0_R-23-24',160),
                ('oscillator_sync_k--1_conn_all-M5-T2000',220),
                ('1994-12-18-mw57-fiji-islands-region-5',110),
                ('epidemic_cumulative_C70-84',120),
                ('spatiotemporal_chaos_alpha-2-0_epsilon-0-3_M20_T100',120)
                ]

hierarchical = False
nnodes = 20

# Normalize adjacency to [0,1]
dd_adj = 1-(dd_adj - dd_adj.min())/(dd_adj.max() - dd_adj.min())
for mvts, cutoff in focal_mvts:
    if hierarchical:
        ts = (0.85,0.65)
        ws=(1,0.05)
        cmodules = fcluster(Z,cutoff,criterion='distance')
        mvtsmodmap = {d:m for d,m in zip(mvtsnames,cmodules)}
        print(f'Computing similarity network for {mvts}')
        try:
            mod = mvtsmodmap[mvts]
            nearest_neighbours = [m for m in mvtsnames if mvtsmodmap[m] == mod]
        except KeyError:
            print(f'Time series {mvts} not in matrix.')
            continue
    else:
        ts = (0.95,0.75)
        ws=(1,0.05)
        try:
            ser = dd_adj[mvts]
        except KeyError:
            print(f'Time series {mvts} not in matrix.')
            continue
        nearest_neighbours = ser.sort_values().dropna()[-nnodes:].index.to_list() + [mvts]

    print(f'Found {len(nearest_neighbours)} nearest neighbours for {mvts}.')

    myadj = dd_adj.loc[nearest_neighbours,nearest_neighbours]
    draw_network(myadj,f=mvts,mvts=database,squared=False,use_kk=True,savedir=None,seed=1,pos=None,labels_on=True,ts=ts,ws=ws,
                        edge_kwargs=dict(min_source_margin=15,min_target_margin=15) )
    plt.suptitle(mvts)

plt.show()