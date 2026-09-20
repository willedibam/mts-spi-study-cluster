import os, random
import pandas as pd, numpy as np
import _pickle as cPickle
from scipy.stats import ks_2samp

random.seed(42)

use_cv = False
n_repeats = 1

cls = 'cml_normal'
# cls = 'var_cml'
# cls = 'var_osc'
# cls = 'osc_cml'
# cls = 'normal_cauchy'
# cls = 'var_noise'

resdir = './results'
datatypes = ['CML','VAR','kuramoto','noise','VAR_5back']
for d in datatypes:
    with open(os.path.join(resdir,d)+'.pkl','rb') as f:
        cf1 = cPickle.load(f)
        try:
            cf.merge(cf1)
        except NameError:
            cf = cf1

feature_matrix = cf.get_feature_matrix(sthresh=0.95,dthresh=0.05,dropduplicates=False)

if cls == 'cml_normal':
    classes = [['normal'],['coupled map lattice']]
elif cls == 'normal_cauchy':
    classes = [['normal'],['cauchy']]
elif cls == 'var_noise':
    classes = [['var'],['normal']]
elif cls == 'var_cml':
    classes = [['var'],['coupled map lattice']]
elif cls == 'var_osc':
    classes = [['var'],['oscillator']]
elif cls == 'osc_cml':
    classes = [['oscillator'],['coupled map lattice']]


cf.set_dgroups(classes=classes)
ids = np.array(cf.get_dgroup_ids(feature_matrix.columns))

cls_idx = np.where(ids>-1)[0]
print(f'Found {len(cls_idx)} datasets with relevant labels.')

data_matrix = feature_matrix.iloc[:,cls_idx]
y = ids[cls_idx]

id0 = y == 0
id1 = y == 1

stats = pd.Series(index=data_matrix.index,data=np.full(data_matrix.shape[0],np.nan))
for i, f in enumerate(data_matrix.index):
    print(f'[{i}/{len(data_matrix.index)}] feature {f}.')
    x0 = data_matrix.loc[f][id0].values
    x1 = data_matrix.loc[f][id1].values
    stats[f], _ = ks_2samp(x0,x1)

stats.to_csv(f'results/ks_{cls}.csv')