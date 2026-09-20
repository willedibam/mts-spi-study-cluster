import os
import _pickle as cPickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, StandardScaler

resdir = './results'
datatypes = ['CML','VAR','kuramoto','noise']
datatypes = ['noise']
for d in datatypes:
    with open(os.path.join(resdir,d)+'.pkl','rb') as f:
        cf1 = cPickle.load(f)
        try:
            cf.merge(cf1)
        except NameError:
            cf = cf1

print('Getting feature matrix...')
fm = cf.get_feature_matrix(sthresh=0.99).T
fm.fillna(0,inplace=True)

X = RobustScaler(unit_variance=True).fit_transform(fm.values)
fm = pd.DataFrame(data=X,columns=fm.columns,index=fm.index)
fm.fillna(0,inplace=True)

print('Done. Generating clustermap...')
sns.clustermap(fm[['normal' in i for i in fm.index]],figsize=(10,2),yticklabels=False,xticklabels=False,cmap='coolwarm',vmin=-2,vmax=2)
sns.clustermap(fm[['cauchy' in i for i in fm.index]],figsize=(10,2),yticklabels=False,xticklabels=False,cmap='coolwarm',vmin=-2,vmax=2)
print('Done.')

plt.show()