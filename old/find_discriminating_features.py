import os, random
import pandas as pd, seaborn as sns, numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt

# Classifier stuff
from sklearn import svm
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

random.seed(42)

use_cv = False
n_repeats = 1

resdir = './results'
datatypes = ['CML','VAR','kuramoto','noise']
for d in datatypes:
    with open(os.path.join(resdir,d)+'.pkl','rb') as f:
        cf1 = cPickle.load(f)
        try:
            cf.merge(cf1)
        except NameError:
            cf = cf1

feature_matrix = cf.get_feature_matrix()

classes = [['var'],['coupled map lattice']]

cf.set_dgroups(classes=classes)
ids = np.array(cf.get_dgroup_ids(feature_matrix.columns))

cls_idx = np.where(ids>-1)[0]
print(f'Found {len(cls_idx)} datasets with relevant labels.')

X = feature_matrix.iloc[:,cls_idx].values.T
y = ids[cls_idx]

scaler = StandardScaler()
clf = svm.SVC(kernel='linear') # Linear Kernel
sfs = SequentialFeatureSelector(clf, n_features_to_select=1,cv=5,scoring='balanced_accuracy')

if use_cv:
    X = scaler.fit_transform(X)
    np.nan_to_num(X,copy=False)

    print(f'Finding reduced feature set...')
    sfs.fit(X, y)

    feature = feature_matrix.index[sfs.get_support()][0]
    print(f'Chosen feature: {feature}')

    X = sfs.transform(X)

    score = np.mean(cross_val_score(clf, X, y, cv=5))
else:
    scores = {}
    for r in range(n_repeats):
        print(f'Repeat {r}/{n_repeats}.')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

        X_train = scaler.fit_transform(X_train)
        np.nan_to_num(X_train,copy=False)
        X_test = scaler.transform(X_test)
        np.nan_to_num(X_test,copy=False)

        sfs.fit(X_train,y_train)

        feature = feature_matrix.index[sfs.get_support()][0]
        print(f'Chosen feature: {feature}')

        X_train = sfs.transform(X_train)
        X_test = sfs.transform(X_test)

        clf.fit(X_train,y_train)
        score = balanced_accuracy_score(y_test,clf.predict(X_test))
        print(f'Score: {score}')
        scores[feature] = score

    feature = max(scores,key=scores.get)
    score = scores[feature]

print(f'Final score: {score}')

feature_name = ', '.join(feature)
data = feature_matrix.loc[feature]
data.name = feature_name
labels = pd.Series(data=cf.get_dgroup_names(feature_matrix.columns),index=feature_matrix.columns,name='label')
reduced_feature_matrix = pd.concat([data,labels],axis=1)
reduced_feature_matrix = reduced_feature_matrix[reduced_feature_matrix['label'] != 'N/A']

sns.histplot(data=reduced_feature_matrix, x=feature_name, hue='label',bins=50)
plt.show()