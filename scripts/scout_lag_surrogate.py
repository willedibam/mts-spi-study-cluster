"""Reproduce the development-only six-probe feasibility scout; never open held members."""
import json
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from scripts.build_lag_surrogate import DATA, OUT


def probes(x):
    m,t=x.shape; mask=~np.eye(m,dtype=bool)
    profiles=[(x@np.roll(x,k,axis=1).T/t)[mask] for k in [0,1,5,10]]
    for lag in [0,5]:
        v=[]
        for i in range(m):
            js=[j for j in range(m) if i!=j]
            v.extend(mutual_info_regression(np.roll(x[js],lag,axis=1).T,x[i],n_neighbors=4,random_state=261013))
        profiles.append(np.array(v))
    v=np.array(profiles); c=np.corrcoef(v)
    return dict(probe_mean=v.mean(axis=1),probe_z=c[np.triu_indices(len(v),1)],
                covariance=np.array([profiles[0].mean(),abs(profiles[0]).mean()]),
                univariate=np.array([np.mean(x**3),np.mean(x**4)]))


def run():
    rows=pd.DataFrame(json.loads((DATA/'manifest.json').read_text())['rows'][:96])
    bank={}
    with np.load(DATA/'observations.npz') as archive:
        for i,r in rows.iterrows():
            for k,v in probes(archive[r.row_id]).items():bank.setdefault(k,[]).append(v)
            if i%16==0:print('Scouted',i,flush=True)
    np.savez_compressed(OUT/'probe-features.npz',row_id=rows.row_id.to_numpy(dtype=str),**bank)
    train=rows.development_part.eq('train').to_numpy();y=rows.label.to_numpy();results=[]
    for key,x in bank.items():
        x=np.array(x);scaler=StandardScaler().fit(x[train]);x=scaler.transform(x)
        for model in [LogisticRegression(C=1),SVC(C=1)]:
            model.fit(x[train],y[train]);results.append(dict(method=key,model=type(model).__name__,BA=balanced_accuracy_score(y[~train],model.predict(x[~train]))))
    pd.DataFrame(results).to_csv(OUT/'probe-metrics.csv',index=False)
    print(pd.DataFrame(results).to_string(index=False))

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
