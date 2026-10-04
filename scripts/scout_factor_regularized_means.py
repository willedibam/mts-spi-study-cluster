"""Check the previously observed regularized-covariance leak before any M32 p90 run."""
import json,warnings
import numpy as np
import pandas as pd
from sklearn.covariance import graphical_lasso
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from scripts.scout_factor_channels import OUT,parent,covariance,CLASSES,SEED
from src.corpus_geometry import fit_geometry_transform


def run():
    (OUT/'regularized-protocol.json').write_text(json.dumps(dict(stage='Additional development-only leakage check after M32 cheap gate passed; no parameter retuning',M=32,alphas=[.01,.1],same_training_validation=True,selection='Require every augmented mean readout<=.65; no p90 if this fails'),indent=2)+'\n')
    off=~np.eye(32,dtype=bool);rows=pd.read_csv(OUT/'rows.csv');keep=rows.M.eq(32).to_numpy();frame=rows[keep].reset_index(drop=True)
    with np.load(OUT/'features.npz') as a:base=a['probe_mean'][keep]
    augmented=[];warning_count=0
    for r in frame.itertuples():
        order=np.random.default_rng(np.random.SeedSequence([SEED,r.block,99,32])).permutation(32)
        x=parent(r.label,r.block)[:,order];values=list(base[r.Index])
        for alpha in [.01,.1]:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');c,p=graphical_lasso(np.cov(x,rowvar=False,bias=True),alpha=alpha,max_iter=1000)
            warning_count+=len(caught);v=[c[off].mean(),(c[off]**2).mean(),p[off].mean(),(p[off]**2).mean()];assert np.isfinite(v).all();values.extend(v)
        augmented.append(values)
    x=np.array(augmented);tr=frame.training.to_numpy();y=frame.label.to_numpy();pos=y==CLASSES[1]
    transform=fit_geometry_transform(x[tr],scaling='standard',minimum_valid_fraction=1);z=np.clip(transform.transform(x),-5,5)
    effect=z[tr & pos].mean(axis=0)-z[tr & ~pos].mean(axis=0);j=int(np.argmax(abs(effect)))
    models=[('augmented_mean_linear',LogisticRegression(C=1,max_iter=5000),z),('augmented_mean_RBF',SVC(C=1),z),('augmented_mean_trees',ExtraTreesClassifier(n_estimators=500,min_samples_leaf=2,random_state=261003,n_jobs=4),z),('augmented_selected_linear',LogisticRegression(C=1),z[:,j,None]),('augmented_selected_stump',DecisionTreeClassifier(max_depth=1,random_state=261021),z[:,j,None])]
    results=[]
    for name,model,values in models:
        model.fit(values[tr],y[tr]);results.append(dict(method=name,BA=balanced_accuracy_score(y[~tr],model.predict(values[~tr]))))
    np.savez_compressed(OUT/'regularized-features.npz',mean=x)
    pd.DataFrame(results).to_csv(OUT/'regularized-metrics.csv',index=False)
    (OUT/'regularized-validity.json').write_text(json.dumps(dict(warnings=warning_count,all_finite=True,selected_column=int(transform.keep_indices[j])),indent=2)+'\n')
    print(pd.DataFrame(results).to_string(index=False));print('Warnings',warning_count)

if __name__=='__main__':
    with threadpool_limits(limits=4):run()
