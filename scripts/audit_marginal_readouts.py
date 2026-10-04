"""Training-selected sparse marginal controls on already-seen development banks."""
import argparse,json,warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from scripts.spi_baseline_exploration import ROOT,sha
from src.corpus_geometry import fit_geometry_transform


def transform_fit(x,train):
    transform=fit_geometry_transform(x[train],scaling='standard',minimum_valid_fraction=.95)
    return transform,np.clip(transform.transform(x),-5,5)


def sparse_fit(x,y,c):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        model=LogisticRegression(C=c,penalty='l1',solver='liblinear',max_iter=5000,random_state=261021).fit(x,y)
    assert not any(issubclass(w.category,ConvergenceWarning) for w in caught)
    return model


def run(run):
    out=ROOT/'results/representation'/run/'development';target=out/'marginal-audit';target.mkdir(exist_ok=True)
    records=[r for r in json.loads((ROOT/'data/representation'/run/'manifest.json').read_text())['rows'] if r['role']=='development']
    full=pd.DataFrame(records);keep=(full.panel.eq('matched').to_numpy() if 'panel' in full else np.ones(len(full),dtype=bool))
    rows=full[keep].reset_index(drop=True);train=rows.development_part.eq('train').to_numpy();y=rows.label.to_numpy();positive=y==sorted(set(y))[-1]
    protocol=dict(status='Exploratory audit of already-seen development; held banks untouched',features_sha256=sha(out/'features.npz'),script_sha256=sha(__file__),
        single='Rank maximum absolute training standardized class mean difference; logisticC1 and depth1 stump fitted on training only',
        sparse='L1 logistic liblinear; C in [.01,.1,1,10], selected by four-fold GroupKFold within training only. Fit preprocessing independently inside each fold; keep paired blocks together. Tie favors smaller C.',
        no_selection='No validation-dependent change to ranking, hyperparameters, feature list, classes, or seeds')
    (target/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    with np.load(out/'features.npz') as a:
        np.testing.assert_array_equal(a['row_id'],full.row_id)
        x=a['mean'][keep];names=a['spi_order'];distributions=a['distribution'][keep]
    results=[];predictions=[];selected=[]
    for key,values,feature_names in [('mean',x,names),('distribution',distributions,np.array([str(n)+'/'+s for n in names for s in ['mean','sd','q10','q25','q50','q75','q90']]))]:
        transform,z=transform_fit(values,train)
        effect=z[train & positive].mean(axis=0)-z[train & ~positive].mean(axis=0);index=int(np.argmax(abs(effect)))
        selected.append(dict(method=key,feature=feature_names[transform.keep_indices[index]],training_effect=effect[index],validation_auc=roc_auc_score(positive[~train],z[~train,index]*np.sign(effect[index]))))
        for name,model in [('selected_linear',LogisticRegression(C=1,max_iter=5000)),('selected_stump',DecisionTreeClassifier(max_depth=1,random_state=261021))]:
            model.fit(z[train,index,None],y[train]);pred=model.predict(z[~train,index,None]);method=key+'_'+name
            results.append(dict(method=method,BA=balanced_accuracy_score(y[~train],pred)))
            f=rows.loc[~train,['row_id','block','label']].copy();f['method']=method;f['predicted']=pred;predictions.append(f)
    candidates=[.01,.1,1.,10.];cv=[];indices=np.flatnonzero(train)
    for fold,(tr,va) in enumerate(GroupKFold(n_splits=4).split(indices,y[train],groups=rows.loc[train,'block'])):
        mask=np.zeros(len(rows),dtype=bool);mask[indices[tr]]=True
        _,z=transform_fit(x,mask)
        for c in candidates:
            model=sparse_fit(z[indices[tr]],y[indices[tr]],c)
            cv.append(dict(fold=fold,C=c,BA=balanced_accuracy_score(y[indices[va]],model.predict(z[indices[va]]))))
    cv=pd.DataFrame(cv);cv.to_csv(target/'training-cv.csv',index=False)
    best_c=float(cv.groupby('C').BA.mean().idxmax());tr,z=transform_fit(x,train);model=sparse_fit(z[train],y[train],best_c);pred=model.predict(z[~train])
    results.append(dict(method='mean_sparse_cv',BA=balanced_accuracy_score(y[~train],pred),selected_C=best_c,nonzero=int(np.count_nonzero(model.coef_))))
    f=rows.loc[~train,['row_id','block','label']].copy();f['method']='mean_sparse_cv';f['predicted']=pred;predictions.append(f)
    pd.DataFrame(selected).to_csv(target/'selected-features.csv',index=False);pd.DataFrame(results).to_csv(target/'metrics.csv',index=False);pd.concat(predictions).to_csv(target/'predictions.csv',index=False)
    print(run);print(pd.DataFrame(results).to_string(index=False));print(pd.DataFrame(selected).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
    with threadpool_limits(limits=4):run(a.run)
