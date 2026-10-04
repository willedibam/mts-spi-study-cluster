"""Exploratory diagnosis: rank one mean feature on training, assess on seen validation."""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
from src.corpus_geometry import fit_geometry_transform
from scripts.build_lag_surrogate import DATA,OUT
from scripts.analyze_band_swap import diagnostic_features


def run():
    frame=pd.DataFrame(json.loads((DATA/'manifest.json').read_text())['rows'][:96])
    train=frame.development_part.eq('train').to_numpy();y=frame.label.eq('shared-phase').to_numpy();target=OUT/'development'
    with np.load(target/'features.npz') as a:
        bank={k:a[k][:96] for k in ['mean','z']};names=a['spi_order']
    diagnostic_features(bank,frame,train,target)
    transform=fit_geometry_transform(bank['mean'][train],scaling='standard',minimum_valid_fraction=.95)
    x=np.clip(transform.transform(bank['mean']),-5,5)
    effect=x[train & y].mean(axis=0)-x[train & ~y].mean(axis=0);selected=int(np.argmax(abs(effect)))
    result=[];predictions=[]
    z=pd.read_csv(target/'predictions.csv').query("method=='z_complete'").set_index('row_id').correct
    for model in [LogisticRegression(C=1),DecisionTreeClassifier(max_depth=1,random_state=261003)]:
        model.fit(x[train,selected,None],y[train]);pred=model.predict(x[~train,selected,None])
        f=frame.loc[~train,['row_id','block']].copy();f['model']=type(model).__name__;f['correct']=pred==y[~train]
        predictions.append(f);delta=(f.row_id.map(z).astype(float)-f.correct.astype(float)).groupby(f.block).mean().to_numpy()
        rng=np.random.default_rng(261020);boot=delta[rng.integers(len(delta),size=(5000,len(delta)))].mean(axis=1);lo,hi=np.quantile(boot,[.025,.975])
        result.append(dict(feature=names[transform.keep_indices[selected]],feature_index=int(transform.keep_indices[selected]),
            selection='Maximum absolute training standardized difference; diagnostic inspected after primary validation outcomes',model=type(model).__name__,
            BA=balanced_accuracy_score(y[~train],pred),AUC=roc_auc_score(y[~train],x[~train,selected]*np.sign(effect[selected])),
            z_minus_selected=delta.mean(),paired_low=lo,paired_high=hi))
    pd.DataFrame(result).to_csv(target/'selected-mean-control.csv',index=False)
    pd.concat(predictions).to_csv(target/'selected-mean-predictions.csv',index=False)
    print(pd.DataFrame(result).to_string(index=False))

if __name__=='__main__':run()
