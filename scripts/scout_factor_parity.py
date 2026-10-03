"""Cheap development-only parameter scout after population graphical-lasso leak."""
import json
import numpy as np
import pandas as pd
from sklearn.covariance import graphical_lasso
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from scripts.build_factor_parity import covariance,simulate,direct_features,CLASSES,OUT

CANDIDATES=[(.35,[.8,.15,.05]),(.25,[.7,.2,.1]),(.2,[.7,.2,.1]),(.3,[.8,.15,.05]),(.35,[.85,.1,.05]),
    (.2,[.85,.1,.05]),(.25,[.85,.1,.05]),(.35,[.9,.075,.025]),(.25,[.9,.075,.025]),(.3,[.9,.075,.025])]


def run():
    OUT.mkdir(exist_ok=True,parents=True)
    (OUT/'scout-protocol.json').write_text(json.dumps(dict(candidates=CANDIDATES,train_blocks=[0,23],validation_blocks=[24,31],
        selection='Require population graphical-lasso mean differences below 1e-5 at alpha .01/.1, then prefer low worst mean-readout BA with z BA at least .8; exploratory development only',
        motivation='Original .35/.7-.2-.1 has population graphical-lasso precision mean gap ~.01 despite identical raw absolute strengths; no factor p90 jobs launched',
        adaptive_extension='Candidates 5–9 added after initial five failed; stronger dominant factor or weaker strength to reduce finite-sample graphical-lasso bias'),indent=2)+'\n')
    records=pd.read_csv(OUT/'scout-results.csv').to_dict('records') if (OUT/'scout-results.csv').exists() else []
    done={r['candidate'] for r in records};mask=~np.eye(16,dtype=bool)
    for index,(strength,weights) in enumerate(CANDIDATES):
        if index in done:continue
        population=[]
        for label in CLASSES:
            c=covariance(label,strength,weights);values=[]
            for alpha in [.01,.1]:
                a,p=graphical_lasso(c,alpha=alpha)
                values.extend([a[mask].mean(),p[mask].mean()])
            population.append(values)
        gap=float(np.max(np.abs(np.diff(population,axis=0))))
        mean=[];zs=[];y=[]
        for block in range(32):
            for label in CLASSES:
                x,_=simulate(label,block,strength,weights);f=direct_features(x);v=list(f['probe_mean'])
                for alpha in [.01,.1]:
                    a,p=graphical_lasso(np.cov(x,rowvar=False,bias=True),alpha=alpha)
                    v.extend([a[mask].mean(),p[mask].mean(),(p[mask]**2).mean()])
                mean.append(v);zs.append(f['probe_z']);y.append(label)
        y=np.array(y);scores={}
        for name,x in [('mean',mean),('z',zs)]:
            x=np.array(x);s=StandardScaler().fit(x[:48]);x=np.clip(s.transform(x),-5,5)
            models={'linear':LogisticRegression(C=1,max_iter=3000)}
            if name=='mean':models.update(rbf=SVC(C=1,gamma='scale'),trees=ExtraTreesClassifier(n_estimators=500,min_samples_leaf=2,random_state=261003,n_jobs=4))
            for model_name,model in models.items():
                model.fit(x[:48],y[:48]);scores[name+'_'+model_name]=balanced_accuracy_score(y[48:],model.predict(x[48:]))
        row=dict(candidate=index,strength=strength,weights=str(weights),population_GL_mean_gap=gap,**scores);records.append(row);print(row,flush=True)
        np.savez_compressed(OUT/f'scout-{index}.npz',mean=mean,z=zs,label=y)
        pd.DataFrame(records).to_csv(OUT/'scout-results.csv',index=False)


if __name__=='__main__':
    with threadpool_limits(limits=4):run()
