"""Fresh-development length check after the T1000 confirmation fell short."""
import json
import numpy as np
import pandas as pd
from sklearn.covariance import graphical_lasso
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from scripts.build_factor_parity import simulate,direct_features,CLASSES,ROOT

OUT=ROOT/'results/representation/factor-length-scout-261004'


def run():
    OUT.mkdir(exist_ok=True,parents=True)
    (OUT/'protocol.json').write_text(json.dumps(dict(lengths=[2000,4000],strength=.25,weights=[.9,.075,.025],rng_development_blocks=[64,95],rng_held_reserved=[96,127],selection='Shortest length with cheap z BA>=.90 and every mean BA<=.65; full289 development still required',motivation='T1000 held centered z .6875 versus primary mean .5625; paired difference unresolved. More samples may improve character estimation without changing population strengths.'),indent=2)+'\n')
    result=[];mask=~np.eye(16,dtype=bool)
    for t in [2000,4000]:
        features=[];z=[];y=[]
        for block in range(32):
            for label in CLASSES:
                x,_=simulate(label,block,.25,[.9,.075,.025],t,64);f=direct_features(x);means=list(f['probe_mean'])
                for alpha in [.01,.1]:
                    c,p=graphical_lasso(np.cov(x,rowvar=False,bias=True),alpha=alpha)
                    means.extend([c[mask].mean(),p[mask].mean(),(p[mask]**2).mean()])
                features.append(means);z.append(f['probe_z']);y.append(label)
        y=np.array(y)
        for name,x in [('mean',features),('z',z)]:
            x=np.array(x);scale=StandardScaler().fit(x[:48]);x=np.clip(scale.transform(x),-5,5)
            models={'linear':LogisticRegression(C=1,max_iter=3000)}
            if name=='mean':models.update(rbf=SVC(C=1,gamma='scale'),trees=ExtraTreesClassifier(n_estimators=500,min_samples_leaf=2,random_state=261003,n_jobs=4))
            for kind,model in models.items():
                model.fit(x[:48],y[:48]);r=dict(T=t,method=name+'_'+kind,BA=balanced_accuracy_score(y[48:],model.predict(x[48:])));result.append(r);print(r,flush=True)
        np.savez_compressed(OUT/f'T{t}.npz',mean=features,z=z,label=y)
        pd.DataFrame(result).to_csv(OUT/'results.csv',index=False)


if __name__=='__main__':
    with threadpool_limits(limits=4):run()
