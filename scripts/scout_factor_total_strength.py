"""One declared follow-up: hold total absolute covariance per channel fixed at M32."""
import json,warnings
import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLasso
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from scripts.scout_factor_channels import signs,CLASSES,WEIGHTS,ROOT
from scripts.build_factor_parity import direct_features
from scripts.spi_baseline_exploration import project_features,sha
from src.corpus_geometry import fit_geometry_transform

OUT=ROOT/'results/representation/factor-total-strength-scout-261004'


def run():
    OUT.mkdir(parents=True,exist_ok=True);m=32;t=1000;seed=261023
    total=.25*(.9*16-1);strength=total/(.9*m-1)
    protocol=dict(status='Single additional development-only check; no held or p90 jobs',M=m,T=t,seed=seed,strength=strength,row_total_absolute_covariance=total,
        rationale='Previous lambda .25 at M32 doubles total row coupling relative to M16. Use lambda=3.35/(.9*M-1), analytically matching the M16 row absolute total. This is one declared candidate, not a parameter sweep.',
        training_blocks=[0,95],validation_blocks=[96,127],weights=WEIGHTS.tolist(),source_sha256=sha(__file__),
        probes='Prior ten means plus four exact fixed-p90 GraphicalLasso means; selected single means, linear/RBF/trees; focused z and signed distribution controls',
        gate='All mean readouts<=.65 and focused centered z>=.85. Stop this channel-count branch if it fails. Full p90 still unverified even if it passes.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');bank={};rows=[];diag=[]
    for block in range(128):
        for label in CLASSES:
            v=signs(label,m);c=(1-strength)*np.eye(m)+strength*np.einsum('k,ki,kj->ij',WEIGHTS,v,v)
            np.testing.assert_allclose(abs(c-np.eye(m)).sum(axis=1),total)
            rng=np.random.default_rng(np.random.SeedSequence([seed,block,CLASSES.index(label)]));factors=rng.normal(size=(3,t));noise=rng.normal(size=(m,t))
            x=(np.sqrt(1-strength)*noise+np.sqrt(strength)*v.T@(np.sqrt(WEIGHTS)[:,None]*factors)).T
            order=np.random.default_rng(np.random.SeedSequence([seed,block,99])).permutation(m);x=x[:,order]
            features=direct_features(x)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');model=GraphicalLasso().fit(x)
            off=~np.eye(m,dtype=bool);a,p=model.covariance_,model.precision_;features['probe_mean']=np.r_[features['probe_mean'],a[off].mean(),(a[off]**2).mean(),p[off].mean(),(p[off]**2).mean()]
            assert all(np.isfinite(value).all() for value in features.values())
            for k,value in features.items():bank.setdefault(k,[]).append(value)
            rows.append(dict(block=block,label=label,training=block<96));diag.append(dict(block=block,label=label,convergence_warnings=sum(issubclass(w.category,ConvergenceWarning) for w in caught),iterations=model.n_iter_,dual_gap=model.costs_[-1][1]))
    frame=pd.DataFrame(rows);bank={k:np.array(v) for k,v in bank.items()};np.savez_compressed(OUT/'features.npz',**bank);frame.to_csv(OUT/'rows.csv',index=False);pd.DataFrame(diag).to_csv(OUT/'estimator-diagnostics.csv',index=False)
    tr=frame.training.to_numpy();y=frame.label.to_numpy();pos=y==CLASSES[1];transform=fit_geometry_transform(bank['probe_mean'][tr],scaling='standard',minimum_valid_fraction=1);z=np.clip(transform.transform(bank['probe_mean']),-5,5)
    effect=z[tr & pos].mean(axis=0)-z[tr & ~pos].mean(axis=0);j=int(np.argmax(abs(effect)))
    models=[('mean_linear',LogisticRegression(C=1,max_iter=5000),z),('mean_RBF',SVC(C=1),z),('mean_trees',ExtraTreesClassifier(n_estimators=500,min_samples_leaf=2,random_state=261003,n_jobs=4),z),('mean_selected_linear',LogisticRegression(C=1),z[:,j,None]),('mean_selected_stump',DecisionTreeClassifier(max_depth=1,random_state=261021),z[:,j,None])]
    scores=[]
    for name,model,x in models:
        model.fit(x[tr],y[tr]);scores.append(dict(method=name,BA=balanced_accuracy_score(y[~tr],model.predict(x[~tr]))))
    for key,standard,name in [('probe_z',False,'z_center'),('probe_z',True,'z_standard'),('raw_covariance',True,'covariance_distribution'),('raw_Pearson',True,'Pearson_distribution')]:
        _,d,h=project_features(bank[key][tr],bank[key][~tr],dimensions=20,standard=standard);model=LogisticRegression(C=1,max_iter=5000).fit(d,y[tr]);scores.append(dict(method=name,BA=balanced_accuracy_score(y[~tr],model.predict(h))))
    pd.DataFrame(scores).to_csv(OUT/'metrics.csv',index=False);print(pd.DataFrame(scores).to_string(index=False));print('lambda',strength,'total',total)

if __name__=='__main__':
    with threadpool_limits(limits=4):run()
