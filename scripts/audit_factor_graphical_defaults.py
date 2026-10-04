"""Replay the actual fixed p90 GraphicalLasso estimator and audit convergence."""
import json,warnings
import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLasso
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from scripts.scout_factor_channels import OUT,parent,CLASSES,SEED
from src.corpus_geometry import fit_geometry_transform


def run():
    f=pd.read_csv(OUT/'rows.csv');keep=f.M.eq(32).to_numpy();f=f[keep].reset_index(drop=True)
    with np.load(OUT/'features.npz') as a:base=a['probe_mean'][keep]
    off=~np.eye(32,dtype=bool);rows=[];values=[]
    for r in f.itertuples():
        order=np.random.default_rng(np.random.SeedSequence([SEED,r.block,99,32])).permutation(32)
        x=parent(r.label,r.block)[:,order]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model=GraphicalLasso().fit(x)
        c,p=model.covariance_,model.precision_;v=np.array([c[off].mean(),(c[off]**2).mean(),p[off].mean(),(p[off]**2).mean()])
        assert np.isfinite(v).all()
        # Independently check finite logdet using a Cholesky factor, rather than warning flags alone.
        chol=np.linalg.cholesky(p);logdet=2*np.log(chol.diagonal()).sum()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');sgn,det=np.linalg.slogdet(p)
        np.testing.assert_allclose(logdet,det,atol=1e-10);assert sgn==1
        values.append(np.r_[base[r.Index],v]);rows.append(dict(block=r.block,label=r.label,iterations=model.n_iter_,convergence_warnings=sum(issubclass(w.category,ConvergenceWarning) for w in caught),runtime_warnings=sum(issubclass(w.category,RuntimeWarning) for w in caught),dual_gap=float(model.costs_[-1][1]),inverse_residual=float(abs(c@p-np.eye(32)).max())))
    x=np.array(values);tr=f.training.to_numpy();y=f.label.to_numpy();transform=fit_geometry_transform(x[tr],scaling='standard',minimum_valid_fraction=1);z=np.clip(transform.transform(x),-5,5);scores=[]
    for name,model in [('mean_linear',LogisticRegression(C=1,max_iter=5000)),('mean_RBF',SVC(C=1)),('mean_trees',ExtraTreesClassifier(n_estimators=500,min_samples_leaf=2,random_state=261003,n_jobs=4))]:
        model.fit(z[tr],y[tr]);scores.append(dict(method=name,BA=balanced_accuracy_score(y[~tr],model.predict(z[~tr]))))
    pd.DataFrame(rows).to_csv(OUT/'graphical-default-diagnostics.csv',index=False);pd.DataFrame(scores).to_csv(OUT/'graphical-default-metrics.csv',index=False);np.savez_compressed(OUT/'graphical-default-features.npz',mean=x)
    summary=dict(estimator='sklearn GraphicalLasso default alpha .01/max_iter100/tol1e-4, as used by p90',convergence_warnings=sum(r['convergence_warnings'] for r in rows),runtime_warnings=sum(r['runtime_warnings'] for r in rows),all_finite=True,independent_cholesky_logdet_agrees=True,max_abs_dual_gap=max(abs(r['dual_gap']) for r in rows),max_inverse_residual=max(r['inverse_residual'] for r in rows))
    (OUT/'graphical-default-audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(pd.DataFrame(scores).to_string(index=False));print(summary)

if __name__=='__main__':
    with threadpool_limits(limits=4):run()
