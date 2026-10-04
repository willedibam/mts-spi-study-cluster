"""Fresh development-only M8/16/32 scout of the existing mean-matched factor model."""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from scripts.build_factor_parity import direct_features,CLASSES
from scripts.spi_baseline_exploration import ROOT,project_features,sha
from src.corpus_geometry import fit_geometry_transform

RUN='factor-channel-scout-261004';OUT=ROOT/'results/representation'/RUN
SEED=261022;WEIGHTS=np.array([.9,.075,.025]);STRENGTH=.25


def signs(label,m):
    indices=[1,2,3] if label==CLASSES[0] else [1,2,4]
    return np.array([[(-1.)**((i&k).bit_count()%2) for i in range(m)] for k in indices])


def covariance(label,m):
    v=signs(label,m)
    return (1-STRENGTH)*np.eye(m)+STRENGTH*np.einsum('k,ki,kj->ij',WEIGHTS,v,v)


def parent(label,block):
    rng=np.random.default_rng(np.random.SeedSequence([SEED,block,CLASSES.index(label)]))
    factors=rng.normal(size=(3,1000));noise=rng.normal(size=(32,1000))
    return (np.sqrt(1-STRENGTH)*noise+np.sqrt(STRENGTH)*signs(label,32).T@(np.sqrt(WEIGHTS)[:,None]*factors)).T


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(status='Prospective local development-only scout; no held data or p90 jobs',seed=SEED,T=1000,M=[8,16,32],training_blocks=[0,95],validation_blocks=[96,127],
        model='Unchanged lambda .25 and weights .9/.075/.025, Walsh linked/unlinked signs. Generate max32 Gaussian latent-factor parents; firstM channels are nested views, then apply label-independent channel permutation. Both classes share population absolute covariance multisets, eigenvalues and signed means within each M.',
        source_sha256=sha(__file__),probe_source_sha256=sha(ROOT/'scripts/build_factor_parity.py'),
        methods='Ten cheap MPI means; full linear/RBF/ExtraTrees; one training-ranked mean logistic/stump; probe z standardized/centered PCA20 logistic; covariance/Pearson distribution summaries secondary.',
        criterion='Consider M32 p90 only if all cheap mean readouts<=.65 and focused z>=.85 at M32. This is a screen, not full-p90 evidence. Do not choose an intermediate unlisted M, T>1000 or new parameters after outcomes.',
        limitation='Signed marginal distributions differ; this is sign-magnitude organization of Gaussian dependence, not nonlinear dynamics. Views within each parent remain dependent, and all reported scores are exploratory development.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    records=[];bank={}
    for m in [8,16,32]:
        a,b=[covariance(label,m) for label in CLASSES];off=~np.eye(m,dtype=bool)
        np.testing.assert_allclose(np.sort(abs(a[off])),np.sort(abs(b[off])))
        np.testing.assert_allclose(np.linalg.eigvalsh(a),np.linalg.eigvalsh(b))
        np.testing.assert_allclose(a.sum(axis=1),b.sum(axis=1))
        for block in range(128):
            order=np.random.default_rng(np.random.SeedSequence([SEED,block,99,m])).permutation(m)
            for label in CLASSES:
                x=parent(label,block)[:,:m][:,order]
                features=direct_features(x)
                for key,value in features.items():bank.setdefault(key,[]).append(value)
                records.append(dict(M=m,T=1000,block=block,label=label,training=block<96))
        print('Scouted M',m,flush=True)
    frame=pd.DataFrame(records);frame.to_csv(OUT/'rows.csv',index=False);bank={k:np.array(v) for k,v in bank.items()};np.savez_compressed(OUT/'features.npz',**bank)
    scores=[]
    for m in [8,16,32]:
        use=frame.M.eq(m).to_numpy();rows=frame[use].reset_index(drop=True);tr=rows.training.to_numpy();y=rows.label.to_numpy();positive=y==CLASSES[1]
        x=bank['probe_mean'][use];transform=fit_geometry_transform(x[tr],scaling='standard',minimum_valid_fraction=1.);z=np.clip(transform.transform(x),-5,5)
        effect=z[tr & positive].mean(axis=0)-z[tr & ~positive].mean(axis=0);j=int(np.argmax(abs(effect)))
        models=[('mean_full',LogisticRegression(C=1,max_iter=5000),z),('mean_RBF',SVC(C=1),z),('mean_trees',ExtraTreesClassifier(n_estimators=500,min_samples_leaf=2,random_state=261003,n_jobs=4),z),('mean_selected_linear',LogisticRegression(C=1),z[:,j,None]),('mean_selected_stump',DecisionTreeClassifier(max_depth=1,random_state=261021),z[:,j,None])]
        for name,model,values in models:
            model.fit(values[tr],y[tr]);scores.append(dict(M=m,method=name,BA=balanced_accuracy_score(y[~tr],model.predict(values[~tr]))))
        for key,standard,name in [('probe_z',True,'z_standard'),('probe_z',False,'z_center'),('raw_covariance',True,'covariance_distribution'),('raw_Pearson',True,'Pearson_distribution')]:
            _,d,h=project_features(bank[key][use][tr],bank[key][use][~tr],dimensions=20,standard=standard)
            model=LogisticRegression(C=1,max_iter=5000).fit(d,y[tr]);scores.append(dict(M=m,method=name,BA=balanced_accuracy_score(y[~tr],model.predict(h))))
    score=pd.DataFrame(scores);score.to_csv(OUT/'metrics.csv',index=False);print(score.pivot(index='method',columns='M',values='BA').round(4).to_string())

if __name__=='__main__':
    with threadpool_limits(limits=4):run()
