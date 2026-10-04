"""Retrospective six-family view of the cached native-gain control."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from umap import UMAP
from threadpoolctl import threadpool_limits
from scripts.analyze_all_proof_coupling import rows as load_rows
from scripts.analyze_band_swap import fitted_logistic
from scripts.spi_baseline_exploration import ROOT,project_features,sha
from scripts.refresh_case_figures import style,save
from src.corpus_geometry import fit_geometry_transform

SOURCE=ROOT/'results/representation/proof-strength-all-261003'
OUT=ROOT/'results/representation/six-family-view-261004'
# First historical inventory entry for each requested family; not outcome-selected.
LABELS={'VAR-legacy-0.95-0.4':'VAR','Wave':'Wave','Gaussian-noise':'Gaussian noise',
        'Cauchy-noise':'Cauchy noise','CML-1.895':'CML','Kuramoto-fast':'Kuramoto'}


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=load_rows();take=rows.label.isin(LABELS).to_numpy();rows=rows.loc[take].reset_index(drop=True)
    train=rows.role.eq('development').to_numpy();y=rows.label.map(LABELS).to_numpy()
    assert len(rows)==144 and train.sum()==72 and rows.M.eq(16).all() and rows['T'].eq(1000).all()
    protocol=dict(scope='retrospective descriptive subset of previously examined data; no new confirmation',
        representatives=LABELS,selection='first historical inventory entry per requested family',M=16,T=1000,
        train_blocks=[0,11],display_blocks=[12,23],points_per_class=12,
        source_sha256=sha(SOURCE/'features.npz'),mean='training-standardized/clipped/PCA20',z='training-centered/PCA20',
        embedding='Training fit, evaluation transform; UMAP30/min_dist.1/seed261003; no point or seed selection',
        gain='Coupled families G=.20+/-.002 under native-coordinate conventions; independent noise G=0')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    projections={};scores=[]
    with np.load(SOURCE/'features.npz') as archive:
        np.testing.assert_array_equal(archive['row_id'][take],rows.row_id)
        for name,key,standard in [('mean','mean',True),('z','z',False),('z_standard','z',True),('z_validity','z_validity',True)]:
            _,d,h=project_features(archive[key][take][train],archive[key][take][~train],standard=standard,dimensions=20)
            model=fitted_logistic(d,y[train]);scores.append(dict(method=name,BA=balanced_accuracy_score(y[~train],model.predict(h))))
            projections[name]=(d,h)
        tr=fit_geometry_transform(archive['mean'][take][train],scaling='standard',minimum_valid_fraction=.95)
        x=np.clip(tr.transform(archive['mean'][take]),-5,5)
        for name,model in [('mean_full',fitted_logistic(x[train],y[train])),('mean_RBF',SVC(C=1,gamma='scale').fit(x[train],y[train])),('mean_trees',ExtraTreesClassifier(n_estimators=500,min_samples_leaf=2,random_state=261003,n_jobs=4).fit(x[train],y[train]))]:
            scores.append(dict(method=name,BA=balanced_accuracy_score(y[~train],model.predict(x[~train]))))
    pd.DataFrame(scores).to_csv(OUT/'metrics.csv',index=False)
    rows.groupby('label').agg(n=('row_id','size'),gain=('strength','mean'),mean_abs_Pearson=('mean_abs_Pearson','mean')).to_csv(OUT/'coupling.csv')
    style();fig,axes=plt.subplots(2,2,figsize=(8.6,8.6),layout='constrained')
    colors=dict(zip(LABELS.values(),['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00','#555555']))
    for col,(name,title) in enumerate([('mean',r'Per-SPI means $m$'),('z',r'SPI–SPI $z$')]):
        d,h=projections[name]
        for row,kind in enumerate(['PCA','UMAP']):
            mapper=PCA(n_components=2) if kind=='PCA' else UMAP(n_neighbors=30,min_dist=.1,random_state=261003,transform_seed=261003,n_jobs=1)
            mapper.fit(d);xy=mapper.transform(h);ax=axes[row,col]
            for label,color in colors.items():
                mask=y[~train]==label;ax.scatter(*xy[mask].T,color=color,s=30,alpha=.8,edgecolors='white',linewidths=.3,label=label)
            ax.set(title=title+' — 6 classes',xlabel=kind+' 1',ylabel=kind+' 2');ax.set_box_aspect(1)
            for spine in ax.spines.values():spine.set_visible(True)
            if kind=='UMAP':ax.set(xticks=[],yticks=[])
    handles,names=axes[0,0].get_legend_handles_labels();fig.legend(handles,names,loc='outside lower center',ncol=3)
    save(fig,OUT,'six-family-embeddings')
    print(pd.DataFrame(scores).to_string(index=False))


if __name__=='__main__':
    with threadpool_limits(limits=4):run()
