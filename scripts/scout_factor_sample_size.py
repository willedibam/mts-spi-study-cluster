"""Exploratory grouped learning curves on the already-seen T1000 bank."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from scripts.analyze_band_swap import fitted_logistic
from scripts.spi_baseline_exploration import ROOT, project_features, sha
from src.corpus_geometry import fit_geometry_transform

RUN = 'factor-parity-dominant-261004'
OUT = ROOT/'results/representation/factor-sample-size-261004'


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT/'results/representation'/RUN/'final/features.npz'
    protocol = dict(source=str(source.relative_to(ROOT)), source_sha256=sha(source),
        status='exploratory reuse of all previously examined records; not new confirmation',
        T=1000, repeats=2, folds=4, train_blocks=[16,32,48], evaluation_blocks=16,
        split_seed=261005, grouping='Both classes from a simulation block remain together',
        methods=['mean','mean_full','mean_RBF','mean_trees','z_center','z_full_center'],
        rationale='Check training-sample limitations before changing the generator or acquiring more recordings',
        readout='Frozen PCA20 and logistic C1; full centered z logistic is a declared diagnostic of PCA loss',
        uncertainty='Overlapping folds/repeats are not independent confirmation; report descriptive averages only')
    protocol_file=OUT/'protocol.json'
    if protocol_file.exists():
        assert json.loads(protocol_file.read_text()) == protocol
    else:
        protocol_file.write_text(json.dumps(protocol,indent=2)+'\n')
    rows=pd.DataFrame(json.loads((ROOT/'data/representation'/RUN/'manifest.json').read_text())['rows'])
    with np.load(source) as archive:
        np.testing.assert_array_equal(archive['row_id'],rows.row_id)
        bank={k:archive[k] for k in ['mean','z']}
    results=[]
    for repeat in range(2):
        rng=np.random.default_rng(np.random.SeedSequence([261005,repeat]))
        order=rng.permutation(64)
        for fold in range(4):
            test_blocks=order[fold*16:(fold+1)*16]
            candidates=rng.permutation(np.setdiff1d(order,test_blocks))
            for size in [16,32,48]:
                train=rows.block.isin(candidates[:size]).to_numpy()
                test=rows.block.isin(test_blocks).to_numpy()
                y=rows.label.to_numpy()
                readouts={}
                for key,standard in [('mean',True),('z',False)]:
                    _,a,b=project_features(bank[key][train],bank[key][test],standard=standard,dimensions=20)
                    readouts['mean' if key=='mean' else 'z_center']=(a,b,'linear')
                    transform=fit_geometry_transform(bank[key][train],scaling='standard' if standard else 'center',minimum_valid_fraction=.95)
                    x=transform.transform(bank[key])
                    if standard:x=np.clip(x,-5,5)
                    if key=='mean':
                        for name,kind in [('mean_full','linear'),('mean_RBF','rbf'),('mean_trees','trees')]:
                            readouts[name]=(x[train],x[test],kind)
                    else:readouts['z_full_center']=(x[train],x[test],'linear')
                for name,(a,b,kind) in readouts.items():
                    if kind=='linear':model=fitted_logistic(a,y[train])
                    elif kind=='rbf':model=SVC(C=1,gamma='scale').fit(a,y[train])
                    else:model=ExtraTreesClassifier(n_estimators=500,min_samples_leaf=2,random_state=261003,n_jobs=4).fit(a,y[train])
                    results.append(dict(repeat=repeat,fold=fold,training_blocks=size,training_recordings=2*size,method=name,BA=balanced_accuracy_score(y[test],model.predict(b))))
                pd.DataFrame(results).to_csv(OUT/'fold-results.csv',index=False)
            print('Completed repeat',repeat,'fold',fold,flush=True)
    summary=pd.DataFrame(results).groupby(['training_recordings','method']).BA.agg(['mean','min','max']).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    print(summary.to_string(index=False))


if __name__=='__main__':
    with threadpool_limits(limits=4):run()
