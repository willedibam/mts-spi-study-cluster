"""Declared development-only readout audit; preserves the failed frozen result."""
import argparse,json
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from scripts.spi_baseline_exploration import ROOT,project_features,sha
from scripts.analyze_band_swap import fitted_logistic
from src.corpus_geometry import fit_geometry_transform


def run(run):
    out=ROOT/'results/representation'/run/'development'
    rows=pd.DataFrame([r for r in json.loads((ROOT/'data/representation'/run/'manifest.json').read_text())['rows'] if r['role']=='development'])
    train=rows.development_part.eq('train').to_numpy();y=rows.label.to_numpy()
    protocol=dict(stage='development only; no held outcomes',reason='More training recordings did not improve centered PCA20 linear z; focused z .8906 warrants checking the fixed broad readout',
        grid='Means: standardized PCA20/full × linear/RBF. z: centered/standardized × PCA20/full × linear/RBF. C1 throughout, RBF gamma scale.',
        criterion='First assess center-only PCA20 RBF, changing only classifier flexibility; require BA>=.80 and all four diagnostic mean readouts<=.65 before considering a separately frozen held confirmation. Do not silently replace the failed primary.',
        features_sha256=sha(out/'features.npz'))
    (out/'readout-audit-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    results=[];predictions=[]
    with np.load(out/'features.npz') as archive:
        np.testing.assert_array_equal(archive['row_id'],rows.row_id)
        for key in ['mean','z']:
            for standard in ([True] if key=='mean' else [False,True]):
                for dimensions in [20,None]:
                    if dimensions:
                        _,d,h=project_features(archive[key][train],archive[key][~train],standard=standard,dimensions=dimensions)
                    else:
                        tr=fit_geometry_transform(archive[key][train],scaling='standard' if standard else 'center',minimum_valid_fraction=.95)
                        x=tr.transform(archive[key]);x=np.clip(x,-5,5) if standard else x;d,h=x[train],x[~train]
                    for kind in ['linear','rbf']:
                        model=fitted_logistic(d,y[train]) if kind=='linear' else SVC(C=1,gamma='scale').fit(d,y[train])
                        pred=model.predict(h)
                        label=f'{key}_{"standard" if standard else "center"}_{dimensions or "full"}_{kind}'
                        results.append(dict(method=label,BA=balanced_accuracy_score(y[~train],pred)))
                        frame=rows.loc[~train,['row_id','block','label']].copy();frame['method']=label;frame['predicted']=pred;predictions.append(frame)
    pd.DataFrame(results).to_csv(out/'readout-audit.csv',index=False)
    pd.concat(predictions).to_csv(out/'readout-audit-predictions.csv',index=False)
    print(pd.DataFrame(results).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
    with threadpool_limits(limits=4):run(a.run)
