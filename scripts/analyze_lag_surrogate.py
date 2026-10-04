"""Use the frozen matched-bank readout, with explicit binary and null gates."""
import argparse
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.metrics import balanced_accuracy_score
from scripts import analyze_pearson_strength_match as readout
from scripts.build_lag_surrogate import DATA,OUT,RUN
from scripts.spi_baseline_exploration import project_features,sha
from scripts.analyze_band_swap import fitted_logistic


def run(action,stage):
    readout.DATA,readout.OUT,readout.RUN=DATA,OUT,RUN
    if action=='extract':
        readout.extract(stage)
        return
    readout.analyze(stage)
    target=OUT/stage
    frame=readout.rows(stage)
    fit=(frame.panel.eq('matched') & (frame.development_part.eq('train') if stage=='development' else frame.role.eq('development'))).to_numpy()
    null=frame.panel.eq('null').to_numpy()
    null_results={}
    with np.load(target/'features.npz') as a:
        for name in ['mean','z']:
            _,d,h=project_features(a[name][fit],a[name][null],dimensions=20,standard=name=='mean',valid=1. if name=='z' else .95)
            model=fitted_logistic(d,frame.loc[fit,'label'].to_numpy())
            null_results[name]=float(balanced_accuracy_score(frame.loc[null,'label'],model.predict(h)))
    score=pd.read_csv(target/'metrics.csv').set_index('method').BA
    gates=dict(z_strong=bool(score.z_complete>=.8),means_weak=bool(score[['mean','mean_full','mean_RBF','mean_trees']].max()<=.65),
        covariance_weak=bool(score[[k for k in score.index if k=='b' or k.startswith('b_') or k.startswith('pearson_two')]].max()<=.6),
        validity_weak=bool(score.z_validity<=.65),shuffle_weak=bool(score.z_shuffled_complete<=.65),null_weak=bool(null_results['z']<=.65))
    report=json.loads((target/'analysis.json').read_text())
    report.update(development_gate=dict(**gates,all_pass=all(gates.values())),null_accuracy=null_results,
        wrapper_sha256=sha(__file__),qualification='Binary paired temporal-organization contrast; development gates supersede the six-class gates in the reused readout. Circular second-order equality is exact. Full p90 means/distributions may encode the difference. No held release unless all gates pass.')
    (target/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Authoritative binary gates:',gates,'Null:',null_results)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['extract','analyze']);p.add_argument('--stage',choices=['development','final'],default='development');a=p.parse_args()
    with threadpool_limits(limits=4):run(a.action,a.stage)
