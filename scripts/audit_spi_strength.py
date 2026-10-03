"""Cached proof diagnostic: remove each MPI's location and scale, retain shape."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from scripts.spi_baseline_exploration import ROOT, DATA, CML, project_features, balance, retrieval, row_keys, sha

OUT = ROOT / 'results/baseline-strength-audit_261003'


def normalized_quantiles(marginal):
    means, sd = marginal[:, :, 0], marginal[:, :, 1]
    valid = np.isfinite(marginal).all(axis=2) & (sd > 0)
    shape = np.full(marginal[:, :, 2:].shape, np.nan)
    np.divide(marginal[:, :, 2:] - means[:, :, None], sd[:, :, None],
              out=shape, where=valid[:, :, None])
    return shape.reshape(len(marginal), -1), valid


def analyze():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(json.loads((DATA/'proof-inputs.json').read_text())['records'])
    with np.load(DATA/'proof-summaries.npz') as a:
        np.testing.assert_array_equal(frame.row_id, a['row_id'])
        shape, valid = normalized_quantiles(a['marginal'])
        frame['strength_proxy'] = a['pearson'][:, 1]
    coordinates = ROOT/'results/representation/cross-mt/cross_mt_transfer_260824/confirmation-coordinates.npz'
    with np.load(coordinates, allow_pickle=True) as a:
        kd = a['development_row_keys'].astype(str)
        ke = row_keys(a['confirmation_y'], a['confirmation_M'], a['confirmation_T'], a['confirmation_instance'])
    lookup = pd.Series(np.arange(len(frame)), index=frame.row_id)
    dev, held = lookup.loc[kd].to_numpy(), lookup.loc[ke].to_numpy()
    metadata = [frame.iloc[i].reset_index(drop=True) for i in (dev, held)]
    bank = ROOT/'results/baseline-comparison_260921/proof-projections.npz'
    with np.load(bank) as a:
        blocks = {k:(a[k+'_dev'], a[k+'_eval']) for k in ('mean','distribution','z')}
    model, d, h = project_features(shape[dev], shape[held])
    blocks['shape'] = balance(d, h)
    vm, d, h = project_features(valid[dev].astype(float), valid[held].astype(float))
    blocks['validity'] = balance(d, h)
    metrics, outcomes = [], []
    for name, (d,h) in blocks.items():
        for scope, classes in [('all14', metadata[0].label.unique()), ('CML5', CML)]:
            take_d, take_h = [f.label.isin(classes).to_numpy() for f in metadata]
            clf = LogisticRegression(C=1, max_iter=3000).fit(d[take_d], metadata[0].label[take_d])
            predicted = clf.predict(h[take_h])
            ap, _ = retrieval(d[take_d], h[take_h], metadata[0][take_d], metadata[1][take_h])
            result = metadata[1].loc[take_h, ['row_id','label','instance','M','T']].copy()
            result['predicted'], result['correct'], result['ap'] = predicted, predicted == result.label, ap
            result['method'], result['scope'] = name, scope
            outcomes.append(result)
            metrics.append(dict(method=name, scope=scope, balanced_accuracy=balanced_accuracy_score(result.label,predicted),
                                hard_mAP=float(ap.mean()), n=len(result), chance=1/len(classes)))
    metrics = pd.DataFrame(metrics)
    old = pd.read_csv(ROOT/'results/baseline-comparison_260921/proof-metrics.csv').set_index(['method','scope'])
    for r in metrics[metrics.method.isin(['mean','distribution','z'])].itertuples():
        np.testing.assert_allclose([r.balanced_accuracy,r.hard_mAP],old.loc[(r.method,r.scope),['balanced_accuracy','hard_mAP']].to_numpy(float),atol=1e-13)
    metrics.to_csv(OUT/'metrics.csv',index=False)
    pd.concat(outcomes).to_csv(OUT/'predictions.csv',index=False)
    summary = frame.query('role == "development"').groupby(['M','T','label']).strength_proxy.agg(['min','median','max'])
    summary.to_csv(OUT/'development-strength-ranges.csv')
    np.savez_compressed(OUT/'projections.npz', **{k+s:v for k,pair in blocks.items() for s,v in zip(('_dev','_eval'),pair)},
                        dev_labels=metadata[0].label.to_numpy(str),eval_labels=metadata[1].label.to_numpy(str))
    provenance = dict(status='exploratory; cached historical proof; no new p90',
        shape_features_retained=len(model.transform.keep_indices),validity_features_retained=len(vm.transform.keep_indices),
        nonpositive_SD_or_invalid_profiles=int((~valid).sum()),
        original_mean_distribution_z_metrics_replayed=True,
        inputs={str(p.relative_to(ROOT)):sha(p) for p in [DATA/'proof-summaries.npz',DATA/'proof-inputs.json',coordinates,bank]},
        script_sha256=sha(Path(__file__)),protocol_sha256=sha(ROOT/'configs/analysis/spi-strength-audit-261003.yaml'))
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(metrics.round(4).to_string(index=False))


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        analyze()
