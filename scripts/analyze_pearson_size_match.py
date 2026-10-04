"""Frozen, parent-grouped exploratory analysis across nine M/T cells."""
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from threadpoolctl import threadpool_limits
from scripts.build_pearson_size_match import DATA, OUT, RUN
from scripts.analyze_native_coupling import extract as extract_common
from scripts.analyze_band_swap import fitted_logistic
from scripts.spi_baseline_exploration import project_features, sha
from src.corpus_geometry import fit_geometry_transform
from src.spi_spi_contract import build_unified_feature_values
from src.utils import slugify


def rows():
    return pd.DataFrame(json.loads((DATA/'manifest.json').read_text())['rows'])


def extract():
    extract_common(DATA, OUT, corpus=RUN, row_limit=360)
    shuffled = []
    rng = np.random.default_rng(261011)
    for r in rows().itertuples():
        folder = DATA/'mpis'/RUN/f"{r.corpus_index+1:04d}-{slugify(r.row_id,'dataset')}"
        meta = json.loads((folder/'meta.json').read_text())
        names = [s['name'] for s in meta['pyspi']['spis']]
        with np.load(folder/'spi_mpis.npz') as archive:
            mask = ~np.eye(r.M, dtype=bool)
            matrices = {}
            for name in names:
                a = archive[name].copy()
                v = a[mask].copy()
                rng.shuffle(v)
                a[mask] = v
                np.testing.assert_array_equal(np.sort(v), np.sort(archive[name][mask]))
                matrices[name] = a
        z, _, _ = build_unified_feature_values(matrices, names)
        shuffled.append(z)
    np.savez_compressed(OUT/'shuffled.npz', z=np.array(shuffled), row_id=rows().row_id.to_numpy())


def bank():
    allrows = rows()
    keep = allrows.panel.eq('matched').to_numpy()
    frame = allrows.loc[keep].reset_index(drop=True)
    with np.load(OUT/'features.npz') as a:
        np.testing.assert_array_equal(a['row_id'], allrows.row_id)
        data = {key: a[key][keep] for key in ['mean', 'distribution', 'z', 'z_validity']}
        index = list(a['spi_order']).index('cov_EmpiricalCovariance')
    np.testing.assert_allclose(data['mean'][:, index], frame.mean_covariance, atol=1e-12)
    data['b'] = data['mean'][:, index, None]
    data['pearson_two'] = np.c_[data['b'], frame.mean_abs_Pearson]
    data['z_complete'] = data['z']
    data['z_standard'] = data['z']
    data['mean_complete'] = data['mean']
    with np.load(OUT/'shuffled.npz') as a:
        np.testing.assert_array_equal(a['row_id'], allrows.row_id)
        data['z_shuffled_complete'] = a['z'][keep]
    return frame, data


def analyze():
    frame, data = bank()
    predictions, diagnostics = [], []
    for parent in range(5):
        train = frame.instance.ne(parent).to_numpy()
        y = frame.label.to_numpy()
        readouts = {}
        for name, x in data.items():
            standard = name not in ['z', 'z_complete', 'z_shuffled_complete']
            try:
                projection, d, h = project_features(x[train], x[~train], standard=standard,
                    dimensions=20, valid=1. if name.endswith('_complete') else .95)
                info = dict(features=len(projection.transform.keep_indices), components=d.shape[1],
                    test_missing_fraction=float(np.mean(~np.isfinite(x[~train][:, projection.transform.keep_indices]))))
            except RuntimeError as error:
                if str(error) != 'no features pass the variance gate':
                    raise
                d, h = np.zeros((train.sum(), 1)), np.zeros(((~train).sum(), 1))
                info = dict(features=0, components=1, test_missing_fraction=0.)
            diagnostics.append(dict(parent=parent, method=name, **info))
            readouts[name] = (d, h, 'linear')
        for name in ['mean', 'b', 'pearson_two']:
            transform = fit_geometry_transform(data[name][train], scaling='standard', minimum_valid_fraction=.95)
            x = np.clip(transform.transform(data[name]), -5, 5)
            for suffix, kind in [('full', 'linear'), ('RBF', 'rbf'), ('trees', 'trees')]:
                readouts[name+'_'+suffix] = (x[train], x[~train], kind)
        for name, (d, h, kind) in readouts.items():
            if kind == 'linear':
                model = fitted_logistic(d, y[train])
            elif kind == 'rbf':
                model = SVC(C=1, gamma='scale').fit(d, y[train])
            else:
                model = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, random_state=261003, n_jobs=4).fit(d, y[train])
            f = frame.loc[~train, ['row_id', 'label', 'instance', 'M', 'T', 'cell']].copy()
            f['method'], f['predicted'] = name, model.predict(h)
            f['correct'] = f.label == f.predicted
            predictions.append(f)
        print('Evaluated held parent', parent, flush=True)
    predictions = pd.concat(predictions)
    predictions.to_csv(OUT/'predictions.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'diagnostics.csv', index=False)
    metrics = []
    for name, f in predictions.groupby('method', sort=False):
        parent_scores = f.groupby('instance').correct.mean()
        metrics.append(dict(method=name, n=len(f), independent_parent_blocks=5,
            BA=balanced_accuracy_score(f.label, f.predicted), parent_min=parent_scores.min(), parent_max=parent_scores.max(), chance=1/6))
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv', index=False)
    predictions.groupby(['method', 'M', 'T']).correct.mean().rename('BA').reset_index().to_csv(OUT/'cell-metrics.csv', index=False)
    # Display fit is explicitly descriptive and never reused in validation.
    coordinates = {}
    for name in ['mean', 'distribution', 'z_complete']:
        _, d, _ = project_features(data[name], data[name], standard=name != 'z_complete', dimensions=20,
                                   valid=1. if name == 'z_complete' else .95)
        coordinates[name] = d
    np.savez_compressed(OUT/'display-projections.npz', **coordinates, row_id=frame.row_id.to_numpy())
    (OUT/'analysis.json').write_text(json.dumps(dict(status='exploratory five-parent leave-one-parent-out evaluation',
        matched_rows=len(frame), controls=90, readout_sha256=sha(__file__), features_sha256=sha(OUT/'features.npz'),
        qualification='Nine nested views per parent are dependent; no precise per-cell or independent-confirmation claim. Display fits all matched data, classifier preprocessing is fitted separately inside every parent fold.'), indent=2)+'\n')
    print(pd.DataFrame(metrics).round(4).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['extract', 'analyze'])
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        globals()[args.action]()
