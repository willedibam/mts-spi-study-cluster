"""Independently replay saved scout preprocessing, PC1 and reported diagnostics.

Verifies saved analysis artifacts without recomputing SPIs or selecting a model.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-9, equal_nan=True)


def replay_pc1(features, train, model):
    """Check source-only preprocessing and the leading eigenvalue via the Gram matrix."""
    train = np.asarray(train)
    mask = np.isfinite(features[train]).sum(0) >= len(train)-1
    np.testing.assert_array_equal(model['mask'], mask)
    values = features[:, mask]
    medians = np.nanmedian(values[train], axis=0)
    close(model['medians'], medians)
    filled = np.where(np.isfinite(values), values, medians)
    means, scales = filled[train].mean(0), filled[train].std(0)
    close(model['means'], means)
    close(model['scales'], scales)
    keep = scales > 1e-10
    np.testing.assert_array_equal(model['keep'], keep)
    space = (filled[:, keep]-means[keep])/scales[keep]
    loading = model['loading']
    close(np.linalg.norm(loading), 1.)
    assert loading[np.argmax(np.abs(loading))] > 0
    source = space[train]
    # The analysis used an SVD. This check uses the small source Gram matrix
    # and a covariance eigenvector residual, avoiding a second identical fit.
    eigenvalues = np.linalg.eigvalsh(source @ source.T)
    eigenvalue = eigenvalues[-1]
    residual = source.T @ (source @ loading) - eigenvalue*loading
    assert np.linalg.norm(residual)/max(eigenvalue, 1.) < 1e-8
    raw = space @ loading
    close(model['score_scale'], raw[train].std())
    score = raw/model['score_scale']
    close(score[train].std(), 1.)
    return score, space, float(eigenvalue/eigenvalues.sum())


def verify(root):
    output = root/'analysis'
    report = json.loads((output/'report.json').read_text())
    rows = json.loads((output/'scores.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    assert report['status'] == 'complete' and sha(root/'manifest.json') == report['manifest_sha256']
    assert len(rows) == len(manifest['rows']) == 64
    for row, original in zip(rows, manifest['rows'], strict=True):
        assert all(row[k] == value for k, value in original.items())
    names = [r['name'] for r in rows]
    primary = [i for i, r in enumerate(rows) if r['T'] == 4000 and r['layout'] == 'coverage_a']
    train, held = primary[:8], primary[8:]
    assert len(train) == len(held) == 8
    assert report['training_names'] == [names[i] for i in train]
    assert report['evaluation_names'] == [names[i] for i in held]
    checks = {}
    with np.load(output/'feature-bank.npz', allow_pickle=False) as bank, np.load(output/'pc1-models.npz', allow_pickle=False) as models:
        assert bank['names'].tolist() == names
        for name in ['z', 'm', 'spectra']:
            model = {key[len(name)+1:]: models[key] for key in models.files if key.startswith(name+'_')}
            q, space, explained = replay_pc1(bank[name], train, model)
            close(q, [r[name] for r in rows])
            summary = report['summaries'][name]
            close(summary['training_explained_variance'], explained)
            assert summary['retained_features'] == space.shape[1]
            associations = summary['held_later_blocks']
            close(associations['load_auc_arbitrary_orientation'], roc_auc_score([rows[i]['memory'] == 2 for i in held], q[held]))
            close(associations['image_auc_arbitrary_orientation'], roc_auc_score([rows[i]['image'] == 2 for i in held], q[held]))
            close(associations['chronological_spearman'], spearmanr(np.arange(8), q[held]).statistic)
            comparisons = summary['observation_comparisons']
            assert [(c['layout'], c['T']) for c in comparisons] == [('coverage_b',4000), ('coverage_a',2000), ('coverage_b',2000)]
            reference = pdist(space[primary])
            for comparison in comparisons:
                alternate = [next(i for i,r in enumerate(rows) if r['block'] == rows[j]['block'] and r['layout'] == comparison['layout'] and r['T'] == comparison['T']) for j in primary]
                close(comparison['block_distance_spearman'], spearmanr(reference, pdist(space[alternate])).statistic)
                close(comparison['median_displacement_over_between_block_distance'], np.median(np.linalg.norm(space[alternate]-space[primary], axis=1))/np.median(reference))
                close(comparison['q_spearman_all_blocks'], spearmanr(q[primary], q[alternate]).statistic)
                close(comparison['q_spearman_held_blocks'], spearmanr(q[held], q[alternate[8:]]).statistic)
            checks[name] = dict(retained_features=space.shape[1], all_64_scores_replayed=True,
                                preprocessing_and_leading_eigenvector_verified=True, all_reported_metrics_replayed=True)
    verification = dict(status='verified', checks=checks, script_sha256=sha(Path(__file__)),
                        artifacts={p.name:sha(p) for p in [root/'manifest.json', output/'report.json', output/'scores.json', output/'feature-bank.npz', output/'pc1-models.npz']},
                        scope='Independent saved preprocessing, Gram-eigenvalue PC1 and diagnostic replay; no new SPI extraction or predictive model selection.')
    (output/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    verify(parser.parse_args().root)
