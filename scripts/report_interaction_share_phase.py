"""Paired raw phase-control results, with strict origin and fit provenance checks."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src.representation_screen import bootstrap_group_means
from src.representation_state_data import file_hash, load_state_data
from src.run_external_corpus import _atomic_json


def report(config, data, results, output):
    protocol = yaml.safe_load(config.read_text())
    arms = protocol['phase_control']['arms']
    families = protocol['generator']['families']
    seeds = protocol['methods']['subset_seeds']
    names = sum((protocol['phase_control'][key] for key in
                 ['statistical_methods', 'raw_methods', 'neural_methods']), [])
    fits, provenance, reference, cohorts = {}, [], None, {}
    for arm in arms:
        manifest, _ = load_state_data(data / arm, config)
        rows = manifest['rows']
        if reference is None:
            reference = rows
        assert rows == reference, 'Origin records differ between arms'
        for path in sorted((results / arm).rglob('*.json')):
            record = json.loads(path.read_text())
            if 'identity' not in record or 'predictions_sha256' not in record:
                continue
            ident = record['identity']
            assert ident['protocol_sha256'] == file_hash(config)
            assert ident['manifest_sha256'] == file_hash(data / arm / 'manifest.json')
            assert record['predictions_sha256'] == file_hash(path.with_suffix('.npz'))
            name, family, seed = (ident[k] for k in ['method', 'source_family', 'seed'])
            assert name in names and family in families and seed in seeds
            assert ident['n_per_coupling'] == 4 and record['labels_total'] == 20
            key = arm, name, family, seed
            assert key not in fits, f'Duplicate fit: {key}'
            with np.load(path.with_suffix('.npz'), allow_pickle=False) as archive:
                ix, train = archive['evaluation_indices'], archive['train_indices']
                np.testing.assert_array_equal(ix, [i for i, row in enumerate(rows)
                                                  if row['role'] == 'evaluation'])
                assert len(train) == len(set(train)) == 20
                assert all(rows[i]['role'] == 'training_pool' and
                           rows[i]['family'] == family and
                           rows[i]['cohort_index'] == seeds.index(seed) for i in train)
                assert not {rows[i]['master_id'] for i in train} & {rows[i]['master_id'] for i in ix}
                cohort_key = family, seed
                if cohort_key in cohorts:
                    np.testing.assert_array_equal(train, cohorts[cohort_key])
                cohorts[cohort_key] = train.copy()
                np.testing.assert_array_equal(archive['row_id'], [rows[i]['row_id'] for i in ix])
                np.testing.assert_array_equal(archive['target'], [rows[i]['target'] for i in ix])
                prediction = archive['prediction']
                assert prediction.shape == archive['target'].shape and np.isfinite(prediction).all()
                fits[key] = (ix.copy(), prediction.copy(), abs(prediction - archive['target']))
            provenance.append(dict(path=str(path), sha256=file_hash(path)))
    expected = len(arms) * len(families) * len(seeds)
    counts = {name: sum(key[1] == name for key in fits) for name in names}
    complete = [name for name in names if counts[name] == expected]
    if not complete:
        raise ValueError('No method complete across arms, families and cohorts')
    # Matched origins and label subsets must give identical invariant-control predictions.
    for name in ['median', 'u-pls']:
        if name in complete:
            for family in families:
                for seed in seeds:
                    for arm in arms[1:]:
                        np.testing.assert_allclose(fits[arm, name, family, seed][1],
                                                   fits[arms[0], name, family, seed][1],
                                                   atol=1e-10, rtol=0)
    summaries, cells, paired, bootstraps, cohort_values = {}, {}, {}, {}, {}
    for scope in ['same_family', 'cross_family']:
        summaries[scope], cells[scope], paired[scope], cohort_values[scope] = {}, {}, {}, {}
        for name in complete:
            for arm in arms:
                family_means, family_boots = [], []
                cohort_values[scope].setdefault(name, {})[arm] = {}
                for family in families:
                    destination = family if scope == 'same_family' else next(f for f in families if f != family)
                    errors = []
                    for seed in seeds:
                        ix, _, error = fits[arm, name, family, seed]
                        mask = np.array([rows[i]['family'] == destination for i in ix])
                        errors.append(error[mask])
                    assert mask.sum() == 200
                    selected = ix[mask]
                    assert len({rows[i]['master_id'] for i in selected}) == len(selected)
                    strata = np.array([rows[i]['coupling_index'] for i in selected])
                    values = np.mean(errors, axis=0)
                    # Shared seed/order gives paired resampling across methods and arms.
                    boot = bootstrap_group_means(values[None, :], strata, 2000,
                                                 1729 + families.index(family))[0]
                    family_means.append(float(values.mean()))
                    family_boots.append(boot)
                    cell = f'{family}->{destination}'
                    cells[scope].setdefault(cell, {}).setdefault(name, {})[arm] = {
                        'MAE': float(values.mean()), 'conditional_95_CI': np.quantile(boot, [.025, .975]).tolist()}
                    cohort_values[scope][name][arm][family] = np.mean(errors, axis=1).tolist()
                boot = np.mean(family_boots, axis=0)
                bootstraps[scope, name, arm] = boot
                summaries[scope].setdefault(name, {})[arm] = {
                    'MAE': float(np.mean(family_means)),
                    'conditional_95_CI': np.quantile(boot, [.025, .975]).tolist()}
            for left, right in [('independent_phase', 'common_phase'), ('common_phase', 'intact')]:
                delta = summaries[scope][name][left]['MAE'] - summaries[scope][name][right]['MAE']
                boot = bootstraps[scope, name, left] - bootstraps[scope, name, right]
                paired[scope].setdefault(name, {})[left + '_minus_' + right] = {
                    'MAE_difference': delta, 'conditional_95_CI': np.quantile(boot, [.025, .975]).tolist()}
    method_contrasts = {}
    for scope in summaries:
        for left, right in [('z-pls', 'shape-pls'), ('z-pls', 'u-pls'), ('z-pls', 'linear'),
                            ('z-pls', 'random_encoder'), ('z-pls', 'neural')]:
            if left not in complete or right not in complete:
                continue
            for arm in arms:
                boot = bootstraps[scope, left, arm] - bootstraps[scope, right, arm]
                method_contrasts.setdefault(scope, {}).setdefault(left + '_minus_' + right, {})[arm] = {
                    'MAE_difference': summaries[scope][left][arm]['MAE'] - summaries[scope][right][arm]['MAE'],
                    'conditional_95_CI': np.quantile(boot, [.025, .975]).tolist()}
    output.mkdir(parents=True, exist_ok=True)
    incomplete = {name: count for name, count in counts.items() if name not in complete}
    _atomic_json(output / 'results.json', dict(
        status='paired_mechanism_exploratory_conditional_on_fitted_models',
        primary='same_family_independent_minus_common_phase', summary=summaries,
        paired_arm_contrasts=paired, method_contrasts=method_contrasts, cells=cells,
        per_source_cohort_MAE=cohort_values, incomplete_methods=incomplete,
        fit_provenance=provenance, protocol_sha256=file_hash(config),
        report_code_sha256=file_hash(Path(__file__)),
        caveat='Origin-system labels, not surrogate Jacobians. No full channel-independence claim. '
               'Intervals condition on fitted models and resample paired evaluation origins; pointwise, unadjusted.'))
    lines = ['# Raw phase-control experiment', '',
             'M16/T1000; 20 source labels including tuning; five disjoint cohorts per family.',
             'Each model is refit in each arm. Lower MAE is better.', '']
    for scope in summaries:
        lines += [scope.replace('_', ' ').capitalize(), '',
                  '| Method | Intact | Common phase | Independent phase | Independent − common [95% CI] |',
                  '|---|---:|---:|---:|---:|']
        for name in complete:
            item = paired[scope][name]['independent_phase_minus_common_phase']
            lo, hi = item['conditional_95_CI']
            lines.append('| ' + name + ' | ' + ' | '.join(
                f"{summaries[scope][name][arm]['MAE']:.4f}" for arm in arms) +
                f" | {item['MAE_difference']:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
        lines.append('')
    lines += ['Intervals are paired across evaluation origins and conditional on the fitted models; '
              'they do not account for training-population uncertainty. All cohort and family results are in results.json.',
              'Labels describe the original systems. Phase randomization does not preserve exact histograms; '
              'independent phases do not imply full independence.',
              f'Incomplete methods (excluded from tables): {incomplete}', '']
    (output / 'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines))
    if not incomplete:
        plot(summaries, paired, complete, arms, output)


def plot(summaries, paired, names, arms, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = {'z-pls': 'SPI–SPI z', 'shape-pls': 'SPI distributions', 'u-pls': 'Autospectra',
              'median': 'Source median', 'linear': 'Linear reference',
              'random_encoder': 'Frozen random encoder', 'neural': 'Trained encoder'}
    colors = ['#30343B', '#267EAB', '#D16C31']
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), gridspec_kw={'width_ratios': [1.25, 1]}, sharey=True)
    for row, scope in enumerate(['same_family', 'cross_family']):
        ax, delta_ax = axes[row]
        for j, arm in enumerate(arms):
            means = np.array([summaries[scope][name][arm]['MAE'] for name in names])
            intervals = np.array([summaries[scope][name][arm]['conditional_95_CI'] for name in names])
            ax.errorbar(means, np.arange(len(names)) + (j - 1) * .2,
                        xerr=np.maximum(np.stack([means - intervals[:, 0], intervals[:, 1] - means]), 0),
                        fmt='o', ms=4, color=colors[j], lw=1,
                        label=arm.replace('_', ' ').capitalize())
        for i, name in enumerate(names):
            item = paired[scope][name]['independent_phase_minus_common_phase']
            value = item['MAE_difference']
            lo, hi = item['conditional_95_CI']
            delta_ax.plot([lo, hi], [i, i], color='#267EAB', lw=1.5)
            delta_ax.plot(value, i, 'o', color='#267EAB', ms=4)
        ax.set_title(scope.replace('_', ' ').capitalize(), loc='left', fontsize=11, fontweight='bold')
        ax.set_yticks(range(len(names)), [labels[name] for name in names])
        ax.set_xlabel('Mean absolute error (lower is better)')
        ax.set_xlim(.065, .22)
        delta_ax.axvline(0, color='#999999', ls='--', lw=1)
        delta_ax.set_xlabel('MAE: independent − common phase')
        delta_ax.set_xlim(-.012, .095)
        for panel in [ax, delta_ax]:
            panel.spines[['top', 'right']].set_visible(False)
            panel.grid(axis='x', alpha=.15)
    axes[0, 0].invert_yaxis()
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.58, .9), ncol=3, fontsize=9, frameon=False)
    fig.suptitle('Which information supports origin-system inference?', fontsize=14, x=.02, ha='left')
    fig.text(.02, .92, 'M = 16, T = 1000 · 20 labels · 5 disjoint cohorts per family · models refit in each condition', fontsize=10)
    fig.text(.02, .015, '95% intervals resample paired evaluation origins, conditional on fitted models. Labels describe the original systems.', fontsize=8)
    fig.tight_layout(rect=[0, .04, 1, .91])
    for suffix in ['png', 'svg']:
        fig.savefig(output / ('phase-controls.' + suffix), dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['config', 'data', 'results', 'output']:
        parser.add_argument('--' + key, required=True, type=Path)
    args = parser.parse_args()
    report(args.config, args.data, args.results, args.output)
