"""Summarize the verified, frozen InceptionTime comparison without new fitting."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path('results/inceptiontime_followup_260911')
SYNTHETIC = {
    'original': 'oscillatory_coorganization_pilot_260909',
    'faster_dynamics': 'oscillatory_coorganization_transfer_260909',
    'direct_mechanism': 'oscillatory_mechanism_transfer_260910',
}
METRICS = ['balanced_accuracy', 'auroc', 'brier']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report_path = ROOT / 'evaluation/report.json'
    report = json.loads(report_path.read_text())
    verified = json.loads((ROOT / 'evaluation/verification.json').read_text())
    assert verified['status'] == 'verified' and verified['report_sha256'] == sha(report_path)
    # Bind downloaded source reports to the exact freeze used for target scoring.
    for remote_stem, expected in report['frozen'].items():
        stem = ROOT / 'fits' / Path(remote_stem).parent.name / Path(remote_stem).name
        assert sha(stem.with_suffix('.json')) == expected['report_sha256']
        assert sha(stem.with_suffix('.pt')) == expected['checkpoint_sha256']
    rows, inputs = [], {str(report_path): sha(report_path)}
    for dataset, folder in SYNTHETIC.items():
        path = Path('results') / folder / 'verified-report/summary.csv'
        inputs[str(path)] = sha(path)
        for row in csv.DictReader(path.open()):
            if int(row['M']) == 16:
                rows.append(dict(dataset=dataset, labels=int(row['labels']), method=row['method'],
                                 aggregation='mean of three cohort metrics',
                                 balanced_accuracy=float(row['balanced_accuracy']),
                                 auroc=float(row['AUROC']), brier=float(row['Brier'])))
        for labels in [10, 20, 40]:
            subset = [s for s in report['scores'] if s['dataset'] == dataset and
                      s['case'].startswith(f'synthetic-n{labels}-')]
            for ensemble in [True, False]:
                selected = [s for s in subset if (s['member'] == 'ensemble') == ensemble]
                assert len(selected) == (3 if ensemble else 15)
                rows.append(dict(dataset=dataset, labels=labels,
                                 method='InceptionTime ensemble' if ensemble else 'InceptionTime members',
                                 aggregation='mean of cohort ensembles' if ensemble else 'mean of cohort/member metrics',
                                 **{m: float(np.mean([s[m] for s in selected])) for m in METRICS}))
    neuro = [s for s in report['scores'] if s['dataset'] == 'neurotycho']
    neuro_members = []
    for member in [11, 23, 47, 71, 101, 'ensemble']:
        selected = [s for s in neuro if s['member'] == member]
        assert len(selected) == 2
        neuro_members.append(dict(member=member, **{m: float(np.mean([s[m] for s in selected])) for m in METRICS}))
    path = Path('results/neurotycho_target_pilot_260910/evaluation/report.json')
    inputs[str(path)] = sha(path)
    original = [s for s in json.loads(path.read_text())['scores'] if s['M'] == 16]
    neuro_baselines = []
    for method in sorted({s['method'] for s in original}):
        selected = [s for s in original if s['method'] == method]
        neuro_baselines.append(dict(method=method, seeds=len(selected),
                                   balanced_accuracy=float(np.mean([s['mean']['balanced_accuracy'] for s in selected])),
                                   auroc=float(np.mean([s['mean']['auroc'] for s in selected])),
                                   brier=float(np.mean([s['mean']['balanced_brier'] for s in selected]))))
    source = []
    for path in sorted((ROOT / 'fits').glob('*/member-*.json')):
        if '.progress.' in path.name:
            continue
        fitted = json.loads(path.read_text())
        selected = min(fitted['candidates'], key=lambda c: c['mean_brier'])
        source.append(dict(case=path.parent.name, member=fitted['member_seed'],
                           selected_epochs=fitted['selected_epochs'], validation_brier=selected['mean_brier'],
                           selected_fold_caps=sum(f['selected_epoch_at_ceiling'] for f in selected['folds']),
                           seconds=fitted['total_seconds']))
    assert len(source) == 55
    summary = dict(input_sha256=inputs, synthetic=rows, neurotycho_members=neuro_members,
                   neurotycho_baselines=neuro_baselines, neurotycho_dates=neuro,
                   source_fits=source, frozen_source_reports_verified=len(report['frozen']),
                   caveat='Full M=16 only; ensembles and averages of seed metrics are distinct estimands.')
    (ROOT / 'evaluation/comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (ROOT / 'evaluation/synthetic-comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    styles = [('z-pca', 'SPI–SPI + PCA', '#0072b2'), ('m-pls', 'Rich SPI summaries', '#d55e00'),
              ('raw:agreement-pls', 'Phase/envelope specialist', '#009e73'),
              ('InceptionTime ensemble', 'InceptionTime (5-member)', '#6a3d9a')]
    for ax, (dataset, _) in zip(axes, SYNTHETIC.items()):
        for method, label, color in styles:
            selected = sorted([r for r in rows if r['dataset'] == dataset and r['method'] == method], key=lambda r: r['labels'])
            ax.plot([r['labels'] for r in selected], [r['balanced_accuracy'] for r in selected], 'o-', label=label, color=color)
        ax.axhline(.5, color='.6', linewidth=.8, linestyle=':')
        ax.set(title=dataset.replace('_', ' ').capitalize(), xlabel='Total labelled source realizations', xticks=[10, 20, 40], ylim=(.4, 1.025))
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Balanced accuracy, full M=16')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .16, 1, 1))
    for extension in ['png', 'svg']:
        fig.savefig(ROOT / f'evaluation/full-size-comparison.{extension}', dpi=180)
    plt.close(fig)
    print(json.dumps(dict(neurotycho_members=neuro_members,
                          selected_fold_caps=sum(s['selected_fold_caps'] for s in source),
                          summed_fit_seconds=sum(s['seconds'] for s in source)), indent=2))


if __name__ == '__main__':
    main()
