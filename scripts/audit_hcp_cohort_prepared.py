"""Verify the full prepared HCP cohort without fitting or inspecting predictions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8*1024**2):
            digest.update(block)
    return digest.hexdigest()


def main(audit_root, cohort_root):
    plan = json.loads((cohort_root/'plan.json').read_text())
    people = sorted(person for pack in plan['packs'] for person in pack['subjects'])
    assert len(people) == len(set(people)) == 24
    runs, errors = [], []
    for person in people:
        for run in ['6-Wrkmem', '7-Wrkmem']:
            root = audit_root/'cohort-prepared'/f'{person}_{run}'
            try:
                audit = json.loads((root/'continuous-audit.json').read_text())
                prepared = json.loads((root/'m32-spatial/preparation.json').read_text())
                identity = json.loads((root/'preparation-identity.json').read_text())
                assert identity['subject'] == audit['subject'] == person and identity['run'] == audit['run'] == run
                assert sha(root/'continuous-audit.json') == prepared['input_audit_sha256']
                archive = root/'m32-spatial/views.npz'
                assert sha(archive) == prepared['archive_sha256']
                assert all(v > 0 for v in audit['condition_block_counts'].values())
                entries = prepared['entries']
                assert len(entries) == 2*len(audit['blocks'])
                maximum_mean, maximum_sd_error = 0., 0.
                with np.load(archive, allow_pickle=False) as data:
                    names = data['__dataset_names__'].tolist()
                    assert names == [row['name'] for row in entries] and len(names) == len(set(names))
                    assert data['__axis_order__'].tolist() == ['observation', 'process']
                    for i, row in enumerate(entries):
                        x = data[row['name']]
                        assert row['M'] == 32 and x.shape == (row['T'], 32)
                        assert data['__shapes__'][i].tolist() == list(x.shape)
                        assert json.loads(data['__labels_json__'][i]) == [f"memory{row['memory']}", f"image{row['image']}", row['layout']]
                        assert np.isfinite(x).all()
                        maximum_mean = max(maximum_mean, float(np.max(np.abs(x.mean(0)))))
                        maximum_sd_error = max(maximum_sd_error, float(np.max(np.abs(x.std(0)-1))))
                    assert maximum_mean < 1e-9 and maximum_sd_error < 1e-9
                runs.append(dict(subject=person, run=run, sfreq=audit['sfreq'], raw_samples=audit['samples'],
                    raw_meg=audit['meg_channels'], good_meg=audit['good_meg_channels'],
                    clean_blocks=len(audit['blocks']), excluded_blocks=len(audit['excluded_blocks']),
                    conditions=audit['condition_block_counts'], views=len(entries),
                    parent_T=sorted({row['T'] for row in entries}),
                    ica_inverse_error=prepared['ica_left_inverse_max_error'], excluded_ica=prepared['excluded_ica_zero_based'],
                    timing_residual=audit['eprime']['max_affine_timing_residual_seconds'],
                    waveform_max_abs_mean=maximum_mean, waveform_max_sd_error=maximum_sd_error,
                    archive_sha256=prepared['archive_sha256'], preparation_sha256=sha(root/'m32-spatial/preparation.json')))
            except Exception as error:
                errors.append(dict(subject=person, run=run, error_type=type(error).__name__, detail=str(error)))
    totals = Counter()
    for row in runs:
        totals.update(row['conditions'])
    report = dict(status='verified' if not errors and len(runs) == 48 else 'requires_review',
                  participants=len(people), verified_runs=len(runs), clean_blocks=sum(r['clean_blocks'] for r in runs),
                  dependent_parent_views=sum(r['views'] for r in runs), conditions=dict(totals),
                  runs=runs, errors=errors, script_sha256=sha(Path(__file__)),
                  scope='Prepared waveform/metadata integrity only; no pyspi extraction, final family split, or model prediction for this cohort.')
    (audit_root/'cohort-prepared-audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'runs'}, indent=2))
    if errors:
        raise RuntimeError('Prepared cohort requires review; successful archives retained')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-root', type=Path, required=True)
    parser.add_argument('--cohort-root', type=Path, required=True)
    args = parser.parse_args()
    main(args.audit_root, args.cohort_root)
