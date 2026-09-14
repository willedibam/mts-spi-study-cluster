"""Inventory raw/ICA payload for metadata-qualified HCP participants; no downloads."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import boto3
from botocore.config import Config


def main(root, profile):
    inventory_path, qc_path = root/'cohort-inventory.json', root/'cohort-qc-audit.json'
    inventory, qc = json.loads(inventory_path.read_text()), json.loads(qc_path.read_text())
    assert inventory['status'] == qc['status'] == 'complete'
    by_person = {}
    for row in qc['runs']:
        by_person.setdefault(row['subject'], []).append(row)
    eligible = sorted(person for person, rows in by_person.items()
                      if len(rows) == 2 and all(row['all_four_conditions_present'] for row in rows))
    assert len(eligible) == qc['summary']['participants_with_both_runs_all_four_clean_conditions']
    client = boto3.Session(profile_name=profile).client('s3', region_name='us-east-1',
        config=Config(max_pool_connections=16, retries={'mode': 'standard', 'max_attempts': 5}))
    raw, requests = [], []
    for participant in inventory['rows']:
        person = participant['subject']
        if person not in eligible:
            continue
        for run in participant['runs']:
            assert run['has_raw'] and run['has_config'] and run['has_eprime']
            for row in run['files']:
                raw.append(dict(row, subject=person, run=run['run'],
                                relative_path=row['key'].removeprefix('HCP_1200/')))
            stem = f"{person}_MEG_{run['run']}"
            key = f'HCP_1200/{person}/MEG/Wrkmem/icaclass/{stem}_icaclass.mat'
            requests.append(dict(subject=person, run=run['run'], key=key))

    def inspect(row):
        try:
            response = client.head_object(Bucket='hcp-openaccess', Key=row['key'])
            return dict(row, size=response['ContentLength'], etag=response['ETag'],
                        relative_path=row['key'].removeprefix('HCP_1200/'))
        except Exception as error:
            return dict(row, error_type=type(error).__name__)

    with ThreadPoolExecutor(max_workers=12) as pool:
        inspected = list(pool.map(inspect, requests))
    errors = [row for row in inspected if 'error_type' in row]
    ica = [row for row in inspected if 'error_type' not in row]
    objects = sorted(raw + ica, key=lambda row: row['key'])
    report = dict(status='complete' if not errors else 'partial', eligible_subjects=eligible,
                  selection='Both runs retain all four clean complete-block condition cells in the existing QC audit; no predictive selection',
                  raw_objects=len(raw), ica_objects=len(ica), objects=objects, errors=errors,
                  raw_bytes=sum(row['size'] for row in raw), ica_bytes=sum(row['size'] for row in ica),
                  total_bytes=sum(row['size'] for row in objects),
                  input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [inventory_path, qc_path]},
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  limitations='Object availability/size only; ICA contents, participant-specific cleaning, family eligibility and split manifest remain unverified. No waveforms or correction matrices downloaded.')
    (root/'cohort-payload-inventory.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in {'objects', 'eligible_subjects', 'errors'}}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--profile', default='hcp')
    args = parser.parse_args()
    main(args.root, args.profile)
