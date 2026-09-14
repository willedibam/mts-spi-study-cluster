"""Stage the public unrelated-list/QC overlap on Gadi using private expiring URLs.

Creates four byte-balanced, whole-participant download packs. No permanent keys
or signed URLs are written locally or logged. Job submission remains separate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

import boto3


REMOTE = '/scratch/ql44/we2614/hcp_working_memory_cohort_260915'
SSH = ['ssh', '-4', '-o', 'BatchMode=yes', '-o', 'IPQoS=none', '-o', 'ConnectTimeout=10',
       '-o', 'ControlMaster=no', '-o', 'ControlPath=none', 'we2614@gadi.nci.org.au']


def send_file(path, content):
    parent = str(Path(path).parent)
    command = f'umask 077; mkdir -p {shlex.quote(parent)} && cat > {shlex.quote(path)}'
    subprocess.run(SSH+[command], input=content, text=True, check=True)


def main(root, profile):
    payload_path, overlap_path = root/'cohort-payload-inventory.json', root/'official-unrelated-overlap.json'
    payload, overlap = json.loads(payload_path.read_text()), json.loads(overlap_path.read_text())
    assert payload['status'] == 'complete' and not payload['errors']
    people = set(overlap['both_runs_four_condition_metadata_overlap'])
    assert len(people) == 24 and '105923' not in people
    rows = [row for row in payload['objects'] if row['subject'] in people]
    assert len(rows) == 192 and len({(row['subject'], row['run']) for row in rows}) == 48
    # One directory per participant avoids duplicating the deep S3 hierarchy
    # under Gadi's tight inode quota; the original object key is retained.
    rows = [dict(row, relative_path=f"{row['subject']}/{row['run']}_{Path(row['key']).name}") for row in rows]
    assert len({row['relative_path'] for row in rows}) == len(rows)
    by_person = {person: [row for row in rows if row['subject'] == person] for person in people}
    packs = [dict(subjects=[], objects=[], bytes=0) for _ in range(4)]
    for person in sorted(people, key=lambda p: (-sum(row['size'] for row in by_person[p]), p)):
        pack = min(packs, key=lambda p: p['bytes'])
        pack['subjects'].append(person)
        pack['objects'].extend(by_person[person])
        pack['bytes'] += sum(row['size'] for row in by_person[person])
    plan = dict(status='staged_for_download', remote_root=REMOTE, packs=packs,
                selection='QC-qualified overlap with the official S900 unrelated list; no model-performance selection; not a final cohort or family-safe split',
                total_bytes=sum(row['size'] for row in rows),
                source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [payload_path, overlap_path]},
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    local = root/'cohort-stage'
    local.mkdir(exist_ok=True)
    encoded = json.dumps(plan, indent=2)+'\n'
    path = local/'plan.json'
    if path.exists():
        assert path.read_text() == encoded, 'Preserve the existing staging selection'
    else:
        path.write_text(encoded)
    send_file(REMOTE+'/plan.json', encoded)
    client = boto3.Session(profile_name=profile).client('s3', region_name='us-east-1')
    for number, pack in enumerate(packs):
        folder = f'{REMOTE}/part-{number}'
        manifest = [{key: row[key] for key in ['key', 'size', 'etag', 'relative_path']} for row in pack['objects']]
        send_file(folder+'/download-manifest.json', json.dumps(manifest, indent=2)+'\n')
        signed = {row['key']: client.generate_presigned_url('get_object',
                  Params={'Bucket': 'hcp-openaccess', 'Key': row['key']}, ExpiresIn=43200)
                  for row in manifest}
        send_file(folder+'/download-auth.json', json.dumps(signed))
        print(json.dumps(dict(pack=number, people=len(pack['subjects']), objects=len(manifest), bytes=pack['bytes'])), flush=True)
    print('Prepared expiring object-scoped authorization on Gadi; permanent credentials remain local.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--profile', default='hcp')
    args = parser.parse_args()
    main(args.root, args.profile)
