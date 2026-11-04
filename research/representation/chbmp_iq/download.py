"""Download CHBMP EEG and cognitive tables from Synapse; MRI is skipped.

Runs standalone in an isolated environment holding synapseclient (see
jobs/download.pbs), on Gadi copyq because compute nodes have no network.
Authentication comes only from ~/.synapseConfig or SYNAPSE_AUTH_TOKEN, never
from arguments or the repository. Re-running skips files whose MD5 already
matches, so an interrupted download resumes. receipt.json records each file's
Synapse ID, version, size, MD5 and SHA-256.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import synapseclient

BIDS = 'syn28129311'    # ds_bids_cbm_loris_24_11_21, the November 2021 BIDS release
SCALES = 'syn22324942'  # CHBMP_Cognitive_Scales (WAIS_III.csv and demographics)
TOP = {'participants.tsv', 'participants.json', 'dataset_description.json', 'README', 'CHANGES'}


def digest(path, name):
    h = hashlib.new(name)
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, payload):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)


def children(syn, parent):
    return list(syn.getChildren(parent, includeTypes=['folder', 'file']))


def fetch(syn, entity_id, dest, root, receipts):
    meta = syn.get(entity_id, downloadFile=False)
    md5 = meta._file_handle.get('contentMd5')
    local = dest / meta.name
    if not (local.exists() and md5 and digest(local, 'md5') == md5):
        dest.mkdir(parents=True, exist_ok=True)
        local = Path(syn.get(entity_id, downloadLocation=str(dest), ifcollision='overwrite.local').path)
        if md5 and digest(local, 'md5') != md5:
            raise IOError(f'MD5 mismatch for {entity_id}')
    receipts[entity_id] = dict(path=str(local.relative_to(root)), version=meta.versionNumber,
                               bytes=local.stat().st_size, md5=md5, sha256=digest(local, 'sha256'))


def main(root, limit=None):
    syn = synapseclient.Synapse(skip_checks=True)
    syn.login(silent=True)
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / 'receipt.json'
    receipts = {}
    bids = root / 'bids'
    top = children(syn, BIDS)
    subjects = sorted((c for c in top if c['name'].startswith('sub-')), key=lambda c: c['name'])[:limit]
    for c in top:
        if c['name'] in TOP:
            fetch(syn, c['id'], bids, root, receipts)
    # Derivatives are listed, not downloaded, so their role can be judged first.
    derivatives = [c for c in top if c['name'] == 'derivatives']
    listing = {d['id']: [x['name'] for x in children(syn, d['id'])] for d in derivatives}
    for c in children(syn, SCALES):
        fetch(syn, c['id'], root / 'cognitive', root, receipts)
    missing = []
    for k, s in enumerate(subjects, 1):
        eeg = [c for c in children(syn, s['id']) if c['name'] == 'eeg']
        if not eeg:
            missing.append(s['name'])
            continue
        for f in children(syn, eeg[0]['id']):
            fetch(syn, f['id'], bids / s['name'] / 'eeg', root, receipts)
        if k % 10 == 0:
            atomic_json(receipt_path, dict(source=BIDS, scales=SCALES, files=receipts, partial=True))
            print(f'{k}/{len(subjects)} subjects', flush=True)
    atomic_json(receipt_path, dict(source=BIDS, scales=SCALES, subjects=len(subjects), without_eeg=missing,
                                   derivatives_listing=listing, synapseclient=synapseclient.__version__,
                                   files=receipts, partial=False))
    print(f'done: {len(subjects)} subjects, {len(missing)} without EEG, {len(receipts)} files', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--limit', type=int, help='first N subjects only (a smoke test)')
    args = parser.parse_args()
    main(args.root, args.limit)
