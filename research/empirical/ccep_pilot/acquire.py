"""Acquire the pinned, bounded CCEP selection on Gadi's transfer service."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def fetch_one(item, raw):
    path = raw / item['path']
    if path.is_file() and path.stat().st_size == item['bytes'] and sha256(path) == item['sha256']:
        return dict(path=item['path'], status='verified-existing', bytes=item['bytes'], sha256=item['sha256'])
    if path.exists():
        raise ValueError(f'Existing file does not match pinned source: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.part')
    if not partial.exists() or partial.stat().st_size != item['bytes']:
        subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location',
                        '--continue-at', '-', '--retry', '2', '--retry-delay', '10',
                        '--connect-timeout', '30', '--max-time', '7200',
                        '--output', str(partial), item['url']], check=True)
    if partial.stat().st_size != item['bytes'] or sha256(partial) != item['sha256']:
        raise ValueError(f'Acquisition checksum/size mismatch; preserved {partial}')
    partial.replace(path)
    print(f"Verified {item['path']} ({item['bytes']} bytes)", flush=True)
    return dict(path=item['path'], status='downloaded-verified', bytes=item['bytes'], sha256=item['sha256'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    for item in selection['files']:
        if Path(item['path']).is_absolute() or '..' in Path(item['path']).parts:
            raise ValueError(item['path'])
    args.root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(lambda item: fetch_one(item, args.root / 'raw'), selection['files']))
    receipt = dict(completed_utc=datetime.now(timezone.utc).isoformat(),
                   dataset=selection['dataset'], git_tree=selection['git_tree'],
                   selection_sha256=sha256(args.selection), files=receipts)
    (args.root / 'acquisition.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
