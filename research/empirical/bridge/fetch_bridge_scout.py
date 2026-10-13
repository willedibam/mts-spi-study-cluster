"""Fetch the frozen bridge metadata, compact indices and one raw configuration."""
import argparse
import hashlib
import json
from pathlib import Path

import requests


def fetch(root):
    root.mkdir(parents=True, exist_ok=True)
    meta_path = root / 'bridge-metadata.json'
    if not meta_path.exists():
        response = requests.get('https://api.figshare.com/v2/articles/27732792/versions/1', timeout=60)
        response.raise_for_status()
        meta_path.write_text(json.dumps(response.json(), indent=2) + '\n')
    metadata = json.loads(meta_path.read_text())
    assert metadata['version'] == 1
    receipts = {}
    for item in metadata['files']:
        if not item['name'].endswith('.pkl'):
            continue
        tail_path = root / (item['name'] + '.tail1m')
        if not tail_path.exists():
            with requests.get(item['download_url'], headers={'Range': 'bytes=-1048576'},
                              stream=True, timeout=60) as response:
                response.raise_for_status()
                assert response.status_code == 206, 'Server did not honour bounded range'
                expected = f"bytes {item['size']-1048576}-{item['size']-1}/{item['size']}"
                assert response.headers['Content-Range'] == expected
                content = response.raw.read(1048577)
                assert len(content) == 1048576
                tail_path.write_bytes(content)
        assert tail_path.stat().st_size == 1048576
        receipts[item['name']] = dict(url=item['download_url'], source_size=item['size'],
            source_md5=item['computed_md5'], tail_sha256=hashlib.sha256(tail_path.read_bytes()).hexdigest())
        if item['name'] == 'B3.pkl':
            dest = root / item['name']
            if not dest.exists():
                temporary = dest.with_suffix('.partial')
                with requests.get(item['download_url'], stream=True, timeout=90) as response:
                    response.raise_for_status()
                    with temporary.open('wb') as stream:
                        for chunk in response.iter_content(1024*1024):
                            stream.write(chunk)
                assert temporary.stat().st_size == item['size']
                assert hashlib.md5(temporary.read_bytes()).hexdigest() == item['computed_md5']
                temporary.rename(dest)
            assert dest.stat().st_size == item['size']
            assert hashlib.md5(dest.read_bytes()).hexdigest() == item['computed_md5']
    (root / 'fetch-receipts.json').write_text(json.dumps(receipts, indent=2) + '\n')
    print('Verified compact indices for four configurations and raw B3.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/representation_application_scout_260908'))
    fetch(parser.parse_args().data)
