"""Acquire open datasets for the consciousness atlas, verified against the repository's own checksums.

Each dataset is resolved from its repository's API into a file list (path, URL, bytes, digest), the files are
fetched with curl into <root>/<dataset>/raw, and every file is checked against the published digest (SHA-256
where the repository gives one, else MD5). <root>/<dataset>/acquisition.json records the resolved list and
the outcome. Run on Gadi's copyq (compute nodes have no internet).

    python -m research.representation.atlas.acquire fly13 doc purdon fly49 --root $MTS_SCRATCH/data/atlas
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

FIGSHARE = 'https://api.figshare.com/v2/articles/{}'
OSF = 'https://api.osf.io/v2/nodes/{}/files/osfstorage/'
DANDI = 'https://api.dandiarchive.org/api/dandisets/{}/versions/{}/assets/?page_size=200&metadata=1'
# name -> (repository, identifier, subfolder or version); see the scoping note for what each dataset is.
DATASETS = dict(fly13=('figshare', '12309950', None), doc=('figshare', '23552964', None),
                fly49=('osf', '8wvsq', 'preprocessed'), purdon=('dandi', '001285', 'draft'))


def get(url):
    out = subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location', '--retry', '3', url],
                         check=True, capture_output=True)
    return json.loads(out.stdout)


def figshare(article, _):
    record = get(FIGSHARE.format(article))
    return [dict(path=f['name'], url=f['download_url'], bytes=f['size'], md5=f['computed_md5']) for f in record['files']]


def osf(node, folder):
    listing = get(OSF.format(node))['data']
    if folder is not None:
        [entry] = [e for e in listing if e['attributes']['name'] == folder]
        listing = get(entry['relationships']['files']['links']['related']['href'])['data']
    items = []
    for e in listing:
        if e['attributes']['kind'] != 'file':
            raise ValueError(f"nested folder {e['attributes']['name']} is not handled")
        a = e['attributes']
        items.append(dict(path=a['name'], url=e['links']['download'], bytes=a['size'], sha256=a['extra']['hashes']['sha256']))
    return items


def dandi(dandiset, version):
    url, items = DANDI.format(dandiset, version), []
    while url:
        page = get(url)
        for asset in page['results']:
            meta = asset['metadata']
            items.append(dict(path=asset['path'], url=f"https://api.dandiarchive.org/api/assets/{asset['asset_id']}/download/",
                              bytes=asset['size'], sha256=meta['digest']['dandi:sha2-256']))
        url = page['next']
    return items


def digest(path, kind):
    h = hashlib.new(kind)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def fetch(item, raw):
    kind = 'sha256' if 'sha256' in item else 'md5'
    path = raw / item['path']
    if Path(item['path']).is_absolute() or '..' in Path(item['path']).parts:
        raise ValueError(item['path'])
    if path.is_file() and path.stat().st_size == item['bytes'] and digest(path, kind) == item[kind]:
        return dict(path=item['path'], status='verified-existing')
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.part')
    subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location', '--continue-at', '-', '--retry', '3',
                    '--retry-delay', '10', '--connect-timeout', '30', '--output', str(partial), item['url']], check=True)
    if partial.stat().st_size != item['bytes'] or digest(partial, kind) != item[kind]:
        raise ValueError(f"size or {kind} mismatch for {item['path']}; kept {partial}")
    partial.replace(path)
    return dict(path=item['path'], status='downloaded-verified')


def acquire(name, root):
    repository, identifier, extra = DATASETS[name]
    items = dict(figshare=figshare, osf=osf, dandi=dandi)[repository](identifier, extra)
    out = root / name
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(lambda item: fetch(item, out / 'raw'), items))
    (out / 'acquisition.json').write_text(json.dumps(dict(
        dataset=name, repository=repository, identifier=identifier, subset=extra,
        completed_utc=datetime.now(timezone.utc).isoformat(), files=items, receipts=receipts), indent=1) + '\n')
    print(name, len(items), 'files,', round(sum(i['bytes'] for i in items) / 1e9, 2), 'GB verified', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datasets', nargs='+', choices=sorted(DATASETS))
    parser.add_argument('--root', type=Path, required=True)
    a = parser.parse_args()
    for name in a.datasets:
        acquire(name, a.root)
