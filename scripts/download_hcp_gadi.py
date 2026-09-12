"""Download a bounded HCP manifest on copyq; never log signed URLs or credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import unquote, urlparse


def verify(path, row):
    if path.stat().st_size != row['size']:
        raise ValueError('Object size mismatch')
    etag = row['etag'].strip('"')
    whole, sha, parts = hashlib.md5(), hashlib.sha256(), []
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            whole.update(block); sha.update(block); parts.append(hashlib.md5(block).digest())
    calculated = (hashlib.md5(b''.join(parts)).hexdigest() + '-' + str(len(parts))) if '-' in etag else whole.hexdigest()
    if calculated != etag:
        raise ValueError('ETag mismatch (multipart verification assumes 8 MiB parts)')
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--auth', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    signed = json.loads(args.auth.read_text())
    args.root.mkdir(parents=True, exist_ok=True)
    report_path = args.root / 'download-verification.json'
    report = {'status': 'running', 'files': [], 'started_unix': time.time()}
    for row in manifest:
        relative = Path(row['relative_path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        path = args.root / 'data' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            url = signed[row['key']]
            parsed = urlparse(url)
            assert parsed.scheme == 'https' and parsed.hostname == 'hcp-openaccess.s3.amazonaws.com'
            assert unquote(parsed.path).lstrip('/') == row['key']
            assert not any(c in url for c in ['"', '\n', '\r', '\\'])
            partial = path.with_name(path.name + '.part')
            if partial.exists() and partial.stat().st_size > row['size']:
                raise ValueError('Oversized partial object')
            result = subprocess.run(['curl', '--proto', '=https', '--tlsv1.2', '--fail', '--silent',
                                     '--show-error', '--location', '--retry', '4', '--retry-delay', '5',
                                     '--connect-timeout', '30', '--max-time', '7200', '--continue-at', '-',
                                     '--output', str(partial), '--config', '-'],
                                    input='url = "' + url + '"\n', text=True, capture_output=True)
            if result.returncode:
                report.update(status='download_failed', failed_key=row['key'], curl_exit=result.returncode)
                report_path.write_text(json.dumps(report, indent=2) + '\n')
                raise RuntimeError('Download failed; curl exit ' + str(result.returncode))
            digest = verify(partial, row)
            partial.replace(path)
        else:
            digest = verify(path, row)
        report['files'].append(dict(key=row['key'], path=str(path), size=row['size'], etag=row['etag'], sha256=digest))
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        print('VERIFIED', row['key'], row['size'], flush=True)
    report.update(status='complete', finished_unix=time.time())
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    args.auth.unlink()  # Remove expiring object-scoped URLs after successful transfer.


if __name__ == '__main__':
    main()
