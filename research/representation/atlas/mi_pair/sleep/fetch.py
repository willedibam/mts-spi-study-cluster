"""Download the MNI Open iEEG Atlas BIDS derivative (NEMAR nm000362 v1.0.2, CC-BY-NC-4.0) files this study uses.

    python fetch.py <out dir> <pattern> [<pattern> ...]     e.g. 'task-wake_acq-1min' 'task-sleepN3_acq-10min' '_events.tsv'

A file is fetched when its path contains any pattern. Git-tracked text comes from GitHub; annexed files (symlinks to
/annex/objects/<key>) from NEMAR's public S3 remote (git-annex branch remote.log: bucket nemar, fileprefix nm000362/objects/).
Every annexed file is checked against the SHA-256 in its key. Existing complete files are skipped.
"""
import sys, os, json, hashlib, subprocess
REPO, REF = 'nemarDatasets/nm000362', 'main'
S3 = 'https://nemar.s3.us-east-2.amazonaws.com/nm000362/objects/'
get = lambda u: subprocess.run(['curl', '-sfL', '--retry', '5', '-m', '1800', u], check=True, capture_output=True).stdout


def main(out, pats):
    tree = json.loads(get(f'https://api.github.com/repos/{REPO}/git/trees/{REF}?recursive=1'))['tree']
    paths = [t['path'] for t in tree if t['type'] == 'blob' and t['path'].startswith('sub-') and any(p in t['path'] for p in pats)]
    print(len(paths), 'files', flush=True)
    for p in paths:
        dst = os.path.join(out, p)
        if os.path.exists(dst): continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        b = get(f'https://raw.githubusercontent.com/{REPO}/{REF}/{p}')
        if b.startswith(b'/annex/objects/') or b'.git/annex/objects/' in b[:200]:
            key = os.path.basename(b.decode().strip()); b = get(S3 + key)
            sha = key.split('--')[1].split('.')[0]
            if hashlib.sha256(b).hexdigest() != sha: raise SystemExit(f'checksum mismatch {p}')
        tmp = dst + '.part'; open(tmp, 'wb').write(b); os.replace(tmp, dst)
    print('done', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
