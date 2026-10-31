"""Retire only the eleven source copies approved on 31 October 2026.

Run on Gadi through PBS. Archives are compared with every source member before
removal. Git worktrees are retired through Git; external symlink targets are not
archived or removed. Scientific run directories are outside this allowlist.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


PROJECT = Path('/home/562/we2614/mts-spi-study-cluster')
SCRATCH = Path('/scratch/ql44/we2614')
GDATA = Path('/g/data/ql44/we2614/mts-spi-study')
DESTINATION = GDATA / 'operations/maintenance/source-retirement-261031'
GROUPS = {
    'proof-refresh': (SCRATCH / 'mts-spi-study/operations/sources', [
        'proof-refresh-aaad401', 'proof-refresh-d094b6b', 'proof-refresh-b9a92f6']),
    'hardware-pilot': (GDATA / 'operations/sources', [
        'hardware-pilot-19c73d8', 'hardware-pilot-18a2231', 'hardware-pilot-24f3436',
        'hardware-pilot-177a82d', 'hardware-pilot-5259c45', 'hardware-pilot-49ecccd',
        'hardware-pilot-04f37c2', 'hardware-pilot-48e0e7f']),
}


def run(*args, **kwargs):
    return subprocess.run(args, check=True, text=True, capture_output=True, **kwargs).stdout


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def entries(path):
    return 1 + sum(len(dirs) + len(files) for _, dirs, files in os.walk(path, followlinks=False))


def check_idle(paths):
    jobs = run('qselect', '-u', 'we2614').split()
    details = run('qstat', '-f', *jobs) if jobs else ''
    for path in paths:
        # A job can mention its source through cwd, environment or input path.
        if path.name in details:
            raise RuntimeError(f'job still references {path}')


def validate_sources(parent, names):
    if parent.resolve() != parent:
        raise ValueError(f'source parent is redirected: {parent}')
    worktrees = run('git', '-C', str(PROJECT), 'worktree', 'list', '--porcelain')
    registered = {line.removeprefix('worktree ') for line in worktrees.splitlines()
                  if line.startswith('worktree ')}
    records = []
    for name in names:
        path = parent / name
        if not path.is_dir() or path.is_symlink() or path.stat().st_uid != os.getuid():
            raise ValueError(f'not an owned, real source directory: {path}')
        git = path / '.git'
        record = dict(path=str(path), entries=entries(path), worktree=str(path) in registered)
        if git.exists() or git.is_symlink():
            if not git.is_file() or git.is_symlink() or not record['worktree']:
                raise ValueError(f'unexpected Git metadata: {path}')
            record['commit'] = run('git', '-C', str(path), 'rev-parse', 'HEAD').strip()
            # Historical commits must remain reachable from the authoritative branch.
            run('git', '-C', str(PROJECT), 'merge-base', '--is-ancestor', record['commit'], 'origin/main')
            record['git_status'] = run('git', '-C', str(path), 'status', '--porcelain=v1')
        elif record['worktree']:
            raise ValueError(f'broken registered worktree: {path}')
        records.append(record)
    check_idle([parent / name for name in names])
    return records


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def archive_group(label, parent, names):
    records = validate_sources(parent, names)
    archive = DESTINATION / f'{label}.tar.gz'
    report_path = DESTINATION / f'{label}.json'
    if archive.exists() or report_path.exists():
        raise FileExistsError('existing retirement evidence; inspect it instead of overwriting')
    partial = archive.with_suffix('.partial')
    if partial.exists():
        raise FileExistsError(partial)
    run('tar', '-I', 'gzip -1', '-cf', str(partial), '-C', str(parent), '--', *names)
    # GNU tar compares member content, type, modes, ownership, size, mtime and links.
    run('tar', '-I', 'gzip -d', '--compare', '-f', str(partial), '-C', str(parent))
    partial.replace(archive)
    result = dict(archive=str(archive), sha256=sha256(archive), bytes=archive.stat().st_size,
                  verification='GNU tar --compare, exit 0', sources=records, removed=[])
    write_json(report_path, result)
    check_idle([parent / name for name in names])
    # Recheck just before retirement. Any changed source keeps its loose copy.
    run('tar', '--compare', '-f', str(archive), '-C', str(parent))
    for record in records:
        path = Path(record['path'])
        if record['worktree']:
            run('git', '-C', str(PROJECT), 'worktree', 'remove', '--force', str(path))
        else:
            if path.is_symlink():
                raise ValueError(f'source changed to a link: {path}')
            shutil.rmtree(path)
        if path.exists() or path.is_symlink():
            raise RuntimeError(f'removal incomplete: {path}')
        result['removed'].append(str(path))
        write_json(report_path, result)
    pointer = parent / f'ARCHIVED-{label}-261031.txt'
    pointer.write_text(f'Inactive source copies archived at {archive}\n'
                       f'SHA256 {result["sha256"]}\n'
                       f'Verification and Git commits: {report_path}\n'
                       f'Restore instructions: {DESTINATION / "README.txt"}\n')
    print(json.dumps(result), flush=True)


def purge_caches():
    report = []
    for kind in ['pip', 'uv']:
        path = SCRATCH / '.cache' / kind
        if not path.exists():
            continue
        if path.is_symlink() or path.stat().st_uid != os.getuid():
            raise ValueError(f'unexpected cache: {path}')
        before = entries(path)
        env = dict(os.environ)
        if kind == 'pip':
            env['PIP_CACHE_DIR'] = str(path)
            run('python3', '-m', 'pip', 'cache', 'purge', env=env)
        else:
            env['UV_CACHE_DIR'] = str(path)
            run(str(Path.home() / '.local/bin/uv'), 'cache', 'clean', env=env)
        report.append(dict(path=str(path), entries_before=before,
                           entries_after=entries(path) if path.exists() else 0))
    write_json(DESTINATION / 'caches.json', report)
    print(json.dumps(report), flush=True)


def main(apply):
    for _, (parent, names) in GROUPS.items():
        print(json.dumps(validate_sources(parent, names)), flush=True)
    if not apply:
        return
    DESTINATION.mkdir(parents=True, exist_ok=False)
    (DESTINATION / 'README.txt').write_text(
        'User-approved retirement of inactive source copies, 31 October 2026.\n'
        'Each JSON records archive SHA256, GNU tar comparison, source paths and Git commits.\n'
        'Archives preserve every directory/file/link without following external symlinks.\n'
        'Proof snapshots: extract the archive into their original parent.\n'
        'Hardware worktrees: git worktree add --detach ORIGINAL_PATH COMMIT first, then\n'
        'extract the matching member into its original parent with --exclude=*/.git\n'
        'so the newly created worktree metadata is preserved. Inspect inode headroom first.\n'
        'Caches were disposable; scientific runs and active environments were not removed.\n')
    for label, (parent, names) in GROUPS.items():
        archive_group(label, parent, names)
    purge_caches()
    print(run('lquota'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='archive, verify and retire the explicit allowlist')
    main(parser.parse_args().apply)
