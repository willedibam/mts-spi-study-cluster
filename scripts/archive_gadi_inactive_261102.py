"""Pack the inactive run directories and source copies approved on 2 November 2026.

Run on Gadi through PBS (jobs/gadi/archive_inactive_261102.pbs); without --apply it only validates.
One archive per directory, so any run can be restored alone. A unit is skipped, and stays loose, if any
entry is newer than CUTOFF, a queued or running job names it, or its Git worktree commit is not on
origin/main. Archives are compared with the source twice (GNU tar --compare) and the member count is
rechecked before removal. Archives stay on the source's own filesystem: gdata had about 15 GiB free,
too little for the Scratch units, so those archives remain subject to Scratch expiry like the originals.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil

from scripts.archive_gadi_source_candidates import PROJECT, check_idle, entries, run, sha256, validate_sources, write_json

S = Path('/scratch/ql44/we2614/mts-spi-study')
G = Path('/g/data/ql44/we2614/mts-spi-study')
DATE, CUTOFF = '261102', '2026-10-19'   # nothing written in the last 14 days
WT = 'worktrees'                          # every child of a container of worktrees is its own unit
UNITS = [  # (workstream, parent, names)
    ('order-parameter-inference', S / 'order-parameter-inference', [
        'chimera-breathing-261006', 'cml2d', 'cross-frequency-locking-261006', 'cross-frequency-locking-confirm-261006',
        'cross-frequency-locking-nuisance-261006', 'cross-frequency-locking-snr-261006', 'dependence-transition-261005',
        'desai-zwanzig', 'lean2-regime-candidates-261006', 'miller-huse', 'neurotycho-induction-261006', 'quadratic-cml',
        't-copula-normal-margins-261006', 't-copula-normal-margins-261006-replication', 'tail-alignment-261005', 'vicsek']),
    ('proof', S / 'proof', [
        'dial-network-261007', 'noise-experiments-261016', 'proof-strength-nuisance-261006',
        'proof-strength-nuisance-m24-261006', 'proof-uncoupled-261007', 'refresh-261014', 'strength-factorial-261015',
        'strength-grid-261015']),
    ('operations', S / 'operations/sources', [  # pyspi-* snapshots stay: HPC.md and audits name them
        'band-marginals-a816549', 'band-swap-24a173f', 'cross-frequency-confirm-69180ed', 'cross-frequency-nuisance-02faa24',
        'cross-frequency-snr-ef8cc5e', 'factor-dominant-7faf1b2', 'factor-parity-aba94ff', 'factor-replication-e11a106',
        'factor-t4000-c5b4096', 'lag-surrogate-669a718', 'lean2-6db6270', 'native-coupling-d03108a', 'native-gain-2b787db',
        'native-gain-3a2f21c', 'neurotycho-induction-7dc6dd4', 'noise-analysis-a3bfcc3', 'noise-experiments-0de0330',
        'pearson-size-a770dba', 'pearson-strength-3c1624d', 'proof-strength-f5ee54c', 'proof-strength-nuisance-5749870',
        'strength-factorial-analysis-e6cd116', 'strength-factorial-da87ef7', 'strength-grid-1cf951a',
        'strength-grid-analysis-5282a8c', 'strength-grid-analysis-d3d50c3']),
    ('operations', S / 'operations/sources/registered-worktrees', WT),
    ('operations', G / 'operations/sources', [
        'chimera-scout-5b91985', 'copula-normal-118582e', 'copula-replication-d30cece', 'cross-frequency-c144491',
        'dependence-analysis-b010fd2', 'dependence-transition-df69e7e', 'hardware-full-p90-e27dfb4', 'hardware-si-ids-9c824dd',
        'hardware-transfer-8cf7640', 'hardware-transfer-fetch-2879845', 'hardware-transfer-fetch-f567bc3']),
    ('operations', G / 'operations/sources/code-snapshots', WT),
    ('order-parameter-inference', G / 'order-parameter-inference', ['cml2d', 'finite-regime']),
    ('proof', G / 'proof', ['noise-experiments-261016']),   # proof/data stays: shared input banks
    ('representation', G / 'representation/empirical', ['hardware-pilot-261012']),
]


def newest(path):
    stamp = max(os.lstat(os.path.join(d, n)).st_mtime for d, ds, fs in os.walk(path) for n in ds + fs + ['.'])
    return datetime.date.fromtimestamp(stamp).isoformat()


def plan():
    units, skipped = [], []
    for workstream, parent, names in UNITS:
        names = sorted(p.name for p in parent.iterdir() if p.is_dir() and not p.is_symlink()) if names == WT else names
        for name in names:
            try:
                record = validate_sources(parent, [name])[0]
                record['newest'] = newest(parent / name)
                if record['newest'] > CUTOFF:
                    raise ValueError(f'written {record["newest"]}, after {CUTOFF}')
                store = G if str(parent).startswith(str(G)) else S
                units.append(dict(record, workstream=workstream, parent=str(parent), name=name,
                                  destination=str(store / 'archives' / workstream / DATE)))
            except Exception as error:  # noqa: BLE001 - any doubt keeps the directory loose
                skipped.append(dict(path=str(parent / name), reason=f'{type(error).__name__}: {error}'))
    return units, skipped


def restore_line(u):
    return f'tar --acls --xattrs -p -I "zstd -d" -xf {u["archive"]} -C {u["parent"]}'


def pack(u):
    parent, name, dest = Path(u['parent']), u['name'], Path(u['destination'])
    dest.mkdir(parents=True, exist_ok=True)
    archive, report = dest / f'{name}.tar.zst', dest / f'{name}.json'
    if archive.exists() or report.exists():
        raise FileExistsError(archive)
    partial = dest / f'{name}.tar.zst.partial'
    run('tar', '--acls', '--xattrs', '-I', 'zstd -T8 -3', '-cf', str(partial), '-C', str(parent), '--', name)
    run('tar', '-I', 'zstd -d', '--compare', '-f', str(partial), '-C', str(parent))
    partial.replace(archive)
    u.update(archive=str(archive), sha256=sha256(archive), bytes=archive.stat().st_size, removed=False,
             verification='GNU tar --compare twice, exit 0; member count equals source entries')
    write_json(report, u)
    check_idle([parent / name])
    run('tar', '-I', 'zstd -d', '--compare', '-f', str(archive), '-C', str(parent))
    members = len(run('tar', '-I', 'zstd -d', '-tf', str(archive)).splitlines())
    if members != entries(parent / name):
        raise RuntimeError(f'{name}: {members} members, {entries(parent / name)} entries on disk')
    if u['worktree']:
        run('git', '-C', str(PROJECT), 'worktree', 'remove', '--force', str(parent / name))
    else:
        shutil.rmtree(parent / name)
    if (parent / name).exists():
        raise RuntimeError(f'removal incomplete: {parent / name}')
    u['removed'] = True
    write_json(report, u)
    with open(dest / 'SHA256SUMS', 'a') as sums:
        sums.write(f'{u["sha256"]}  {archive.name}\n')
    with open(parent / 'ARCHIVED.md', 'a') as pointer:
        pointer.write(f'- `{name}`: verified archive `{archive}` ({DATE}). Restore: `{restore_line(u)}`. '
                      f'Check `SHA256SUMS` there first.\n')


def readme(done):
    for dest in sorted({u['destination'] for u in done}):
        here = [u for u in done if u['destination'] == dest]
        lines = [f'Inactive directories packed on 2 November 2026 (user decision, to free inodes). One archive per original',
                 f'directory; `<name>.json` records its source path, entry count, newest file, Git commit if a worktree, and',
                 f'verification. Check with `sha256sum -c SHA256SUMS`, then restore into the original parent:', '']
        for u in here:
            restore = restore_line(u)
            if u['worktree']:
                restore = (f'git -C {PROJECT} worktree add --detach {u["parent"]}/{u["name"]} {u["commit"]}, then '
                           f'{restore} --exclude={u["name"]}/.git')
            lines.append(f'- `{u["name"]}` ({u["entries"]} entries, newest {u["newest"]}): `{restore}`')
        Path(dest, 'README.md').write_text('\n'.join(lines) + '\n')


def main(apply):
    units, skipped = plan()
    print(json.dumps(dict(units=len(units), entries=sum(u['entries'] for u in units), skipped=skipped), indent=1), flush=True)
    for u in units:
        print(u['entries'], u['newest'], u['worktree'], u['parent'] + '/' + u['name'], '->', u['destination'], flush=True)
    if not apply:
        return
    done = []
    for u in units:
        try:
            pack(u)
            done.append(u)
            print('packed', u['name'], u['bytes'], flush=True)
        except Exception as error:  # noqa: BLE001 - one failure must not stop the rest; its source stays
            print('FAILED', u['parent'] + '/' + u['name'], f'{type(error).__name__}: {error}', flush=True)
    readme(done)
    print(run('lquota'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='pack, verify and remove the approved directories')
    main(parser.parse_args().apply)
