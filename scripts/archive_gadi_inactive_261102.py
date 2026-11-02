"""Pack the inactive run directories and source copies approved on 2 November 2026.

Run on Gadi through PBS (jobs/gadi/archive_inactive_261102.pbs); without --apply it only validates.
One archive per directory, so any run can be restored alone. A unit is skipped, and stays loose, if any
entry is newer than CUTOFF, a queued or running job names it, its Git worktree commit is not on
origin/main, or gdata lacks headroom for it. Archives are compared with the source twice (GNU tar
--compare) and the member count is rechecked before removal.

Destinations: gdata units go to gdata `archives/<workstream>/261102/`; Scratch units go to
`<original parent>/packed-261102/` on Scratch, where Scratch expiry still applies. Scratch's `archives`
is a link to gdata's, so the first run (2 November) wrote Scratch archives to gdata and exceeded its
quota; --repair moves those to Scratch, verifying each against its recorded SHA-256, and removes the
partial files that run left. Each destination's README.md and SHA256SUMS are rebuilt from its records.
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
GDATA_CEILING_GIB = 95                  # never fill gdata beyond this (quota 100)
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


def on_gdata(path):
    return str(Path(path).resolve()).startswith('/g/data/')


def destination(workstream, parent):
    dest = G / 'archives' / workstream / DATE if on_gdata(parent) else parent / f'packed-{DATE}'
    # The resolved destination must sit on the source's own filesystem (Scratch `archives` links to gdata).
    if on_gdata(dest) != on_gdata(parent):
        raise ValueError(f'{dest} is not on the same filesystem as {parent}')
    return dest


def gdata_used_gib():
    for line in run('lquota').splitlines():
        f = line.split()
        if len(f) > 3 and f[0] == 'ql44' and f[1] == 'gdata':
            return float(f[2]) * {'GiB': 1, 'TiB': 1024, 'MiB': 1 / 1024}[f[3]]
    raise RuntimeError('gdata usage not found in lquota')


def newest(path):
    stamp = max(os.lstat(os.path.join(d, n)).st_mtime for d, ds, fs in os.walk(path) for n in ds + fs + ['.'])
    return datetime.date.fromtimestamp(stamp).isoformat()


def plan():
    units, skipped = [], []
    for workstream, parent, names in UNITS:
        names = sorted(p.name for p in parent.iterdir() if p.is_dir() and not p.is_symlink()
                       and not p.name.startswith('packed-')) if names == WT else names
        for name in names:
            try:
                if not (parent / name).exists():
                    raise FileNotFoundError('already packed or removed')
                record = validate_sources(parent, [name])[0]
                record['newest'] = newest(parent / name)
                if record['newest'] > CUTOFF:
                    raise ValueError(f'written {record["newest"]}, after {CUTOFF}')
                units.append(dict(record, workstream=workstream, parent=str(parent), name=name,
                                  destination=str(destination(workstream, parent))))
            except Exception as error:  # noqa: BLE001 - any doubt keeps the directory loose
                skipped.append(dict(path=str(parent / name), reason=f'{type(error).__name__}: {error}'))
    return units, skipped


def restore_line(u):
    return f'tar --acls --xattrs -p -I "zstd -d" -xf {u["archive"]} -C {u["parent"]}'


def refresh(dest):
    """Rebuild README.md and SHA256SUMS from the records in one destination."""
    dest = Path(dest)
    records = [json.loads(p.read_text()) for p in sorted(dest.glob('*.json'))]
    records = [r for r in records if r.get('removed')]
    for name in ('README.md', 'SHA256SUMS'):
        (dest / name).unlink(missing_ok=True)
    if not records:
        return
    (dest / 'SHA256SUMS').write_text(''.join(f'{r["sha256"]}  {Path(r["archive"]).name}\n' for r in records))
    lines = ['Inactive directories packed on 2–3 November 2026 (user decision, to free inodes). One archive per original',
             'directory; `<name>.json` records its source path, entry count, newest file, Git commit if a worktree, and',
             'verification. Check with `sha256sum -c SHA256SUMS`, then restore into the original parent:', '']
    for r in records:
        restore = restore_line(r)
        if r['worktree']:
            restore = (f'git -C {PROJECT} worktree add --detach {r["parent"]}/{r["name"]} {r["commit"]}, then '
                       f'{restore} --exclude={r["name"]}/.git')
        lines.append(f'- `{r["name"]}` ({r["entries"]} entries, newest {r["newest"]}): `{restore}`')
    (dest / 'README.md').write_text('\n'.join(lines) + '\n')


def pointer(u):
    with open(Path(u['parent']) / 'ARCHIVED.md', 'a') as stream:
        stream.write(f'- `{u["name"]}`: verified archive `{u["archive"]}` ({DATE}). Restore: `{restore_line(u)}`. '
                     f'Check `SHA256SUMS` there first.\n')


def pack(u):
    parent, name, dest = Path(u['parent']), u['name'], Path(u['destination'])
    if on_gdata(dest) and gdata_used_gib() + int(run('du', '-sk', str(parent / name)).split()[0]) / 2**20 > GDATA_CEILING_GIB:
        raise RuntimeError('not enough gdata headroom')
    dest.mkdir(parents=True, exist_ok=True)
    archive, report = dest / f'{name}.tar.zst', dest / f'{name}.json'
    if archive.exists() or report.exists():
        raise FileExistsError(archive)
    partial = dest / f'{name}.tar.zst.partial'
    try:
        run('tar', '--acls', '--xattrs', '-I', 'zstd -T8 -3', '-cf', str(partial), '-C', str(parent), '--', name)
        run('tar', '-I', 'zstd -d', '--compare', '-f', str(partial), '-C', str(parent))
        partial.replace(archive)
    finally:
        partial.unlink(missing_ok=True)
    u.update(archive=str(archive), sha256=sha256(archive), bytes=archive.stat().st_size, removed=False,
             verification='GNU tar --compare twice, exit 0; member count equals source entries')
    try:
        write_json(report, u)
        check_idle([parent / name])
        run('tar', '-I', 'zstd -d', '--compare', '-f', str(archive), '-C', str(parent))
        members = len(run('tar', '-I', 'zstd -d', '-tf', str(archive)).splitlines())
        if members != entries(parent / name):
            raise RuntimeError(f'{name}: {members} members, {entries(parent / name)} entries on disk')
    except Exception:
        archive.unlink(missing_ok=True)   # the source is untouched, so an unverified archive is only clutter
        report.unlink(missing_ok=True)
        raise
    if u['worktree']:
        run('git', '-C', str(PROJECT), 'worktree', 'remove', '--force', str(parent / name))
    else:
        shutil.rmtree(parent / name)
    if (parent / name).exists():
        raise RuntimeError(f'removal incomplete: {parent / name}')
    u['removed'] = True
    write_json(report, u)
    pointer(u)


def repair():
    """Clean the first run's leftovers and move Scratch-sourced archives from gdata to Scratch."""
    parents = {name: parent for _, parent, names in UNITS if names != WT for name in names}
    for _, parent, names in UNITS:
        if names == WT:
            parents.update({p.name: parent for p in parent.iterdir() if p.is_dir()})
    touched = set()
    for dest in sorted((G / 'archives').glob(f'*/{DATE}')):
        for p in list(dest.glob('*.partial')) + list(dest.glob('*.tmp')):
            print('remove leftover', p, flush=True); p.unlink(); touched.add(dest)
        for archive in sorted(dest.glob('*.tar.zst')):
            name = archive.name.removesuffix('.tar.zst')
            if (dest / f'{name}.json').exists():
                continue
            source = parents.get(name)
            if source is None or not (source / name).is_dir():
                print('KEEP orphan archive, source not found:', archive, flush=True)
                continue
            print('remove unverified archive (source still loose)', archive, flush=True); archive.unlink(); touched.add(dest)
        for report in sorted(dest.glob('*.json')):
            r = json.loads(report.read_text())
            if on_gdata(r['parent']) or not r.get('removed'):
                continue
            old = Path(r['archive'])
            new_dest = destination(r['workstream'], Path(r['parent']))
            new_dest.mkdir(parents=True, exist_ok=True)
            new, partial = new_dest / old.name, new_dest / (old.name + '.partial')
            shutil.copyfile(old, partial)
            if sha256(partial) != r['sha256']:
                partial.unlink()
                raise RuntimeError(f'copy of {old} does not match its recorded SHA-256')
            partial.replace(new)
            r.update(archive=str(new), destination=str(new_dest))
            write_json(new_dest / report.name, r)
            note = Path(r['parent']) / 'ARCHIVED.md'
            note.write_text(note.read_text().replace(str(old), str(new)))
            old.unlink(); report.unlink()
            touched.update({dest, new_dest})
            print('moved', old, '->', new, flush=True)
    for dest in touched:
        refresh(dest)
    print(f'gdata now {gdata_used_gib():.1f} GiB' if shutil.which('lquota') else 'run lquota on a login node', flush=True)


def main(apply):
    units, skipped = plan()
    print(json.dumps(dict(units=len(units), entries=sum(u['entries'] for u in units), skipped=skipped), indent=1), flush=True)
    for u in units:
        print(u['entries'], u['newest'], u['worktree'], u['parent'] + '/' + u['name'], '->', u['destination'], flush=True)
    if not apply:
        return
    touched = set()
    for u in units:
        try:
            pack(u)
            print('packed', u['name'], u['bytes'], flush=True)
        except Exception as error:  # noqa: BLE001 - one failure must not stop the rest; its source stays
            print('FAILED', u['parent'] + '/' + u['name'], f'{type(error).__name__}: {error}', flush=True)
        touched.add(u['destination'])
    for dest in touched:
        if Path(dest).exists():
            refresh(dest)
    print(run('lquota'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--apply', action='store_true', help='pack, verify and remove the approved directories')
    group.add_argument('--repair', action='store_true', help='clean the first run and move Scratch archives off gdata')
    a = parser.parse_args()
    repair() if a.repair else main(a.apply)
