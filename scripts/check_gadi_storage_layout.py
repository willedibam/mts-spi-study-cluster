"""Read-only checks for Gadi project roots, indexed locations and shared environment."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

PROJECT_DIRS = {'proof', 'order-parameter-inference', 'zenodo', 'representation',
                'archives', 'operations', 'environments', 'dev', 'documentation',
                'legacy-non-p90'}
OLD_ROOTS = {'mts-spi-data', 'mts-spi-data-v2', 'mts-spi-archives', 'mts-spi-logs',
             'venvs', 'environments', 'archives', 'order-parameter-inference',
             'order-parameter-models', 'spi-spi-direction-v2', 'spi-spi-unified-v3'}


def check(roots):
    errors = []
    for root in roots:
        if not root.is_dir():
            errors.append(f'Missing store: {root}')
            continue
        for item in root.iterdir():
            if item.name in OLD_ROOTS or item.name.startswith(('cml2d-', 'finite-regime-source-', 'spi-spi-cross-mt', 'mts-spi-cross-mt')):
                errors.append(f'Obsolete root entry: {item}')
        project = root / 'mts-spi-study'
        if not project.is_dir():
            errors.append(f'Missing project: {project}')
            continue
        for item in project.iterdir():
            if item.is_dir() and item.name not in PROJECT_DIRS:
                errors.append(f'Unrecognised project category: {item}')
            if item.is_symlink() and not item.exists():
                errors.append(f'Broken project link: {item}')
        index = project / 'documentation/storage-index.csv'
        if not index.exists():
            errors.append(f'Missing index: {index}')
        else:
            with index.open() as handle:
                for row in csv.DictReader(handle):
                    if row['status'] == 'live' and not Path(row['physical_path']).exists():
                        errors.append(f'Missing indexed artifact: {row["physical_path"]}')
    envs = [r / 'mts-spi-study/environments/mts-spi-v3-631de27' for r in roots]
    physical = {p.resolve() for p in envs if p.exists()}
    if len(physical) != 1:
        errors.append(f'Expected one shared MTS environment, found {len(physical)}')
    elif 'uv = ' not in (next(iter(physical)) / 'pyvenv.cfg').read_text():
        errors.append('Shared environment is not recorded as uv-created')
    return sorted(set(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', nargs='+', type=Path, default=[Path('/scratch/ql44/we2614'), Path('/g/data/ql44/we2614')])
    args = parser.parse_args()
    errors = check(args.roots)
    if errors:
        print('\n'.join(errors))
        raise SystemExit(1)
    print('Storage layout OK: project categories, indexed artifacts, shared uv environment.')


if __name__ == '__main__':
    main()
