"""Resolve a recorded historical Gadi path without filesystem compatibility aliases."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

DEFAULT_MAP = Path('/g/data/ql44/we2614/mts-spi-study/documentation/path-map.json')


def resolve_path(path: str, mapping: dict) -> str:
    """Apply the longest path-prefix match; mappings contain final destinations."""
    path = path.rstrip('/')
    for old in sorted(mapping['paths'], key=len, reverse=True):
        if path == old or path.startswith(old + '/'):
            entry = mapping['paths'][old]
            if entry['status'] != 'live':
                raise ValueError(f"{entry['status']}: {old}; {entry.get('note', '')}")
            return entry['path'] + path[len(old):]
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path')
    parser.add_argument('--map', type=Path, default=DEFAULT_MAP)
    parser.add_argument('--must-exist', action='store_true')
    args = parser.parse_args()
    try:
        result = resolve_path(args.path, json.loads(args.map.read_text()))
        if args.must_exist and not Path(result).exists():
            raise ValueError(f'Path does not exist: {result}')
    except (ValueError, OSError) as exc:
        parser.exit(1, str(exc) + '\n')
    print(result)


if __name__ == '__main__':
    main()
