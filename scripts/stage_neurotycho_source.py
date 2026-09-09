"""Stage predefined KTMD source channels; never download propofol waveforms."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np
from scipy.io import loadmat

from src.neurotycho_zip import directory, read_member


def montage(path, count=16):
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith('.mat'))
        data = loadmat(io.BytesIO(archive.read(name)), simplify_cells=True)
    xy = np.column_stack([data['X'], data['Y']])
    if xy.shape != (128, 2) or not np.isfinite(xy).all():
        raise ValueError('unexpected electrode map')
    distances = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    np.fill_diagonal(distances, np.inf)
    maximum = 1.5 * np.median(distances.min(axis=1))
    candidates = sorted((distances[i, j], i, j) for i in range(128)
                        for j in range(i + 1, 128) if distances[i, j] <= maximum)
    used, pairs = set(), []
    for distance, i, j in candidates:
        if i not in used and j not in used:
            pairs.append((i, j)); used.update((i, j))
    if len(pairs) < count:
        raise ValueError('too few short nonoverlapping pairs')
    midpoints = np.array([xy[[i, j]].mean(axis=0) for i, j in pairs])
    # Geometry only: start at leftmost midpoint, then maximize spatial coverage.
    chosen = [int(np.lexsort((midpoints[:, 1], midpoints[:, 0]))[0])]
    while len(chosen) < count:
        separation = np.linalg.norm(midpoints[:, None] - midpoints[chosen], axis=-1).min(axis=1)
        separation[chosen] = -np.inf
        chosen.append(int(separation.argmax()))
    selected = [pairs[i] for i in chosen]
    return dict(pairs=[[i + 1, j + 1] for i, j in selected],
                coordinates=xy.tolist(), maximum_pair_distance=float(maximum),
                distance_units='provider image pixels; not cortical geodesic distance',
                map_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main(args):
    audit = json.loads((args.audit / 'audit.json').read_text())
    if args.all_source:
        records = [r for r in audit['availability'] if r['agent'] == 'KTMD']
    else:
        records = [r for r in audit['annotation_samples'] if r['agent'] == 'KTMD']
    args.output.mkdir(parents=True, exist_ok=True)
    maps = {animal: montage(args.audit / 'montage' / f'{animal}.zip')
            for animal in sorted({r['animal'] for r in records})}
    plan = dict(phase='source waveform QC; no PF downloads',
                source_names=[r['name'] for r in records], montages=maps)
    plan_path = args.output / ('all-source-plan.json' if args.all_source else 'scout-plan.json')
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('existing staging plan differs')
    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    for record in records:
        root = args.output / record['name']
        root.mkdir(exist_ok=True)
        members = directory(record['url'])
        meta = [m for m in members if m.filename.endswith('Condition.mat') or
                Path(m.filename).name.startswith('Info-')]
        sessions = set()
        for member in meta:
            session = Path(member.filename).parent.name
            path = root / session / Path(member.filename).name
            raw = read_member(record['url'], member, path)
            if member.filename.endswith('Condition.mat'):
                labels = np.atleast_1d(loadmat(io.BytesIO(raw), simplify_cells=True)['ConditionLabel'])
                if any(str(label).startswith(('AwakeEyesClosed-', 'Anesthetized-')) for label in labels):
                    sessions.add(session)
        channels = set(np.asarray(maps[record['animal']]['pairs']).ravel().tolist())
        selected = []
        for member in members:
            filename = Path(member.filename)
            match = re.fullmatch(r'ECoG_ch(\d+)\.mat', filename.name)
            if filename.parent.name in sessions and (filename.name == 'ECoGTime.mat' or
                    (match and int(match[1]) in channels)):
                selected.append(member)
        if len(selected) != len(sessions) * (len(channels) + 1) or not sessions:
            raise ValueError('incomplete selected channel/time inventory')
        def fetch(member):
            path = root / Path(member.filename).parent.name / Path(member.filename).name
            read_member(record['url'], member, path)
            print(f'{record["animal"]} {path.relative_to(root)} verified', flush=True)
        print(f'Staging {record["name"]}: {len(selected)} members', flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(fetch, selected))
        (root / 'complete.json').write_text(json.dumps(dict(name=record['name'], animal=record['animal'],
            sessions=sorted(sessions), channels=sorted(channels), members=len(selected)), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=Path('results/neurotycho_audit_260910'))
    parser.add_argument('--output', type=Path, default=Path('data/neurotycho_source_260910'))
    parser.add_argument('--all-source', action='store_true')
    main(parser.parse_args())
