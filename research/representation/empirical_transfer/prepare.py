"""Prepare the two bounded empirical corpora on Gadi; never alter old banks.

Downloads retain published checksums and source bytes in one ZIP. NeuroTycho
restores only six named input files from its verified historical archive.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
import time
import urllib.request
import zipfile

import numpy as np
import yaml

from mtsspi.provenance import array_sha256, atomic_json, atomic_savez, file_sha256

FIGSHARE = 'https://api.figshare.com/v2/articles/12971162/versions/2'
LOOKUP = ('https://raw.githubusercontent.com/DynamicsAndNeuralSystems/pyspi_paper_code/'
          '7bb124fd500cd887b8a5afe3b528105bf577b330/data/Yeo_parc_info.csv')
NEUROTYCHO = {
    'neurotycho_source_f64_scout_260910': '546eac25b2f5af718c14303f5d81556a2041e3323f9b9ad251d26cdb00100767',
    'neurotycho_source_f64_extra_260910': '3d815d1048319104082213edb9f4ed84428bb421a625446f99a598f56660a63f',
    'neurotycho_target_pf_260910': 'd6b15943e7cbf2cb2f88d0ecfacae77354ccf93aca63521fc0b52f09b02769ea',
}


def fetch(url):
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'mts-spi-study'})
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def write_corpus(root, name, rows, arrays, provenance):
    """Write a homogeneous, hashed archive and an explicit current-run config."""
    if not rows or len(rows) != len(arrays):
        raise ValueError('empty or misaligned corpus')
    names = [r['row_id'] for r in rows]
    if len(set(names)) != len(names) or set(names) != set(arrays):
        raise ValueError('duplicate/misaligned row names')
    shapes = {np.shape(arrays[n]) for n in names}
    if len(shapes) != 1:
        raise ValueError('farm inputs must have one M,T size')
    root.mkdir(parents=True, exist_ok=True)
    archive = root / 'views.npz'
    if archive.exists():
        raise FileExistsError(archive)
    for index, row in enumerate(rows, 1):
        x = arrays[row['row_id']]
        if x.dtype != np.float64 or not np.isfinite(x).all():
            raise ValueError('prepared inputs must be finite float64')
        row.update(corpus_index=index, T=len(x), M=x.shape[1], array_sha256=array_sha256(x))
    payload = dict(arrays)
    payload.update(__dataset_names__=np.array(names),
                   __labels_json__=np.array([json.dumps([r['group'], f'state-{r["target"]}']) for r in rows]),
                   __shapes__=np.array([arrays[n].shape for n in names]),
                   __axis_order__=np.array(['observation', 'process']))
    atomic_savez(archive, payload)
    digest = file_sha256(archive)
    config = dict(name=name, source=dict(format='named-npz-v1', archive=str(archive.resolve()),
                  sha256=digest, axis_order=['observation', 'process']),
                  base_output_dir=str((root / 'features').resolve()),
                  pyspi_config='benchmarked_p90', normalise=False, random_seed=261031)
    (root / 'corpus.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    atomic_json(root / 'manifest.json', dict(name=name, rows=rows, archive_sha256=digest,
                preparation_sha256=file_sha256(Path(__file__)), provenance=provenance))
    print(json.dumps(dict(corpus=str(root / 'corpus.yaml'), rows=len(rows), shape=list(next(iter(shapes))),
                          archive_sha256=digest)), flush=True)


def restore_neurotycho(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    required = {(name, file) for name in NEUROTYCHO for file in ('views.npz', 'manifest.json')}
    missing = {(name, file) for name, file in required if not (destination / name / file).exists()}
    if not missing:
        return
    process = subprocess.Popen(['zstd', '-dc', str(archive)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as stream:
            for member in stream:
                parts = Path(member.name).parts
                key = tuple(parts[-2:])
                if key not in missing:
                    continue
                if not member.isfile():
                    raise ValueError(f'expected regular archived input: {member.name}')
                target = destination / key[0] / key[1]
                target.parent.mkdir(exist_ok=True)
                partial = target.with_suffix(target.suffix + '.partial')
                with stream.extractfile(member) as source, partial.open('wb') as output:
                    while block := source.read(1024 * 1024):
                        output.write(block)
                partial.replace(target)
                missing.remove(key)
                print('Restored', member.name, flush=True)
                if not missing:
                    break
        if missing:
            raise ValueError(f'archive lacks required inputs: {sorted(missing)}')
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        process.wait()


def neurotycho(args):
    originals = args.root / 'historical-inputs'
    restore_neurotycho(args.archive, originals)
    rows, arrays, receipts = [], {}, []
    for name, expected in NEUROTYCHO.items():
        path = originals / name / 'views.npz'
        if file_sha256(path) != expected:
            raise ValueError(f'changed historical archive: {path}')
        manifest_path = path.with_name('manifest.json')
        manifest = json.loads(manifest_path.read_text())
        if manifest['archive_sha256'] != expected:
            raise ValueError('historical manifest/archive mismatch')
        role = 'source' if '_source_' in name else 'target'
        with np.load(path, allow_pickle=False) as bank:
            for original in manifest['rows']:
                if (original['M'], original['T']) != (16, 2000):
                    continue
                x = bank[original['row_id']]
                if array_sha256(x) != original['array_sha256']:
                    raise ValueError('historical waveform identity changed')
                if original['row_id'] in arrays:
                    raise ValueError('overlapping source partitions')
                row = dict(original, group=original['animal'], role=role, historical_corpus=name)
                rows.append(row)
                arrays[row['row_id']] = x
        receipts.append(dict(corpus=name, archive_sha256=expected, manifest_sha256=file_sha256(manifest_path)))
    if len(rows) != 480 or sum(r['role'] == 'source' for r in rows) != 352:
        raise ValueError('expected 352 source and 128 target primary recordings')
    write_corpus(args.root / 'neurotycho-primary', 'neurotycho-primary-r10-261031', rows, arrays,
                 dict(restored_from=str(args.archive), historical_inputs=receipts,
                      waveforms='unchanged historical float64 primary views', sample_rate_hz=250,
                      target_status='previously inspected PF; version replication and interpretation'))


def aligned_networks(native_text, lookup_text, column='Yeo_17_Networks_Bilateral'):
    native = [r['name'].strip() for r in csv.DictReader(io.StringIO(native_text)) if r.get('name')]
    lookup = list(csv.DictReader(io.StringIO(lookup_text)))
    named = {r['Full_Region_Name']: r for r in lookup}
    if len(native) != 114 or len(named) != 114 or set(native) != set(named):
        raise ValueError('parcel identity mismatch')
    if any(int(named[n]['Index']) != i + 1 for i, n in enumerate(native)):
        raise ValueError('native parcel names/index disagree')
    groups = sorted({named[n][column] for n in native})
    return groups, [np.array([i for i, n in enumerate(native) if named[n][column] == g]) for g in groups]


def prepare_fmri_array(x, regions, length=947):
    if x.ndim != 2 or x.shape[1] != 114:
        raise ValueError('expected native T,114 array')
    # Do not join finite fragments across motion-censored gaps.
    if not np.isfinite(x).all():
        return None, 'all_missing' if not np.isfinite(x).any() else 'partial_missing'
    if len(x) < length:
        return None, 'too_short'
    start = (len(x) - length) // 2
    raw = x[start:start + length]
    coarse = np.column_stack([raw[:, ix].mean(axis=1) for ix in regions])
    std = coarse.std(axis=0)
    if not np.isfinite(std).all() or np.any(std <= 1e-12):
        return None, 'constant_network'
    return np.ascontiguousarray((coarse - coarse.mean(axis=0)) / std, dtype=np.float64), None


def assign_fmri_roles(rows, seed=261031):
    people = sorted({r['group'] for r in rows})
    eligible = [p for p in people if all({r['target'] for r in rows
                if r['group'] == p and r['run_set'] == visit} == {0, 1} for visit in ('early', 'late'))]
    if len(eligible) < 6:
        raise ValueError('fewer than six participants with both states in both run sets')
    count = 18 if len(eligible) == 29 else round(2 * len(eligible) / 3)
    development = set(np.random.default_rng(seed).permutation(eligible)[:count])
    retained = []
    for original in rows:
        if original['group'] not in eligible:
            continue
        row = dict(original)
        if row['group'] in development:
            row['role'] = 'source' if row['run_set'] == 'early' else 'development_shift'
        else:
            row['role'] = 'confirmation_subject_only' if row['run_set'] == 'early' else 'confirmation_primary'
        retained.append(row)
    return retained, dict(seed=seed, development=sorted(development),
                         confirmation=sorted(set(eligible) - development),
                         excluded_incomplete_people=sorted(set(people) - set(eligible)))


def fmri(args):
    args.root.mkdir(parents=True, exist_ok=True)
    raw = args.root / 'fmri-source-v2.zip'
    inventory_path = args.root / 'fmri-acquisition.json'
    if raw.exists():
        inventory = json.loads(inventory_path.read_text())
    else:
        inventory = json.loads(fetch(FIGSHARE))
        atomic_json(inventory_path, inventory)
        files = inventory['files']
        if len(files) != 233:
            raise ValueError('unexpected Figshare version-2 inventory')
        def download(item):
            content = fetch(item['download_url'])
            if len(content) != item['size'] or hashlib.md5(content).hexdigest() != item['computed_md5']:
                raise ValueError(f'published checksum mismatch: {item["name"]}')
            return item['name'], content
        temporary = raw.with_suffix('.partial')
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as output, ThreadPoolExecutor(4) as pool:
            for index, (name, content) in enumerate(pool.map(download, files), 1):
                output.writestr(name, content)
                if index % 32 == 0:
                    print('Downloaded', index, '/', len(files), flush=True)
        temporary.replace(raw)
    lookup_path = args.root / 'fmri-network-lookup.csv'
    if not lookup_path.exists():
        lookup_path.write_bytes(fetch(LOOKUP))
    lookup = lookup_path.read_text()
    rows, arrays, audit = [], {}, []
    with zipfile.ZipFile(raw) as archive:
        groups, regions = aligned_networks(archive.read('parc_info.txt').decode(), lookup)
        for item in sorted(inventory['files'], key=lambda f: f['name']):
            match = re.fullmatch(r'(sub\d+)\.(rest|movie)\.([1-4])\.txt', item['name'])
            if match is None:
                continue
            person, state, run_text = match.groups()
            content = archive.read(item['name'])
            if hashlib.md5(content).hexdigest() != item['computed_md5']:
                raise ValueError('cached source bytes fail published checksum')
            x = np.loadtxt(io.BytesIO(content), delimiter=',')
            prepared, reason = prepare_fmri_array(x, regions)
            audit.append(dict(file=item['name'], shape=list(x.shape), excluded=reason,
                              finite_rows=int(np.isfinite(x).all(axis=1).sum())))
            if reason:
                continue
            run = int(run_text)
            row_id = f'{person}-{state}-{run}-m17-t947'
            rows.append(dict(row_id=row_id, group=person, target=int(state == 'movie'), run=run,
                             run_set='early' if run <= 2 else 'late', native_file=item['name'],
                             native_md5=item['computed_md5'], crop_start=(len(x) - 947) // 2))
            arrays[row_id] = prepared
    retained, split = assign_fmri_roles(rows)
    arrays = {r['row_id']: arrays[r['row_id']] for r in retained}
    atomic_json(args.root / 'fmri-input-audit.json', dict(files=audit, split=split, usable_scans_before_person_qc=len(rows),
                prepared_scans=len(retained), network_order=groups, native_indices=[v.tolist() for v in regions]))
    write_corpus(args.root / 'fmri-primary', 'rest-movie-primary-r10-261031', retained, arrays,
                 dict(dataset=FIGSHARE, raw_zip_sha256=file_sha256(raw), lookup_url=LOOKUP,
                      lookup_sha256=file_sha256(lookup_path), network_order=groups, split=split,
                      sample_rate_hz=1 / .813, crop='central contiguous 947 native samples',
                      standardization='per-network per-record population z-score after parcel averaging',
                      run_interpretation='indices 1/2 versus 3/4; acquisition dates not present in public files',
                      limitations=['motion covariates absent from this release',
                                   'no cross-site acquisition claim', 'different scans are not independent people']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['neurotycho', 'fmri'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    if args.dataset == 'neurotycho' and args.archive is None:
        parser.error('--archive is required for NeuroTycho')
    {'neurotycho': neurotycho, 'fmri': fmri}[args.dataset](args)
