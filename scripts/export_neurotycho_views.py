"""Export verified source windows for p90 without synthetic-generator semantics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from src.run_external_corpus import _array_sha256


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    rows, arrays, sources = [], {}, []
    excluded = {} if args.exclude_manifest is None else {
        r['row_id']: r['array_sha256'] for r in json.loads(args.exclude_manifest.read_text())['rows']}
    for meta_path in sorted(args.input.glob('201*.json')):
        meta = json.loads(meta_path.read_text())
        if not meta['usable']:
            continue
        path = meta_path.with_suffix('.npz')
        bank = np.load(path)
        sources.append(dict(metadata=str(meta_path), metadata_sha256=sha(meta_path),
                            bank=str(path), bank_sha256=sha(path)))
        for record in meta['records']:
            if not record['quality']['accepted']:
                continue
            original = bank['x'][record['array_row']]
            for m, t in [(16, 2000), (8, 1000)]:
                x = original[:m, -t:].T.astype(np.float64)
                x = np.ascontiguousarray((x - x.mean(axis=0)) / x.std(axis=0))
                if not np.isfinite(x).all():
                    raise ValueError('invalid standardized observation')
                name = f'{record["archive"]}--{record["session"]}--y{record["target"]}--w{record["window"]:02d}--m{m}-t{t}'
                if name in excluded:
                    if _array_sha256(x) != excluded[name]:
                        raise ValueError('previously exported observation changed')
                    continue
                if name in arrays:
                    raise ValueError('duplicate record')
                arrays[name] = x
                rows.append(dict(row_id=name, corpus_index=len(rows)+1, animal=record['animal'],
                    archive=record['archive'], session=record['session'], target=record['target'],
                    window=record['window'], start=record['start'], M=m, T=t,
                    array_sha256=_array_sha256(x)))
    if not rows:
        raise ValueError('no source windows')
    arrays.update(__dataset_names__=np.array([r['row_id'] for r in rows]),
                  __shapes__=np.array([[r['M'], r['T']] for r in rows]),
                  __labels_json__=np.array(json.dumps([[r['animal'], f'state-{r["target"]}'] for r in rows])))
    args.output.mkdir(parents=True)
    archive = args.output / 'views.npz'
    np.savez_compressed(archive, **arrays)
    manifest = dict(name=args.name, phase='KTMD source only', rows=rows, sources=sources,
        archive_sha256=sha(archive), catalogue_sha256=sha(Path('configs/pyspi/benchmarked_p90.yaml')),
        exporter_sha256=sha(Path(__file__)),
        excluded_manifest=None if args.exclude_manifest is None else str(args.exclude_manifest),
        preprocessing_access='8/4second analysis windows inherit28seconds of filtered raw context; not strict recording-duration transfer')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    config = dict(name=args.name, source=dict(format='named-npz-v1', archive=str(args.remote / 'views.npz'),
        sha256=manifest['archive_sha256'], axis_order=['observation','process']),
        base_output_dir=str(args.remote / 'pyspi'), pyspi_config='configs/pyspi/benchmarked_p90.yaml',
        normalise=False, random_seed=260910407)
    config_path = Path('configs/external') / (args.name + '.yaml')
    if config_path.exists():
        raise FileExistsError(config_path)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps(dict(rows=len(rows), archive_sha256=manifest['archive_sha256'], config=str(config_path))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--remote', type=Path, required=True)
    parser.add_argument('--exclude-manifest', type=Path)
    main(parser.parse_args())
