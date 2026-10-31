"""Pack audited current-version MPIs without fitting or imputing any features."""
import argparse
import json
from pathlib import Path

import numpy as np

from mtsspi.corpus import ArchiveCorpus
from mtsspi.provenance import array_sha256, atomic_json, atomic_savez, file_sha256
from mtsspi.run_corpus import completion_error, execution_identity
from mtsspi.spi_spi_contract import build_unified_feature_values, build_unified_schema, schema_json, schema_sha256
from research.representation.shared.mpi_representation_baselines import summarize_mpis
from research.representation.shared.representation_attribution import rich_marginals


def raw_features(x):
    """Aligned Pearson and power-band features, plus channel-pooled spectra.

    Bins are equal-width in native FFT frequency, excluding DC. Both aligned
    and pooled spectral versions are retained; no task labels choose bands.
    """
    pearson = np.corrcoef(x.T)[np.triu_indices(x.shape[1], 1)]
    power = np.abs(np.fft.rfft(x, axis=0)[1:]) ** 2 / len(x)
    bands = np.log1p(np.stack([v.mean(0) for v in np.array_split(power, 32)]))
    pooled = np.concatenate([bands.mean(1), bands.std(1), *np.quantile(bands, [.1, .5, .9], axis=1)])
    return pearson, bands.ravel(), pooled


def main(root):
    output = root / 'bank.npz'
    if output.exists():
        raise FileExistsError(output)
    corpus = ArchiveCorpus.from_file(root / 'corpus.yaml')
    corpus.validate()
    manifest = json.loads((root / 'manifest.json').read_text())
    rows = manifest['rows']
    identity = execution_identity(corpus)
    arrays = {k: [] for k in ['x', 'm', 'g', 'z', 'validity', 'aligned', 'pearson', 'spectral', 'spectral_pooled']}
    order, versions, receipts = None, set(), []
    with np.load(corpus.archive, allow_pickle=False) as raw:
        for row, entry in zip(rows, corpus.entries(), strict=True):
            error = completion_error(corpus, entry, identity=identity)
            if error:
                raise ValueError(f'{entry.name}: {error}')
            directory = entry.output_dir(corpus)
            meta_path = directory / 'meta.json'
            mpi_path = directory / 'spi_mpis.npz'
            meta = json.loads(meta_path.read_text())
            names = [s['name'] for s in meta['pyspi']['spis']]
            if order is None:
                order = names
            if names != order or row['row_id'] != entry.name:
                raise ValueError('inconsistent catalogue or row order')
            x = raw[entry.name]
            if array_sha256(x) != row['array_sha256'] or meta['source']['member_sha256'] != row['array_sha256']:
                raise ValueError('prepared and computed waveforms differ')
            with np.load(mpi_path, allow_pickle=False) as bank:
                mpis = {k: bank[k] for k in order}
            _, graph, valid = summarize_mpis(mpis, order)
            z, _, _ = build_unified_feature_values(mpis, order)
            pearson, spectral, pooled = raw_features(x)
            mask = ~np.eye(entry.M, dtype=bool)
            values = [x.astype(np.float32), rich_marginals(mpis, order), graph, z, valid,
                      np.concatenate([mpis[k][mask] for k in order]), pearson, spectral, pooled]
            for key, value in zip(arrays, values, strict=True):
                arrays[key].append(value)
            versions.add(json.dumps(meta['pyspi']['version'], sort_keys=True))
            receipts.append(dict(row_id=entry.name, mpi_sha256=file_sha256(mpi_path),
                                 metadata_sha256=file_sha256(meta_path), valid_spis=int(valid.sum())))
            if len(receipts) % 32 == 0:
                print('Packed', len(receipts), '/', len(rows), flush=True)
    if len(versions) != 1:
        raise ValueError('mixed computation versions')
    schema = build_unified_schema(order)
    payload = {k: np.asarray(v) for k, v in arrays.items()}
    payload.update(y=np.array([r['target'] for r in rows]),
                   M=np.array([r['M'] for r in rows]), T=np.array([r['T'] for r in rows]),
                   spi_order=np.array(order), schema_sha256=np.array(schema_sha256(schema)))
    for key in ['row_id', 'group', 'role']:
        payload[key] = np.array([r[key] for r in rows])
    atomic_savez(output, payload)
    (root / 'feature-schema.json').write_text(schema_json(schema) + '\n')
    atomic_json(root / 'bank.json', dict(sha256=file_sha256(output), manifest_sha256=file_sha256(root / 'manifest.json'),
                identity=identity, pyspi_versions=sorted(versions), source_receipts=receipts,
                shapes={k: list(v.shape) for k, v in payload.items()},
                spectral_scope='standardized prepared recordings, not historical unstandardized ECoG powers',
                bank_builder_sha256=file_sha256(Path(__file__))))
    print('Verified bank', output, len(rows), 'recordings;', len(order), 'SPIs', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
