"""Promote a complete, verified instrumented profile without recomputation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyspi import _parallel
from pyspi.calculator import Calculator

from src.compute import ComputeResult, SPIInfo, _spi_info
from src.run_external_corpus import (ExternalCorpusConfig, load_inventory, load_timeseries,
                                    _atomic_json, _atomic_savez, _execution_identity,
                                    _metadata, completion_error)


def main(config_path, index, profile):
    config = ExternalCorpusConfig.from_file(config_path)
    entry = load_inventory(config)[index-1]
    summary = json.loads((profile/'summary.json').read_text())
    recorded = json.loads((profile/'identity.json').read_text())
    assert summary['status'] == 'complete' and not summary['errors']
    data, source = load_timeseries(config, entry)
    calc = Calculator(dataset=data.T, config=str(config.pyspi_config), zscore=config.normalise, verbose=False)
    assert recorded['run_digest'] == calc.run_digest
    assert recorded['name'] == entry.name and recorded['source'] == source
    owner, reason = _parallel.checkpoint_owner_matches(profile/'checkpoints', calc.run_digest)
    assert owner, reason
    names = list(calc.spis)
    saved, missing = _parallel.load_checkpoints(profile/'checkpoints', names, entry.M)
    assert not missing and set(summary['timings']) == set(names) and len(saved) == 289
    events = [json.loads(line) for line in (profile/'trace.jsonl').read_text().splitlines()]
    completed = [event for event in events if event['event'] == 'complete']
    assert [event['key'] for event in completed] == names
    assert all(event['error'] is None for event in completed)
    matrices, metadata = {}, []
    for key, info in _spi_info(calc.spis).items():
        matrix = np.array(saved[key][0], dtype=float, copy=True)
        np.fill_diagonal(matrix, 0)
        if not info['directed']:
            matrix = .5*(matrix+matrix.T)
        matrices[key] = matrix
        metadata.append(SPIInfo(name=key, **info))
    result = ComputeResult(pd.DataFrame(), matrices, metadata, summary['timings'], {})
    directory = entry.output_dir(config)
    assert not (directory/'meta.json').exists() and not (directory/'spi_mpis.npz').exists(), 'Never overwrite a production output'
    identity = _execution_identity(config)
    meta = _metadata(config, entry, result, source, identity, sum(summary['timings'].values()))
    meta['profile_provenance'] = dict(directory=str(profile), identity=recorded,
        trace_sha256=hashlib.sha256((profile/'trace.jsonl').read_bytes()).hexdigest(),
        summary_sha256=hashlib.sha256((profile/'summary.json').read_bytes()).hexdigest(),
        packaging_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        computation='Original instrumented serial run; all checkpoint matrices reused, no SPI recomputed',
        warning_note='Per-SPI warnings were emitted by the original run but not retained in its checkpoint sidecars')
    _atomic_savez(directory/'spi_mpis.npz', matrices)
    with np.load(directory/'spi_mpis.npz') as archive:
        for key in names:
            np.testing.assert_array_equal(archive[key], matrices[key])
    _atomic_json(directory/'meta.json', meta)
    assert completion_error(config, entry) is None
    print(json.dumps(dict(status='complete', index=index, directory=str(directory), n_spis=len(names))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    args = parser.parse_args()
    main(args.config, args.index, args.profile)
