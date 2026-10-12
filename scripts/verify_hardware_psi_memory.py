"""Check the existing PSI memory repair on measured hardware before retrying p90."""
import argparse
import hashlib
import json
from pathlib import Path
import resource

import numpy as np
from pyspi import _parallel
from pyspi.calculator import Calculator

from src.hcp_spectral_memory import bounded_psi_memory
from src.run_external_corpus import ExternalCorpusConfig, load_inventory, load_timeseries


def psi(data, config):
    calc = Calculator(dataset=data.T, config=str(config.pyspi_config),
                      zscore=config.normalise, verbose=False)
    matrices = {}
    for key, spi in calc.spis.items():
        if not key.startswith('psi_multitaper_'):
            continue
        matrix, error, _, _ = _parallel.run_spi(spi, calc.dataset, key, data.shape[1])
        if error:
            raise RuntimeError(f'{key}: {error}')
        matrices[key] = matrix
    if not matrices:
        raise ValueError('No multitaper PSI variants found')
    return matrices


def main(config_path, output):
    config = ExternalCorpusConfig.from_file(config_path)
    entries = load_inventory(config)
    rows = []
    for index in (1, 5):
        entry = entries[index - 1]
        data, _ = load_timeseries(config, entry)
        # The unmodified formula is affordable on a 512-sample window retaining
        # all 28 channels. This window is validation only, never the pilot input.
        old = psi(data[:512], config)
        with bounded_psi_memory() as repair:
            new = psi(data[:512], config)
            full = psi(data, config)
        differences = {}
        for key in old:
            np.testing.assert_allclose(new[key], old[key], rtol=1e-9, atol=1e-9)
            differences[key] = float(np.nanmax(np.abs(new[key] - old[key])))
            edge = ~np.eye(entry.M, dtype=bool)
            if not np.isfinite(full[key][edge]).all():
                raise ValueError(f'Nonfinite full-record PSI: {entry.name}/{key}')
        rows.append(dict(index=index, name=entry.name, M=entry.M, T=entry.T,
                         comparison_T=512, differences=differences,
                         full_record_finite=True))
    report = dict(status='passed', repair=repair, recordings=rows,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  archive_sha256=config.archive_sha256,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.config, args.output)
