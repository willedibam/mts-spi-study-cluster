"""Validate the opt-in PSI memory repair against a completed HCP profile."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import resource

import numpy as np
from pyspi import _parallel
from pyspi.calculator import Calculator

from src.hcp_spectral_memory import bounded_psi_memory
from src.run_external_corpus import ExternalCorpusConfig, load_inventory, load_timeseries
from src.spi_spi_contract import build_unified_features


def main(config_path, index, profile, output):
    config = ExternalCorpusConfig.from_file(config_path)
    entry = load_inventory(config)[index - 1]
    data, _ = load_timeseries(config, entry)
    summary = json.loads((profile / 'summary.json').read_text())
    assert summary['status'] == 'complete' and not summary['errors']
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    calc = Calculator(dataset=data.T, config=str(config.pyspi_config), zscore=config.normalise, verbose=False)
    identity = json.loads((profile / 'identity.json').read_text())
    assert identity['run_digest'] == calc.run_digest
    owner, reason = _parallel.checkpoint_owner_matches(profile / 'checkpoints', calc.run_digest)
    assert owner, reason
    names = list(calc.spis)
    saved, missing = _parallel.load_checkpoints(profile / 'checkpoints', names, entry.M)
    assert not missing and len(saved) == 289
    old = {key: item[0] for key, item in saved.items()}
    new = dict(old)
    differences = {}
    with bounded_psi_memory() as repair:
        for key in names:
            if not key.startswith('psi_multitaper_'):
                continue
            matrix, error, warnings, elapsed = _parallel.run_spi(calc.spis[key], calc.dataset, key, entry.M)
            assert error is None, (key, error)
            np.testing.assert_allclose(matrix, old[key], rtol=1e-9, atol=1e-9)
            differences[key] = dict(max_abs_difference=float(np.nanmax(np.abs(matrix - old[key]))),
                                    seconds=elapsed, warnings=[str(w) for w in warnings])
            new[key] = matrix
    assert differences
    before = build_unified_features(old, names, metric='pearson').z
    after = build_unified_features(new, names, metric='pearson').z
    np.testing.assert_allclose(after, before, rtol=1e-9, atol=1e-9)
    report = dict(status='passed', index=index, M=entry.M, T=entry.T,
                  repair=repair, matrix_differences=differences,
                  z_max_abs_difference=float(np.nanmax(np.abs(after-before))),
                  z_finite=int(np.isfinite(before).sum()),
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  original_profile_peak_rss_kib=summary['peak_rss_kib'],
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Only affected PSI computations replayed; all other completed matrices reused; no band or estimator amendment')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.config, args.index, args.profile, args.output)
