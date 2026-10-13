"""Hardware p90 with the verified PSI memory repair and diagnostic checkpoints."""
from contextlib import contextmanager
from functools import partial
import hashlib
import json
import os
from pathlib import Path

from src.hcp_spectral_memory import bounded_psi_memory
from src import run_external_corpus as corpus


def pyspi_bounds_psi_memory():
    """Whether this pyspi applies the multitaper PSI band (computation r8 on).

    It then thins the spectrum to about fifty frequencies before the backend's
    all-pairs sum, so the repair is not needed -- and must not be applied: it
    reorders that sum, and a patch made here reaches forked workers but not
    spawned ones, so results would depend on the start method.
    """
    from pyspi.statistics.spectral import PhaseSlopeIndex
    return bool(getattr(PhaseSlopeIndex(fs=1, fmin=0, fmax=0.5), '_recompute', False))


@contextmanager
def _no_repair():
    yield {'applied': False,
           'reason': 'pyspi applies the PSI band; the all-pairs sum is over a thinned spectrum'}


def main():
    args = corpus.parse_args()
    bounded = pyspi_bounds_psi_memory()
    if args.job_index is None:
        raise ValueError('Give one recording with --job-index')
    if args.n_jobs != 1 and not (bounded and corpus.pyspi_pins_estimator_seeds()):
        raise ValueError('This pyspi revision needs one serial worker per recording')
    config = corpus.ExternalCorpusConfig.from_file(args.config)
    entries = corpus.load_inventory(config)
    if not 1 <= args.job_index <= len(entries):
        raise ValueError('Dataset index outside inventory')
    entry = entries[args.job_index - 1]
    if args.skip_existing and corpus.completion_error(config, entry) is None:
        print(f'[SKIP] {entry.index}/{entry.name}')
        return
    attempt = os.environ.get('PBS_JOBID', f'local-{os.getpid()}')
    checkpoints = config.path.parent / 'checkpoints' / attempt / f'{entry.index:04d}'
    if checkpoints.exists():
        raise FileExistsError(f'Preserve previous diagnostic checkpoints: {checkpoints}')
    original = corpus.run_pyspi
    # No checkpoint replay: a fresh serial RNG trajectory avoids inheriting
    # uncertain estimator state after a killed process. Completed SPIs survive
    # as diagnostics even if the eventual production archive cannot be written.
    corpus.run_pyspi = partial(original, checkpoint_dir=checkpoints, resume=False)
    try:
        with (_no_repair() if bounded else bounded_psi_memory()) as repair:
            directory = corpus.run_dataset(
                config, entry, n_jobs=args.n_jobs, dry_run=args.dry_run)
    finally:
        corpus.run_pyspi = original
    if not args.dry_run:
        path = directory / 'meta.json'
        meta = json.loads(path.read_text())
        meta['spectral_memory_repair'] = dict(repair,
            runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        meta['diagnostic_checkpoints'] = str(checkpoints)
        corpus._atomic_json(path, meta)


if __name__ == '__main__':
    main()
