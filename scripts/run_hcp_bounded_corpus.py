"""HCP external runner with a verified, process-local PSI memory repair."""
import hashlib
import json
from pathlib import Path

from src.hcp_spectral_memory import bounded_psi_memory
from src import run_external_corpus as corpus


def main():
    args = corpus.parse_args()
    if args.job_index is None:
        raise ValueError('This opt-in runner only accepts individual dataset jobs')
    if args.n_jobs != 1:
        raise ValueError('Use one worker per recording')
    config = corpus.ExternalCorpusConfig.from_file(args.config)
    entries = corpus.load_inventory(config)
    if not 1 <= args.job_index <= len(entries):
        raise ValueError('Dataset index outside inventory')
    entry = entries[args.job_index-1]
    if args.skip_existing and corpus.completion_error(config, entry) is None:
        print(f'[SKIP] {entry.index}/{entry.name}')
        return
    with bounded_psi_memory() as repair:
        directory = corpus.run_dataset(config, entry, n_jobs=1, dry_run=args.dry_run)
    if not args.dry_run:
        meta_path = directory/'meta.json'
        meta = json.loads(meta_path.read_text())
        meta['spectral_memory_repair'] = repair
        meta['spectral_memory_repair']['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        corpus._atomic_json(meta_path, meta)


if __name__ == '__main__':
    main()
