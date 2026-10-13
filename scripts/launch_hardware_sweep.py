"""Release the fixed sweep only after successful smoke and resource validation."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess

import numpy as np


def resource_plan(peak_gib, seconds, workers=29):
    # The aggregate two-worker peak upper-bounds either individual worker.
    # Keep 50% above that bound and at least 8 GiB per new recording.
    per_worker = max(8, math.ceil(1.5 * peak_gib))
    per_node = 180 // per_worker
    if per_node < 1:
        raise ValueError('Unexpected smoke memory requirement')
    nodes = math.ceil(workers / per_node)
    if nodes > 4:
        raise ValueError('Resource request exceeds the bounded pilot envelope')
    memory = math.ceil(workers / nodes) * per_worker * nodes
    hours = max(6, math.ceil((1.5 * seconds + 1800) / 3600))
    return dict(workers=workers, nodes=nodes, ncpus=workers if nodes == 1 else 48 * nodes,
                mem_gb=memory, per_worker_gib=per_worker, walltime=f'{hours:02d}:00:00')


def validate_sampling(root):
    report = json.loads((root / 'scope-er-intermediate/sampling-decision.json').read_text())
    if report['records'] != 29 or report['smallest_passing_T'] != 5000:
        raise ValueError('Review intermediate sampling and matched SPI structure before releasing T=5000')
    return report['smallest_passing_T']


def main(root, smoke_job, dry_run=False):
    from src.corpus import ArchiveCorpus, load_inventory
    from src.run_corpus import completion_error
    marker = root / 'sweep-launch.json'
    if marker.exists() and not dry_run:
        raise FileExistsError(f'Inspect existing launch receipt before retrying: {marker}')
    validate_sampling(root)
    status = subprocess.check_output(['qstat', '-fx', smoke_job], text=True)
    if not re.search(r'job_state = F', status) or not re.search(r'Exit_status = 0(?:\s|$)', status):
        raise ValueError('Smoke did not finish successfully')
    match = re.search(r'resources_used.mem = (\d+)(kb|mb|gb|b)', status)
    if not match:
        raise ValueError('Final PBS memory measurement unavailable')
    peak = int(match[1]) * {'b':1, 'kb':1024, 'mb':1024**2, 'gb':1024**3}[match[2]] / 1024**3
    config = ArchiveCorpus.from_file(root / 'pilot/external.yaml')
    entries = load_inventory(config)
    reports, names = [], None
    for index in (1, 5):
        entry = entries[index - 1]
        error = completion_error(config, entry)
        if error:
            raise ValueError(f'{entry.name}: {error}')
        directory = entry.output_dir(config)
        meta = json.loads((directory / 'meta.json').read_text())
        order = [s['name'] for s in meta['pyspi']['spis']]
        if len(order) != 289 or (names is not None and names != order):
            raise ValueError('Smoke catalogue mismatch')
        names = order
        with np.load(directory / 'spi_mpis.npz') as bank:
            edge = ~np.eye(entry.M, dtype=bool)
            finite = float(np.mean([np.isfinite(bank[s][edge]).mean() for s in names]))
        if len(meta['pyspi']['errors']) > 15 or finite < .9:
            raise ValueError(f'Smoke coverage requires review: {entry.name}, finite={finite}')
        reports.append(dict(index=index, seconds=meta['job']['compute_seconds'],
                            finite_edge_fraction=finite, errors=meta['pyspi']['errors']))
    # Keep T=5000: changing resolution requires a separate estimator check.
    sweep = ArchiveCorpus.from_file(root / 'sweep29/external.yaml')
    target = load_inventory(sweep)
    if len(target) != 29 or any(e.M != 28 or e.T != 5000 for e in target):
        raise ValueError('Expanded corpus differs from the authorised bank')
    plan = resource_plan(peak, max(r['seconds'] for r in reports))
    receipt = dict(status='validated', smoke_job=smoke_job, smoke_peak_gib=peak,
                   smoke=reports, resources=plan, source_snapshot=str(Path.cwd()))
    if dry_run:
        print(json.dumps(receipt, indent=2)); return
    # Preserve an ambiguous submission outcome rather than risking duplicates.
    receipt['status'] = 'submitting'
    marker.write_text(json.dumps(receipt, indent=2) + '\n')
    env = dict(os.environ, NCPUS=str(plan['ncpus']), WORKERS=str(plan['workers']),
               MEM_GB=str(plan['mem_gb']), WALLTIME=plan['walltime'])
    env.pop('TASK_TIMEOUT', None)
    job = subprocess.check_output(['bash', 'jobs/gadi/submit_farm.sh',
                                   str(sweep.path)], env=env, text=True).strip()
    if not re.fullmatch(r'\d+\.gadi-pbs', job):
        raise ValueError(f'Unexpected qsub response; inspect before retrying: {job}')
    receipt.update(status='submitted', job_id=job)
    marker.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--smoke-job', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    main(args.run_root, args.smoke_job, args.dry_run)
