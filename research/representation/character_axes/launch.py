"""Finite PBS chain for one prepared corpus: measurement, sized farm, bank, archive."""
import argparse
import json
import math
from pathlib import Path
import subprocess

from mtsspi.provenance import atomic_json
from research.representation.dataset_screen.launch import batch, farm

JOBS = 'research/representation/character_axes/jobs'


def main(root, stage, measurement_job=None, measurement_seconds=3600):
    receipt = root / f'{stage}-chain.json'
    if receipt.exists():
        raise FileExistsError(receipt)
    count = len(json.loads((root / 'manifest.json').read_text())['rows'])
    record = dict(root=str(root), snapshot=str(Path.cwd()), stage=stage, records=count, jobs={})
    atomic_json(receipt, record)

    def save(name, job):
        record['jobs'][name] = job; atomic_json(receipt, record); return job

    if stage == 'pilot':
        pilot = save('measurement', farm(root, 1, 1, 6, measurement_seconds))
        save('release_farm', batch(root, f'{JOBS}/chain.pbs', pilot, MEASUREMENT_JOB=pilot))
        return
    status = json.loads(subprocess.check_output(['qstat', '-xf', '-F', 'json', measurement_job], text=True))['Jobs']
    status = next(iter(status.values()))
    if status['job_state'] != 'F' or status['Exit_status'] != 0:
        raise ValueError('measurement did not complete successfully')
    used = status['resources_used']
    peak_gib = int(used['mem'][:-2]) / 1024 ** 2
    h, m, s = map(int, used['walltime'].split(':'))
    tasks = count - 1
    cpus = tasks if tasks < 48 else min(384, (tasks // 48) * 48)
    per_cpu = max(3.0, 1.25 * peak_gib)
    if per_cpu > 3.9:
        raise ValueError('per-recording memory exceeds a normal-node core share; plan multi-core tasks')
    memory = math.ceil(cpus * per_cpu)
    walltime = math.ceil((2 * (h * 3600 + m * 60 + s) * math.ceil(tasks / cpus) + 300) / 60) * 60
    if walltime > 12 * 3600:
        raise ValueError('measurement requires a different farm plan')
    record.update(measurement_job=measurement_job, measurement_resources=used,
                  farm_request=dict(cpus=cpus, memory_GB=memory, walltime_seconds=walltime))
    atomic_json(receipt, record)
    job = save('farm', farm(root, count, cpus, memory, walltime, start=2))
    bank = save('bank', batch(root, 'research/representation/empirical_transfer/jobs/bank.pbs', job))
    save('archive', batch(root, f'{JOBS}/archive.pbs', bank))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=['pilot', 'farm'], required=True)
    parser.add_argument('--measurement-job')
    parser.add_argument('--measurement-seconds', type=int, default=3600, help='walltime of the one-recording pilot')
    args = parser.parse_args()
    main(args.root, args.stage, args.measurement_job, args.measurement_seconds)
