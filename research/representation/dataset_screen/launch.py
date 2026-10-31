"""Finite PBS dependency chain: measurement, sized farm, audited bank, discovery fits."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess

from mtsspi.provenance import atomic_json


def submit(args, env=None):
    result = subprocess.check_output(args, text=True, env=env)
    jobs = re.findall(r'^\d+\.gadi-pbs$', result, flags=re.M)
    if len(jobs) != 1:
        raise RuntimeError(f'cannot identify submission: {result}')
    print('Submitted', jobs[0], flush=True)
    return jobs[0]


def batch(root, script, dependency=None, **variables):
    command = ['qsub']
    if dependency: command += ['-W', f'depend=afterok:{dependency}']
    variables = dict(RUN_ROOT=str(root), **variables)
    command += ['-v', ','.join(f'{k}={v}' for k,v in variables.items()), script]
    return submit(command)


def farm(root, count, cpus, memory, seconds, start=1):
    env = dict(os.environ, START_INDEX=str(start), END_INDEX=str(count), CORES_PER_RECORDING='1',
               NCPUS=str(cpus), MEM_GB=str(memory), WALLTIME=f'{seconds//3600:02d}:{seconds%3600//60:02d}:{seconds%60:02d}')
    return submit(['bash', 'jobs/gadi/submit_farm.sh', str(root/'corpus.yaml')], env)


def main(root, stage, measurement_job):
    receipt = root / f'{stage}-chain.json'
    if receipt.exists(): raise FileExistsError(receipt)
    count = len(json.loads((root/'manifest.json').read_text())['rows'])
    record = dict(root=str(root), snapshot=str(Path.cwd()), stage=stage, records=count, jobs={})
    atomic_json(receipt, record)
    def save(name, job):
        record['jobs'][name] = job; atomic_json(receipt, record); return job
    if stage == 'pilot':
        pilot = save('measurement', farm(root,1,1,4,2400))
        save('release_farm', batch(root,'research/representation/dataset_screen/jobs/chain.pbs',
                                  pilot, STAGE='farm', MEASUREMENT_JOB=pilot))
        return
    status = json.loads(subprocess.check_output(['qstat','-xf','-F','json',measurement_job],text=True))['Jobs']
    status = next(iter(status.values()))
    if status['job_state'] != 'F' or status['Exit_status'] != 0:
        raise ValueError('measurement did not complete successfully')
    used = status['resources_used']
    if not used['mem'].endswith('kb'): raise ValueError('unrecognized PBS memory unit')
    peak_gib = int(used['mem'][:-2]) / 1024**2
    hours, minutes, seconds = map(int, used['walltime'].split(':'))
    elapsed = hours*3600 + minutes*60 + seconds
    # At most four normal nodes, with no rounded-up idle fifth node.
    tasks = count - 1
    cpus = tasks if tasks < 48 else min(192, (tasks//48)*48)
    memory = cpus * max(3, math.ceil(1.25*peak_gib))
    walltime = math.ceil((2*elapsed*math.ceil(tasks/cpus) + 300)/60)*60
    if walltime > 12*3600: raise ValueError('measurement requires a different farm plan')
    record.update(measurement_job=measurement_job, measurement_resources=used,
                  farm_request=dict(cpus=cpus,memory_GB=memory,walltime_seconds=walltime))
    atomic_json(receipt,record)
    job = save('farm',farm(root,count,cpus,memory,walltime,start=2))
    bank = save('bank',batch(root,'research/representation/empirical_transfer/jobs/bank.pbs',job))
    save('statistical',batch(root,'research/representation/dataset_screen/jobs/statistical.pbs',bank))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',choices=['pilot','farm'],required=True)
    parser.add_argument('--measurement-job')
    args=parser.parse_args()
    if args.stage == 'farm' and not args.measurement_job: parser.error('farm stage needs --measurement-job')
    main(args.root,args.stage,args.measurement_job)
