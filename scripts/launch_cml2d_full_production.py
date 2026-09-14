"""One-shot PBS continuation after successful representative p90 batches.

Only launches the two already specified 680-record arms and their reports.
It is not a poller, reminder, seed search or adaptive scientific experiment.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess


def resource_plan(seconds, tasks=680):
    if len(seconds) != 24 or min(seconds) <= 0:
        raise ValueError('24 successful representative timings required')
    median = statistics.median(seconds)
    peak = max(seconds)
    workers = 680 if peak / median <= 2 else 336
    nodes = math.ceil(workers * 8 / 190)  # 8 GB per concurrent recording
    wall = max(1800, math.ceil((1.5*peak*math.ceil(tasks/workers)+600)/900)*900)
    if wall > 8*3600:
        raise ValueError('runtime tail requires manual review before production')
    return dict(workers=workers, ncpus=48*nodes, memory_gb=190*nodes,
        wall_seconds=wall, task_timeout=math.ceil(1.5*peak+300),
        median_task_seconds=median, maximum_task_seconds=peak)


def launch(root, source, commit, sizes=(6, 8)):
    actual = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'], text=True).strip()
    assert actual == commit
    plans = {}
    for L in sizes:
        arm = root / f'L{L}'
        expected = set(map(int, (arm/'smoke-indices.txt').read_text().split()))
        expected.update(map(int, (arm/'node-indices.txt').read_text().split()))
        times = {}
        for path in (arm/'primary/mpi/primary').glob('*/meta.json'):
            meta = json.loads(path.read_text())
            index = meta['job']['index']
            if index in expected:
                assert meta['status']=='complete' and meta['pyspi']['n_spis']==289
                assert meta['M']==L*L and meta['T']==1000
                times[index] = meta['job']['compute_seconds']
        assert set(times)==expected and len(expected)==24
        plans[str(L)] = resource_plan(list(times.values()))
    # Atomic one-shot guard; retain it after failures for manual reconciliation.
    tag = '-'.join(f'L{L}' for L in sizes)
    (root/f'production-launch-lock-{tag}').mkdir()
    record = dict(source_commit=commit, plans=plans, jobs={})
    ledger = root/f'production-submission-{tag}.json'
    def save():
        ledger.write_text(json.dumps(record,indent=2)+'\n')
    def qsub(args):
        return subprocess.check_output(['qsub',*args],cwd=source,text=True).strip()
    save()
    for L in sizes:
        plan = plans[str(L)]
        arm = root / f'L{L}'
        hours, remainder = divmod(plan['wall_seconds'],3600)
        wall = f'{hours:02}:{remainder//60:02}:00'
        job = qsub(['-N',f'cml2d-full-L{L}', '-l',
            f"ncpus={plan['ncpus']},mem={plan['memory_gb']}GB,walltime={wall}", '-v',
            f"CORPUS_CONFIG={arm}/corpus.yaml,START_INDEX=1,END_INDEX=680,WORKERS={plan['workers']},TASK_TIMEOUT={plan['task_timeout']}",
            'jobs/gadi/run_external_corpus_farm.pbs'])
        record['jobs'][f'L{L}_p90']=job
        save()
        report = qsub(['-N',f'cml2d-full-L{L}-report','-W',f'depend=afterok:{job}','-v',
            f'SOURCE_DIR={source},EXPECTED_COMMIT={commit},STAGE=report,L={L},DIAGNOSTIC_ROOT={root},'
            'PHYSICS_DIR=/scratch/ql44/we2614/mts-spi-data/order_parameter/cml2d_full_observation_260914/physics,'
            'FROZEN_DIR=/scratch/ql44/we2614/mts-spi-data/order_parameter/cml2d_period_doubling_260911/primary-analysis',
            'jobs/gadi/run_cml2d_full_observation_stage.pbs'])
        record['jobs'][f'L{L}_report']=report
        save()
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--commit',required=True)
    p.add_argument('--L',type=int,nargs='+',choices=[6,8],default=[6,8])
    a=p.parse_args()
    if len(set(a.L)) != len(a.L):p.error('duplicate lattice size')
    launch(a.root,a.source,a.commit,tuple(a.L))
