"""Release only this pilot's remaining records after a verified size-class smoke."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import numpy as np

EXPECTED_SOURCE='/scratch/ql44/we2614/mts-spi-study/operations/sources/pyspi-pair-sampling-fd0de71'


def recording_peak_mib(peak, workers):
    """Runner metadata stores parent and largest child, not one scalar RSS."""
    return float(peak['parent']+workers*peak['largest_worker']) if isinstance(peak,dict) else float(peak)


def resource_plan(peak_mib, measured_workers, seconds, remaining, M):
    """Fixed workers, PBS whole-job memory and at most one target-worker pilot."""
    workers=8 if M<=64 else 12
    peak_gib=float(peak_mib)/1024
    if not np.isfinite(peak_gib) or peak_gib<=0: raise ValueError('Invalid PBS memory')
    if not remaining: return {'complete':True}
    pilot=measured_workers!=workers
    # Only for the first fixed-worker pilot: conservative current HPC.md allowance.
    forecast=peak_gib+max(0,workers-measured_workers)*2
    per_record=math.ceil(forecast*1.25)
    capacity=min(48//workers,190//per_record)
    if capacity<1: raise ValueError('One recording exceeds normal-node memory; reassess')
    groups=1 if pilot else min(remaining,capacity)
    wall=min(48*3600,max(600,math.ceil(2*seconds/60)*60))
    return dict(workers=workers,groups=groups,mem_gb=groups*per_record,
                walltime=f'{wall//3600:02d}:{wall//60%60:02d}:00',
                fixed_worker_pilot=pilot,complete=False)


def pbs_job(job_id):
    result=subprocess.run(['qstat','-xf','-F','json',job_id],check=True,text=True,capture_output=True)
    return json.loads(result.stdout)['Jobs'][job_id]


def job_indices(job):
    variables=job['Variable_List']
    if variables.get('INDEX_FILE'):
        return [int(x) for x in Path(variables['INDEX_FILE']).read_text().split()]
    return list(range(int(variables['START_INDEX']),int(variables['END_INDEX'])+1))


def pbs_measurement(job,config,index,workers):
    assert job['job_state']=='F' and int(job['Exit_status'])==0
    variables=job['Variable_List']
    assert Path(variables['CORPUS_CONFIG']).resolve()==config.resolve()
    assert variables['PYSPI_SOURCE']==EXPECTED_SOURCE
    assert job_indices(job)==[index], 'Memory must come from a one-recording job'
    assert int(variables['CORES_PER_RECORDING'])==workers
    memory=re.fullmatch(r'([0-9.]+)(kb|mb|gb|b)',job['resources_used']['mem'].lower())
    if not memory: raise ValueError('Unrecognised PBS memory units')
    peak=float(memory[1])*{'b':1/2**20,'kb':1/1024,'mb':1,'gb':1024}[memory[2]]
    h,m,s=map(int,job['resources_used']['walltime'].split(':'))
    return peak,h*3600+m*60+s


def assert_no_active_farm(config):
    result=subprocess.run(['qstat','-f','-F','json','-u',os.environ['USER']],check=True,text=True,capture_output=True)
    for jid,job in json.loads(result.stdout).get('Jobs',{}).items():
        configured=job.get('Variable_List',{}).get('CORPUS_CONFIG')
        if job.get('Job_Name')=='spi-farm' and configured and Path(configured).resolve()==config.resolve() and job['job_state'] not in ('F','X'):
            raise RuntimeError(f'Existing farm owns this corpus: {jid} ({job["job_state"]})')


def validate_table(corpus,entry):
    from mtsspi.run_corpus import completion_error
    error=completion_error(corpus,entry)
    if error: raise ValueError(error)
    metadata=json.loads((entry.output_dir(corpus)/'meta.json').read_text())
    assert metadata['pyspi']['version']['computation']=='3.0.0.r9'
    with np.load(entry.output_dir(corpus)/'spi_mpis.npz') as bank:
        edge=~np.eye(entry.M,dtype=bool)
        coverage=float(np.mean([np.isfinite(bank[key][edge]).mean() for key in bank.files]))
        assert len(bank.files)==289
    if coverage<.9 or len(metadata['pyspi']['errors'])>28: raise ValueError('Table fails coverage')
    return metadata,coverage


def release(root,config,after_index,after_job,smoke_index=None,smoke_job=None):
    from mtsspi.corpus import ArchiveCorpus
    from mtsspi.run_corpus import completion_error
    root,config=Path(root),Path(config)
    assert os.environ.get('PYSPI_SOURCE')==EXPECTED_SOURCE
    active=json.loads((Path(__file__).parent/'configs/analysis-selection.json').read_text())['corpora']
    assert config.resolve()==(root/'corpora'/config.parent.name/'external.yaml').resolve() and config.parent.name in active
    corpus=ArchiveCorpus.from_file(config);assert corpus.pair_tolerance is None
    entries=corpus.entries();assert len(entries)<=4 and len({(e.M,e.T) for e in entries})==1
    folder=root/'releases'/config.parent.name;folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/f'fixed-after-{after_index}.json'
    if receipt.exists(): print(receipt.read_text());return
    dependency=pbs_job(after_job)
    assert dependency['job_state']=='F' and int(dependency['Exit_status'])==0
    assert Path(dependency['Variable_List']['CORPUS_CONFIG']).resolve()==config.resolve()
    assert after_index in job_indices(dependency)
    for index in job_indices(dependency): validate_table(corpus,entries[index-1])
    smoke_index=smoke_index or after_index;smoke_job=smoke_job or after_job
    smoke=entries[smoke_index-1]
    metadata,coverage=validate_table(corpus,smoke)
    assert_no_active_farm(config)
    remaining=[e.index for e in entries if completion_error(corpus,e) is not None]
    peak,seconds=pbs_measurement(pbs_job(smoke_job),config,smoke_index,metadata['job']['n_jobs'])
    plan=resource_plan(peak,metadata['job']['n_jobs'],seconds,len(remaining),smoke.M)
    record=dict(config=str(config),after_index=after_index,after_job=after_job,smoke_index=smoke_index,
                smoke_job=smoke_job,coverage=coverage,pbs_peak_mib=peak,pbs_seconds=seconds,
                plan=plan,remaining=remaining,status='validated')
    with receipt.open('x') as stream: json.dump(record,stream,indent=2)
    if plan['complete']: return
    selected=remaining[:plan['groups']]
    index_file=folder/f'fixed-after-{after_index}-indices.txt'
    index_file.write_text(''.join(f'{index}\n' for index in selected))
    env=dict(os.environ,MEM_GB=str(plan['mem_gb']),WORKERS=str(plan['groups']),
             CORES_PER_RECORDING=str(plan['workers']),WALLTIME=plan['walltime'])
    env.pop('NCPUS',None)
    try:
        result=subprocess.run(['bash','jobs/gadi/submit_farm.sh',str(config),str(index_file)],env=env,text=True,capture_output=True,check=True)
        record['submission_stdout']=result.stdout;record['submission_stderr']=result.stderr
        match=re.findall(r'\b\d+\.gadi-pbs\b',result.stdout)
        if len(match)!=1: raise RuntimeError('Ambiguous submission; inspect queue and receipt')
        record.update(farm_job=match[0],status='submitted',indices=selected)
        receipt.write_text(json.dumps(record,indent=2)+'\n')
        if len(selected)<len(remaining):
            next_smoke_index=selected[0] if plan['fixed_worker_pilot'] else smoke_index
            next_smoke_job=match[0] if plan['fixed_worker_pilot'] else smoke_job
            variables=f'RUN_ROOT={root},CORPUS_CONFIG={config},AFTER_INDEX={selected[-1]},AFTER_JOB={match[0]},SMOKE_INDEX={next_smoke_index},SMOKE_JOB={next_smoke_job},PYSPI_SOURCE={EXPECTED_SOURCE}'
            follow=subprocess.run(['qsub','-W',f'depend=afterok:{match[0]}','-v',variables,
                'research/empirical/ccep_pilot/jobs/release.pbs'],text=True,capture_output=True,check=True)
            record['next_release_job']=follow.stdout.strip()
        receipt.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
    except Exception as exc:
        record.update(status='requires-inspection',error=str(exc))
        if isinstance(exc,subprocess.CalledProcessError): record.update(stdout=exc.stdout,stderr=exc.stderr)
        receipt.write_text(json.dumps(record,indent=2)+'\n');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--after-index',type=int,required=True)
    parser.add_argument('--after-job',required=True);parser.add_argument('--smoke-index',type=int);parser.add_argument('--smoke-job')
    args=parser.parse_args();release(args.root,args.config,args.after_index,args.after_job,args.smoke_index,args.smoke_job)
