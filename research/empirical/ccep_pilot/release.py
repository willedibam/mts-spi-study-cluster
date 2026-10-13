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


def resource_plan(peak_mib, workers, seconds, remaining):
    peak_gib=recording_peak_mib(peak_mib,workers)/1024
    if not np.isfinite(peak_gib) or peak_gib<=0: raise ValueError('Invalid smoke memory measurement')
    if not remaining: return {'complete':True}
    required=max(workers,math.ceil(peak_gib/4))
    options=(1,2,3,4,6,8,12,16,24,48)
    # More workers can multiply memory. Forecast conservatively before a remeasurement.
    safe=[k for k in options if recording_peak_mib(peak_mib,k)/1024*1.25<=190]
    if not safe: raise ValueError('No within-node worker count fits the measured memory bound plus margin')
    target=next((k for k in options if k>=required),48)
    requested=min(target,max(safe))
    remeasure=requested!=workers
    count=1 if remeasure else remaining
    per_node_capacity=min(48//requested,math.floor(190/(peak_gib*1.25))) if not remeasure else 1
    nodes=math.ceil(count/per_node_capacity)
    per_node=math.ceil(count/nodes)
    # nci-parallel maps by node; allow for the node with the extra group.
    memory=max(180,math.ceil(recording_peak_mib(peak_mib,requested)/1024*1.25)) if remeasure else math.ceil(peak_gib*per_node*nodes*1.25)
    if memory/nodes>190: raise ValueError('Measured simultaneous peaks exceed one-node memory; reassess worker count')
    wall_seconds=min(48*3600,max(600,math.ceil(2*seconds/60)*60))
    result=dict(workers=requested,groups=count,ncpus=nodes*48,mem_gb=memory,
                walltime=f'{wall_seconds//3600:02d}:{wall_seconds//60%60:02d}:00',
                remeasure=remeasure,complete=False)
    if requested<target: result['memory_limited_workers']=True
    return result


def release(root,config,after_index):
    from mtsspi.corpus import ArchiveCorpus
    from mtsspi.run_corpus import completion_error
    root,config=Path(root),Path(config)
    assert os.environ.get('PYSPI_SOURCE')==EXPECTED_SOURCE
    assert config.resolve().is_relative_to((root/'corpora').resolve())
    corpus=ArchiveCorpus.from_file(config)
    assert corpus.pair_tolerance is None
    entries=corpus.entries()
    assert len(entries)<=4 and len({(e.M,e.T) for e in entries})==1
    smoke=entries[after_index-1]
    error=completion_error(corpus,smoke)
    if error: raise ValueError(error)
    metadata=json.loads((smoke.output_dir(corpus)/'meta.json').read_text())
    assert metadata['pyspi']['version']['computation']=='3.0.0.r9'
    with np.load(smoke.output_dir(corpus)/'spi_mpis.npz') as bank:
        edge=~np.eye(smoke.M,dtype=bool)
        coverage=float(np.mean([np.isfinite(bank[key][edge]).mean() for key in bank.files]))
        assert len(bank.files)==289
    if coverage<.9 or len(metadata['pyspi']['errors'])>28:
        raise ValueError(f'Smoke fails coverage: {coverage}, {len(metadata["pyspi"]["errors"])} failed SPIs')
    remaining=[e.index for e in entries if completion_error(corpus,e) is not None]
    job=metadata['job']
    plan=resource_plan(job['peak_memory_mib'],job['n_jobs'],job['compute_seconds'],len(remaining))
    folder=root/'releases'/config.parent.name;folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/f'after-{after_index}.json'
    if receipt.exists():
        print(receipt.read_text());return
    record=dict(config=str(config),after_index=after_index,coverage=coverage,
                measured_job=job,plan=plan,remaining=remaining,status='validated')
    # Exclusive receipt prevents duplicate dispatch on repeated heartbeat/dependency calls.
    with receipt.open('x') as stream:json.dump(record,stream,indent=2)
    if plan['complete']:return
    selected=remaining[:1] if plan['remeasure'] else remaining
    index_file=folder/f'after-{after_index}-indices.txt'
    index_file.write_text(''.join(f'{index}\n' for index in selected))
    env=dict(os.environ,NCPUS=str(plan['ncpus']),MEM_GB=str(plan['mem_gb']),WORKERS=str(plan['groups']),
             CORES_PER_RECORDING=str(plan['workers']),WALLTIME=plan['walltime'])
    try:
        result=subprocess.run(['bash','jobs/gadi/submit_farm.sh',str(config),str(index_file)],env=env,
                              text=True,capture_output=True,check=True)
        record['submission_stdout']=result.stdout;record['submission_stderr']=result.stderr
        match=re.findall(r'\b\d+\.gadi-pbs\b',result.stdout)
        if len(match)!=1:raise RuntimeError('Ambiguous submission outcome; inspect receipt/queue before retry')
        record['farm_job']=match[0];record['status']='submitted';record['indices']=selected
        receipt.write_text(json.dumps(record,indent=2)+'\n')
        if plan['remeasure']:
            variables=f'RUN_ROOT={root},CORPUS_CONFIG={config},AFTER_INDEX={selected[0]},PYSPI_SOURCE={EXPECTED_SOURCE}'
            follow=subprocess.run(['qsub','-W',f'depend=afterok:{match[0]}','-v',variables,
                'research/empirical/ccep_pilot/jobs/release.pbs'],text=True,capture_output=True,check=True)
            record['next_release_job']=follow.stdout.strip()
        receipt.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record,indent=2))
    except Exception as exc:
        record['status']='requires-inspection';record['error']=str(exc)
        if isinstance(exc,subprocess.CalledProcessError):
            record['stdout']=exc.stdout;record['stderr']=exc.stderr
        receipt.write_text(json.dumps(record,indent=2)+'\n');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--after-index',type=int,required=True)
    args=parser.parse_args();release(args.root,args.config,args.after_index)
