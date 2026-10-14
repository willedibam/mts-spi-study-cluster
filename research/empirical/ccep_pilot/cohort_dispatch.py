"""One fixed-worker measurement per participant/size, then disjoint cohort farms."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
from .acquire import sha256
from .release import EXPECTED_SOURCE, assert_no_active_farm, pbs_job, pbs_measurement, validate_table


def plan(peak_mib, seconds, M, count):
    workers=8 if M<=64 else 12
    per_record=math.ceil(float(peak_mib)/1024*1.25)
    if per_record<=0 or per_record>190: raise ValueError('Invalid or excessive per-record memory')
    groups=min(count,48//workers,190//per_record)
    wall=min(48*3600,max(600,math.ceil(2*seconds/60)*60))
    return dict(workers=workers,groups=groups,mem_gb=groups*per_record,
                walltime=f'{wall//3600:02d}:{wall//60%60:02d}:00')


def cohorts(root):
    from mtsspi.corpus import ArchiveCorpus
    groups={}
    for name in json.loads((root/'active-configs.json').read_text()):
        config=Path(name)
        assert config.resolve()==(root/'corpora'/config.parent.name/'external.yaml').resolve()
        corpus=ArchiveCorpus.from_file(config);entries=corpus.entries()
        assert corpus.pair_tolerance is None and 2<=len(entries)<=4
        manifest=json.loads((config.parent/'manifest.json').read_text())
        assert len({(e.M,e.T) for e in entries})==1 and entries[0].T==4096
        subject=manifest['rows'][0]['subject'];M=entries[0].M
        assert all(r['subject']==subject for r in manifest['rows'])
        groups.setdefault((subject,M),[]).append((config,corpus,entries))
    return groups


def submitted(command, env=None):
    result=subprocess.run(command,env=env,text=True,capture_output=True,check=True)
    ids=re.findall(r'\b\d+\.gadi-pbs\b',result.stdout)
    if len(ids)!=1: raise RuntimeError(f'Ambiguous submission: {result.stdout} {result.stderr}')
    return ids[0]


def farm(config, indices, folder, resources):
    index_file=folder/'indices.txt';index_file.write_text(''.join(f'{i}\n' for i in indices))
    env=dict(os.environ,MEM_GB=str(resources['mem_gb']),WORKERS=str(len(indices)),
             CORES_PER_RECORDING=str(resources['workers']),WALLTIME=resources['walltime'])
    env.pop('NCPUS',None)
    return submitted(['bash','jobs/gadi/submit_farm.sh',str(config),str(index_file)],env)


def save(path, record):
    path.write_text(json.dumps(record,indent=2)+'\n')


def dispatch(root, subject=None, M=None):
    from mtsspi.run_corpus import completion_error
    root=Path(root)
    assert os.environ.get('PYSPI_SOURCE')==EXPECTED_SOURCE
    groups=cohorts(root)
    for (who,size), group in groups.items():
        if subject is not None and (who,size)!=(subject,M):continue
        folder=root/'dispatch'/f'{who}-m{size}';folder.mkdir(parents=True,exist_ok=True)
        receipt=folder/'measurement.json'
        if receipt.exists(): continue  # Includes ambiguous outcomes: inspect, never blindly retry.
        for config,corpus,entries in group:
            assert_no_active_farm(config)
            assert all(completion_error(corpus,e) for e in entries), 'Unexpected existing output; reconcile first'
        config,corpus,entries=group[0];workers=8 if size<=64 else 12
        resources=dict(workers=workers,groups=1,mem_gb=32 if size<=64 else 64,walltime='08:00:00')
        record=dict(status='submitting',subject=who,M=size,config=str(config),index=1,
                    active_configs_sha256=sha256(root/'active-configs.json'),resources=resources)
        with receipt.open('x') as stream: json.dump(record,stream,indent=2)
        try:
            record['job']=farm(config,[1],folder,resources);record['status']='measurement-submitted';save(receipt,record)
            variables=f'RUN_ROOT={root},SUBJECT={who},CHANNELS={size},PYSPI_SOURCE={EXPECTED_SOURCE}'
            record['controller']=submitted(['qsub','-W',f"depend=afterok:{record['job']}",'-v',variables,
                          'research/empirical/ccep_pilot/jobs/cohort-release.pbs'])
            save(receipt,record);print(json.dumps(record),flush=True)
        except Exception as exc:
            record.update(status='requires-inspection',error=str(exc));save(receipt,record);raise


def release(root, subject, M):
    from mtsspi.corpus import ArchiveCorpus
    from mtsspi.run_corpus import completion_error
    root=Path(root);assert os.environ.get('PYSPI_SOURCE')==EXPECTED_SOURCE
    folder=root/'dispatch'/f'{subject}-m{M}'
    measurement=json.loads((folder/'measurement.json').read_text())
    assert measurement['active_configs_sha256']==sha256(root/'active-configs.json')
    config=Path(measurement['config']);corpus=ArchiveCorpus.from_file(config)
    meta,coverage=validate_table(corpus,corpus.entries()[0])
    workers=8 if M<=64 else 12
    assert meta['job']['n_jobs']==workers
    peak,seconds=pbs_measurement(pbs_job(measurement['job']),config,1,workers)
    group=cohorts(root)[(subject,M)]
    for config,corpus,entries in group:
        prefix=folder/config.parent.name;prefix.mkdir(exist_ok=True)
        existing=list(prefix.glob('batch-*/receipt.json'))
        if existing:
            owned=set()
            for path in existing:
                old=json.loads(path.read_text())
                if old['status']!='submitted': raise RuntimeError(f'Inspect ambiguous receipt: {path}')
                owned.update(old['indices'])
            outstanding={e.index for e in entries if completion_error(corpus,e)}
            if not outstanding.issubset(owned): raise RuntimeError(f'Incomplete dispatch: inspect {prefix}')
            continue
        assert_no_active_farm(config)
        remaining=[]
        for e in entries:
            if completion_error(corpus,e):remaining.append(e.index)
            else:validate_table(corpus,e)
        if not remaining:continue
        capacity=plan(peak,seconds,M,len(remaining))['groups']
        for offset in range(0,len(remaining),capacity):
            indices=remaining[offset:offset+capacity];batch=prefix/f'batch-{offset//capacity}'
            batch.mkdir();receipt=batch/'receipt.json'
            resources=plan(peak,seconds,M,len(indices))
            record=dict(status='submitting',config=str(config),indices=indices,resources=resources,
                        measurement_job=measurement['job'],pbs_peak_mib=peak,pbs_seconds=seconds,coverage=coverage)
            with receipt.open('x') as stream:json.dump(record,stream,indent=2)
            try:
                record['job']=farm(config,indices,batch,resources);record['status']='submitted';save(receipt,record)
                print(json.dumps(record),flush=True)
            except Exception as exc:
                record.update(status='requires-inspection',error=str(exc));save(receipt,record);raise
    (folder/'release-complete.json').write_text(json.dumps(dict(status='dispatch-complete',measurement_job=measurement['job']))+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['dispatch','release'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--subject');parser.add_argument('--M',type=int)
    args=parser.parse_args()
    if args.command=='dispatch': dispatch(args.root,args.subject,args.M)
    else:release(args.root,args.subject,args.M)
