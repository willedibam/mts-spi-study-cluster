"""Gadi submissions gated by resource pilots, with a persistent duplicate guard."""
import argparse,json,math,os,re,subprocess
from pathlib import Path
from research.proof.baselines.noise_experiments import REMOTE,PYSPI,ROOT,RUN

def main(stage):
    root=Path(REMOTE);ledger=root/'jobs.json';jobs=json.loads(ledger.read_text()) if ledger.exists() else []
    manifest=json.loads((root/'manifest.json').read_text())
    os.chdir(ROOT)
    for experiment in ['dose','random']:
        role='pilot' if stage=='pilot' else 'production'
        if any(j['experiment']==experiment and j['role']==role for j in jobs):continue
        if stage=='pilot':
            indices=[int(x) for x in (root/(experiment+'-pilot.txt')).read_text().split()]
            cpus=len(indices);mem=cpus*4;wall='00:30:00';timeout=1800
        else:
            pilot=next(j for j in jobs if j['experiment']==experiment and j['role']=='pilot')
            status=json.loads(subprocess.check_output(['qstat','-xf','-F','json',pilot['job']],text=True))['Jobs'][pilot['job']]
            if status['job_state']!='F':print(experiment,'pilot still',status['job_state'],flush=True);continue
            assert status['Exit_status']==0,status
            times=[];errors={}
            for row in manifest['rows']:
                if row['experiment']!=experiment or row['local_index'] not in pilot['indices']:continue
                from mtsspi.utils import slugify
                folder=Path(row['mpi_root'])/row['corpus_name']/f'{row["local_index"]:04d}-{slugify(row["row_id"],"dataset")}'
                d=json.loads((folder/'meta.json').read_text())
                assert d['status']=='complete' and d['random_seed']==2610152027
                assert d['source']['archive_sha256']==row['archive_sha256'] and (d['M'],d['T'])==(16,1000)
                assert d['pyspi']['version']['computation']=='3.0.0.r9' and len(d['pyspi']['spis'])==289
                assert not any(any(x in message for x in ['ImportError:','ModuleNotFoundError:','TypeError:','NameError:','AttributeError:','KeyError:']) for message in d['pyspi'].get('errors',{}).values())
                times.append(d['job']['compute_seconds'])
                for message in d['pyspi'].get('errors',{}).values():errors[message.split(':')[0]]=errors.get(message.split(':')[0],0)+1
            assert len(times)==len(pilot['indices'])
            pilot.update(exit_status=0,resources_used=status['resources_used'],max_seconds=max(times),error_types=errors)
            indices=[int(x) for x in (root/(experiment+'-remaining.txt')).read_text().split()]
            cpus=384 if experiment=='dose' else 192
            # One core/record; 2 GiB/record comfortably exceeds the audited pilot peak.
            peak_gb=int(status['resources_used']['mem'].removesuffix('kb'))/1024**2/len(times)
            per_record=max(2,math.ceil(1.5*peak_gb));mem=cpus*per_record
            seconds=max(1800,math.ceil((2*math.ceil(len(indices)/cpus)*max(times)+300)/1800)*1800)
            wall=f'{seconds//3600:02}:{seconds%3600//60:02}:00';timeout=max(1800,math.ceil(max(times)*3))
        env=dict(os.environ,PYSPI_SOURCE=PYSPI,NCPUS=str(cpus),MEM_GB=str(mem),CORES_PER_RECORDING='1',WALLTIME=wall,TASK_TIMEOUT=str(timeout))
        result=subprocess.check_output(['bash','jobs/gadi/submit_farm.sh',str(root/(experiment+'.yaml')),str(root/(experiment+('-pilot.txt' if stage=='pilot' else '-remaining.txt')))],env=env,text=True)
        ids=re.findall(r'^\d+\.gadi-pbs$',result,re.M);assert len(ids)==1,result
        jobs.append(dict(experiment=experiment,role=role,job=ids[0],records=len(indices),indices=indices if stage=='pilot' else None,ncpus=cpus,mem_gb=mem,walltime=wall,task_timeout=timeout,source=str(ROOT),pyspi_source=PYSPI))
        ledger.write_text(json.dumps(jobs,indent=2)+'\n');print(jobs[-1],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['pilot','production']);main(p.parse_args().stage)
