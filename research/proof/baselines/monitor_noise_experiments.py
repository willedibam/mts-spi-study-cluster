"""Resume-safe local orchestration of the authorized Gadi noise experiments."""
import json,subprocess,time
from research.proof.baselines.noise_experiments import ROOT,REMOTE,OUT,PYSPI

COMPUTE='/scratch/ql44/we2614/mts-spi-study/operations/sources/noise-experiments-0de0330'
ANALYSIS='/scratch/ql44/we2614/mts-spi-study/operations/sources/noise-analysis-a3bfcc3'

def remote_python(code):
    return subprocess.check_output(['ssh','-o','BatchMode=yes','we2614@gadi.nci.org.au','python3 -'],input=code,text=True)

def snapshot():
    return json.loads(remote_python(f'''import json,subprocess
from pathlib import Path
r=Path({REMOTE!r});jobs=json.loads((r/'jobs.json').read_text());p=r/'extraction-job.txt'
extract=p.read_text().strip() if p.exists() else None
ids=[j['job'] for j in jobs]+([extract] if extract else [])
a=json.loads(subprocess.check_output(['qstat','-xf','-F','json',*ids],text=True))['Jobs']
print(json.dumps(dict(jobs=jobs,extraction=extract,status={{k:{{field:v.get(field) for field in ['job_state','Exit_status','resources_used','comment']}} for k,v in a.items()}})))
'''))

def main():
    last=None
    while True:
        try:s=snapshot()
        except subprocess.CalledProcessError:
            print('Transient status failure; retrying.',flush=True);time.sleep(55);continue
        state=[(k,v['job_state'],v['Exit_status']) for k,v in s['status'].items()]
        if state!=last:print(state,flush=True);last=state
        for job,status in s['status'].items():
            if status['job_state']=='F' and status['Exit_status']!=0:raise RuntimeError((job,status))
        pilots=s['jobs'];ready=any(j['role']=='pilot' and s['status'][j['job']]['job_state']=='F' and not any(p['role']=='production' and p['experiment']==j['experiment'] for p in pilots) for j in pilots)
        if ready:
            subprocess.run(['ssh','-o','BatchMode=yes','we2614@gadi.nci.org.au',f'cd {COMPUTE} && source jobs/gadi/env.sh && python -m research.proof.baselines.run_noise_experiments production'],check=True)
            continue
        production=[j for j in pilots if j['role']=='production']
        if len(production)==2 and not s['extraction']:
            print(remote_python(f'''import json,subprocess,os
from pathlib import Path
r=Path({REMOTE!r});p=r/'extraction-job.txt'
if p.exists():print(p.read_text())
else:
 jobs=[j['job'] for j in json.loads((r/'jobs.json').read_text()) if j['role']=='production'];assert len(jobs)==2
 os.chdir({ANALYSIS!r})
 out=subprocess.check_output(['qsub','-W','depend=afterok:'+':'.join(jobs),'-v','PYSPI_SOURCE={PYSPI}','research/proof/baselines/extract_noise_experiments.pbs'],text=True)
 p.write_text(out);print(out)
'''),flush=True)
            continue
        if s['extraction'] and s['status'][s['extraction']]['job_state']=='F':break
        time.sleep(55)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'cluster-completion.json').write_text(json.dumps(s,indent=2)+'\n')
    subprocess.run(['rsync','-az','we2614@gadi-dm.nci.org.au:'+REMOTE+'/analysis/',str(OUT)+'/'],check=True)
    for module,args in [('analyze_noise_experiments',['analyze']),('report_noise_experiments',['plots'])]:
        print('Running',module,flush=True)
        subprocess.run([str(ROOT/'.venv/bin/python'),'-m','research.proof.baselines.'+module,*args],cwd=ROOT,check=True)
    print('PLOTS READY: inspect, interpret, append and commit.',flush=True)

if __name__=='__main__':main()
