"""Metadata-only HCP block/QC audit; no predictive outcomes or final eligibility."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat


def scalar(text, field):
    match=re.search(re.escape(field)+r'\s*=\s*([\d.]+)',text)
    return float(match[1]) if match else None


def audit(root):
    manifest=json.loads((root/'cohort-qc-downloads.json').read_text())
    groups=defaultdict(dict)
    for row in manifest['rows']:
        path=Path(row['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        groups[(row['subject'],row['run'])][row['kind']]=path
    runs,errors=[],[]
    for (subject,run),files in sorted(groups.items()):
        try:
            assert len(files)==5
            trl=loadmat(files['tmegpreproc_trialinfo.mat'],simplify_cells=True)['trlInfo']
            locks=list(np.atleast_1d(trl['lockNames']))
            data=np.asarray(trl['lockTrl'][locks.index('TIM')])
            task=data[np.isin(data[:,3],[1,2])&np.isin(data[:,4],[1,2])&np.isin(data[:,8],np.arange(1,11))]
            raw_qc=files['baddata_rawtrialinfo_QC.txt'].read_text()
            text=files['baddata_badsegments.txt'].read_text()
            match=re.search(r'badsegment\.all\s*=\s*\[([^\]]*)\]',text)
            assert match is not None
            segments=np.asarray([int(v) for v in re.findall(r'\d+',match[1])],dtype=int).reshape(-1,2)
            badnames=sorted(set(re.findall(r'A\d+',files['baddata_badchannels.txt'].read_text())))
            blocks=[]
            for block in sorted(np.unique(task[:,1])):
                rows=task[task[:,1]==block]
                conditions=np.unique(rows[:,[3,4]],axis=0)
                start,stop=int(rows[:,6].min()),int(rows[:,7].max())
                overlap=sum(max(0,min(stop,b)-max(start,a)+1) for a,b in segments)
                complete=len(rows)==10 and set(rows[:,8])==set(range(1,11))
                blocks.append(dict(block=int(block),trials=len(rows),complete_ten_trial_sequence=complete,
                    consistent_condition=len(conditions)==1,
                    conditions=conditions.astype(int).tolist(),start_sample_one_based=start,stop_sample_inclusive=stop,
                    marked_bad_overlap_samples=int(overlap),clean_complete=bool(complete and len(conditions)==1 and overlap==0)))
            conditions=Counter(f"image{b['conditions'][0][0]}_memory{b['conditions'][0][1]}" for b in blocks if b['clean_complete'])
            annotation=files['icaclass_vs.txt'].read_text()
            raw_trials=scalar(raw_qc,'trialSummary.Ntrials')
            runs.append(dict(subject=subject,run=run,retained_task_rows=len(task),raw_qc_trials=raw_trials,
                retained_trial_difference=None if raw_trials is None else raw_trials-len(task),task_blocks=len(blocks),
                clean_complete_blocks=sum(b['clean_complete'] for b in blocks),clean_complete_condition_counts=dict(conditions),
                all_four_conditions_present=len(conditions)==4,marked_bad_sensor_count=len(badnames),
                ica_total=scalar(annotation,'vs.total_ic_number'),ica_brain=scalar(annotation,'vs.brain_ic_number'),
                ica_visual_flag=scalar(annotation,'vs.flag'),blocks=blocks))
        except Exception as exc:
            errors.append(dict(subject=subject,run=run,error_type=type(exc).__name__,detail=str(exc)[:200]))
    subjects=defaultdict(list)
    for row in runs:subjects[row['subject']].append(row)
    summary=dict(runs_parsed=len(runs),participants_parsed=len(subjects),raw_qc_trials=sum(r['raw_qc_trials'] or 0 for r in runs),
                 retained_task_rows=sum(r['retained_task_rows'] for r in runs),task_blocks=sum(r['task_blocks'] for r in runs),
                 clean_complete_blocks=sum(r['clean_complete_blocks'] for r in runs),
                 participants_with_both_runs_all_four_clean_conditions=sum(len(v)==2 and all(r['all_four_conditions_present'] for r in v) for v in subjects.values()),
                 runs_missing_retained_trials=sum((r['retained_trial_difference'] or 0)>0 for r in runs),
                 visual_ica_flags=dict(Counter(str(r['ica_visual_flag']) for r in runs)),
                 clean_complete_blocks_per_run_range=[min(r['clean_complete_blocks'] for r in runs),max(r['clean_complete_blocks'] for r in runs)])
    result=dict(status='complete' if not errors else 'partial',summary=summary,runs=runs,errors=errors,
                manifest_sha256=hashlib.sha256((root/'cohort-qc-downloads.json').read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                limitations='Metadata-only conservative whole-block audit. No raw headers, waveform QC, sensor geometry, final window exclusions or family independence established. Do not turn clean_complete into final eligibility without the preprocessing protocol.')
    (root/'cohort-qc-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],summary=summary,errors=errors),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    audit(parser.parse_args().root)
