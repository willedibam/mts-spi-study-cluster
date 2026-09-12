import json,hashlib,subprocess
from pathlib import Path
import numpy as np
r=Path('results/minirocket_synthetic_followup_260912');data=Path('results/inceptiontime_followup_260911');report=json.loads((r/'evaluation/report.json').read_text());manifest=json.loads((data/'manifest.json').read_text());source=np.load(data/'source-synthetic.npz');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
original_code=hashlib.sha256(subprocess.check_output(['git','show','32fbb41:scripts/run_minirocket_synthetic_followup.py'])).hexdigest();assert report['identity']['script_sha256']==original_code
assert report['identity']['protocol_sha256']==sha(Path('docs/minirocket-synthetic-followup.md'))
assert report['identity']['source_sha256']==sha(data/'source-synthetic.npz')
for stem,digest in report['frozen'].items():
 p=r/'fits'/stem;assert sha(p.with_suffix('.json'))==digest
 f=json.loads(p.with_suffix('.json').read_text());assert f['identity']==report['identity'];assert sha(p.with_suffix('.joblib'))==f['model_sha256'];assert sha(p.with_suffix('.npz'))==f['predictions_sha256']
 case=next(c for c in manifest['cases'] if c['name']==f['case']['name']);assert f['case']==case;assert not f['target_data_used'];ix=case['train'];assert f['training_ids']==source['record_id'][ix].tolist()
 with np.load(p.with_suffix('.npz')) as a:np.testing.assert_array_equal(a['y'],source['y'][ix]);np.testing.assert_array_equal(a['record_id'],source['record_id'][ix])
for row in report['scores']:
 p=r/'evaluation'/row['prediction_file'];assert sha(p)==report['prediction_hashes'][p.name]
 with np.load(p) as a,np.load(data/f'target-{row["dataset"]}.npz') as target:
  np.testing.assert_array_equal(a['record_id'],target['record_id']);np.testing.assert_array_equal(a['y'],target['y']);assert not set(target['master_id'])&set(source['master_id'])
  y=a['y'];s=a['score'];diff=s[y==1][:,None]-s[y==0][None,:];auc=float(np.mean((diff>0)+.5*(diff==0)));ba=float(((s[y==1]>0).mean()+(s[y==0]<=0).mean())/2)
  assert abs(auc-row['auroc'])<1e-12 and abs(ba-row['balanced_accuracy'])<1e-12
for row in report['summary']:
 xs=[x for x in report['scores'] if x['dataset']==row['dataset'] and x['labels']==row['labels']];assert len(xs)==9
 for key in ['balanced_accuracy','auroc']:assert abs(row[key]-np.mean([x[key] for x in xs]))<1e-12
assert len(report['scores'])==81 and len(report['frozen'])==27
result=dict(status='verified',source_models=27,target_metric_rows=81,report_sha256=sha(r/'evaluation/report.json'),exact_pinned_source_code=True,maximum_batch_replay_error=report['maximum_replay_error'])
(r/'evaluation/independent-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result);print(report['summary'])
