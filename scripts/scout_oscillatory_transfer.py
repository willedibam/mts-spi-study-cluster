"""Prespecified raw-only feasibility gate for one faster dynamics regime."""
import argparse,json
from pathlib import Path
import numpy as np,yaml
from sklearn.metrics import roc_auc_score
from src.oscillatory_coorganization import simulate,observed_controls
from src.representation_state_data import observed_view,file_hash


def main(config,output):
    if output.exists():raise FileExistsError(output)
    cfg=yaml.safe_load(config.read_text());s=cfg['transfer'];checks=[];scores={16:[],8:[]};truth=[]
    for condition in [0,1]:
        for replicate in range(s['scout_per_class']):
            seed=[s['scout_seed'],condition,replicate]
            x,meta=simulate(bool(condition),seed,ranges=s['ranges']);truth.append(condition)
            record=dict(seed=seed,target=condition,parameters=meta,controls={})
            for m,t in [(16,1000),(8,500)]:
                c=observed_controls(observed_view(x,m,t));values=c['direct_agreement']
                assert np.isfinite(values).all();scores[m].append(float(values[0]));record['controls'][str(m)]=values.tolist()
            checks.append(record)
    auc={str(m):float(roc_auc_score(truth,values)) for m,values in scores.items()}
    output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(passed=all(v>=s['minimum_agreement_AUROC'] for v in auc.values()),AUROC=auc,
        config_sha256=file_hash(config),code_sha256={p:file_hash(Path(p)) for p in [__file__,'src/oscillatory_coorganization.py']},records=checks)
    output.write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='records'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.config,a.output)
