"""Report all available fits, preserving observation and state-process scopes."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def report(data,inputs,output):
    manifest=json.loads((data/'manifest.json').read_text());rows=manifest['rows']
    fit_rows=[];losses=[]
    for item in inputs:
        label,path=item.split('=',1) if '=' in item else ('',item)
        for p in sorted(Path(path).rglob('*.json')):
            info=json.loads(p.read_text())
            if 'identity' not in info or 'labels_total' not in info:continue
            ident=info['identity'];method=label or ident['method'];family=ident['source_family'];seed=ident['seed']
            with np.load(p.with_suffix('.npz'),allow_pickle=False) as a:
                indices=a['evaluation_indices'];target=a['target'];prediction=a['prediction']
                np.testing.assert_array_equal(a['row_id'],[rows[i]['row_id'] for i in indices])
                np.testing.assert_allclose(target,[rows[i]['target'] for i in indices],atol=0,rtol=0)
                for source in [True,False]:
                    for m in [16,8]:
                        scope=('same' if source else 'other')+('-full' if m==16 else '-reduced')
                        mask=np.asarray([(rows[i]['family']==family)==source and rows[i]['M']==m for i in indices])
                        error=np.abs(prediction-target)
                        fit_rows.append(dict(method=method,labels=info['labels_total'],source=family,seed=seed,
                                             scope=scope,MAE=float(error[mask].mean()),evaluation_records=int(mask.sum())))
                        for pos in np.flatnonzero(mask):
                            r=rows[indices[pos]]
                            losses.append(dict(method=method,labels=info['labels_total'],source=family,seed=seed,scope=scope,
                                               master=r['master_id'],evaluation_family=r['family'],error=float(error[pos])))
    frame=pd.DataFrame(fit_rows)
    if frame.empty:raise ValueError('No complete fits')
    assert not frame.duplicated(['method','labels','source','seed','scope']).any()
    output.mkdir(parents=True,exist_ok=True)
    frame.to_csv(output/'per-fit.csv',index=False)
    summary=frame.groupby(['scope','method','labels']).agg(MAE=('MAE','mean'),fits=('MAE','size'),
                                                        cohort_SD=('MAE','std')).reset_index()
    summary.to_csv(output/'summary.csv',index=False)
    loss=pd.DataFrame(losses);comparisons=[]
    for scope in ['same-full','same-reduced','other-full','other-reduced']:
        for n in sorted(loss['labels'].unique()):
            subset=loss[(loss.scope==scope)&(loss.labels==n)]
            wide=subset.pivot(index=['source','seed','master','evaluation_family'],columns='method',values='error')
            if 'z-pls' not in wide:continue
            for other in ['shape-pls','m-pls','moment-calibrated','raw:covariance+cumulant+window-pls','neural-pair','neural-aligned']:
                if other not in wide:continue
                delta=(wide['z-pls']-wide[other]).dropna()
                per_master=delta.groupby(['master','evaluation_family']).mean()
                rng=np.random.default_rng(260909)
                boot=[]
                for fam in ['persistent','iid']:
                    values=per_master.xs(fam,level='evaluation_family').to_numpy()
                    boot.append(values[rng.integers(len(values),size=(2000,len(values)))].mean(1))
                lo,hi=np.quantile(np.mean(boot,axis=0),[.025,.975])
                comparisons.append(dict(scope=scope,labels=int(n),contrast='z-pls minus '+other,
                                        mean=float(per_master.mean()),low=float(lo),high=float(hi),
                                        matched_fit_predictions=len(delta),independent_evaluation_masters=len(per_master)))
    (output/'paired-contrasts.json').write_text(json.dumps(dict(comparisons=comparisons,
        interval='pointwise95%, resample independent test masters within process; condition on fitted models'),indent=2)+'\n')
    print(summary[summary.scope=='same-reduced'].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report(a.data,a.inputs,a.output)
