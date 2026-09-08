"""Matched binary pilot metrics and conditional paired accuracy/Brier contrasts."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
from src.representation_state_data import file_hash


def main(data,inputs,output):
    rows=json.loads((data/'manifest.json').read_text())['rows'];manifest_hash=file_hash(data/'manifest.json')
    scores=[];losses=[];training={};identity=[]
    for item in inputs:
        label,directory=item.split('=',1) if '=' in item else ('',item)
        for path in sorted(Path(directory).rglob('*.json')):
            info=json.loads(path.read_text())
            if 'identity' not in info or 'labels_total' not in info:continue
            ident=info['identity'];name=label or ident['method'];seed=ident['seed'];n=info['labels_total']
            assert ident['manifest_sha256']==manifest_hash and info['predictions_sha256']==file_hash(path.with_suffix('.npz'))
            identity.append(dict(fit=str(path),sha256=file_hash(path)))
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as a:
                train=a['train_indices'];evaluation=a['evaluation_indices'];target=a['target'];prediction=a['prediction']
                assert training.setdefault((seed,n),tuple(train))==tuple(train)
                assert not {rows[i]['master_id'] for i in train}&{rows[i]['master_id'] for i in evaluation}
                np.testing.assert_array_equal(target,[rows[i]['target'] for i in evaluation])
                np.testing.assert_array_equal(a['row_id'],[rows[i]['row_id'] for i in evaluation])
                for m in [16,8]:
                    mask=np.array([rows[i]['M']==m for i in evaluation]);truth=target[mask];pred=prediction[mask]
                    scores.append(dict(method=name,labels=n,seed=seed,M=m,
                        balanced_accuracy=float(balanced_accuracy_score(truth,pred>=.5)),AUROC=float(roc_auc_score(truth,pred)),
                        Brier=float(np.mean((truth-pred)**2)),MAE=float(np.mean(abs(truth-pred)))))
                    for pos in np.flatnonzero(mask):
                        losses.append(dict(method=name,labels=n,seed=seed,M=m,master=rows[evaluation[pos]]['master_id'],
                            target=int(target[pos]),error=float((prediction[pos]>=.5)!=target[pos]),Brier=float((prediction[pos]-target[pos])**2)))
    frame=pd.DataFrame(scores);assert not frame.duplicated(['method','labels','seed','M']).any()
    summary=frame.groupby(['M','method','labels']).agg(fits=('seed','size'),balanced_accuracy=('balanced_accuracy','mean'),
        AUROC=('AUROC','mean'),Brier=('Brier','mean'),MAE=('MAE','mean'),cohort_BA_SD=('balanced_accuracy','std')).reset_index()
    output.mkdir(parents=True,exist_ok=True);frame.to_csv(output/'per-fit.csv',index=False);summary.to_csv(output/'summary.csv',index=False)
    all_losses=pd.DataFrame(losses);comparisons=[]
    pairs=[('z-pls','m-pls'),('z-pls','shape-pls'),('m+z-pls','m-pls'),('shape+z-pls','shape-pls'),
           ('z-pls','neural-aligned'),('z-pls','neural-pair'),('z-pls','raw:agreement-pls')]
    for m in [16,8]:
        for n in [10,20,40]:
            subset=all_losses[(all_losses.M==m)&(all_losses.labels==n)]
            for metric in ['error','Brier']:
                wide=subset.pivot(index=['seed','master','target'],columns='method',values=metric)
                for left,right in pairs:
                    if left not in wide or right not in wide:continue
                    diff=(wide[left]-wide[right]).dropna().groupby(['master','target']).mean()
                    if not len(diff):continue
                    rng=np.random.default_rng(260909223);means=[]
                    for target in [0,1]:
                        values=diff.xs(target,level='target').to_numpy()
                        means.append(values[rng.integers(len(values),size=(2000,len(values)))].mean(1))
                    boot=np.mean(means,axis=0);sign=-1 if metric=='error' else 1
                    lo,hi=np.quantile(sign*boot,[.025,.975]);comparisons.append(dict(M=m,labels=n,
                        metric='balanced_accuracy' if metric=='error' else metric,contrast=left+' minus '+right,
                        mean=float(sign*diff.mean()),low=float(lo),high=float(hi)))
    (output/'contrasts.json').write_text(json.dumps(dict(comparisons=comparisons,
        interval='pointwise95%; resample independent test masters within class; conditional on matched fitted cohorts; no multiplicity correction',
        inputs=identity,reporter_sha256=file_hash(Path(__file__))),indent=2)+'\n')
    print(summary[summary.M==8].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.data,a.inputs,a.output)
