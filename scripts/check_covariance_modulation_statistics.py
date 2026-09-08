"""Replay every statistical method in both processes, including the rank-one fold."""
import json
from pathlib import Path
import numpy as np
import yaml
from src.interaction_share_learning import fit_statistical, select_statistical, standardized_marginal_shapes
from src.representation_state_data import file_hash


def main():
    root=Path('results/covariance_modulation_260909')
    data=Path('data/covariance_modulation_260909')
    config=Path('configs/analysis/covariance-modulation-260909.yaml')
    cfg=yaml.safe_load(config.read_text())
    rows=json.loads((data/'manifest.json').read_text())['rows']
    target=np.array([r['target'] for r in rows]);strata=np.array([r['coupling_index'] for r in rows])
    path=root/'gadi-analysis/features.npz'
    assert file_hash(path)==json.loads(path.with_suffix('.json').read_text())['artifact_sha256']
    with np.load(path,allow_pickle=False) as a:bank={k:a['X_'+k] for k in ['m','g','z','validity']}
    shapes=standardized_marginal_shapes(bank['m'],bank['validity'])
    fits=sorted((root/'gadi-analysis/statistical-rank-fixed').glob('*/*.json'))
    assert len(fits)==144
    checks=[]
    for p in fits:
        info=json.loads(p.read_text());ident=info['identity']
        # All methods at40labels in both processes; also the previously failing fold.
        if not (ident['seed']==11 and info['labels_total']==40) and not (
            ident['method']=='validity-pls' and ident['seed']==47 and info['labels_total']==10):continue
        assert ident['manifest_sha256']==file_hash(data/'manifest.json')
        assert ident['feature_bank_sha256']==file_hash(path)
        assert info['predictions_sha256']==file_hash(p.with_suffix('.npz'))
        with np.load(p.with_suffix('.npz'),allow_pickle=False) as a:
            train,evaluation,previous=a['train_indices'],a['evaluation_indices'],a['prediction']
        assert not {rows[i]['master_id'] for i in train}&{rows[i]['master_id'] for i in evaluation}
        view,head=ident['method'].rsplit('-',1);active=bank
        if 'shape' in view:active={**bank,'m':shapes};view=view.replace('shape','m')
        chosen,details=select_statistical(active,view,train,target,strata,cfg['methods'],ident['seed'],head)
        assert list(chosen)==info['details']['chosen']
        cv_error=max(abs(x-y) for a,b in zip(details['candidates'],info['details']['candidates'],strict=True)
                     for x,y in zip(a['MAE'],b['MAE'],strict=True))
        transform,model=fit_statistical(active,view,train,target,cfg['methods']['preprocessing'],head,*chosen)
        prediction=np.clip(model.predict(transform.transform(active,evaluation)).reshape(-1),0,1)
        error=float(abs(prediction-previous).max())
        assert error<1e-9 and cv_error<1e-9,(p,error,cv_error)
        checks.append(dict(fit=str(p),prediction_difference=error,CV_difference=cv_error))
    assert len(checks)==18
    result=dict(status='passed',checks=checks,checker_sha256=file_hash(Path(__file__)))
    (root/'statistical-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(dict(fits=len(checks),maximum_prediction_difference=max(r['prediction_difference'] for r in checks),
               maximum_CV_difference=max(r['CV_difference'] for r in checks)))


if __name__=='__main__':main()
