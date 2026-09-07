"""Replay every confirmation statistical pipeline for one cohort per family."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src.interaction_share_learning import fit_statistical, select_statistical, standardized_marginal_shapes
from src.representation_state_data import file_hash, load_state_data


def check(config,data,analysis,output):
    p=yaml.safe_load(config.read_text());manifest,_=load_state_data(data,config)
    rows=manifest['rows'];target=np.asarray([r['target'] for r in rows])
    strata=np.asarray([r['coupling_index'] for r in rows])
    meta=json.loads((analysis/'features.json').read_text())
    assert file_hash(analysis/'features.npz')==meta['artifact_sha256']
    with np.load(analysis/'features.npz') as archive:
        bank={name:archive['X_'+name] for name in ['m','z','validity']}
    shapes={**bank,'m':standardized_marginal_shapes(bank['m'],bank['validity'])}
    seed=p['methods']['subset_seeds'][0]
    n=max(p['sampling']['labelled_training_masters_per_coupling'])
    checks=[]
    for family in p['generator']['families']:
        for directory,method,view,head,active in [
            ('statistical','m-pls','m','pls',bank),
            ('statistical','z-pca','z','pca',bank),
            ('statistical','z-pls','z','pls',bank),
            ('shape','shape-pls','m','pls',shapes),
            ('shape','shape+z-pls','m+z','pls',shapes)]:
            stem=analysis/directory/family/f'{method.replace("+","_")}-n{n}-s{seed}'
            record=json.loads(stem.with_suffix('.json').read_text())
            assert record['predictions_sha256']==file_hash(stem.with_suffix('.npz'))
            with np.load(stem.with_suffix('.npz')) as a:
                train,evaluation,old=a['train_indices'],a['evaluation_indices'],a['prediction']
            chosen,details=select_statistical(active,view,train,target,strata,p['methods'],seed,head)
            assert list(chosen)==record['details']['chosen']
            candidate_difference=max(abs(x-y) for new,previous in zip(details['candidates'],record['details']['candidates'],strict=True)
                                     for x,y in zip(new['MAE'],previous['MAE'],strict=True))
            transform,model=fit_statistical(active,view,train,target,p['methods']['preprocessing'],head,*chosen)
            prediction=np.clip(model.predict(transform.transform(active,evaluation)).reshape(-1),0,1)
            np.testing.assert_allclose(prediction,old,atol=1e-10,rtol=0)
            assert candidate_difference<1e-10
            checks.append(dict(source_family=family,method=method,chosen=chosen,
                max_candidate_MAE_difference=candidate_difference,
                max_prediction_difference=float(abs(prediction-old).max())))
    result=dict(status='passed',bank_sha256=meta['artifact_sha256'],seed=seed,
        total_labels=n*len(p['generator']['nominal_shares']),checks=checks,
        verifier_sha256=file_hash(Path(__file__)))
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['config','data','analysis','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();check(args.config,args.data,args.analysis,args.output)
