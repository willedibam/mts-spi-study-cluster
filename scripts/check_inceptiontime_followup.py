"""Independent split/metric audit and sampled CPU checkpoint prediction replay."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
import torch
import yaml

from src.inceptiontime_baseline import InceptionNetwork


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with np.load(path) as data:
        return {k:data[k] for k in data.files}


def independent_metrics(y,p):
    positive,negative=y==1,y==0
    n1,n0=positive.sum(),negative.sum()
    return dict(balanced_accuracy=float(((p[positive]>=.5).mean()+(p[negative]<.5).mean())/2),
        auroc=float((rankdata(p)[positive].sum()-n1*(n1+1)/2)/(n1*n0)),
        brier=float(np.square(y-p).mean()))


def replay(model,x):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x)).sigmoid().numpy()


def main(root):
    torch.set_num_threads(2)
    config_path=Path('configs/analysis/inceptiontime-followup-260911.yaml')
    config=yaml.safe_load(config_path.read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    report=json.loads((root/'evaluation/report.json').read_text())
    assert manifest['config_sha256']==report['config_sha256']==sha(config_path)
    assert report['manifest_sha256']==sha(root/'manifest.json')
    for name,digest in manifest['artifacts'].items():
        assert sha(root/name)==digest
    for name,digest in report['prediction_hashes'].items():
        assert sha(root/'evaluation'/name)==digest
    errors=[];cap_counts=0;fit_count=0;metrics_checked=0
    for case in manifest['cases']:
        source=load(root/case['source']);train=np.asarray(case['train'])
        if case['domain']=='neurotycho':
            assert case['excluded_animal'] not in source['animal'][train]
        models=[]
        for seed in config['member_seeds']:
            stem=root/'fits'/case['name']/f'member-{seed}'
            fitted=json.loads(stem.with_suffix('.json').read_text());saved=load(stem.with_suffix('.npz'))
            assert fitted['target_data_used'] is False
            assert sha(stem.with_suffix('.pt'))==fitted['checkpoint_sha256']
            assert sha(stem.with_suffix('.npz'))==fitted['predictions_sha256']
            assert fitted['identity']['source_sha256']==manifest['artifacts'][case['source']]
            assert fitted['identity']['case']==case
            np.testing.assert_array_equal(saved['training_ids'],source['record_id'][train])
            np.testing.assert_array_equal(saved['y'],source['y'][train])
            for candidate in fitted['candidates']:
                values=[]
                for expected,fold in zip(case['folds'],candidate['folds'],strict=True):
                    assert not set(expected['fit'])&set(expected['validation'])
                    assert set(expected['fit'])|set(expected['validation'])==set(train)
                    for key,field in [('fit','training_ids'),('validation','validation_ids')]:
                        assert fold[field]==source['record_id'][expected[key]].tolist()
                    y=np.asarray(fold['validation_y']);p=np.asarray(fold['validation_probability'])
                    np.testing.assert_array_equal(y,source['y'][expected['validation']])
                    brier=float(np.mean([np.square(p[y==c]-c).mean() for c in [0,1]]))
                    np.testing.assert_allclose(brier,fold['validation_brier'],atol=1e-7,rtol=0)
                    values.append(fold['validation_brier'])
                np.testing.assert_allclose(np.mean(values),candidate['mean_brier'],atol=1e-12,rtol=0)
            selected=min(fitted['candidates'],key=lambda c:c['mean_brier'])
            assert (selected['lr'],selected['decay'])==(fitted['selected_lr'],fitted['selected_decay'])
            assert fitted['selected_epochs']==max(1,int(np.median([f['best_epoch'] for f in selected['folds']])))
            cap_counts+=sum(f['selected_epoch_at_ceiling'] for f in selected['folds'])
            checkpoint=torch.load(stem.with_suffix('.pt'),map_location='cpu',weights_only=True)
            model=InceptionNetwork(checkpoint['spec']);model.load_state_dict(checkpoint['state_dict'])
            sample=np.concatenate([np.flatnonzero(saved['y']==c)[:4] for c in [0,1]])
            error=float(np.abs(replay(model,source['x'][train[sample]])-saved['probability'][sample]).max())
            assert error<1e-4
            errors.append(error);models.append(model);fit_count+=1
        datasets=config['evaluation']['synthetic'] if case['domain']=='synthetic' else ['neurotycho']
        for dataset in datasets:
            target=load(root/f'target-{dataset}.npz')
            if case['domain']=='neurotycho':
                ix=target['animal']==case['excluded_animal'];target={k:v[ix] for k,v in target.items()}
            else:
                assert not set(source['master_id'][train])&set(target['master_id'])
            saved=load(root/'evaluation'/f'{case["name"]}-{dataset}.npz')
            np.testing.assert_array_equal(saved['record_id'],target['record_id'])
            np.testing.assert_array_equal(saved['y'],target['y'])
            np.testing.assert_allclose(saved['ensemble'],saved['probability'].mean(0),rtol=0,atol=0)
            sample=np.concatenate([np.flatnonzero(saved['y']==c)[:4] for c in [0,1]])
            for model,p in zip(models,saved['probability'],strict=True):
                error=float(np.abs(replay(model,target['x'][sample])-p[sample]).max())
                assert error<1e-4
                errors.append(error)
            for member,p in [*zip(config['member_seeds'],saved['probability']),('ensemble',saved['ensemble'])]:
                row=next(r for r in report['scores'] if (r['case'],r['dataset'],r['member'])==(case['name'],dataset,member))
                scores=[independent_metrics(saved['y'][saved['archive']==g],p[saved['archive']==g]) for g in np.unique(saved['archive'])]
                for metric in ['balanced_accuracy','auroc','brier']:
                    np.testing.assert_allclose(np.mean([s[metric] for s in scores]),row[metric],rtol=0,atol=1e-7)
                metrics_checked+=1
    assert fit_count==55 and metrics_checked==174
    result=dict(report_sha256=sha(root/'evaluation/report.json'),source_models=fit_count,
        metric_rows=metrics_checked,selected_fold_epoch_caps=cap_counts,
        sampled_cpu_replays=len(errors),maximum_replay_error=max(errors),status='verified')
    (root/'evaluation/verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/inceptiontime_followup_260911'))
    main(parser.parse_args().root)
