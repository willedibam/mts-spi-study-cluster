"""Independent alignment/metric checks and serialized target prediction replay."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from src.neurotycho_library_baselines import predict_model

ROOT=Path('results/neurotycho_library_followup_260911')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y,p):
    p0,p1=p[y==0],p[y==1]
    delta=p1[:,None]-p0[None,:]
    return dict(balanced_accuracy=float(((p0<.5).mean()+(p1>=.5).mean())/2),
        auroc=float(((delta>0)+.5*(delta==0)).mean()),
        balanced_brier=float(((p0**2).mean()+((p1-1)**2).mean())/2))


def main():
    report=json.loads((ROOT/'report/report.json').read_text())
    assert report['post_PF_inspection_followup'] and not report['target_calibration'] and report['threshold']==.5
    assert sha(ROOT/'report/predictions.npz')==report['prediction_sha256']
    original=Path('results/neurotycho_target_pilot_260910/evaluation/report.json')
    assert sha(original)=='d7d697f9c716ee0fe4bcbe7ca9f1aa1ad9e33725f2195a5fe08699cfc1daa402'
    with np.load(ROOT/'report/predictions.npz') as d:p={k:d[k] for k in d.files}
    assert len(p['y'])==640 and len(set(zip(p['method'],p['seed'],p['record_id'])))==640
    with np.load(ROOT/'source/bank.npz') as d:source={k:d[k] for k in ['record_id','animal','y']}
    with np.load(ROOT/'evaluation/bank.npz') as d:target={k:d[k] for k in ['record_id','animal','archive','y','x','tsfresh','catch22']}
    labels={}
    for path in Path('results/neurotycho_target_pilot_260910').glob('201*.json'):
        for r in json.loads(path.read_text())['records']:
            if r['quality']['accepted']:
                rid=f'{r["archive"]}/{r["session"]}/{r["state"]}/{r["start"]}'
                labels[rid]=(r['animal'],r['archive'],r['target'])
    for i,rid in enumerate(p['record_id']):
        assert labels[rid]==(p['animal'][i],p['archive'][i],p['y'][i])
    maximum=0.;replay_max=0.;source_cv_max=0.
    lookup={r:i for i,r in enumerate(source['record_id'])}
    assert len(report['models'])==10 and len(report['scores'])==5
    for audit in report['models']:
        path=ROOT/'fits'/audit['model'];fit=json.loads(path.read_text());excluded=fit['identity']['animal']
        assert sha(path)==audit['report_sha256'] and sha(path.with_suffix('.joblib'))==audit['model_sha256']
        assert not fit['target_data_used']
        assert set(fit['training_ids'])==set(source['record_id'][source['animal']!=excluded])
        for candidate in fit['candidates']:
            values=[]
            for fold in candidate['folds']:
                group=fold['animal']
                expected_train=set(source['record_id'][(source['animal']!=group)&(source['animal']!=excluded)])
                expected_valid=set(source['record_id'][source['animal']==group])
                assert set(fold['training_ids'])==expected_train and set(fold['validation_ids'])==expected_valid
                np.testing.assert_array_equal(fold['y'],source['y'][[lookup[r] for r in fold['validation_ids']]])
                value=metrics(np.asarray(fold['y']),np.asarray(fold['probability']))['balanced_brier']
                source_cv_max=max(source_cv_max,abs(value-fold['brier']));values.append(value)
            source_cv_max=max(source_cv_max,abs(np.mean(values)-candidate['mean_brier']))
        selected=min(fit['candidates'],key=lambda c:c['mean_brier'])
        assert all(selected[k]==fit['selected'][k] for k in ['C','layout','mean_brier'])
        ix=target['animal']==excluded;tb={k:v[ix] for k,v in target.items()}
        replay=predict_model(joblib.load(path.with_suffix('.joblib')),tb)
        pix=(p['method']==fit['identity']['method'])&(p['seed']==fit['identity']['seed'])&(p['animal']==excluded)
        np.testing.assert_array_equal(tb['record_id'],p['record_id'][pix])
        replay_max=max(replay_max,float(np.max(np.abs(replay-p['probability'][pix]))))
    for score in report['scores']:
        ix=(p['method']==score['method'])&(p['seed']==score['seed'])
        assert ix.sum()==128 and set(p['record_id'][ix])==set(labels)
        animals=[]
        for animal in score['animals']:
            ds=[]
            for date in [d for d in score['dates'] if d['animal']==animal['animal']]:
                jx=ix&(p['archive']==date['archive']);assert jx.sum()==32
                computed=metrics(p['y'][jx],p['probability'][jx]);ds.append(computed)
                maximum=max(maximum,max(abs(computed[k]-date[k]) for k in computed))
            mean={k:float(np.mean([d[k] for d in ds])) for k in computed};animals.append(mean)
            maximum=max(maximum,max(abs(mean[k]-animal[k]) for k in mean))
        maximum=max(maximum,max(abs(np.mean([a[k] for a in animals])-score['mean'][k]) for k in mean))
    assert maximum<1e-12 and source_cv_max<1e-12 and replay_max<1e-12
    result=dict(status='passed',models=10,target_predictions=640,metric_max_difference=maximum,
        source_cv_max_difference=source_cv_max,target_serialized_replay_max_difference=replay_max,
        original_report_unchanged=True,report_sha256=sha(ROOT/'report/report.json'))
    (ROOT/'report/verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
