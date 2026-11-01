"""Audit saved discovery predictions and isolate fusion from readout retuning.

The new no-z fits are diagnostics at the already selected fusion settings.
They do not replace the original comparators or select a new model on targets.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone

from mtsspi.provenance import atomic_json, file_sha256
from research.representation.dataset_screen.common import weights, probabilities
from research.representation.dataset_screen.statistical import load_bank, feature_bank, predict


def grouped_scores(y, p, groups):
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    if p.shape[0] != len(y) or p.ndim != 2 or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1):
        raise ValueError('invalid saved probabilities')
    labels = set(y)
    losses = dict(brier=.5 * np.sum((p - np.eye(p.shape[1])[y]) ** 2, axis=1),
                  balanced_accuracy=(p.argmax(1) == y).astype(float),
                  log_loss=-np.log(np.maximum(p[np.arange(len(y)), y], 1e-15)))
    scores, strata = {}, []
    for group in np.unique(groups):
        group_labels = set(y[groups == group])
        if group_labels != labels and len(group_labels) != 1:
            raise ValueError('unsupported incomplete class coverage in a group')
        strata.append(str(next(iter(group_labels))) if len(group_labels) == 1 else 'all')
        scores[str(group)] = {key: float(np.mean([value[(groups == group) & (y == label)].mean()
                            for label in sorted(group_labels)])) for key,value in losses.items()}
    if 'all' in strata and set(strata) != {'all'}:
        raise ValueError('mixed single-class and complete-class group types')
    return scores, np.array(strata)


def aggregate(values, strata):
    return float(np.mean([np.asarray(values)[strata == label].mean() for label in np.unique(strata)]))


def bootstrap(values, strata, repetitions=10000):
    rng = np.random.default_rng(261102)
    result = np.zeros(repetitions)
    for label in np.unique(strata):
        members = np.flatnonzero(strata == label)
        draws = rng.choice(members, (repetitions, len(members)), replace=True)
        result += np.asarray(values)[draws].mean(1) / len(np.unique(strata))
    return result


def contrast(a, b, strata):
    result = {}
    for metric in ['brier', 'balanced_accuracy']:
        direction = -1 if metric == 'brier' else 1
        differences = direction * np.array([a[g][metric]-b[g][metric] for g in a])
        result[metric] = dict(improvement=aggregate(differences, strata),
            conditional_95_interval=np.quantile(bootstrap(differences,strata),[.025,.975]).tolist(),
            groups_better=int(np.sum(differences > 1e-12)), groups_worse=int(np.sum(differences < -1e-12)))
    return result


def assert_close(actual, expected, message, tolerance=1e-10):
    if not np.allclose(actual, expected, atol=tolerance, rtol=tolerance):
        raise ValueError(message)


def validate_source(report, manifest, raw=False):
    rows = [r for r in manifest['rows'] if r['role'] == 'source']
    ids = [r['row_id'] for r in rows]
    lookup = {r['row_id']:r for r in rows}
    identity = report['identity'] if raw else report
    if set(identity['source_rows']) != set(ids) or identity['target_data_used'] is not False:
        raise ValueError('source-only provenance mismatch')
    selected = min(report['candidates'],key=lambda r:r['mean_brier'])
    assert_close(selected['mean_brier'],report['source_validation_brier'],'source selection mismatch')
    validation = []
    for fold in selected['folds']:
        train = [ids[i] for i in fold['training_rows']] if raw else fold['training_ids']
        valid = [ids[i] for i in fold['validation_rows']] if raw else fold['validation_ids']
        if set(train) & set(valid) or set(train) | set(valid) != set(ids):
            raise ValueError('source fold row leakage')
        if {lookup[r]['group'] for r in train} & {lookup[r]['group'] for r in valid}:
            raise ValueError('source fold group leakage')
        validation.extend(valid)
        y = np.array([lookup[r]['target'] for r in valid])
        g = np.array([lookup[r]['group'] for r in valid])
        per_group,strata = grouped_scores(y,fold['probability'],g)
        value=aggregate([v['brier'] for v in per_group.values()],strata)
        assert_close(value,fold['brier'],'source fold metric mismatch')
    if sorted(validation) != sorted(ids): raise ValueError('source validation coverage mismatch')


def matched_null(fitted, bank, train, held):
    """Keep baseline projection, alpha and (for RBF) kernel bandwidth fixed."""
    mapped,_ = feature_bank(bank,fitted['name'],fitted['spi_index'],fitted['spi_count'])
    transform = fitted['transform'].transforms[0]
    x, hx = transform.transform(mapped,train), transform.transform(mapped,held)
    head = clone(fitted['model'])
    target = np.eye(int(bank['y'].max())+1)[bank['y'][train]]
    head.fit(x,target,sample_weight=weights(bank['y'][train],bank['group'][train]))
    return probabilities(head.predict(hx))


def review(root):
    manifest = json.loads((root/'manifest.json').read_text())
    bank = load_bank(root)
    train, held = [np.flatnonzero(bank['role'] == role) for role in ['source','screen']]
    if set(bank['group'][train]) & set(bank['group'][held]): raise ValueError('held-out group leakage')
    locked = json.loads((root/'statistical-source-lock.json').read_text())
    if locked['bank_sha256'] != file_sha256(root/'bank.npz'): raise ValueError('locked bank changed')
    saved = json.loads((root/'statistical-screen.json').read_text())
    if saved['source_lock_sha256'] != file_sha256(root/'statistical-source-lock.json'):
        raise ValueError('source lock changed')
    expected_ids = bank['row_id'][held].tolist()
    if saved['row_id'] != expected_ids or saved['y'] != bank['y'][held].tolist() or saved['group'] != bank['group'][held].tolist():
        raise ValueError('statistical prediction row mismatch')
    predictions, source, replays, dimensions = {}, {}, {}, {}
    for name, outcome in saved['methods'].items():
        path=root/'screen-models'/f'{name}.joblib'
        report=json.loads(path.with_suffix('.json').read_text())
        if file_sha256(path) != locked['models'][name] or report['model_sha256'] != locked['models'][name]:
            raise ValueError('statistical model changed')
        validate_source(report,manifest)
        fitted=joblib.load(path)
        p=predict(fitted,bank,held)
        assert_close(p,outcome['probability'],'statistical replay mismatch')
        replays[name]=float(np.max(np.abs(p-np.asarray(outcome['probability']))))
        predictions[name]=p
        transforms=getattr(fitted['transform'],'transforms',[fitted['transform']])
        dimensions[name]=[int(t.pca.n_components_) if t.pca is not None else 0 for t in transforms]
        source[name]=dict(selected=report['selected'],brier=report['source_validation_brier'])
        if name in ['m_plus_z','baseline_plus_z']:
            predictions[name+'_matched_no_z']=matched_null(fitted,bank,train,held)
        if name=='z':
            keep=fitted['transform'].blocks[0].keep
            availability=dict(retained_z_columns=len(keep),
                retained_source_missing_fraction=float((~np.isfinite(bank['z'][train][:,keep])).mean()),
                retained_screen_missing_fraction=float((~np.isfinite(bank['z'][held][:,keep])).mean()),
                groups={str(g):dict(mean_valid_spis=float(bank['validity'][held][bank['group'][held]==g].sum(1).mean()),
                    retained_z_missing_fraction=float((~np.isfinite(bank['z'][held][bank['group'][held]==g][:,keep])).mean()))
                    for g in np.unique(bank['group'][held])})
    from research.representation.dataset_screen.raw import input_manifest,load_role,make_network,network_prediction
    import torch
    x,_,_,_=load_role(root,input_manifest(root),'screen')
    raw_reports={}
    for name in ['minirocket','inceptiontime','aligned_encoder']:
        path=root/f'{name}-screen.json';outcome=json.loads(path.read_text())
        report_path=root/'screen-models'/f'{name}.json';report=json.loads(report_path.read_text())
        if file_sha256(report_path)!=outcome['source_report_sha256'] or outcome['row_id']!=expected_ids:
            raise ValueError('raw prediction provenance mismatch')
        if outcome['y']!=bank['y'][held].tolist() or outcome['group']!=bank['group'][held].tolist():
            raise ValueError('raw prediction label/group mismatch')
        if report['identity']['manifest_sha256']!=file_sha256(root/'manifest.json'):
            raise ValueError('raw manifest changed')
        validate_source(report,manifest,raw=True)
        members=[]
        for member in report['members']:
            path=root/'screen-models'/member['path']
            if file_sha256(path)!=member['sha256']: raise ValueError('raw model changed')
            if name!='minirocket':
                checkpoint=torch.load(path,map_location='cpu',weights_only=True)
                model=make_network(name,checkpoint['spec']);model.load_state_dict(checkpoint['state_dict'])
                tensor=(torch.as_tensor(x[:16])-checkpoint['mean'])/checkpoint['scale']
                members.append(network_prediction(model,tensor))
        predictions[name]=np.array(outcome['probability'])
        if members:
            value=np.mean(members,axis=0)
            assert_close(value,predictions[name][:16],'neural replay mismatch',1e-6)
            replays[name]=dict(records=len(value),maximum_error=float(np.max(np.abs(value-predictions[name][:16]))))
        source[name]=dict(brier=report['source_validation_brier'],final_epochs=report.get('final_epochs'),
            selected_lr=report.get('selected',{}).get('lr'),members=len(report['members']),
            folds_at_epoch_limit=sum(f.get('reached_epoch_limit',False) for c in report['candidates'] for f in c['folds']))
        raw_reports[name]=outcome
    methods={}
    for name,p in predictions.items():
        per_group,strata=grouped_scores(bank['y'][held],p,bank['group'][held])
        overall={key:aggregate([v[key] for v in per_group.values()],strata)
                 for key in ['brier','balanced_accuracy','log_loss']}
        original=saved['methods'].get(name,raw_reports.get(name))
        if original:
            for metric,value in overall.items(): assert_close(value,original[metric],'saved metric mismatch')
        methods[name]=dict(**overall,groups=per_group)
    baseline=locked['selected_baseline']
    comparisons=[('baseline_plus_z',baseline),('m_plus_z','m'),
                 ('baseline_plus_z','baseline_plus_z_matched_no_z'),('m_plus_z','m_plus_z_matched_no_z'),
                 *[('z',m) for m in ['m','aligned','univariate_rbf','minirocket','inceptiontime','aligned_encoder']]]
    result=dict(dataset=root.name,screen_records=len(held),screen_groups=len(np.unique(bank['group'][held])),
        source_selected_baseline=baseline,source_selected_spi=locked['selected_spi'],methods=methods,
        contrasts={f'{a} versus {b}':contrast(methods[a]['groups'],methods[b]['groups'],strata) for a,b in comparisons},
        source_selection=source,final_projection_dimensions=dimensions,availability=availability,replays=replays,
        provenance=dict(bank_sha256=locked['bank_sha256'],source_lock_sha256=file_sha256(root/'statistical-source-lock.json'),
                        review_sha256=file_sha256(Path(__file__))),
        scope='Exploratory, pointwise paired whole-group bootstrap, conditional on fitted models; no adjustment for dataset/model selection and no source-training uncertainty. LANL stratifies configurations by class.',
        diagnostic='Matched no-z removes z while retaining the fusion baseline projection, ridge penalty and RBF bandwidth; head refitted on sources only. It is a post-result attribution diagnostic, not a new selected primary baseline.')
    atomic_json(root/'discovery-review.json',result)
    print(root.name,'review verified',baseline,flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    result={name:review(args.root/name) for name in ['chbmit','hmc','grabmyo','lanl3ss']}
    atomic_json(args.root/'discovery-review.json',result)
