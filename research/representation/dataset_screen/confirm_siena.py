"""Predict Siena using checksum-frozen CHB-MIT models; never fit on Siena."""
import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import subprocess

import joblib
import numpy as np

from mtsspi.provenance import atomic_json,file_sha256
from research.representation.dataset_screen.common import probabilities


def checked_lock(root):
    lock=json.loads((root/'source-lock.json').read_text())
    source=Path(lock['source_root'])
    for name,digest in lock['files'].items():
        if file_sha256(source/name)!=digest: raise ValueError(f'frozen source changed: {name}')
    return lock,source


def inputs(root):
    lock,source=checked_lock(root)
    metadata=json.loads((root/'bank.json').read_text())
    if metadata['sha256']!=file_sha256(root/'bank.npz'):raise ValueError('confirmation bank changed')
    source_meta=json.loads((source/'bank.json').read_text())
    for field in ['pyspi_config_sha256','pyspi_version']:
        if metadata['identity'][field]!=source_meta['identity'][field]:raise ValueError('source/confirmation computation mismatch')
    keys=['m','z','validity','pearson','aligned','y','group','row_id','role','spi_order']
    with np.load(root/'bank.npz',allow_pickle=False) as archive:bank={k:archive[k] for k in keys}
    with np.load(source/'bank.npz',allow_pickle=False) as archive:
        if not np.array_equal(bank['spi_order'],archive['spi_order']):raise ValueError('SPI order differs')
    if set(bank['role'])!={'confirmation'}:raise ValueError('unexpected target roles')
    audit=json.loads((root/'input-audit.json').read_text())
    if file_sha256(root/'univariate.npz')!=audit['univariate_sha256']:raise ValueError('target summary changed')
    with np.load(root/'univariate.npz',allow_pickle=False) as archive:
        if archive['row_id'].tolist()!=bank['row_id'].tolist():raise ValueError('target summary order differs')
        bank['u']=archive['u']
    return lock,source,bank


def predict(root,kind):
    output=root/f'predictions-{kind}.json'
    if output.exists():raise FileExistsError(output)
    lock,source,bank=inputs(root)
    ix=np.arange(len(bank['y']));result={};coverage=None
    if kind=='statistical':
        from research.representation.dataset_screen.statistical import predict as stat_predict
        for name,item in lock['models'].items():
            if item['kind']!='statistical':continue
            fitted=joblib.load(source/item['weights'][0])
            result[name]=stat_predict(fitted,bank,ix)
            if name=='z':
                keep=fitted['transform'].blocks[0].keep
                coverage=dict(retained_columns=len(keep),missing_fraction=float((~np.isfinite(bank['z'][:,keep])).mean()),
                    by_patient={str(g):float((~np.isfinite(bank['z'][bank['group']==g][:,keep])).mean()) for g in np.unique(bank['group'])})
    else:
        from research.representation.dataset_screen.raw import input_manifest,load_role
        x,y,groups,rows=load_role(root,input_manifest(root),'confirmation')
        if [r['row_id'] for r in rows]!=bank['row_id'].tolist():raise ValueError('raw target row mismatch')
        if kind=='neural':
            import torch
            from research.representation.dataset_screen.raw import make_network,network_prediction
            torch.set_num_threads(4);torch.set_num_interop_threads(1)
        for name,item in lock['models'].items():
            if item['kind']!=kind:continue
            members=[];report=json.loads((source/item['report']).read_text())
            if kind=='minirocket':
                for package,expected in report['identity']['versions'].items():
                    if version(package)!=expected:raise ValueError(f'MiniRocket runtime differs: {package}')
            for relative in item['weights']:
                path=source/relative
                if kind=='neural':
                    saved=torch.load(path,map_location='cpu',weights_only=True)
                    model=make_network(name,saved['spec']);model.load_state_dict(saved['state_dict'])
                    tensor=(torch.as_tensor(x)-saved['mean'])/saved['scale']
                    members.append(network_prediction(model,tensor))
                else:
                    fitted=joblib.load(path)
                    raw=((x-fitted['mean'])/fitted['scale']).transpose(0,2,1)
                    features=fitted['scaler'].transform(fitted['rocket'].transform(raw).astype(np.float64))
                    members.append(probabilities(fitted['model'].predict(features)))
            result[name]=np.mean(members,axis=0)
    atomic_json(output,dict(probability={k:v.tolist() for k,v in result.items()},row_id=bank['row_id'].tolist(),
        y=bank['y'].tolist(),group=bank['group'].tolist(),source_lock_sha256=file_sha256(root/'source-lock.json'),
        bank_sha256=file_sha256(root/'bank.npz'),fitted_on_target=False,retained_z_coverage=coverage,
        prediction_runner_sha256=file_sha256(Path(__file__))))
    print('Frozen confirmation predictions',kind,len(ix),flush=True)


def summarize(root):
    from research.representation.dataset_screen.review import grouped_scores,aggregate,contrast
    lock,source,bank=inputs(root);methods={};prediction_hashes={};coverage=None
    for kind in ['statistical','neural','minirocket']:
        path=root/f'predictions-{kind}.json';report=json.loads(path.read_text());prediction_hashes[kind]=file_sha256(path)
        if report['source_lock_sha256']!=file_sha256(root/'source-lock.json') or report['bank_sha256']!=file_sha256(root/'bank.npz'):
            raise ValueError('confirmation prediction identity differs')
        if report['row_id']!=bank['row_id'].tolist() or report['y']!=bank['y'].tolist() or report['group']!=bank['group'].tolist():
            raise ValueError('confirmation prediction rows differ')
        if report['fitted_on_target'] is not False:raise ValueError('target fitting was not prohibited')
        if report.get('retained_z_coverage'):coverage=report['retained_z_coverage']
        for name,p in report['probability'].items():
            per_group,strata=grouped_scores(bank['y'],p,bank['group'])
            methods[name]=dict(groups=per_group,**{key:aggregate([v[key] for v in per_group.values()],strata)
                for key in ['brier','balanced_accuracy','log_loss']})
    if set(methods)!=set(lock['models']):raise ValueError('confirmation comparator panel incomplete')
    baseline=json.loads((source/'statistical-source-lock.json').read_text())['selected_baseline']
    comparisons=[('m_plus_z','m'),('baseline_plus_z',baseline),*[('z',name) for name in ['m','aligned','univariate','univariate_rbf','inceptiontime','aligned_encoder','minirocket','minirocket_extended']]]
    atomic_json(root/'confirmation-summary.json',dict(methods=methods,patients=len(np.unique(bank['group'])),records=len(bank['y']),
        contrasts={f'{a} versus {b}':contrast(methods[a]['groups'],methods[b]['groups'],strata) for a,b in comparisons},
        source_selected_baseline=baseline,prediction_sha256=prediction_hashes,retained_z_coverage=coverage,
        source_lock_sha256=file_sha256(root/'source-lock.json'),
        scope='Independent Siena cohort after exploratory CHB-MIT discovery. Frozen source models; no target calibration. Pointwise paired whole-patient bootstrap conditional on fitted models; no source-training uncertainty or multiplicity adjustment.'))
    print('Siena confirmation summarized',flush=True)


def submit_chain(root):
    from research.representation.dataset_screen.launch import farm,batch
    lock,source=checked_lock(root)
    path=root/'computation-jobs.json'
    if path.exists():raise FileExistsError(path)
    manifest=json.loads((root/'manifest.json').read_text());count=len(manifest['rows'])
    if {r['M'] for r in manifest['rows']}!={16} or {r['T'] for r in manifest['rows']}!={2000}:
        raise ValueError('source timing applies only to M16/T2000')
    measured=json.loads((source/'farm-chain.json').read_text())['measurement_resources']
    h,m,s=map(int,measured['walltime'].split(':'));seconds=h*3600+m*60+s
    cpus=count if count<48 else min(96,(count//48)*48)
    memory=cpus*max(3,math.ceil(1.25*int(measured['mem'][:-2])/1024**2))
    wall=math.ceil((2*seconds*math.ceil(count/cpus)+300)/60)*60
    jobs={};record=dict(snapshot=str(Path.cwd()),records=count,source_measurement=measured,jobs=jobs)
    def save(key,value):jobs[key]=value;atomic_json(path,record);return value
    job=save('farm',farm(root,count,cpus,memory,wall))
    bank=save('bank',batch(root,'research/representation/empirical_transfer/jobs/bank.pbs',job))
    outputs=[]
    for kind in ['statistical','neural','minirocket']:
        outputs.append(save(kind,batch(root,'research/representation/dataset_screen/jobs/siena_predict.pbs',bank,MODE=kind,
            WHEELHOUSE='/scratch/ql44/we2614/mts-spi-study/representation/empirical/character-transfer-261031/aeon-wheels')))
    save('summary',batch(root,'research/representation/dataset_screen/jobs/siena_predict.pbs',':'.join(outputs),MODE='summarize'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--mode',choices=['submit','statistical','neural','minirocket','summarize'],required=True)
    args=parser.parse_args()
    if args.mode=='submit':submit_chain(args.root)
    elif args.mode=='summarize':summarize(args.root)
    else:predict(args.root,args.mode)
