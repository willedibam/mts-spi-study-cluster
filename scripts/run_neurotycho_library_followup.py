"""Extract, fit, and evaluate the declared three-library follow-up separately."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import yaml

from src.neurotycho_library_baselines import array_sha, fit_model, predict_model, standardized


ROOT = Path('results/neurotycho_library_followup_260911')
CONFIG = Path('configs/analysis/neurotycho-library-followup-260911.yaml')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def versions():
    return {k:importlib.metadata.version(k) for k in ['aeon','tsfresh','pycatch22','numpy','scipy','pandas','scikit-learn','numba']}


def windows(phase):
    folder = Path('results/neurotycho_source_pilot_260910/float64/all-source' if phase == 'source'
                  else 'results/neurotycho_target_pilot_260910')
    rows, xs, provenance = [], [], []
    for path in sorted(folder.glob('201*.json')):
        meta = json.loads(path.read_text())
        assert meta['usable']
        assert ('ktmd' if phase == 'source' else 'pf') in meta['archive']
        provenance.append(dict(path=str(path), metadata_sha256=sha(path), array_sha256=sha(path.with_suffix('.npz'))))
        with np.load(path.with_suffix('.npz')) as data:
            for r in meta['records']:
                if not r['quality']['accepted']:
                    continue
                xs.append(data['x'][r['array_row']])
                rows.append(dict(record_id=f'{r["archive"]}/{r["session"]}/{r["state"]}/{r["start"]}',
                    animal=r['animal'], archive=r['archive'], y=r['target']))
    assert len(rows) == (352 if phase == 'source' else 128)
    bank = {k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    bank['x'] = np.asarray(xs)
    assert bank['x'].shape == (len(rows), 16, 2000) and bank['x'].dtype == np.float64
    if phase == 'source':
        with np.load('data/neurotycho_neural_260910/matched.npz') as old:
            lookup = {r:i for i,r in enumerate(old['record_id'])}
            ix = [lookup[r] for r in bank['record_id']]
            np.testing.assert_array_equal(bank['y'], old['y'][ix])
            np.testing.assert_array_equal(standardized(bank['x']).transpose(0,2,1), old['x'][ix])
    return bank, provenance


def extract_one(task):
    import pandas as pd
    import pycatch22
    from tsfresh import extract_features
    from tsfresh.feature_extraction.settings import EfficientFCParameters
    index, x, destination = task
    path = Path(destination) / f'{index:04d}.npz'
    digest = array_sha(x)
    if path.exists():
        with np.load(path) as old:
            assert str(old['input_sha256']) == digest
            assert str(old['script_sha256']) == sha(Path(__file__))
        return index, 'cached'
    started = time.perf_counter()
    catch = [pycatch22.catch22_all(channel, catch24=False) for channel in x]
    catch_values = np.asarray([r['values'] for r in catch])
    assert all(r['names'] == catch[0]['names'] for r in catch)
    frame = pd.DataFrame(dict(id=np.repeat(np.arange(16), 2000), t=np.tile(np.arange(2000), 16), value=x.ravel()))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        fresh = extract_features(frame, column_id='id', column_sort='t', column_value='value',
            default_fc_parameters=EfficientFCParameters(), n_jobs=0, disable_progressbar=True, show_warnings=False)
    fresh = fresh.reindex(index=np.arange(16), columns=sorted(fresh.columns))
    np.savez_compressed(path, catch22=catch_values, tsfresh=fresh.to_numpy(),
        catch22_names=np.asarray(catch[0]['names']), tsfresh_names=np.asarray(fresh.columns, dtype=str),
        input_sha256=digest, script_sha256=sha(Path(__file__)), seconds=time.perf_counter()-started)
    return index, round(time.perf_counter()-started, 3)


def extract(args):
    bank, provenance = windows(args.phase)
    config = yaml.safe_load(CONFIG.read_text())
    assert all(versions()[k] == v for k,v in config['packages'].items())
    folder = ROOT / args.phase
    cache = folder / 'records'; cache.mkdir(parents=True, exist_ok=True)
    tasks = [(i,x,str(cache)) for i,x in enumerate(bank['x'])]
    if args.limit:
        tasks = tasks[:args.limit]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, elapsed in executor.map(extract_one, tasks):
            print(args.phase, index, elapsed, flush=True)
    if args.limit:
        return
    arrays = {method:[] for method in ['catch22','tsfresh']}; timings=[]; names={}
    for i in range(len(bank['x'])):
        with np.load(cache / f'{i:04d}.npz') as data:
            for method in arrays:
                current = data[method+'_names']
                if method in names:
                    np.testing.assert_array_equal(names[method], current)
                else:
                    names[method] = current
                arrays[method].append(data[method])
            timings.append(float(data['seconds']))
    path = folder/'bank.npz'
    np.savez_compressed(path, **bank, **{k:np.asarray(v) for k,v in arrays.items()},
                        **{k+'_names':v for k,v in names.items()})
    (folder/'bank.json').write_text(json.dumps(dict(phase=args.phase, sha256=sha(path), rows=len(bank['x']),
        sources=provenance, config_sha256=sha(CONFIG), versions=versions(), script_sha256=sha(Path(__file__)),
        seconds=timings),indent=2)+'\n')


def load_bank(phase):
    path = ROOT/phase/'bank.npz'
    meta = json.loads(path.with_suffix('.json').read_text())
    assert sha(path) == meta['sha256'] and sha(CONFIG) == meta['config_sha256']
    with np.load(path) as data:
        return {k:data[k] for k in ['record_id','animal','archive','y','x','catch22','tsfresh']}


def fit(args):
    config = yaml.safe_load(CONFIG.read_text()); bank = load_bank('source')
    out = ROOT/'fits'; out.mkdir(exist_ok=True)
    for method in config['methods']:
        for animal in config['target_animals']:
            seeds = config['minirocket_seeds'] if method == 'minirocket' else [0]
            for seed in seeds:
                stem = out/f'{method}-{animal}-s{seed}'
                identity = dict(method=method, animal=animal, seed=seed, config_sha256=sha(CONFIG),
                    bank_sha256=sha(ROOT/'source'/'bank.npz'), module_sha256=sha(Path('src/neurotycho_library_baselines.py')))
                if stem.with_suffix('.json').exists():
                    previous = json.loads(stem.with_suffix('.json').read_text())
                    assert previous['identity'] == identity and sha(stem.with_suffix('.joblib')) == previous['model_sha256']
                    continue
                ix = bank['animal'] != animal
                source = {k:v[ix] for k,v in bank.items()}
                started = time.perf_counter()
                model, report = fit_model(source, method, seed, config, args.workers)
                probability = predict_model(model, source)
                joblib.dump(model, stem.with_suffix('.joblib'))
                replay = predict_model(joblib.load(stem.with_suffix('.joblib')), source)
                np.testing.assert_allclose(replay, probability, atol=1e-12, rtol=0)
                np.savez_compressed(stem.with_suffix('.npz'), record_id=source['record_id'], y=source['y'], probability=probability)
                report.update(identity=identity, versions=versions(), target_data_used=False,
                    model_sha256=sha(stem.with_suffix('.joblib')), prediction_sha256=sha(stem.with_suffix('.npz')),
                    source_replay_max_difference=float(np.max(np.abs(replay-probability))), seconds=time.perf_counter()-started)
                stem.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
                print('COMPLETE',stem.name,report['selected'],flush=True)
    assert len(list(out.glob('*.json'))) == 10


def evaluate(args):
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from src.neurotycho_library_baselines import brier
    assert len(list((ROOT/'fits').glob('*.json'))) == 10
    bank = load_bank('evaluation'); rows=[]; audits=[]
    for path in sorted((ROOT/'fits').glob('*.json')):
        report = json.loads(path.read_text()); identity=report['identity']
        assert identity['config_sha256'] == sha(CONFIG) and not report['target_data_used']
        assert sha(path.with_suffix('.joblib')) == report['model_sha256']
        ix=bank['animal'] == identity['animal']; target={k:v[ix] for k,v in bank.items()}
        p=predict_model(joblib.load(path.with_suffix('.joblib')), target)
        for i,probability in enumerate(p):
            rows.append(dict(method=identity['method'],seed=identity['seed'],probability=float(probability),
                **{k:target[k][i].item() for k in ['record_id','animal','archive','y']}))
        audits.append(dict(model=path.name, report_sha256=sha(path), model_sha256=report['model_sha256']))
    flat={k:np.asarray([r[k] for r in rows]) for k in rows[0]}; scores=[]
    for method in sorted(set(flat['method'])):
        for seed in sorted(set(flat['seed'][flat['method']==method])):
            ix=(flat['method']==method)&(flat['seed']==seed)
            assert ix.sum()==128
            dates=[]; animals=[]
            for archive in sorted(set(flat['archive'][ix])):
                jx=ix & (flat['archive']==archive); y=flat['y'][jx]; p=flat['probability'][jx]
                dates.append(dict(archive=str(archive),animal=str(flat['animal'][jx][0]),n=int(jx.sum()),
                    balanced_accuracy=float(balanced_accuracy_score(y,p>=.5)),
                    auroc=float(roc_auc_score(y,p)),balanced_brier=brier(y,p)))
            for animal in ['Chibi','George']:
                ds=[d for d in dates if d['animal']==animal]
                animals.append(dict(animal=animal,**{k:float(np.mean([d[k] for d in ds]))
                    for k in ['balanced_accuracy','auroc','balanced_brier']}))
            scores.append(dict(method=method,seed=int(seed),M=16,T=2000,dates=dates,animals=animals,
                mean={k:float(np.mean([a[k] for a in animals])) for k in ['balanced_accuracy','auroc','balanced_brier']}))
    out=ROOT/'report';out.mkdir(exist_ok=False)
    np.savez_compressed(out/'predictions.npz',**flat)
    (out/'report.json').write_text(json.dumps(dict(scores=scores,models=audits,versions=versions(),
        config_sha256=sha(CONFIG), prediction_sha256=sha(out/'predictions.npz'),
        target_bank_sha256=sha(ROOT/'evaluation'/'bank.npz'), post_PF_inspection_followup=True, threshold=.5, target_calibration=False, script_sha256=sha(Path(__file__))),
        indent=2)+'\n')
    print(json.dumps(scores,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['extract','fit','evaluate'])
    p.add_argument('--phase',choices=['source','evaluation'],default='source')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--limit',type=int)
    args=p.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
    globals()[args.stage](args)
