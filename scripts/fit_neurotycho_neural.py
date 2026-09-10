"""Fit one source-only outer-animal/initialization neural model with grouped CV."""
import argparse
import hashlib
import itertools
import json
import platform
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from src.neurotycho_learning import fit_binary,predict_binary


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    protocol=yaml.safe_load(args.config.read_text());training=protocol['neural_training']
    torch.set_num_threads(training['threads'])
    if args.animal not in protocol['target_animals'] or args.seed not in protocol['neural_seeds']:
        raise ValueError('fit outside declared protocol')
    spec=protocol['raw_encoder']
    args.output.mkdir(parents=True,exist_ok=True)
    stem=args.output/f'{args.kind}-{args.animal}-s{args.seed}'
    paths=[args.data/'matched.npz']+([args.data/'dense.npz'] if args.kind=='enriched' else [])
    hashes={p.name:sha(p) for p in paths}
    identity=dict(config_sha256=sha(args.config),inputs=hashes,animal=args.animal,seed=args.seed,kind=args.kind,
        modules={str(p):sha(p) for p in [Path(__file__),Path('src/neurotycho_learning.py'),Path('src/representation_state_neural.py')]},
        torch=torch.__version__,device='cpu',platform=platform.platform())
    if stem.with_suffix('.json').exists():
        prior=json.loads(stem.with_suffix('.json').read_text())
        if prior['identity']!=identity or sha(stem.with_suffix('.pt'))!=prior['checkpoint_sha256']:
            raise ValueError('resume identity/checkpoint mismatch')
        print('Verified existing completed fit',stem,flush=True);return
    with np.load(paths[0]) as bank:
        matched={k:bank[k] for k in bank.files}
    if args.kind=='enriched':
        with np.load(paths[1]) as bank:source={k:bank[k] for k in bank.files}
    else:source=matched
    source_mask=source['animal']!=args.animal
    source={k:v[source_mask] for k,v in source.items()}
    # Discard excluded-animal data before any tensor construction or fitting.
    matched_mask=matched['animal']!=args.animal
    matched={k:v[matched_mask] for k,v in matched.items()}
    sx=torch.from_numpy(source['x']);vx=torch.from_numpy(matched['x'])
    groups=sorted(set(source['animal']))
    if len(groups)!=3 or set(groups)!=set(matched['animal']):
        raise ValueError('expected three source animals')
    choices=[];started=time.perf_counter()
    for lr,decay in itertools.product(training['learning_rates'],sorted(training['weight_decays'],reverse=True)):
        folds=[]
        for fold,animal in enumerate(groups):
            fit=np.flatnonzero(source['animal']!=animal);valid=np.flatnonzero(matched['animal']==animal)
            assert args.animal not in source['animal'][fit] and animal not in source['animal'][fit]
            model,report=fit_binary(sx[fit],source['y'][fit],source['archive'][fit],spec,training,
                lr,decay,args.seed+1000*fold,validation=(vx[valid],matched['y'][valid]),enriched=args.kind=='enriched')
            probability=predict_binary(model,vx[valid],training['batch_size'])
            folds.append(dict(animal=animal,training_ids=source['record_id'][fit].tolist(),
                validation_ids=matched['record_id'][valid].tolist(),validation_y=matched['y'][valid].tolist(),
                validation_probability=probability.tolist(),**report))
            print(dict(kind=args.kind,target=args.animal,seed=args.seed,lr=lr,decay=decay,
                validation_animal=animal,epochs=report['epochs_run'],best_epoch=report['best_epoch'],
                brier=report['best_validation_brier'],seconds=report['seconds']),flush=True)
            del model
        choices.append(dict(lr=lr,decay=decay,folds=folds,
            mean_brier=float(np.mean([f['best_validation_brier'] for f in folds]))))
        stem.with_suffix('.progress.json').write_text(json.dumps(dict(identity=identity,candidates=choices),indent=2)+'\n')
    selected=min(choices,key=lambda c:c['mean_brier'])
    epochs=int(np.median([f['best_epoch'] for f in selected['folds']]))
    model,report=fit_binary(sx,source['y'],source['archive'],spec,training,selected['lr'],selected['decay'],
        args.seed,epochs=epochs,enriched=args.kind=='enriched')
    torch.save(dict(state_dict=model.state_dict(),spec=spec),stem.with_suffix('.pt'))
    np.savez_compressed(stem.with_suffix('.npz'),training_ids=source['record_id'],y=source['y'],
        probability=predict_binary(model,sx,training['batch_size']))
    result=dict(identity=identity,source_animals=groups,training_windows=len(sx),selected_lr=selected['lr'],
        selected_decay=selected['decay'],selected_epochs=epochs,candidates=choices,final=report,
        checkpoint_sha256=sha(stem.with_suffix('.pt')),predictions_sha256=sha(stem.with_suffix('.npz')),
        total_seconds=time.perf_counter()-started,target_data_used=False)
    stem.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Completed source fit',stem,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/analysis/neurotycho-transfer-260910.yaml'))
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--animal',required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--kind',choices=['matched','enriched'],required=True)
    main(parser.parse_args())
