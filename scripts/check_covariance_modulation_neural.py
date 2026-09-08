"""Check all pilot neural checkpoints, CPU replays and source-only folds."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from src.representation_state_data import observed_view,file_hash
from src.representation_state_neural import make_encoder,predict


def main(tags, expected_fits, output, root, data):
    torch.set_num_threads(2)
    manifest=json.loads((data/'manifest.json').read_text());rows=manifest['rows']
    masters=np.load(data/'masters.npy',mmap_mode='r');checks=[]
    for tag, expected in zip(tags, expected_fits, strict=True):
        paths=sorted((root/tag).glob('*/*.json'))
        assert len(paths)==expected,(tag,len(paths),expected)
        for p in paths:
            info=json.loads(p.read_text());details=info['details'];ident=info['identity']
            assert details['checkpoint_sha256']==file_hash(p.with_suffix('.pt'))
            assert info['predictions_sha256']==file_hash(p.with_suffix('.npz'))
            train=set(info['train_indices']);evaluation=set(info['evaluation_indices'])
            assert not {rows[i]['master_id'] for i in train}&{rows[i]['master_id'] for i in evaluation}
            for fold in details['folds']:
                a,b=set(fold['fit']),set(fold['validation'])
                assert not a&b and a|b==train
            # Trusted checkpoints created by this study, including runtime metadata.
            ck=torch.load(p.with_suffix('.pt'),map_location='cpu',weights_only=False)
            model=make_encoder(ck['spec']);model.load_state_dict(ck['state_dict']);model.eval()
            with np.load(p.with_suffix('.npz'),allow_pickle=False) as saved:
                positions=[0,1,198,199,200,201,398,399];replay=[]
                for pos in positions:
                    r=rows[int(saved['evaluation_indices'][pos])]
                    x=torch.tensor(observed_view(masters[r['master_index']],r['M'],r['T'])[None])
                    replay.append(predict(model,x,1)[0])
                delta=float(np.max(abs(np.asarray(replay)-saved['prediction'][positions])))
                assert delta<1e-5,(p,delta)
            chosen=next(c for c in details['candidates'] if c['learning_rate']==details['chosen_learning_rate']
                        and c['weight_decay']==details['chosen_weight_decay'])
            checks.append(dict(fit=str(p),cpu_mps_max_difference=delta,
                               selected_folds_at_cap=sum(f['best_epoch']==ck['spec']['maximum_epochs'] for f in chosen['folds']),
                               training_MAE=info['training_MAE']))
    result=dict(fits=len(checks),replays_per_fit=8,all_source_splits_checked=True,
                maximum_difference=max(r['cpu_mps_max_difference'] for r in checks),
                selected_folds_at_cap=sum(r['selected_folds_at_cap'] for r in checks),checks=checks,
                checker_sha256=file_hash(Path(__file__)))
    (root/output).write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='checks'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tags',nargs='+',default=['neural-pair','neural-aligned'])
    parser.add_argument('--expected-fits',nargs='+',type=int,default=[18,18])
    parser.add_argument('--output',default='neural-verification.json')
    parser.add_argument('--root',type=Path,default=Path('results/covariance_modulation_260909'))
    parser.add_argument('--data',type=Path,default=Path('data/covariance_modulation_260909'))
    args=parser.parse_args();main(args.tags,args.expected_fits,args.output,args.root,args.data)
