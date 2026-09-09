"""Audit frozen-model references and replay neural transfer from raw views/MPIs."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from src.representation_state_data import file_hash
from src.representation_state_neural import make_encoder,predict
from src.spi_edge_pool import standardize_edges,pack_inputs


def main(root, data):
    rows=json.loads((data/'manifest.json').read_text())['rows'];lookup={r['row_id']:i for i,r in enumerate(rows)}
    with np.load(root/'gadi-analysis/normalized-edges.npz',allow_pickle=False) as a:order=a['spi_order'].tolist()
    samples={}
    with np.load(data/'views.npz',allow_pickle=False) as raw:
        for p in sorted((root/'replay-mpis').glob('*/meta.json')):
            name=json.loads(p.read_text())['dataset_name']
            with np.load(p.parent/'spi_mpis.npz',allow_pickle=False) as a:x,v=standardize_edges(a,order)
            samples[lookup[name]]=(torch.tensor(raw[name][None]),torch.tensor(pack_inputs(x[None],v[None])))
    assert len(samples)==8;checks=[];torch.set_num_threads(2)
    for p in sorted((root/'predictions').glob('*.json')):
        info=json.loads(p.read_text());source=Path(info['source_fit']);ident=info['identity']
        assert file_hash(source)==ident['source_fit_sha256']
        assert file_hash(p.with_suffix('.npz'))==info['predictions_sha256']
        old=json.loads(source.read_text())
        with np.load(p.with_suffix('.npz'),allow_pickle=False) as a,np.load(source.with_suffix('.npz'),allow_pickle=False) as b:
            np.testing.assert_array_equal(a['train_indices'],b['train_indices'])
            np.testing.assert_array_equal(a['row_id'],[r['row_id'] for r in rows])
            np.testing.assert_array_equal(a['target'],[r['target'] for r in rows])
            saved=a['prediction']
        error=None
        if ident['method'].startswith('neural-') or ident['method']=='learned-pooling':
            assert file_hash(source.with_suffix('.pt'))==old['details']['checkpoint_sha256']
            ck=torch.load(source.with_suffix('.pt'),map_location='cpu',weights_only=False)
            model=make_encoder(ck['spec']);model.load_state_dict(ck['state_dict']);model.eval()
            col=int(ident['method']=='learned-pooling')
            error=max(abs(float(predict(model,inputs[col],1)[0])-saved[i]) for i,inputs in samples.items())
            assert error<1e-5,(p,error)
        else:
            assert info['details']['source_chosen_settings']==old['details']['chosen']
            assert info['source_replay_max_difference']<1e-9
        checks.append(dict(fit=str(p),method=ident['method'],neural_raw_or_MPI_replay_max_difference=error))
    assert len(checks)==99
    neural=[c for c in checks if c['neural_raw_or_MPI_replay_max_difference'] is not None];assert len(neural)==27
    result=dict(status='passed',source_references_checked=99,statistical_original_prediction_replays=72,
        neural_checkpoints_replayed=27,replays_per_checkpoint=8,
        max_neural_replay_difference=max(c['neural_raw_or_MPI_replay_max_difference'] for c in neural),
        checks=checks,checker_sha256=file_hash(Path(__file__)))
    (root/'frozen-model-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='checks'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/oscillatory_coorganization_transfer_260909'))
    parser.add_argument('--data',type=Path,default=Path('data/oscillatory_coorganization_transfer_260909'))
    args=parser.parse_args();main(args.root,args.data)
