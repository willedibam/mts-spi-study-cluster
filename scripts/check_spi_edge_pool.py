"""Verify edge-bank provenance, source-only splits and frozen neural predictions."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from src.representation_state_data import file_hash
from src.representation_state_neural import make_encoder,predict
from src.spi_edge_pool import pack_inputs


def main(data,bank,fits,output):
    rows=json.loads((data/'manifest.json').read_text())['rows']
    meta=json.loads(bank.with_suffix('.json').read_text());assert meta['artifact_sha256']==file_hash(bank)
    assert meta['manifest_sha256']==file_hash(data/'manifest.json')
    with np.load(bank,allow_pickle=False) as a:
        np.testing.assert_array_equal(a['row_id'],[r['row_id'] for r in rows]);x,valid,lengths=a['edges'],a['validity'],a['lengths']
    checks=[];torch.set_num_threads(2)
    for p in sorted(fits.rglob('*.json')):
        info=json.loads(p.read_text());details=info['details'];ident=info['identity']
        assert ident['edge_bank_sha256']==file_hash(bank)
        assert details['checkpoint_sha256']==file_hash(p.with_suffix('.pt'))
        assert info['predictions_sha256']==file_hash(p.with_suffix('.npz'))
        train=set(info['train_indices']);ev=set(info['evaluation_indices'])
        assert not {rows[i]['master_id'] for i in train}&{rows[i]['master_id'] for i in ev}
        assert all(rows[i]['role']=='training_pool' for i in train)
        for fold in details['folds']:
            a,b=set(fold['fit']),set(fold['validation']);assert not a&b and a|b==train
        ck=torch.load(p.with_suffix('.pt'),map_location='cpu',weights_only=False)
        assert ck['identity']==ident and ck['spec']['architecture']=='spi_edge_pool'
        model=make_encoder(ck['spec']);model.load_state_dict(ck['state_dict']);model.eval()
        with np.load(p.with_suffix('.npz'),allow_pickle=False) as a:
            idx=a['evaluation_indices'];positions=[0,1,198,199,200,201,398,399];pred=[]
            np.testing.assert_array_equal(a['row_id'],[rows[i]['row_id'] for i in idx])
            for pos in positions:
                i=idx[pos];tensor=torch.tensor(pack_inputs(x[i:i+1,:lengths[i]],valid[i:i+1]));pred.append(predict(model,tensor,1)[0])
            delta=float(np.max(abs(np.array(pred)-a['prediction'][positions])));assert delta<1e-5
        chosen=next(c for c in details['candidates'] if c['learning_rate']==details['chosen_learning_rate'] and c['weight_decay']==details['chosen_weight_decay'])
        checks.append(dict(fit=str(p),max_cpu_replay_difference=delta,parameter_count=sum(p.numel() for p in model.parameters()),
            selected_folds_at_cap=sum(f['best_epoch']==ck['spec']['maximum_epochs'] for f in chosen['folds'])))
    assert len(checks)==9
    result=dict(status='passed',checks=checks,z_recovery_max_difference=meta['z_recovery_max_difference'],
        max_cpu_replay_difference=max(c['max_cpu_replay_difference'] for c in checks),selected_folds_at_cap=sum(c['selected_folds_at_cap'] for c in checks),checker_sha256=file_hash(Path(__file__)))
    output.write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='checks'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','bank','fits','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();main(a.data,a.bank,a.fits,a.output)
