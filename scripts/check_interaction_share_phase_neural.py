"""Audit phase-study neural selection and replay checkpoints on CPU."""
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.representation_state_data import file_hash, load_state_data
from src.representation_state_neural import AlignedChannelEncoder, predict


def check():
    config = Path('configs/analysis/interaction-share-phase-260908.yaml')
    protocol = yaml.safe_load(config.read_text())
    root = Path('results/interaction_share_phase_260908')
    torch.set_num_threads(2)
    audits = []
    for arm in protocol['phase_control']['arms']:
        _, masters = load_state_data(Path('data/interaction_share_phase_260908') / arm, config)
        for family in protocol['generator']['families']:
            for seed in protocol['methods']['subset_seeds']:
                stem = root / arm / 'neural' / family / f'neural-n4-s{seed}'
                record = json.loads(stem.with_suffix('.json').read_text())
                details = record['details']
                assert file_hash(stem.with_suffix('.pt')) == details['checkpoint_sha256']
                selected = min(details['candidates'], key=lambda c: c['mean_MAE'])
                assert selected['learning_rate'] == details['chosen_learning_rate']
                assert selected['weight_decay'] == details['chosen_weight_decay']
                expected_epochs = max(1, int(np.median([f['best_epoch'] for f in selected['folds']]) + .5))
                assert details['refit_epochs'] == expected_epochs
                train = set(record['train_indices'])
                assert len(train) == 20 and not train & set(record['evaluation_indices'])
                for fold in details['folds']:
                    fit, validation = set(fold['fit']), set(fold['validation'])
                    assert len(fit) == len(validation) == 10 and not fit & validation
                    assert fit | validation == train
                with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
                    checkpoint = torch.load(stem.with_suffix('.pt'), map_location='cpu', weights_only=True)
                assert checkpoint['identity'] == record['identity']
                model = AlignedChannelEncoder(checkpoint['spec'])
                model.load_state_dict(checkpoint['state_dict'])
                with np.load(stem.with_suffix('.npz')) as saved:
                    positions = np.arange(0, 400, 40)
                    ix = saved['evaluation_indices'][positions]
                    x = torch.tensor(np.asarray(masters[ix]), dtype=torch.float32)
                    replay = predict(model, x, 10)
                    error = float(abs(replay - saved['prediction'][positions]).max())
                    np.testing.assert_allclose(replay, saved['prediction'][positions], rtol=0, atol=2e-6)
                audits.append(dict(arm=arm, family=family, seed=seed, maximum_CPU_MPS_error=error,
                    parameters=sum(p.numel() for p in model.parameters()),
                    selected_best_epochs=[f['best_epoch'] for f in selected['folds']],
                    selected_epochs_run=[f['epochs_run'] for f in selected['folds']],
                    refit_epochs=expected_epochs, training_MAE=record['training_MAE']))
    result = dict(fits=len(audits), replays_per_fit=10,
                  maximum_CPU_MPS_error=max(a['maximum_CPU_MPS_error'] for a in audits),
                  selected_folds_hitting_cap=sum(e >= 200 for a in audits for e in a['selected_epochs_run']),
                  fits_detail=audits, audit_code_sha256=file_hash(Path(__file__)))
    (root / 'neural-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print({k: v for k, v in result.items() if k != 'fits_detail'})


if __name__ == '__main__':
    check()
