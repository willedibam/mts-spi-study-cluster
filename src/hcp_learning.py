"""Source-only HCP neural fitting with family folds and participant weighting.

This adapter reuses the declared architectures without changing the completed
NeuroTycho/synthetic trainers. It never accepts a confirmation dataset.
"""
import itertools
import time

import numpy as np
import torch
from torch.nn import functional as F

from src.inceptiontime_baseline import InceptionNetwork
from src.neurotycho_learning import predict_binary
from src.representation_state_neural import make_encoder, seed_torch


from src.hcp_grouping import _ids, family_folds, participant_balanced_brier, participant_state_weights


def fit_network(x, y, participants, model_name, spec, training, lr, decay, seed,
                *, validation=None, epochs=None):
    """One source fold or a source-only refit; no augmentation/pretraining."""
    if (validation is None) == (epochs is None):
        raise ValueError('Supply either source validation or source-selected refit epochs')
    y = np.asarray(y)
    weights = participant_state_weights(y, participants)
    if x.ndim != 3 or len(x) != len(y) or not torch.isfinite(x).all():
        raise ValueError('Expected finite B,T,M source tensor matching targets')
    maximum = training['maximum_epochs'] if epochs is None else epochs
    if maximum < 1 or int(maximum) != maximum:
        raise ValueError('Epoch count must be a positive integer')
    if validation is not None:
        vx, vy, vp = validation
        participant_state_weights(vy, vp)
        if (vx.ndim != 3 or len(vx) != len(vy) or vx.shape[1:] != x.shape[1:]
                or not torch.isfinite(vx).all()):
            raise ValueError('Expected matching finite source-validation input')
        if set(_ids(participants, len(y))) & set(_ids(vp, len(vy))):
            raise ValueError('Training and validation participants overlap')
    seed_torch(seed)
    if model_name == 'aligned_channel':
        model = make_encoder(spec).to(x.device)
        epsilon = 1e-8
    elif model_name == 'inceptiontime':
        model = InceptionNetwork(spec).to(x.device)
        epsilon = training['adam_epsilon']
    else:
        raise ValueError(f'Unknown declared HCP network: {model_name}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay, eps=epsilon)
    scheduler = None
    if model_name == 'inceptiontime':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=.5, patience=50, min_lr=.0001, threshold=1e-4, threshold_mode='abs')
    generator = torch.Generator().manual_seed(seed)
    target = torch.as_tensor(y, dtype=torch.float32, device=x.device)
    weights = torch.as_tensor(weights, dtype=torch.float32, device=x.device)
    best, best_epoch, state = float('inf'), 0, None
    history, started = [], time.perf_counter()
    for epoch in range(1, int(maximum) + 1):
        model.train()
        total = 0.
        for ix in torch.randperm(len(x), generator=generator).to(x.device).split(training['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            loss = (F.binary_cross_entropy_with_logits(model(x[ix]), target[ix], reduction='none') * weights[ix]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ix)
        record = dict(epoch=epoch, training_bce=total/len(x), learning_rate=optimizer.param_groups[0]['lr'])
        if scheduler is not None:
            scheduler.step(record['training_bce'])
        if validation is not None:
            score = participant_balanced_brier(vy, predict_binary(model, vx, training['batch_size']), vp)
            record['validation_brier'] = score
            if score < best - 1e-8:
                best, best_epoch = score, epoch
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(record)
        if validation is not None and epoch >= training['minimum_epochs'] and epoch-best_epoch >= training['patience']:
            break
    if validation is not None:
        if state is None:
            raise FloatingPointError('No finite source-validation checkpoint')
        model.load_state_dict(state)
    else:
        best_epoch = int(maximum)
    return model, dict(history=history, best_epoch=best_epoch, epochs_run=len(history),
                       validation_brier=None if validation is None else best,
                       selected_epoch_at_ceiling=validation is not None and best_epoch == maximum,
                       seconds=time.perf_counter()-started,
                       parameters=sum(p.numel() for p in model.parameters()))


def fit_source_member(x, y, participants, families, record_ids, model_name, candidates,
                      member_seed, fold_seed, n_splits=4):
    """Select/refit one member using source records only; return auditable OOF scores."""
    y = np.asarray(y)
    participants, families = _ids(participants, len(y)), _ids(families, len(y))
    record_ids = _ids(record_ids, len(y))
    if len(np.unique(record_ids)) != len(y):
        raise ValueError('Source record identifiers must be unique')
    participant_state_weights(y, participants)
    if len(x) != len(y):
        raise ValueError('Source tensor/target lengths differ')
    folds = family_folds(participants, families, n_splits, fold_seed)
    spec, training = candidates['architecture'], candidates['training']
    reports = []
    for lr, decay in itertools.product(training['learning_rates'], training['weight_decays']):
        oof = np.full(len(y), np.nan)
        logs = []
        for number, (train, valid) in enumerate(folds):
            model, log = fit_network(x[train], y[train], participants[train], model_name, spec, training,
                                     lr, decay, member_seed+1000*number,
                                     validation=(x[valid], y[valid], participants[valid]))
            probability = predict_binary(model, x[valid], training['batch_size'])
            np.testing.assert_allclose(participant_balanced_brier(y[valid], probability, participants[valid]),
                                       log['validation_brier'], rtol=0, atol=1e-7)
            oof[valid] = probability
            logs.append(dict(**log, training_ids=record_ids[train].tolist(),
                             validation_ids=record_ids[valid].tolist(),
                             validation_probability=probability.tolist()))
            del model
        # Pool out-of-fold predictions before averaging participant scores;
        # equal fold weighting would overweight smaller validation folds.
        reports.append(dict(lr=lr, decay=decay, folds=logs, oof_probability=oof.tolist(),
                            mean_participant_brier=participant_balanced_brier(y, oof, participants)))
    chosen = min(reports, key=lambda item: item['mean_participant_brier'])
    epochs = max(1, int(np.median([fold['best_epoch'] for fold in chosen['folds']])))
    model, log = fit_network(x, y, participants, model_name, spec, training, chosen['lr'], chosen['decay'],
                             member_seed, epochs=epochs)
    return model, dict(candidates=reports, selected_lr=chosen['lr'], selected_decay=chosen['decay'],
                       selected_epochs=epochs, final=log, member_seed=member_seed, fold_seed=fold_seed,
                       record_ids=record_ids.tolist(), y=y.tolist(), participants=participants.tolist(),
                       family_count=len(np.unique(families)), participant_count=len(np.unique(participants)),
                       target_data_used=False, weighting='equal participant, then equal class')
