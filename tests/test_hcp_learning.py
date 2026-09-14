from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.hcp_learning import (family_folds, fit_source_member, participant_balanced_brier,
                              participant_state_weights)
from src.inceptiontime_baseline import InceptionNetwork
from src.neurotycho_learning import predict_binary
from src.representation_state_neural import make_encoder


def test_equal_participant_class_mass_and_replication_invariance():
    y = np.array([0, 1, 0, 0, 0, 1])
    person = np.array(['a', 'a', 'b', 'b', 'b', 'b'])
    p = np.array([.1, .7, .3, .5, .4, .8])
    weights = participant_state_weights(y, person)
    for identifier in ('a', 'b'):
        for label in (0, 1):
            assert weights[(person == identifier) & (y == label)].sum() == pytest.approx(len(y)/4)
    expected = np.mean([(.1**2+.3**2)/2, (np.mean(np.array([.3, .5, .4])**2)+.2**2)/2])
    assert participant_balanced_brier(y, p, person) == pytest.approx(expected)
    ix = np.array([0, 1, 2, 3, 4, 2, 3, 4, 5])
    assert participant_balanced_brier(y[ix], p[ix], person[ix]) == pytest.approx(expected)
    with pytest.raises(ValueError, match='lacks class'):
        participant_state_weights([0, 0, 1], ['a', 'a', 'b'])
    with pytest.raises(ValueError, match='finite probabilities'):
        participant_balanced_brier(y, np.full(6, np.nan), person)


def test_twins_and_all_views_stay_in_one_fold():
    person = np.repeat(['a', 'b', 'c', 'd', 'e'], 4)
    family = np.repeat(['twins', 'twins', 'f2', 'f3', 'f4'], 4)
    folds = family_folds(person, family, 4, 17)
    visited = []
    for train, valid in folds:
        assert not set(family[train]) & set(family[valid])
        assert not set(person[train]) & set(person[valid])
        assert sorted(np.r_[train, valid]) == list(range(len(person)))
        visited.extend(valid)
    assert sorted(visited) == list(range(len(person)))
    for first, second in zip(folds, family_folds(person, family, 4, 17), strict=True):
        np.testing.assert_array_equal(first[1], second[1])
    broken = family.copy()
    broken[0] = 'other'
    with pytest.raises(ValueError, match='multiple families'):
        family_folds(person, broken, 4, 17)


@pytest.mark.parametrize('method', ['aligned_channel', 'inceptiontime'])
def test_actual_architecture_grouped_fit_and_checkpoint_replay(method):
    torch.set_num_threads(1)
    config = yaml.safe_load(Path('configs/analysis/hcp-neural-candidates-260915.yaml').read_text())
    candidates = deepcopy(config[method])
    # This is a software test on random short inputs, not an HCP model fit.
    candidates['training'].update(maximum_epochs=2, minimum_epochs=1, patience=1,
                                  batch_size=4, learning_rates=[.001], weight_decays=[.0001])
    x = torch.from_numpy(np.random.default_rng(17).normal(size=(16, 64, 32)).astype('float32'))
    y = np.tile([0, 1, 0, 1], 4)
    people = np.repeat(['a', 'b', 'c', 'd'], 4)
    families = np.repeat(['twins', 'twins', 'f2', 'f3'], 4)
    ids = np.array([f'record-{i}' for i in range(16)])
    model, report = fit_source_member(x, y, people, families, ids, method, candidates, 11, 123, n_splits=2)
    assert report['target_data_used'] is False
    assert report['family_count'] == 3 and report['participant_count'] == 4
    assert 1 <= report['selected_epochs'] <= 2
    choice = report['candidates'][0]
    manual = []
    for person in np.unique(people):
        probability = np.asarray(choice['oof_probability'])
        manual.append(np.mean([np.mean((probability[(people == person) & (y == label)]-label)**2)
                               for label in (0, 1)]))
    assert choice['mean_participant_brier'] == pytest.approx(np.mean(manual))
    for fold in choice['folds']:
        a = np.array([int(i.split('-')[1]) for i in fold['training_ids']])
        b = np.array([int(i.split('-')[1]) for i in fold['validation_ids']])
        assert not set(families[a]) & set(families[b])
    replay = (InceptionNetwork(candidates['architecture']) if method == 'inceptiontime'
              else make_encoder(candidates['architecture']))
    replay.load_state_dict(model.state_dict())
    np.testing.assert_array_equal(predict_binary(model, x, 4), predict_binary(replay, x, 4))
