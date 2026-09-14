import numpy as np

from src.hcp_evaluation import evaluate_with_schedule_slice, participant_scores, summarize


def test_participant_scores_do_not_overweight_people_with_more_blocks():
    y = np.array([0, 1, 0, 0, 0, 1])
    people = np.array(['a', 'a', 'b', 'b', 'b', 'b'])
    p = np.array([0, 1, 1, 1, 1, 0])[:, None]
    _, _, values = participant_scores(y, p, people, people)
    np.testing.assert_array_equal(values[:, 0], [[1, 1, 0], [0, 0, 1]])
    result = summarize(y, p, people, people, ['model'], seed=41)
    assert result['models']['model']['balanced_accuracy']['mean'] == .5


def test_bootstrap_keeps_families_and_models_paired():
    people = np.repeat(['a', 'b', 'c'], 2)
    family = np.repeat(['related', 'related', 'other'], 2)
    y = np.tile([0, 1], 3)
    p = np.array([0., 1., 0., 1., 1., 0.])
    result = summarize(y, np.column_stack([p, p]), people, family, ['first', 'same'], seed=9)
    for metric in result['paired_differences'][0]['metrics'].values():
        assert metric == dict(mean=0., lower=0., upper=0.)
    assert result['families'] == 2 and result['participants'] == 3
    assert result['models']['first']['balanced_accuracy']['mean'] == 2/3
    # Independent explicit resampling reproduces the interval with unequal family sizes.
    rng = np.random.default_rng(9)
    memberships = [np.array([2]), np.array([0, 1])]  # sorted family order
    scores = np.array([1., 1., 0.])
    manual = [scores[np.concatenate([memberships[j] for j in draw])].mean()
              for draw in rng.integers(2, size=(2000, 2))]
    expected = np.quantile(manual, [.025, .975])
    observed = result['models']['first']['balanced_accuracy']
    np.testing.assert_array_equal(expected, [observed['lower'], observed['upper']])


def test_schedule_subset_exclusions_are_shared_and_explicit():
    y = np.tile([0, 1, 0, 1], 3)
    people = np.repeat(['a', 'b', 'c'], 4)
    blocks = np.tile([1, 10, 2, 4], 3)
    blocks[-3] = 3  # c has only one class at the discordant positions.
    p = np.column_stack([y*.8+.1, np.full(len(y), .5)])
    result = evaluate_with_schedule_slice(y, p, people, people, blocks, ['accurate', 'constant'],
                                          discordant_blocks=[1, 10, 13, 22], seed=29)
    diagnostic = result['discordant_positions']
    assert diagnostic['observations_at_positions'] == 5
    assert diagnostic['retained_observations'] == 4
    assert diagnostic['excluded_observations_for_class_coverage'] == 1
    assert diagnostic['excluded_participants'] == 1
    assert diagnostic['result']['participants'] == 2
    assert diagnostic['result']['models']['constant']['balanced_accuracy']['mean'] == .5
