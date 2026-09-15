import numpy as np
import pytest

from scripts.analyze_hcp_short import frozen_pc1
from scripts.verify_hcp_short_analysis import replay_pc1


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_independent_replay_and_held_data_independence(dtype):
    rng = np.random.default_rng(49)
    features = rng.normal(size=(64, 120)).astype(dtype)
    features[:, 0] = 3.
    features[0, 1] = np.nan
    features[:2, 2] = np.nan
    train = np.arange(8)
    expected, expected_space, info, model = frozen_pc1(features, train)
    actual, space, explained = replay_pc1(features, train, model)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(space, expected_space)
    np.testing.assert_allclose(explained, info['training_explained_variance'])
    changed = features.copy()
    changed[8:] *= 100
    replay, _, _ = replay_pc1(changed, train, model)
    np.testing.assert_array_equal(replay[:8], actual[:8])
    assert not np.allclose(replay[8:], actual[8:])


def test_replay_rejects_wrong_source_transform_and_component():
    features = np.random.default_rng(27).normal(size=(64, 60))
    _, _, _, model = frozen_pc1(features, np.arange(8))
    bad = {**model, 'means': model['means'] + .1}
    with pytest.raises(AssertionError):
        replay_pc1(features, np.arange(8), bad)
    loading = np.zeros_like(model['loading'])
    loading[0] = 1.
    with pytest.raises(AssertionError):
        replay_pc1(features, np.arange(8), {**model, 'loading': loading})
