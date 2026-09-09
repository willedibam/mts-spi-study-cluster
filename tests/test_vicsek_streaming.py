import numpy as np
import pytest
import yaml

from scripts.scout_vicsek_streaming import advance, main, observation_ids, simulate


def reference(position, heading, noise, side, speed):
    delta = position[:, None] - position[None, :]
    delta -= side * np.rint(delta / side)
    neighbors = (delta**2).sum(axis=2) <= 1
    angle = np.angle(neighbors @ np.exp(1j * heading)) + noise
    new_position = (position + speed * np.column_stack([np.cos(angle), np.sin(angle)])) % side
    return new_position, (angle + np.pi) % (2 * np.pi) - np.pi


@pytest.mark.parametrize('side', [3., 3.7, 8.])
def test_linked_cells_agree_with_dense_oracle(side):
    rng = np.random.default_rng(924)
    position = rng.uniform(0, side, (80, 2))
    position[:2] = [[.01, .01], [side - .01, side - .01]]
    heading = rng.uniform(-np.pi, np.pi, 80)
    noise = rng.uniform(-1, 1, 80)
    before = position.copy()
    actual = advance(position, heading, noise, side, .5)
    expected = reference(position, heading, noise, side, .5)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, atol=1e-12)
    np.testing.assert_array_equal(position, before)


def test_zero_noise_aligned_state_and_new_velocity_streaming():
    position = np.array([[.1, .2], [1.5, 2.], [3.9, 3.9]])
    angles = np.full(3, .7)
    p, h = advance(position, angles, np.zeros(3), 4., .5)
    np.testing.assert_allclose(h, angles)
    np.testing.assert_allclose(p, (position + .5 * np.array([np.cos(.7), np.sin(.7)])) % 4)
    _, turned = advance(position, angles, np.full(3, .3), 4., .5)
    np.testing.assert_allclose(turned, 1.)


def test_views_are_unique_and_nested_rectangles():
    position = np.random.default_rng(5).uniform(0, 32, (100, 2))
    particles, bins = observation_ids(position, 32., 4., 12)
    assert particles.shape == bins.shape == (2, 32)
    for view in (*particles, *bins):
        assert len(np.unique(view)) == 32
    for m, nx, ny in [(8, 2, 4), (16, 4, 4), (32, 4, 8)]:
        assert len(np.unique(bins[1, :m] // 8)) == nx
        assert len(np.unique(bins[1, :m] % 8)) == ny


def test_reproducibility_and_physical_bounds():
    config = dict(density=.125, speed=.5, burn=3, samples=16, stride=1, bin_width=4.)
    case = dict(L=32, eta=.5, start='random', seed=432)
    a, _ = simulate(case, config)
    b, _ = simulate(case, config)
    for key in a:
        if isinstance(a[key], np.ndarray):
            np.testing.assert_array_equal(a[key], b[key])
    assert np.all((a['phi'] >= 0) & (a['phi'] <= 1))
    np.testing.assert_allclose(np.linalg.norm(a['particle_velocity'], axis=-1), 1., atol=1e-6)
    assert np.all(a['bin_counts'] >= 0)
    assert np.all(np.linalg.norm(a['bin_current'], axis=-1) <= a['bin_counts'] / 16 + 1e-6)


def test_full_noise_has_independent_heading_polarization_floor():
    config = dict(density=.125, speed=.5, burn=0, samples=512, stride=1, bin_width=4.)
    arrays, meta = simulate(dict(L=32, eta=1., start='ordered', seed=924), config)
    # Independent uniform headings give E(phi^2)=1/N; loose stochastic smoke bound.
    assert .75 < np.mean(arrays['phi']**2) * meta['N'] < 1.25


def test_cli_case_selection_and_overwrite_protection(tmp_path, monkeypatch):
    config = tmp_path / 'cases.yaml'
    config.write_text(yaml.safe_dump({
        'simulation': dict(density=.125, speed=.5, burn=0, samples=8, stride=1, bin_width=4.),
        'cases': [dict(L=32, eta=.5, start=start, seed=924) for start in ('ordered', 'random')]}))
    output = tmp_path / 'output'
    monkeypatch.setattr('sys.argv', ['scout', '--config', str(config), '--output-dir',
                                    str(output), '--case-index', '1'])
    main()
    assert sorted(p.name for p in output.iterdir()) == ['case-001.npz']
    original = (output / 'case-001.npz').read_bytes()
    with pytest.raises(FileExistsError):
        main()
    assert (output / 'case-001.npz').read_bytes() == original
