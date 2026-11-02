import numpy as np

from mtsspi.character_axes import AXES, axis_scores, is_antisymmetric, profile_correlation
from mtsspi.surrogates import multivariate_iaaft, phase_randomized


def coupled(rng, T=2000, M=4):
    e = rng.normal(size=(T, M))
    x = e.copy()
    for m in range(1, M):
        x[:, m] = .6 * np.roll(x[:, m - 1], 3) + .8 * e[:, m]
    return (x - x.mean(0)) / x.std(0)


def test_phase_randomized_preserves_cross_spectrum():
    rng = np.random.default_rng(0)
    x = coupled(rng)
    s = phase_randomized(x, rng)
    fx, fs = np.fft.rfft(x, axis=0), np.fft.rfft(s, axis=0)
    cx = np.einsum('fi,fj->fij', fx, fx.conj())
    cs = np.einsum('fi,fj->fij', fs, fs.conj())
    assert np.allclose(cx, cs, atol=1e-8 * np.abs(cx).max())
    assert not np.allclose(x, s)


def test_iaaft_keeps_marginals_and_linear_coupling_but_removes_nonlinear():
    rng = np.random.default_rng(1)
    x = coupled(rng)
    x[:, 3] = x[:, 2] ** 2 + .3 * rng.normal(size=len(x))   # nonlinear, uncorrelated link
    s, info = multivariate_iaaft(x, rng)
    assert np.allclose(np.sort(s, 0), np.sort(x, 0))
    assert info['amplitude_relative_error'] < .05
    lagged = lambda a, b: np.corrcoef(a[3:], b[:-3])[0, 1]
    assert abs(lagged(x[:, 1], x[:, 0]) - lagged(s[:, 1], s[:, 0])) < .05
    sq = lambda a, b: np.corrcoef(a ** 2, b)[0, 1]
    assert sq(x[:, 2], x[:, 3]) > .5 and abs(sq(s[:, 2], s[:, 3])) < .2


def test_antisymmetric_profiles_enter_as_magnitudes():
    rng = np.random.default_rng(2)
    M = 5
    mask = ~np.eye(M, dtype=bool)
    sym = rng.random((M, M)); sym = sym + sym.T
    anti = np.triu(rng.random((M, M)) + .5, 1); anti = anti - anti.T
    anti_abs = np.abs(anti)
    names = ['cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-5', 'pli_multitaper_mean_fs-1_fmin-0_fmax-0-5']
    assert is_antisymmetric(names[1]) and not is_antisymmetric(names[0])
    # Raw signed antisymmetric profile is orthogonal to any symmetric one.
    assert abs(profile_correlation(sym[mask][None], anti[mask][None])[0]) < 1e-12
    profiles = np.stack([sym[mask], anti[mask]])[None]
    plv = 'plv_multitaper_mean_fs-1_fmin-0_fmax-0-5'
    _, values, pairs, _ = axis_scores(profiles, [plv, names[1]])
    k = pairs.index(('phase_lag', plv, names[1]))
    assert np.isclose(values[0, k], profile_correlation(sym[mask][None], anti_abs[mask][None])[0])


def test_axes_reference_distinct_pairs():
    for axis in AXES:
        assert len(set(axis.pairs)) == len(axis.pairs), axis.name
        assert all(a != b for a, b in axis.pairs), axis.name


def test_iaaft_spectrum_final_matches_spectra_exactly():
    rng = np.random.default_rng(3)
    x = coupled(rng)
    x[:, 0] = np.sign(x[:, 0]) * x[:, 0] ** 2
    s, info = multivariate_iaaft(x, rng, final='spectrum')
    assert info['amplitude_relative_error'] < 1e-10
    fx, fs = np.fft.rfft(x, axis=0), np.fft.rfft(s, axis=0)
    cross = lambda f: f[:, 0] * f[:, 1].conj()
    assert np.allclose(cross(fx), cross(fs), atol=1e-8 * np.abs(cross(fx)).max())


def test_core_transpose_and_levels():
    from mtsspi.character_axes import CORE, core_scores, mt, sgc
    rng = np.random.default_rng(4)
    M = 5
    mask = ~np.eye(M, dtype=bool)
    sym = rng.random((M, M)); sym = sym + sym.T
    directed = rng.random((M, M))
    hi, lo = rng.random((M, M)) + .1, rng.random((M, M)) + .1
    names = [sgc('mean', 'broad'), mt('cohmag', 'mean', 'high'), mt('cohmag', 'mean', 'low')]
    profiles = np.stack([sym[mask], hi[mask], lo[mask]])[None]
    scores, used = core_scores(profiles, names, M)
    assert used == ['band_high_share', 'direction']
    assert np.isclose(scores[0, used.index('direction')], 1.0)            # symmetric matrix: A equals A^T
    assert np.isclose(scores[0, used.index('band_high_share')], np.median((hi / (hi + lo))[mask]))
    profiles = np.stack([directed[mask], hi[mask], lo[mask]])[None]
    expected = np.corrcoef(directed[mask], directed.T[mask])[0, 1]
    assert np.isclose(core_scores(profiles, names, M)[0][0, 1], expected)
    assert all(role in ('core', 'sub') for role, *_ in CORE.values())
