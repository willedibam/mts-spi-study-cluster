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
