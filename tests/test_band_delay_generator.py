import numpy as np
import pytest

from mtsspi.generators import (
    band_delay_cross_spectrum,
    band_spectrum,
    generate_band_delayed_mts,
    generate_series,
)

FAMILIES = ("synchronous", "delayed", "dispersive")


def _population(internals, n=2**14, lag_max=None):
    """Zero-lag correlation, max cross-correlation and P from the population spectrum."""
    f = np.fft.rfftfreq(n)
    S = band_delay_cross_spectrum(internals, f)
    w = np.full(len(f), 2.0)
    w[[0, -1]] = 1.0
    power = np.einsum("f,fjj->j", w, S).real
    N = np.sqrt(np.outer(power, power))
    c = np.fft.irfft(S, n, axis=0) * n / N
    lag_max = n // 4 if lag_max is None else lag_max
    window = np.concatenate([c[: lag_max + 1], c[n - lag_max:]])
    return c[0], window.max(axis=0), np.einsum("f,fjk->jk", w, np.abs(S)) / N


def test_band_spectrum_integrates_to_variance():
    f = np.fft.fftfreq(2**16)
    for centre, half_width in ((0.0, 0.08), (0.25, 0.15), (0.05, 0.05)):
        assert np.isclose(band_spectrum(f, centre, half_width, 1.7).mean(), 1.7, rtol=1e-6)


def test_families_share_spectral_magnitudes_and_nest():
    f = np.fft.rfftfreq(2**12)
    its = {fam: generate_band_delayed_mts(9, 1, family=fam, rng=0, return_internals=True)[1] for fam in FAMILIES}
    ref = np.abs(band_delay_cross_spectrum(its["synchronous"], f))
    iu = np.triu_indices(9, 1)
    for fam, it in its.items():
        assert np.allclose(np.abs(band_delay_cross_spectrum(it, f)), ref, atol=1e-14)
        r, xc, P = _population(it)
        assert np.all(r[iu] <= xc[iu] + 1e-12) and np.all(xc[iu] <= P[iu] + 1e-12)
        if fam != "dispersive":  # zero or linear phase: one delay aligns every frequency
            assert np.max(np.abs(xc - P)[iu]) < 1e-12
        else:
            assert np.mean(xc[iu] / P[iu]) < 0.9
    assert not its["synchronous"].delays.any()
    d = its["delayed"].delays
    assert np.array_equal(d[0], d[1]) and d.min() == 0 and d.max() == 15
    disp = its["dispersive"].delays
    assert np.array_equal(disp[0], d[0]) and np.array_equal(np.sort(disp[1]), np.sort(d[0]))
    assert not np.array_equal(disp[1], d[0])


def test_finite_sample_matches_population_spectrum():
    M, T = 6, 60_000
    data, it = generate_band_delayed_mts(M, T, family="dispersive", rng=4, return_internals=True)
    assert data.shape == (T, M)
    r, _, _ = _population(it)
    iu = np.triu_indices(M, 1)
    assert np.abs(np.corrcoef(data.T) - r)[iu].max() < 0.03
    # Slow-band-only check: low-pass the data, then the lagged covariance peaks at the slow delay difference.
    lo = np.fft.irfft(np.fft.rfft(data, axis=0) * (np.fft.rfftfreq(T) < 0.08)[:, None], T, axis=0)
    j, k = 0, M - 1
    dslow = it.delays[0, k] - it.delays[0, j]
    lags = np.arange(-30, 31)
    c = [np.dot(lo[max(0, -h): T - max(0, h), j], lo[max(0, h): T - max(0, -h), k]) for h in lags]
    assert lags[int(np.argmax(c))] == dslow


def test_validation_and_registry():
    with pytest.raises(ValueError):
        generate_band_delayed_mts(5, 10, family="lagged")
    with pytest.raises(ValueError):
        generate_band_delayed_mts(5, 10, bands=((0.0, 0.1), (0.15, 0.1)))  # overlap
    with pytest.raises(ValueError):
        generate_band_delayed_mts(5, 10, bands=((0.05, 0.1), (0.3, 0.1)))  # centre < half-width
    x = generate_series("band_delayed_mts", seed=2, M=5, T=64, family="delayed", lag_range=4)
    y = generate_series("band_delayed_mts", seed=2, M=5, T=64, family="delayed", lag_range=4)
    assert x.shape == (64, 5) and np.array_equal(x, y)
    _, a = generate_band_delayed_mts(5, 8, rng=1, return_internals=True)
    _, b = generate_band_delayed_mts(5, 8, rng=2, return_internals=True)
    assert np.array_equal(a.loadings, b.loadings) and np.array_equal(a.delays, b.delays)
    _, s = generate_band_delayed_mts(5, 8, shared_loadings=True, rng=1, return_internals=True)
    assert np.array_equal(s.loadings[0], s.loadings[1])
