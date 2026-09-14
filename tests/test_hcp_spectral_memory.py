import numpy as np
import pytest
import spectral_connectivity.connectivity as backend

from src.hcp_spectral_memory import bounded_psi_memory, inner_combination_linear_memory


@pytest.mark.parametrize("shape,axis", [((1, 17, 4, 4), -3), ((9, 3, 4), 0), ((3, 4, 11), 2)])
def test_matches_all_pairs_with_missing_values(shape, axis):
    rng = np.random.default_rng(127)
    x = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    for missing in (False, True):
        if missing:
            x.flat[1] = np.nan
        np.testing.assert_allclose(inner_combination_linear_memory(x, axis),
                                   backend._inner_combination(x, axis), rtol=1e-12, atol=1e-12)


def test_actual_multitaper_and_restoration():
    from spectral_connectivity import Multitaper, Connectivity
    rng = np.random.default_rng(418)
    x = rng.normal(size=(256, 1, 5))
    x[2:, 0, 1] += x[:-2, 0, 0]
    conn = Connectivity.from_multitaper(Multitaper(x, sampling_frequency=1))
    original = backend._inner_combination
    expected = conn.phase_slope_index()
    with bounded_psi_memory():
        np.testing.assert_allclose(conn.phase_slope_index(), expected, rtol=1e-11, atol=1e-11)
    assert backend._inner_combination is original
    with pytest.raises(ValueError):
        with bounded_psi_memory():
            raise ValueError("deliberate")
    assert backend._inner_combination is original


@pytest.mark.parametrize("n", [0, 1])
def test_preserves_degenerate_frequency_failure(n):
    x = np.ones((1, n, 2, 2), dtype=complex)
    with pytest.raises(IndexError):
        backend._inner_combination(x)
    with pytest.raises(IndexError):
        inner_combination_linear_memory(x)
