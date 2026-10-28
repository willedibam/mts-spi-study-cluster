import numpy as np
import pytest
from scipy.stats import spearmanr

from mtsspi.generators import (
    band_delay_structure,
    generate_drifting_delay_mts,
    generate_form_ladder_mts,
    generate_series,
)


def test_form_ladder_monotone_map_moves_only_pearson():
    """Noise enters before the map: rank structure is identical across linear and monotone blocks."""
    lin = generate_form_ladder_mts(8, 4000, maps=("linear", "linear"), rng=1)
    mono = generate_form_ladder_mts(8, 4000, maps=("linear", "monotone"), rng=1)
    assert np.allclose(spearmanr(lin)[0], spearmanr(mono)[0], atol=1e-12)
    r_lin, r_mono = np.corrcoef(lin.T), np.corrcoef(mono.T)
    assert np.allclose(r_lin[:4, :4], r_mono[:4, :4])          # linear block untouched
    assert np.all(r_mono[:4, 4:] < 0.6 * r_lin[:4, 4:])         # linear-monotone pairs lose r


def test_form_ladder_fold_blocks_and_curvature_zero():
    x, it = generate_form_ladder_mts(12, 20000, rng=2, return_internals=True)
    r = np.corrcoef(x.T)
    assert np.max(np.abs(r[:4, 8:])) < 0.05 and np.mean(np.abs(r[4:8, 8:])) < 0.05   # fold pairs: r = 0 by symmetry
    assert it.types == ["linear"] * 4 + ["monotone"] * 4 + ["fold"] * 4
    assert np.allclose(it.noise_stds[:4], it.noise_stds[8:])
    assert np.allclose(generate_form_ladder_mts(8, 500, maps=("linear", "monotone"), curvature=0, rng=3),
                       generate_form_ladder_mts(8, 500, maps=("linear", "linear"), rng=3))
    with pytest.raises(ValueError):
        generate_form_ladder_mts(10, 100)


def test_dispersion_knob_interpolates_and_keeps_defaults():
    d0, _ = band_delay_structure(12, "dispersive", dispersion=0.0)
    dd, _ = band_delay_structure(12, "delayed")
    d1, _ = band_delay_structure(12, "dispersive")
    dh, _ = band_delay_structure(12, "dispersive", dispersion=0.5)
    assert np.array_equal(d0, dd)
    assert np.array_equal(d1, band_delay_structure(12, "dispersive", dispersion=1.0)[0])
    assert np.array_equal(dh[0], d1[0]) and not np.array_equal(dh[1], d1[1]) and not np.array_equal(dh[1], d1[0])
    with pytest.raises(ValueError):
        band_delay_structure(12, "dispersive", dispersion=1.5)


def test_drifting_delay_structure_and_registry():
    x, it = generate_drifting_delay_mts(6, 800, max_lag=3, drift=2, rng=0, return_internals=True)
    assert x.shape == (800, 6)
    assert np.allclose(it.delays, np.linspace(0, 3, 6)) and np.all(it.drift_amp <= 2)
    _, it2 = generate_drifting_delay_mts(6, 800, max_lag=3, drift=2, rng=9, return_internals=True)
    assert np.allclose(it.drift_amp, it2.drift_amp)              # structure fixed by structure_seed
    sync = generate_drifting_delay_mts(6, 4000, max_lag=0, drift=0, noise_low=1e-3, noise_high=1e-3, rng=1)
    assert np.corrcoef(sync.T).min() > 0.99
    assert generate_series("drifting_delay_mts", M=4, T=300, seed=0).shape == (300, 4)
    assert generate_series("form_ladder_mts", M=6, T=300, seed=0).shape == (300, 6)
