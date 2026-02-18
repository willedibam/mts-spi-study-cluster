"""MTS generators package — split from the monolithic src/generate.py."""
from ._common import GeneratorFn, _maybe_zscore, _resolve_rng, _zscore_channels
from .cases import (
    generate_case_i,
    generate_case_ii,
    generate_case_iii,
    generate_mts_master,
)
from .dynamical import (
    generate_cml_logistic,
    generate_kuramoto,
    generate_kuramoto_all_to_all,
    generate_kuramoto_bidirectional_list,
    generate_kuramoto_grid_four,
    generate_mackey_glass,
)
from .linear import (
    generate_cauchy_noise,
    generate_exponential_noise,
    generate_gaussian_noise,
    generate_gbm,
    generate_varma,
    generate_varma_shuffled,
)
from .pde import generate_wave_1d, generate_wave_2d
from .registry import GENERATOR_REGISTRY, available_generators, generate_series

__all__ = [
    "GENERATOR_REGISTRY",
    "GeneratorFn",
    "available_generators",
    "generate_case_i",
    "generate_case_ii",
    "generate_case_iii",
    "generate_cauchy_noise",
    "generate_cml_logistic",
    "generate_exponential_noise",
    "generate_gaussian_noise",
    "generate_gbm",
    "generate_kuramoto",
    "generate_kuramoto_all_to_all",
    "generate_kuramoto_bidirectional_list",
    "generate_kuramoto_grid_four",
    "generate_mackey_glass",
    "generate_mts_master",
    "generate_series",
    "generate_varma",
    "generate_varma_shuffled",
    "generate_wave_1d",
    "generate_wave_2d",
]
