from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ._common import GeneratorFn, _resolve_rng
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

GENERATOR_REGISTRY: Dict[str, GeneratorFn] = {
    "varma": generate_varma,
    "var": generate_varma,
    "varma_shuffled": generate_varma_shuffled,
    "cml_logistic": generate_cml_logistic,
    "gbm": generate_gbm,
    "geometric_brownian_motion": generate_gbm,
    "mackey_glass": generate_mackey_glass,
    "kuramoto": generate_kuramoto,
    "kuramoto_all_to_all": generate_kuramoto_all_to_all,
    "kuramoto_bidirectional_list": generate_kuramoto_bidirectional_list,
    "kuramoto_grid_four": generate_kuramoto_grid_four,
    "gaussian_noise": generate_gaussian_noise,
    "cauchy_noise": generate_cauchy_noise,
    "exponential_noise": generate_exponential_noise,
    "wave_1d": generate_wave_1d,
    "wave_2d": generate_wave_2d,
    "case_i": generate_case_i,
    "case_ii": generate_case_ii,
    "case_iii": generate_case_iii,
    "mts_master": generate_mts_master,
}


def available_generators() -> list[str]:
    return sorted(GENERATOR_REGISTRY.keys())


def generate_series(name: str, *, seed: int | None = None, **params: Any) -> np.ndarray:
    if name not in GENERATOR_REGISTRY:
        raise KeyError(f"Unknown generator '{name}'. Known: {available_generators()}")
    rng = _resolve_rng(seed)
    gen = GENERATOR_REGISTRY[name]
    return gen(rng=rng, **params)
