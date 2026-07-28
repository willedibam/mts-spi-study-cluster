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
    generate_kuramoto_explosive,
    generate_kuramoto_grid_four,
    generate_mackey_glass,
)
from .linear import (
    generate_cauchy_noise,
    generate_exponential_noise,
    generate_filter_roll_mts,
    generate_gaussian_noise,
    generate_gbm,
    generate_lagged_mts,
    generate_lagged_warping_mts,
    generate_sin_mts,
    generate_sin_mts_mother,
    generate_sin_mts_smooth,
    generate_varma,
    generate_varma_shuffled,
    generate_warping_mts,
)
from .chat import (
    generate_var_chat_a,
    generate_var_nonlinear_a,
    generate_var_obs_nonlinear_a,
    generate_var_chat_b,
    generate_var_chat_c,
    generate_var_chat_d,
)
from .pde import generate_wave_1d, generate_wave_1d_pulse, generate_wave_2d
from .topology import (
    generate_topology_chain,
    generate_topology_clustered,
    generate_topology_hub_spoke,
    generate_topology_uniform,
    generate_var_fourring,
    generate_var_lattice,
    generate_var_ring,
    generate_var_tworing,
)

GENERATOR_REGISTRY: Dict[str, GeneratorFn] = {
    "sin_mts": generate_sin_mts,
    "sin_mts_mother": generate_sin_mts_mother,
    "sin_mts_smooth": generate_sin_mts_smooth,
    "filter_roll_mts": generate_filter_roll_mts,
    "warping_mts": generate_warping_mts,
    "lagged_mts": generate_lagged_mts,
    "lagged_warping_mts": generate_lagged_warping_mts,
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
    "kuramoto_explosive": generate_kuramoto_explosive,
    "kuramoto_grid_four": generate_kuramoto_grid_four,
    "gaussian_noise": generate_gaussian_noise,
    "cauchy_noise": generate_cauchy_noise,
    "exponential_noise": generate_exponential_noise,
    "wave_1d": generate_wave_1d,
    "wave_1d_pulse": generate_wave_1d_pulse,
    "wave_2d": generate_wave_2d,
    "case_i": generate_case_i,
    "case_ii": generate_case_ii,
    "case_iii": generate_case_iii,
    "mts_master": generate_mts_master,
    "topology_uniform": generate_topology_uniform,
    "topology_hub_spoke": generate_topology_hub_spoke,
    "topology_chain": generate_topology_chain,
    "topology_clustered": generate_topology_clustered,
    "var_ring": generate_var_ring,
    "var_tworing": generate_var_tworing,
    "var_fourring": generate_var_fourring,
    "var_lattice": generate_var_lattice,
    "var_chat_a": generate_var_chat_a,
    "var_nonlinear_a": generate_var_nonlinear_a,
    "var_obs_nonlinear_a": generate_var_obs_nonlinear_a,
    "var_chat_b": generate_var_chat_b,
    "var_chat_c": generate_var_chat_c,
    "var_chat_d": generate_var_chat_d,
}


def available_generators() -> list[str]:
    return sorted(GENERATOR_REGISTRY.keys())


def generate_series(name: str, *, seed: int | None = None, **params: Any) -> np.ndarray:
    if name not in GENERATOR_REGISTRY:
        raise KeyError(f"Unknown generator '{name}'. Known: {available_generators()}")
    rng = _resolve_rng(seed)
    gen = GENERATOR_REGISTRY[name]
    return gen(rng=rng, **params)
