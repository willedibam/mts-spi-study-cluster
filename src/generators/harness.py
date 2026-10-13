"""What the experiment runner needs from a generator beyond its series.

Most generators return only a (T, M) array and need nothing here. A generator
that also yields saved artefacts or metadata registers one adapter below, and
the runner stays free of generator names:

- ``generate(spec)`` returns ``(data, extras)``. ``extras`` is JSON metadata,
  except three transient keys the runner writes to files and removes:
  ``_ground_truth`` (dict of arrays, ``ground_truth.npz``), ``_full_lattice``
  (``full_lattice.npy``) and ``_mother`` (``mother.npy``).
- ``expected_files(spec)`` names the artefacts a finished dataset must have.
- ``semantics(spec, extras)`` is the generator's descriptive metadata block.

``spec`` needs ``generator``, ``M``, ``T``, ``rng_seed`` and ``generator_params``.
"""
from __future__ import annotations

import inspect
from typing import Any, Dict

import numpy as np

from .. import generators as generate

_OUTPUTS: dict = {}
_SEMANTICS: dict = {}
_GROUND_TRUTH: set = set()


def _outputs(*names, ground_truth=False):
    def register(function):
        for name in names:
            _OUTPUTS[name] = function
            if ground_truth:
                _GROUND_TRUTH.add(name)
        return function
    return register


def _semantics(name):
    def register(function):
        _SEMANTICS[name] = function
        return function
    return register


def generate_from_spec(spec) -> tuple[np.ndarray, dict]:
    """Generate one dataset's series and extras; no disk I/O or caching."""
    generator_params = dict(spec.generator_params)
    data, gen_extras = _OUTPUTS.get(spec.generator, _plain)(spec, generator_params)
    gen_extras.setdefault("resolved_params", resolve_params(spec.generator, generator_params))
    return data, gen_extras


def expected_files(spec) -> list[str]:
    """Artefacts, besides the series, that this dataset's generator writes."""
    files = []
    if spec.generator in _GROUND_TRUTH:
        files.append("ground_truth.npz")
    if spec.generator == "cml_logistic" and spec.generator_params.get("return_full_lattice", False):
        files.append("full_lattice.npy")
    return files


def semantics(spec, gen_extras: Dict[str, Any]) -> Dict[str, Any]:
    describe = _SEMANTICS.get(spec.generator)
    return describe(spec, gen_extras) if describe else {}


def resolve_params(generator: str | None, provided: dict) -> dict:
    """Full effective parameter set: generator-signature defaults overlaid with
    provided params. Records what actually ran, including unspecified defaults.
    Best-effort: returns just `provided` if the signature can't be introspected.
    """
    skip = {"M", "T", "rng", "seed", "return_internals", "return_final_state", "init_state"}
    try:
        fn = generate.GENERATOR_REGISTRY.get(generator)
        sig = inspect.signature(fn)
        resolved: dict = {}
        for name, p in sig.parameters.items():
            if name in skip or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            if p.default is not inspect._empty:
                resolved[name] = p.default
        resolved.update(provided)
        return resolved
    except (TypeError, ValueError):
        return dict(provided)


def _plain(spec, generator_params):
    data = generate.generate_series(
        spec.generator,
        seed=spec.rng_seed,
        M=spec.M,
        T=spec.T,
        **generator_params,
    )
    return data, {}


@_outputs("sin_mts_smooth")
def _sin_mts_smooth(spec, generator_params):
    data, internals = generate.generate_sin_mts_smooth(
        M=spec.M,
        T=spec.T,
        rng=np.random.default_rng(spec.rng_seed),
        return_internals=True,
        **generator_params,
    )
    gen_extras = {"a_values": internals.a_values.tolist()}
    return data, gen_extras


@_outputs("lagged_mts")
def _lagged_mts(spec, generator_params):
    data, internals = generate.generate_lagged_mts(
        M=spec.M,
        T=spec.T,
        rng=np.random.default_rng(spec.rng_seed),
        return_internals=True,
        **generator_params,
    )
    gen_extras = {"lags": internals.lags.tolist()}
    return data, gen_extras


@_outputs("lagged_warping_mts")
def _lagged_warping_mts(spec, generator_params):
    data, internals = generate.generate_lagged_warping_mts(
        M=spec.M,
        T=spec.T,
        rng=np.random.default_rng(spec.rng_seed),
        return_internals=True,
        **generator_params,
    )
    gen_extras = {"lags": internals.lags.tolist()}
    return data, gen_extras


@_outputs("filter_roll_mts")
def _filter_roll_mts(spec, generator_params):
    data, internals = generate.generate_filter_roll_mts(
        M=spec.M,
        T=spec.T,
        rng=np.random.default_rng(spec.rng_seed),
        return_internals=True,
        **generator_params,
    )
    gen_extras = {
        "types": internals.types,
        "betas": [None if np.isnan(b) else float(b) for b in internals.betas],
        "noise_stds": internals.noise_stds.tolist(),
        # transient: persisted to mother.npy by _ensure_timeseries, popped before meta.json
        "_mother": internals.mother,
    }
    return data, gen_extras


@_outputs("var_chat_a", "var_chat_b", "var_chat_c", "var_chat_d")
def _var_chat(spec, generator_params):
    _chat_gen = {
        "var_chat_a": generate.generate_var_chat_a,
        "var_chat_b": generate.generate_var_chat_b,
        "var_chat_c": generate.generate_var_chat_c,
        "var_chat_d": generate.generate_var_chat_d,
    }[spec.generator]
    data, internals = _chat_gen(
        M=spec.M,
        T=spec.T,
        rng=np.random.default_rng(spec.rng_seed),
        return_internals=True,
        **generator_params,
    )
    gen_extras = {
        "motif_node_indices": internals.motif_node_indices,
        "motif_edges": [list(e) for e in internals.motif_edges],
        "class_label": internals.class_label,
        "coupling_values": internals.coupling_values,
    }
    return data, gen_extras


@_outputs("cml_logistic")
def _cml_logistic(spec, generator_params):
    if not generator_params.get("return_full_lattice", False):
        return _plain(spec, generator_params)
    if generator_params.get("return_final_state", False):
        raise ValueError(
            "The experiment harness cannot persist both CML return_full_lattice "
            "and return_final_state; use direct generator calls for continuation."
        )
    generated = generate.generate_cml_logistic(
        M=spec.M,
        T=spec.T,
        rng=np.random.default_rng(spec.rng_seed),
        **generator_params,
    )
    if generator_params.get("return_observation_indices", False):
        data, full_lattice, observation_indices = generated
        gen_extras = {
            "_full_lattice": full_lattice,
            "observation_indices": np.asarray(
                observation_indices, dtype=np.int32
            ).tolist(),
        }
    else:
        data, full_lattice = generated
        gen_extras = {"_full_lattice": full_lattice}
    return data, gen_extras


@_outputs("kuramoto_order_parameter", ground_truth=True)
def _kuramoto_order_parameter(spec, generator_params):
    generator_params.pop("return_internals", None)
    store_full_phases = bool(generator_params.pop("store_full_phases", True))
    data, internals = generate.generate_kuramoto_order_parameter(
        M=spec.M,
        T=spec.T,
        rng=np.random.default_rng(spec.rng_seed),
        return_internals=True,
        store_full_phases=store_full_phases,
        **generator_params,
    )
    ground_truth = {
        "r_full": internals.r_full.astype(np.float32),
        "r_observed": internals.r_observed.astype(np.float32),
        "r_unobserved": internals.r_unobserved.astype(np.float32),
        "frequencies": internals.frequencies.astype(np.float32),
        "observation_indices": internals.observation_indices.astype(np.int32),
        "sensor_offsets": internals.sensor_offsets.astype(np.float32),
        "initial_phases": internals.initial_phases.astype(np.float32),
        "final_phases": internals.final_phases.astype(np.float32),
        "critical_coupling": np.array(internals.critical_coupling),
    }
    if internals.full_phases is not None:
        ground_truth["full_phases"] = internals.full_phases.astype(np.float32)
    gen_extras = {
        "_ground_truth": ground_truth,
        "resolved_params": resolve_params(
            spec.generator,
            {**generator_params, "store_full_phases": store_full_phases},
        ),
    }
    if internals.r_full_future.size:
        gen_extras["_ground_truth"].update(
            {
                "r_full_future": internals.r_full_future.astype(np.float32),
                "r_unobserved_future": internals.r_unobserved_future.astype(np.float32),
            }
        )
    return data, gen_extras


@_outputs("miller_huse", ground_truth=True)
def _miller_huse(spec, generator_params):
    generator_params.pop("return_internals", None)
    store_full_field = bool(generator_params.pop("store_full_field", False))
    persist_future_truth_series = bool(
        generator_params.pop("persist_future_truth_series", True)
    )
    data, internals = generate.generate_miller_huse(
        M=spec.M,
        T=spec.T,
        rng=np.random.default_rng(spec.rng_seed),
        return_internals=True,
        store_full_field=store_full_field,
        **generator_params,
    )
    primary = (
        internals.spin_magnetization_future
        if internals.spin_magnetization_future.size
        else internals.spin_magnetization
    )
    hidden = (
        internals.spin_magnetization_unobserved_future
        if internals.spin_magnetization_unobserved_future.size
        else internals.spin_magnetization_unobserved
    )
    second = float(np.mean(primary**2))
    fourth = float(np.mean(primary**4))
    binder = 1.0 - fourth / (3.0 * second**2) if second > 0.0 else np.nan
    susceptibility = internals.final_field.size * (
        second - float(np.mean(np.abs(primary))) ** 2
    )
    truth_block_count = min(8, len(primary))
    primary_blocks = np.asarray(
        [
            np.mean(np.abs(block))
            for block in np.array_split(primary, truth_block_count)
        ],
        dtype=np.float32,
    )
    hidden_blocks = np.asarray(
        [
            np.mean(np.abs(block))
            for block in np.array_split(hidden, truth_block_count)
        ],
        dtype=np.float32,
    )
    ground_truth = {
        "magnetization": internals.magnetization.astype(np.float32),
        "spin_magnetization": internals.spin_magnetization.astype(np.float32),
        "spin_magnetization_unobserved": (
            internals.spin_magnetization_unobserved.astype(np.float32)
        ),
        "q_spin_rms": np.array(np.sqrt(np.mean(primary**2)), dtype=np.float32),
        "q_spin_abs": np.array(np.mean(np.abs(primary)), dtype=np.float32),
        "q_spin_abs_blocks": primary_blocks,
        "q_spin_rms_unobserved": np.array(
            np.sqrt(np.mean(hidden**2)), dtype=np.float32
        ),
        "q_spin_abs_unobserved": np.array(
            np.mean(np.abs(hidden)), dtype=np.float32
        ),
        "q_spin_abs_unobserved_blocks": hidden_blocks,
        "spin_m2": np.array(second, dtype=np.float32),
        "spin_m4": np.array(fourth, dtype=np.float32),
        "spin_binder_cumulant": np.array(binder, dtype=np.float32),
        "spin_susceptibility": np.array(susceptibility, dtype=np.float32),
        "patch_indices": internals.patch_indices.astype(np.int32),
        "initial_field": internals.initial_field.astype(np.float32),
        "final_field": internals.final_field.astype(np.float32),
    }
    if persist_future_truth_series:
        ground_truth.update(
            {
                "magnetization_future": internals.magnetization_future.astype(
                    np.float32
                ),
                "spin_magnetization_future": (
                    internals.spin_magnetization_future.astype(np.float32)
                ),
                "spin_magnetization_unobserved_future": (
                    internals.spin_magnetization_unobserved_future.astype(np.float32)
                ),
            }
        )
    if internals.full_field is not None:
        ground_truth["full_field"] = internals.full_field.astype(np.float32)
    gen_extras = {
        "_ground_truth": ground_truth,
        "resolved_params": resolve_params(
            spec.generator,
            {
                **generator_params,
                "store_full_field": store_full_field,
                "persist_future_truth_series": persist_future_truth_series,
            },
        ),
    }
    return data, gen_extras


@_outputs("desai_zwanzig", ground_truth=True)
def _desai_zwanzig(spec, generator_params):
    generator_params.pop("return_internals", None)
    store_full_states = bool(generator_params.pop("store_full_states", False))
    data, internals = generate.generate_desai_zwanzig(
        M=spec.M,
        T=spec.T,
        rng=np.random.default_rng(spec.rng_seed),
        return_internals=True,
        store_full_states=store_full_states,
        **generator_params,
    )
    primary = (
        internals.mean_field_future
        if internals.mean_field_future.size
        else internals.mean_field
    )
    truth_block_count = min(8, len(primary))
    ground_truth = {
        "mean_field": internals.mean_field.astype(np.float32),
        "mean_field_future": internals.mean_field_future.astype(np.float32),
        "q_mean_abs": np.array(np.mean(np.abs(primary)), dtype=np.float32),
        "q_mean_rms": np.array(np.sqrt(np.mean(primary**2)), dtype=np.float32),
        "q_mean_signed": np.array(np.mean(primary), dtype=np.float32),
        "q_mean_abs_blocks": np.asarray(
            [
                np.mean(np.abs(block))
                for block in np.array_split(primary, truth_block_count)
            ],
            dtype=np.float32,
        ),
        "observation_indices": internals.observation_indices.astype(np.int32),
        "initial_state": internals.initial_state.astype(np.float32),
        "final_state": internals.final_state.astype(np.float32),
        "reference_mean_field_sigma_c": np.array(
            generate.DESAI_ZWANZIG_REFERENCE_SIGMA_C, dtype=np.float32
        ),
    }
    if internals.full_states is not None:
        ground_truth["full_states"] = internals.full_states.astype(np.float32)
    gen_extras = {
        "_ground_truth": ground_truth,
        "resolved_params": resolve_params(
            spec.generator,
            {**generator_params, "store_full_states": store_full_states},
        ),
    }
    return data, gen_extras


@_outputs("stuart_landau", ground_truth=True)
def _stuart_landau(spec, generator_params):
    generator_params.pop("return_internals", None)
    store_full_states = bool(generator_params.pop("store_full_states", False))
    data, internals = generate.generate_stuart_landau(
        M=spec.M,
        T=spec.T,
        rng=np.random.default_rng(spec.rng_seed),
        return_internals=True,
        store_full_states=store_full_states,
        **generator_params,
    )
    primary_order = (
        internals.order_parameter_future
        if internals.order_parameter_future.size
        else internals.order_parameter
    )
    primary_activity = (
        internals.mean_activity_future
        if internals.mean_activity_future.size
        else internals.mean_activity
    )
    ground_truth = {
        "order_parameter_R": np.abs(internals.order_parameter).astype(np.float32),
        "order_parameter_phase": np.angle(internals.order_parameter).astype(np.float32),
        "mean_activity": internals.mean_activity.astype(np.float32),
        "order_parameter_R_future": np.abs(
            internals.order_parameter_future
        ).astype(np.float32),
        "order_parameter_phase_future": np.angle(
            internals.order_parameter_future
        ).astype(np.float32),
        "mean_activity_future": internals.mean_activity_future.astype(np.float32),
        "q_R_mean": np.array(np.mean(np.abs(primary_order)), dtype=np.float32),
        "q_R_std": np.array(np.std(np.abs(primary_order)), dtype=np.float32),
        "q_activity_mean": np.array(np.mean(primary_activity), dtype=np.float32),
        "frequencies": internals.frequencies.astype(np.float32),
        "observation_indices": internals.observation_indices.astype(np.int32),
        "initial_state_real": internals.initial_state.real.astype(np.float32),
        "initial_state_imag": internals.initial_state.imag.astype(np.float32),
        "final_state_real": internals.final_state.real.astype(np.float32),
        "final_state_imag": internals.final_state.imag.astype(np.float32),
    }
    if internals.full_states is not None:
        ground_truth["full_states_real"] = internals.full_states.real.astype(np.float32)
        ground_truth["full_states_imag"] = internals.full_states.imag.astype(np.float32)
    gen_extras = {
        "_ground_truth": ground_truth,
        "resolved_params": resolve_params(
            spec.generator,
            {**generator_params, "store_full_states": store_full_states},
        ),
    }
    return data, gen_extras


@_outputs("quadratic_cml_order_parameter", ground_truth=True)
def _quadratic_cml_order_parameter(spec, generator_params):
    generator_params.pop("return_internals", None)
    store_truth_field = bool(generator_params.pop("store_truth_field", False))
    data, internals = generate.generate_quadratic_cml_order_parameter(
        M=spec.M,
        T=spec.T,
        rng=np.random.default_rng(spec.rng_seed),
        return_internals=True,
        store_truth_field=store_truth_field,
        **generator_params,
    )
    summary = internals.truth_summary
    turbulent_fraction = summary["turbulent_fraction"]
    ground_truth = {
        "q_temporal_spectral_entropy": np.array(
            summary["temporal_spectral_entropy"], dtype=np.float32
        ),
        "q_dynamical_spatial_pattern_entropy": np.array(
            summary["dynamical_spatial_pattern_entropy"], dtype=np.float32
        ),
        "q_selected_band_power": np.array(
            summary["selected_band_power"], dtype=np.float32
        ),
        "q_period2_activity": np.array(
            summary["period2_activity"], dtype=np.float32
        ),
        "q_turbulent_fraction_0p05": np.array(
            turbulent_fraction["0.05"], dtype=np.float32
        ),
        "q_regime_vector": np.array(
            [
                summary["temporal_spectral_entropy"],
                summary["dynamical_spatial_pattern_entropy"],
                summary["selected_band_power"],
                summary["period2_activity"],
            ],
            dtype=np.float32,
        ),
        "spatial_power_distribution": np.asarray(
            summary["spatial_power_distribution"], dtype=np.float32
        ),
        "spatial_correlation": np.asarray(
            summary["spatial_correlation"], dtype=np.float32
        ),
        "observation_indices": internals.observation_indices.astype(np.int32),
        "final_state": internals.final_state.astype(np.float32),
    }
    if internals.truth_field is not None:
        ground_truth["truth_field"] = internals.truth_field.astype(np.float32)
    gen_extras = {
        "_ground_truth": ground_truth,
        "resolved_params": resolve_params(
            spec.generator,
            {**generator_params, "store_truth_field": store_truth_field},
        ),
    }
    return data, gen_extras


@_semantics("kuramoto_order_parameter")
def _kuramoto_semantics(spec, gen_extras: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(gen_extras.get("resolved_params") or spec.generator_params)
    distribution = str(params.get("frequency_distribution", "gaussian"))
    omega_std = float(params.get("omega_std", 1.0))
    critical = float(
        (gen_extras.get("ground_truth") or {}).get(
            "critical_coupling",
            generate.kuramoto_critical_coupling(distribution, omega_std),
        )
    )
    coupling = float(params.get("K", critical))
    future_truth = int(params.get("future_truth_T", 0)) > 0
    return {
        "control": {
            "name": "K",
            "value": coupling,
            "continuum_critical_value": critical,
            "reduced_name": "kappa",
            "reduced_value": coupling / critical,
        },
        "order_parameter": {
            "name": "Kuramoto phase coherence",
            "definition": "R_N(t)=abs(mean_j(exp(i*theta_j(t))))",
            "canonical_full_array": "r_full",
            "primary_analysis_array": "r_full_future" if future_truth else "r_full",
            "hidden_complement_sensitivity_array": (
                "r_unobserved_future" if future_truth else "r_unobserved"
            ),
            "observed_subset_diagnostic_array": "r_observed",
            "future_truth_disjoint_from_input_window": future_truth,
            "included_in_timeseries_input": False,
        },
    }


@_semantics("miller_huse")
def _miller_huse_semantics(spec, gen_extras: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(gen_extras.get("resolved_params") or spec.generator_params)
    future_truth = int(params.get("future_truth_T", 0)) > 0
    return {
        "control": {
            "name": "coupling",
            "value": float(params.get("coupling", 0.205)),
            "path_parameter": "mu",
            "path_value": float(params.get("mu", 3.0)),
        },
        "order_parameter": {
            "name": "Miller--Huse spin magnetization",
            "definition": "m_s(t)=mean_r(1[x_r(t)>=0]-1[x_r(t)<0])",
            "finite_system_scalar": "mean_t(abs(m_s(t)))",
            "primary_analysis_array": (
                "spin_magnetization_future"
                if future_truth and bool(params.get("persist_future_truth_series", True))
                else None
            ),
            "primary_scalar": "q_spin_abs",
            "primary_block_summary": "q_spin_abs_blocks",
            "hidden_complement_sensitivity_array": (
                "spin_magnetization_unobserved_future"
                if future_truth
                else "spin_magnetization_unobserved"
            ),
            "hidden_complement_scalar": "q_spin_abs_unobserved",
            "hidden_complement_block_summary": "q_spin_abs_unobserved_blocks",
            "rms_magnetization_sensitivity_scalar": "q_spin_rms",
            "future_truth_disjoint_from_input_window": future_truth,
            "included_in_timeseries_input": False,
        },
    }


@_semantics("desai_zwanzig")
def _desai_zwanzig_semantics(spec, gen_extras: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(gen_extras.get("resolved_params") or spec.generator_params)
    future_truth = int(params.get("future_truth_T", 0)) > 0
    return {
        "control": {
            "name": "sigma",
            "value": float(params.get("sigma", 1.890)),
            "plane_parameter": "theta",
            "plane_value": float(params.get("theta", 4.0)),
        },
        "order_parameter": {
            "name": "Desai--Zwanzig first moment",
            "definition": "M1(t)=mean_i(x_i(t))",
            "finite_system_scalar": "mean_t(abs(M1(t)))",
            "primary_analysis_array": (
                "mean_field_future" if future_truth else "mean_field"
            ),
            "primary_scalar": "q_mean_abs",
            "primary_block_summary": "q_mean_abs_blocks",
            "signed_scalar": "q_mean_signed",
            "rms_sensitivity_scalar": "q_mean_rms",
            "mean_field_reference_sigma_c": 1.890,
            "future_truth_disjoint_from_input_window": future_truth,
            "included_in_timeseries_input": False,
        },
    }


@_semantics("stuart_landau")
def _stuart_landau_semantics(spec, gen_extras: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(gen_extras.get("resolved_params") or spec.generator_params)
    future_truth = int(params.get("future_truth_T", 0)) > 0
    return {
        "control": {
            "name": "frequency_half_width",
            "value": float(params.get("frequency_half_width", 0.8)),
            "plane_parameter": "coupling",
            "plane_value": float(params.get("coupling", 0.8)),
        },
        "order_parameter": {
            "name": "Stuart--Landau complex mean field",
            "definition": "Z(t)=mean_j(z_j(t)); R(t)=abs(Z(t))",
            "primary_analysis_array": (
                "order_parameter_R_future" if future_truth else "order_parameter_R"
            ),
            "primary_scalar": "q_R_mean",
            "variability_scalar": "q_R_std",
            "activity_scalar": "q_activity_mean",
            "future_truth_disjoint_from_input_window": future_truth,
            "included_in_timeseries_input": False,
        },
    }


@_semantics("quadratic_cml_order_parameter")
def _quadratic_cml_semantics(spec, gen_extras: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(gen_extras.get("resolved_params") or spec.generator_params)
    return {
        "control": {
            "name": "alpha",
            "value": float(params.get("alpha", 1.8)),
            "path_parameter": "eps",
            "path_value": float(params.get("eps", 0.3)),
        },
        "order_parameter": {
            "name": "quadratic-CML physical regime-coordinate vector",
            "canonical_scalar_order_parameter": False,
            "definition": (
                "(temporal spectral entropy, dynamical spatial-pattern entropy, "
                "selected-band power, period-two residual) on a disjoint "
                "future full-lattice window"
            ),
            "primary_vector": "q_regime_vector",
            "scalar_diagnostics": [
                "q_temporal_spectral_entropy",
                "q_dynamical_spatial_pattern_entropy",
                "q_selected_band_power",
                "q_period2_activity",
            ],
            "future_truth_disjoint_from_input_window": True,
            "included_in_timeseries_input": False,
        },
    }
