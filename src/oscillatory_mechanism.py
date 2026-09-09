"""Direct noisy phase coupling with the existing co-organization observation law."""
from __future__ import annotations

import numpy as np
from numba import njit

from src.oscillatory_coorganization import partitions, stationary_ar


def draw_parameters(seed, settings):
    rng = np.random.default_rng(seed)
    parameters = {
        name: float(rng.uniform(*settings[config_name]))
        for name, config_name in [
            ("carrier", "carrier_range"),
            ("phase_sd", "group_phase_sd_range"),
            ("coupling", "coupling_range"),
            ("amp_rho", "amp_rho_range"),
            ("amp_sd", "amp_sd_range"),
            ("phase_jitter", "phase_jitter_range"),
            ("noise", "noise_range"),
        ]
    }
    parameters["offsets"] = rng.uniform(-np.pi, np.pi, 32).tolist()
    parameters["gains"] = np.exp(rng.uniform(-.3, .3, 32)).tolist()
    parameters["order"] = rng.permutation(32).tolist()
    return parameters


@njit(cache=True)
def integrate_phase(initial, brownian, coupling, sigma, fine_dt, stride, save_every):
    """Integrate in the rotating frame; coarser paths sum fine Brownian increments."""
    state = initial.copy()
    dt = fine_dt * stride
    out = np.empty((len(brownian) // save_every, 32))
    for start in range(0, len(brownian), stride):
        sine, cosine = np.sin(state), np.cos(state)
        mean_sine, mean_cosine = np.zeros(4), np.zeros(4)
        for i in range(32):
            mean_sine[i // 8] += sine[i] / 8
            mean_cosine[i // 8] += cosine[i] / 8
        for i in range(32):
            noise = 0.0
            for j in range(stride):
                noise += brownian[start + j, i]
            g = i // 8
            state[i] += coupling * dt * (
                mean_sine[g] * cosine[i] - mean_cosine[g] * sine[i]
            ) + sigma * noise
        if (start + stride) % save_every == 0:
            out[(start + stride) // save_every - 1] = state
    return out


def simulate(aligned, seed, parameters, settings, t=1000, *, dt=None,
             coupling=None, return_latent=False):
    """Return observed channels; optional latent arrays are diagnostics only."""
    if settings["N"] != 32 or settings["phase_group_size"] != 8:
        raise ValueError("this frozen model requires four groups of eight")
    fs, fine_dt = settings["fs"], settings["fine_dt"]
    dt = settings["integration_dt"] if dt is None else dt
    stride = int(round(dt / fine_dt))
    save_every = int(round(1 / fs / fine_dt))
    if (stride < 1 or not np.isclose(stride * fine_dt, dt)
            or not np.isclose(save_every * fine_dt, 1 / fs)
            or save_every % stride or t < 2):
        raise ValueError("integration steps must divide the observation interval")
    burn = int(round(settings["burn_seconds"] * fs))
    if not np.isclose(burn / fs, settings["burn_seconds"]):
        raise ValueError("burn-in must contain whole observation intervals")
    init_rng, path_rng, amp_rng, jitter_rng, noise_rng = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)
    ]
    initial = init_rng.uniform(-np.pi, np.pi, 32)
    brownian = path_rng.normal(size=((t + burn) * save_every, 32)) * np.sqrt(fine_dt)
    sigma = np.sqrt(8 * fs) * parameters["phase_sd"]
    strength = parameters["coupling"] if coupling is None else coupling
    phase = integrate_phase(initial, brownian, strength, sigma, fine_dt, stride, save_every)[burn:]
    phase += 2 * np.pi * parameters["carrier"] * (np.arange(t) + burn + 1)[:, None] / fs
    phase_group, amp_group = partitions(aligned)
    global_amp = stationary_ar(amp_rng, t, 4, parameters["amp_rho"])
    local_amp = stationary_ar(amp_rng, t, 32, parameters["amp_rho"])
    logamp = parameters["amp_sd"] * (
        np.sqrt(.8) * global_amp[:, amp_group] + np.sqrt(.2) * local_amp
    )
    measured_phase = (phase + np.asarray(parameters["offsets"])
                      + parameters["phase_jitter"] * jitter_rng.normal(size=(t, 32)))
    x = np.asarray(parameters["gains"]) * (
        np.exp(logamp - .5 * parameters["amp_sd"] ** 2) * np.cos(measured_phase)
        + parameters["noise"] * noise_rng.normal(size=(t, 32))
    )
    order = np.asarray(parameters["order"])
    x = x[:, order].astype(np.float32)
    if not return_latent:
        return x
    return x, dict(phase=phase[:, order], logamp=logamp[:, order],
                   phase_group=phase_group[order], envelope_group=amp_group[order])


def phase_matrix(phase):
    unit = np.exp(1j * phase)
    return np.abs(unit.conj().T @ unit / len(unit))


def group_contrast(matrix, groups):
    offdiag = ~np.eye(len(groups), dtype=bool)
    within = groups[:, None] == groups[None, :]
    return float(matrix[offdiag & within].mean() - matrix[offdiag & ~within].mean())
