"""Physics-only, fully observed six-state Rössler pair (PRL 76, 1804, 1996).

All durations are physical time, not integration steps. The future reference is
streamed in ten blocks; only the M=6 observation and compact diagnostics remain.
This measures finite-time entrainment, not proof of an asymptotic locking threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from numba import njit

CHANNELS = ["x1", "y1", "z1", "x2", "y2", "z2"]
SOURCE = "https://doi.org/10.1103/PhysRevLett.76.1804"


@njit(cache=True)
def rhs(state, coupling, mismatch=.015):
    out = np.empty(6)
    for i in range(2):
        j = 3 * i
        other = 3 * (1 - i)
        omega = 1.0 + mismatch * (1 - 2 * i)
        x, y, z = state[j:j + 3]
        out[j] = -omega * y - z + coupling * (state[other] - x)
        out[j + 1] = omega * x + .15 * y
        out[j + 2] = .2 + z * (x - 10.)
    return out


@njit(cache=True)
def rk4_step(state, coupling, dt, mismatch=.015):
    k1 = rhs(state, coupling, mismatch)
    k2 = rhs(state + .5 * dt * k1, coupling, mismatch)
    k3 = rhs(state + .5 * dt * k2, coupling, mismatch)
    k4 = rhs(state + dt * k3, coupling, mismatch)
    return state + dt / 6. * (k1 + 2 * k2 + 2 * k3 + k4)


@njit(cache=True)
def advance(state, coupling, dt, steps):
    state = state.copy()
    for _ in range(steps):
        state = rk4_step(state, coupling, dt)
    return state


@njit(cache=True)
def measure(state, coupling, dt, steps, stride=0):
    """Stream step-resolved phases and crossings; optionally retain full states."""
    observed = np.empty((6, steps // stride if stride else 0))
    phase = np.array([np.arctan2(state[1], state[0]), np.arctan2(state[4], state[3])])
    gain = np.zeros(2)
    crossings = np.zeros(2)
    backwards = np.zeros(2)
    min_radius = np.full(2, np.inf)
    min_velocity = np.full(2, np.inf)
    max_increment = np.zeros(2)
    real_sum = 0.
    imag_sum = 0.
    initial_difference = phase[0] - phase[1]
    difference_min = initial_difference
    difference_max = initial_difference
    slip_anchor = initial_difference
    slips = np.zeros(2)
    for step in range(steps):
        previous = state
        state = rk4_step(state, coupling, dt)
        if not np.isfinite(state).all() or np.max(np.abs(state)) > 1e8:
            raise ValueError("Rössler trajectory became nonfinite or unbounded")
        for i in range(2):
            j = 3 * i
            angle = np.arctan2(state[j + 1], state[j])
            increment = (angle - phase[i] + np.pi) % (2 * np.pi) - np.pi
            gain[i] += increment
            phase[i] = angle
            min_radius[i] = min(min_radius[i], np.hypot(state[j], state[j + 1]))
            min_velocity[i] = min(min_velocity[i], increment / dt)
            max_increment[i] = max(max_increment[i], abs(increment))
            backwards[i] += increment < 0
            if previous[j + 1] <= 0 < state[j + 1] and state[j] > 0:
                crossings[i] += 1
        difference = initial_difference + gain[0] - gain[1]
        real_sum += np.cos(difference)
        imag_sum += np.sin(difference)
        difference_min = min(difference_min, difference)
        difference_max = max(difference_max, difference)
        # Full 2pi excursions from the previous slip anchor, not noisy recrossings
        # of an arbitrary bin edge. Slip counts reset at each reference block.
        if difference - slip_anchor >= 2 * np.pi:
            slips[0] += 1
            slip_anchor += 2 * np.pi
        elif difference - slip_anchor <= -2 * np.pi:
            slips[1] += 1
            slip_anchor -= 2 * np.pi
        if stride and (step + 1) % stride == 0:
            observed[:, (step + 1) // stride - 1] = state
    stats = np.concatenate((gain, crossings, backwards, min_radius, min_velocity,
                            max_increment, slips,
                            np.array([real_sum, imag_sum, difference_min,
                                      difference_max, initial_difference])))
    return state, observed, stats


def summarize(stats, steps, dt):
    duration = steps * dt
    frequencies = stats[:2] / duration
    crossing_frequencies = stats[2:4] * (2 * np.pi / duration)
    return dict(duration=duration, omega=frequencies.tolist(),
                frequency_difference_signed=float(frequencies[0] - frequencies[1]),
                Q=float(abs(frequencies[0] - frequencies[1])),
                PLV=float(np.hypot(stats[14], stats[15]) / steps),
                mean_phase_vector=(stats[14:16] / steps).tolist(),
                phase_gain=stats[:2].tolist(), winding_difference=float((stats[0] - stats[1]) / (2 * np.pi)),
                phase_difference_min=float(stats[16]), phase_difference_max=float(stats[17]),
                phase_difference_span=float(stats[17] - stats[16]),
                positive_slips=int(stats[12]), negative_slips=int(stats[13]),
                slip_rate=float((stats[12] + stats[13]) / duration),
                poincare_crossings=stats[2:4].astype(int).tolist(),
                poincare_omega=crossing_frequencies.tolist(),
                poincare_Q=float(abs(crossing_frequencies[0] - crossing_frequencies[1])),
                poincare_frequency_discrepancy=np.abs(frequencies - crossing_frequencies).tolist(),
                backward_phase_fraction=(stats[4:6] / steps).tolist(),
                minimum_radius=stats[6:8].tolist(), minimum_angular_velocity=stats[8:10].tolist(),
                maximum_phase_increment=stats[10:12].tolist())


def combine_stats(blocks):
    """Combine equal-length block sufficient statistics without phase wrap loss."""
    total = blocks[0].copy()
    offset = 0.
    for previous, block in zip(blocks[:-1], blocks[1:]):
        offset += previous[0] - previous[1]
        alignment = blocks[0][18] + offset - block[18]
        total[16] = min(total[16], block[16] + alignment)
        total[17] = max(total[17], block[17] + alignment)
        total[:6] += block[:6]
        total[6:10] = np.minimum(total[6:10], block[6:10])
        total[10:12] = np.maximum(total[10:12], block[10:12])
        total[12:16] += block[12:16]
    return total


def _steps(duration, dt, label):
    count = int(round(duration / dt))
    if count < 0 or not np.isclose(count * dt, duration, rtol=0, atol=1e-9):
        raise ValueError(f"{label} must be a nonnegative integer multiple of dt")
    return count


def initial_state(seed, coupling):
    # Independent starts per (seed, control), identical starts for dt refinement.
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(round(coupling * 1e9))]))
    phases = rng.uniform(0, 2 * np.pi, 2)
    radii = rng.uniform(3., 8., 2)
    state = np.empty(6)
    state[0::3] = radii * np.cos(phases)
    state[1::3] = radii * np.sin(phases)
    state[2::3] = rng.uniform(.01, .1, 2)
    return state


def simulate(coupling, seed, *, dt=.01, sample_dt=.2, observation_samples=2000,
             burn=2000., reference=100000., blocks=10):
    if not np.isfinite([coupling, dt, sample_dt, burn, reference]).all():
        raise ValueError("parameters must be finite")
    if coupling < 0 or dt <= 0 or sample_dt <= 0 or reference <= 0:
        raise ValueError("invalid coupling or time parameters")
    if observation_samples < 2 or blocks < 2 or blocks % 2:
        raise ValueError("at least two samples and an even number of reference blocks required")
    stride = _steps(sample_dt, dt, "sample_dt")
    burn_steps = _steps(burn, dt, "burn")
    reference_steps = _steps(reference, dt, "reference")
    if stride == 0 or reference_steps % blocks or reference_steps < blocks:
        raise ValueError("reference must partition into nonempty equal blocks")
    start = time.perf_counter()
    initial = initial_state(seed, coupling)
    state = advance(initial, coupling, dt, burn_steps)
    state, X, observation_stats = measure(state, coupling, dt, observation_samples * stride, stride)
    reference_start = state.copy()
    stats = []
    for _ in range(blocks):
        state, _, block = measure(state, coupling, dt, reference_steps // blocks)
        stats.append(block)
    stats = np.asarray(stats)
    summary = summarize(combine_stats(stats), reference_steps, dt)
    block_summaries = [summarize(b, reference_steps // blocks, dt) for b in stats]
    halves = [summarize(combine_stats(b), reference_steps // 2, dt)
              for b in np.array_split(stats, 2)]
    summary.update(Q_blocks=[b["Q"] for b in block_summaries],
                   Q_first_half=halves[0]["Q"], Q_second_half=halves[1]["Q"],
                   Q_half_difference=abs(halves[0]["Q"] - halves[1]["Q"]),
                   PLV_half_difference=abs(halves[0]["PLV"] - halves[1]["PLV"]),
                   block_Q_sd=float(np.std([b["Q"] for b in block_summaries], ddof=1)))
    metadata = dict(coupling=coupling, seed=seed, dt=dt, sample_dt=sample_dt,
                    observation_samples=observation_samples, burn=burn, reference=reference,
                    reference_blocks=blocks, M=6, N_state=6, N_oscillators=2, channels=CHANNELS,
                    parameters=dict(a=.15, b=.2, c=10., omega=[1.015, .985]), source=SOURCE,
                    time_units="physical time; integration dt differs from observation sample_dt",
                    observation_interval=[burn + sample_dt, burn + observation_samples * sample_dt],
                    reference_interval=[burn + observation_samples * sample_dt,
                                        burn + observation_samples * sample_dt + reference],
                    observation=summarize(observation_stats, observation_samples * stride, dt),
                    reference_summary=summary, reference_block_summaries=block_summaries,
                    reference_half_summaries=halves,
                    slip_definition="completed 2pi excursions; anchor resets per reference block, so total is conservative",
                    interpretation="finite-time frequency entrainment; no-slip records do not prove asymptotic phase locking",
                    input_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    elapsed_seconds=time.perf_counter() - start)
    return dict(X=X, initial_state=initial, reference_start_state=reference_start,
                final_state=state, reference_stats=stats), metadata


def pilot_cases(seed_base=260915101, seeds=8):
    cases = [dict(coupling=float(c), seed=seed_base + s, dt=.01, arm="primary")
             for c in np.linspace(.015, .04, 21) for s in range(seeds)]
    cases += [dict(coupling=c, seed=seed_base, dt=.005, arm="half-dt")
              for c in (.015, .0275, .04)]
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--coupling", type=float)
    parser.add_argument("--seed", type=int, default=260915101)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--dt", type=float, default=.01)
    parser.add_argument("--sample-dt", type=float, default=.2)
    parser.add_argument("--observation-samples", type=int, default=2000)
    parser.add_argument("--burn", type=float, default=2000.)
    parser.add_argument("--reference", type=float, default=100000.)
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    cases = pilot_cases(args.seed, args.seeds)
    if args.count_only:
        print(len(cases))
        return
    if args.manifest_only:
        print(json.dumps(dict(cases=cases, common=dict(sample_dt=args.sample_dt,
              observation_samples=args.observation_samples, burn=args.burn,
              reference=args.reference)), indent=2))
        return
    if args.output_dir is None:
        parser.error("--output-dir required")
    if args.coupling is not None:
        if args.case_index is not None:
            parser.error("choose --coupling or --case-index, not both")
        selected = [(0, dict(coupling=args.coupling, seed=args.seed, dt=args.dt, arm="single"))]
    elif args.case_index is not None:
        if not 0 <= args.case_index < len(cases):
            parser.error("case index outside pilot grid")
        selected = [(args.case_index, cases[args.case_index])]
    else:
        selected = list(enumerate(cases))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, case in selected:
        stem = args.output_dir / f"case-{index:04d}"
        if stem.with_suffix(".npz").exists() or stem.with_suffix(".json").exists():
            raise FileExistsError(f"refusing to overwrite existing case {stem}")
        arrays, metadata = simulate(case["coupling"], case["seed"], dt=case["dt"],
            sample_dt=args.sample_dt, observation_samples=args.observation_samples,
            burn=args.burn, reference=args.reference)
        metadata.update(case_index=index, arm=case["arm"])
        np.savez_compressed(stem.with_suffix(".npz"), **arrays)
        stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        print(json.dumps(dict(case_index=index, Q=metadata["reference_summary"]["Q"],
                              PLV=metadata["reference_summary"]["PLV"],
                              elapsed_seconds=metadata["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    main()
