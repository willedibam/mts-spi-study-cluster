"""Physics-only N=8 Lorenz-96 crisis/classifier validation, not a fitted q assay.

Equation and approximate F_c=-6.4717: Barone et al., Chaos 35, 103119 (2025).
The alternating spatial mode and hysteretic residence definitions here are OUR
operational diagnostics; the paper does not specify its region classifier.
No classifier is promoted to physical truth before precrisis-anchor validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from numba import njit

SOURCE = "https://doi.org/10.1063/5.0287572"
VARIANTS = ["raw_h0", "raw_h0.1", "raw_h0.25", "raw_h0.5", "ema_tau0.5_h0.25"]
THRESHOLDS = np.array([0., .1, .25, .5, .25])


@njit(cache=True)
def rhs(x, forcing):
    n = len(x)
    result = np.empty(n)
    for j in range(n):
        result[j] = (x[(j + 1) % n] - x[(j - 2) % n]) * x[(j - 1) % n] - x[j] + forcing
    return result


@njit(cache=True)
def tangent_rhs(state, forcing):
    n = len(state) // 2
    out = np.empty_like(state)
    for j in range(n):
        jp, jm, jmm = (j + 1) % n, (j - 1) % n, (j - 2) % n
        out[j] = (state[jp] - state[jmm]) * state[jm] - state[j] + forcing
        out[n+j] = ((state[n+jp] - state[n+jmm]) * state[jm]
                    + (state[jp] - state[jmm]) * state[n+jm] - state[n+j])
    return out


@njit(cache=True)
def rk4_step(state, forcing, dt):
    k1 = tangent_rhs(state, forcing)
    k2 = tangent_rhs(state + .5 * dt * k1, forcing)
    k3 = tangent_rhs(state + .5 * dt * k2, forcing)
    k4 = tangent_rhs(state + dt * k3, forcing)
    return state + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)


@njit(cache=True)
def advance(state, forcing, dt, steps, renormalize):
    state = state.copy()
    n = len(state) // 2
    for step in range(steps):
        state = rk4_step(state, forcing, dt)
        if (step + 1) % renormalize == 0:
            state[n:] /= np.linalg.norm(state[n:])
    return state


@njit(cache=True)
def alternating_mode(x):
    return np.sum(x[::2] - x[1::2]) / len(x)


@njit(cache=True)
def label_update(value, previous, threshold):
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return previous


@njit(cache=True)
def reference_stream(state, forcing, dt, steps, renormalize, blocks,
                     anchor_steps, anchor_stride):
    n, variants = len(state) // 2, len(THRESHOLDS)
    anchor = np.empty((n, anchor_steps // anchor_stride))
    mode_trace = np.empty(anchor_steps // anchor_stride)
    labels = np.zeros(variants, dtype=np.int64)
    switches = np.zeros(variants, dtype=np.int64)
    last_switch = np.zeros(variants)
    first_switch = np.full(variants, -1.)
    completed_count = np.zeros((variants, 2), dtype=np.int64)
    completed_duration = np.zeros((variants, 2))
    completed_duration_sq = np.zeros((variants, 2))
    occupancy = np.zeros((variants, 2))
    unknown = np.zeros(variants)
    switch_blocks = np.zeros((blocks, variants), dtype=np.int64)
    mode_blocks = np.zeros((blocks, 4))  # mean accumulator, square, min, max
    mode_blocks[:, 2] = np.inf
    mode_blocks[:, 3] = -np.inf
    lyapunov_blocks = np.zeros(blocks)
    ema = alternating_mode(state[:n])
    smoothing = 1 - np.exp(-dt / .5)
    block_steps = steps // blocks
    for step in range(steps):
        state = rk4_step(state, forcing, dt)
        if not np.isfinite(state).all() or np.max(np.abs(state[:n])) > 1e8:
            raise ValueError("trajectory or tangent became nonfinite/unbounded")
        block = step // block_steps
        if (step + 1) % renormalize == 0:
            norm = np.linalg.norm(state[n:])
            if norm <= 0 or not np.isfinite(norm):
                raise ValueError("invalid tangent norm")
            lyapunov_blocks[block] += np.log(norm)
            state[n:] /= norm
        value = alternating_mode(state[:n])
        ema += smoothing * (value - ema)
        mode_blocks[block, 0] += value
        mode_blocks[block, 1] += value**2
        mode_blocks[block, 2] = min(mode_blocks[block, 2], value)
        mode_blocks[block, 3] = max(mode_blocks[block, 3], value)
        now = (step + 1) * dt
        for v in range(variants):
            signal = ema if v == variants - 1 else value
            label = label_update(signal, labels[v], THRESHOLDS[v])
            if label != 0 and labels[v] != 0 and label != labels[v]:
                if switches[v] > 0:
                    column = 0 if labels[v] == -1 else 1
                    duration = now - last_switch[v]
                    completed_count[v, column] += 1
                    completed_duration[v, column] += duration
                    completed_duration_sq[v, column] += duration**2
                else:
                    first_switch[v] = now
                switches[v] += 1
                switch_blocks[block, v] += 1
                last_switch[v] = now
            labels[v] = label
            if label == 0:
                unknown[v] += dt
            else:
                occupancy[v, 0 if label == -1 else 1] += dt
        if step < anchor_steps and (step + 1) % anchor_stride == 0:
            index = (step + 1) // anchor_stride - 1
            anchor[:, index] = state[:n]
            mode_trace[index] = value
    mode_blocks[:, :2] /= block_steps
    lyapunov_blocks /= block_steps * dt
    return (state, anchor, mode_trace, switches, first_switch, last_switch,
            completed_count, completed_duration, completed_duration_sq, occupancy,
            unknown, switch_blocks, mode_blocks, lyapunov_blocks)


def steps_for(duration, dt):
    result = int(round(duration / dt))
    if result < 0 or not np.isclose(result * dt, duration, rtol=0, atol=1e-9):
        raise ValueError("all physical durations must be nonnegative multiples of dt")
    return result


def simulate(forcing, seed, *, dt=.01, sample_dt=.1, observation_samples=2000,
             burn=10000., reference=100000., anchor_duration=0., blocks=10):
    if not np.isfinite([forcing, dt, sample_dt, burn, reference, anchor_duration]).all():
        raise ValueError("finite parameters required")
    if dt <= 0 or sample_dt <= 0 or reference <= 0 or observation_samples < 2 or blocks < 2 or blocks % 2:
        raise ValueError("invalid time/count parameters")
    stride = steps_for(sample_dt, dt)
    renorm = steps_for(1., dt)
    nref, nburn = steps_for(reference, dt), steps_for(burn, dt)
    nanchor = steps_for(anchor_duration, dt)
    if (stride < 1 or renorm < 1 or nref % (blocks * renorm)
            or nburn % renorm or (observation_samples * stride) % renorm
            or nanchor > nref or nanchor % stride or nref < blocks * renorm):
        raise ValueError("block/burn/observation endpoints must align with 1-unit tangent renormalization; anchor with sample_dt")
    start = time.perf_counter()
    control_seed = int(round(abs(forcing) * 1e9))
    rng = np.random.default_rng(np.random.SeedSequence([seed, control_seed]))
    initial = rng.normal(forcing, 1., 8)
    tangent = rng.normal(size=8)
    tangent /= np.linalg.norm(tangent)
    state = advance(np.r_[initial, tangent], forcing, dt, nburn, renorm)
    X = np.empty((8, observation_samples))
    # Run full observation in one-step batches; only 2000 Python calls by default.
    for t in range(observation_samples):
        state = advance(state, forcing, dt, stride, stride)
        X[:, t] = state[:8]
    state[8:] /= np.linalg.norm(state[8:])
    refstart = state[:8].copy()
    values = reference_stream(state, forcing, dt, nref, renorm, blocks, nanchor, stride)
    (state, anchor, mode, switches, first, last, counts, durations, duration_sq,
     occupancy, unknown, switch_blocks, mode_blocks, lyapunov) = values
    diagnostics = {}
    for i, name in enumerate(VARIANTS):
        count = int(counts[i].sum())
        total = float(durations[i].sum())
        diagnostics[name] = dict(switch_count=int(switches[i]), switch_rate=float(switches[i] / reference),
            mean_completed_residence=total / count if count else None,
            reciprocal_completed_residence=count / total if total else None,
            completed_count_by_state=counts[i].tolist(), completed_duration_by_state=durations[i].tolist(),
            completed_duration_square_sum_by_state=duration_sq[i].tolist(),
            occupancy_fraction_by_state=(occupancy[i] / reference).tolist(),
            unknown_duration=float(unknown[i]),
            first_residence_left_censored=True,
            first_switch_time=float(first[i]) if first[i] >= 0 else None,
            last_residence_right_censored_duration=float(reference - last[i]),
            no_switch_window_censored=bool(switches[i] == 0),
            switch_rate_blocks=(switch_blocks[:, i] / (reference / blocks)).tolist())
    meta = dict(forcing=forcing, seed=seed, M=8, N=8, dt=dt, sample_dt=sample_dt,
        observation_samples=observation_samples, burn=burn, reference=reference, blocks=blocks,
        anchor_duration=anchor_duration, elapsed_seconds=time.perf_counter() - start,
        equation="dx_j=(x_(j+1)-x_(j-2))*x_(j-1)-x_j+F; periodic indices",
        source=SOURCE, published_crisis_approx=-6.4717,
        observation_interval=[burn + sample_dt, burn + observation_samples * sample_dt],
        reference_interval=[burn + observation_samples * sample_dt,
                            burn + observation_samples * sample_dt + reference],
        mode_definition="A=(sum_even_zero_based(x)-sum_odd_zero_based(x))/8",
        classifier_status="UNVALIDATED operational symmetry-mode candidate; not the paper's unspecified classifier",
        classifier_definition="enter + state above +h, - state below -h; retain label in deadband; EMA uses tau=.5 physical units",
        statistical_warning="zero observed switches is censored, not proof of absent switching; completed-only residence averages can be biased",
        residence_diagnostics=diagnostics,
        mode_block_mean=mode_blocks[:, 0].tolist(), mode_block_second_moment=mode_blocks[:, 1].tolist(),
        mode_block_min=mode_blocks[:, 2].tolist(), mode_block_max=mode_blocks[:, 3].tolist(),
        largest_lyapunov=float(lyapunov.mean()), largest_lyapunov_blocks=lyapunov.tolist(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(X.tobytes()).hexdigest())
    return dict(X=X, initial_state=initial, reference_start_state=refstart, final_state=state[:8],
                anchor_X=anchor, anchor_mode=mode), meta


def pilot_cases(seed_base=260915201, seeds=8):
    controls = np.linspace(-6.6, -6.4, 21)
    anchors = (0, 13, 20)  # F=-6.6,-6.47,-6.4, chosen before outcomes
    cases = [dict(forcing=float(f), seed=seed_base+s, dt=.01, arm="primary",
                  anchor_duration=10000. if j in anchors and s == 0 else 0.)
             for j, f in enumerate(controls) for s in range(seeds)]
    cases += [dict(forcing=float(controls[j]), seed=seed_base, dt=.005,
                   arm="half-dt", anchor_duration=10000.) for j in anchors]
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--seed-base", type=int, default=260915201)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--burn", type=float, default=10000.)
    parser.add_argument("--reference", type=float, default=100000.)
    args = parser.parse_args()
    cases = pilot_cases(args.seed_base, args.seeds)
    if args.count_only:
        print(len(cases))
        return
    if args.manifest_only:
        print(json.dumps(dict(cases=cases, common=dict(burn=args.burn, reference=args.reference,
                         sample_dt=.1, observation_samples=2000)), indent=2))
        return
    if args.output_dir is None:
        parser.error("--output-dir required")
    if args.case_index is not None and not 0 <= args.case_index < len(cases):
        parser.error("case-index outside pilot")
    selected = [(args.case_index, cases[args.case_index])] if args.case_index is not None else list(enumerate(cases))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, case in selected:
        stem = args.output_dir / f"case-{index:04d}"
        if stem.with_suffix(".npz").exists() or stem.with_suffix(".json").exists():
            raise FileExistsError(f"refusing to overwrite {stem}")
        arrays, meta = simulate(case["forcing"], case["seed"], dt=case["dt"],
            burn=args.burn, reference=args.reference, anchor_duration=case["anchor_duration"])
        meta.update(arm=case["arm"], case_index=index)
        np.savez_compressed(stem.with_suffix(".npz"), **arrays)
        stem.with_suffix(".json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")
        print(json.dumps(dict(case_index=index, elapsed_seconds=meta["elapsed_seconds"],
            largest_lyapunov=meta["largest_lyapunov"], classifier_status=meta["classifier_status"])), flush=True)


if __name__ == "__main__":
    main()
