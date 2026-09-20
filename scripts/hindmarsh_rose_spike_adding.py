"""Fully observed Hindmarsh--Rose spike-deletion physics scout.

Barrio & Shilnikov (2011), doi:10.1186/2190-8567-1-6, Eq.1/Fig.6.
Q counts large action-potential peaks per complete burst, not all local maxima:
the paper explicitly distinguishes the small terminal subthreshold oscillation.
Threshold sensitivities and all local-maximum heights are retained. This is a
dynamical observable, not a thermodynamic order parameter or exact bifurcation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from numba import njit

SOURCE = "https://doi.org/10.1186/2190-8567-1-6"
PARAMETERS = dict(a=1., c=1., d=5., s=4., x0=-1.6, epsilon=.01, I=2.4)
CHANNELS = ["membrane_potential_x", "fast_current_y", "slow_current_z"]
# Includes the published near-boundary illustration, without treating it as an
# analytically exact transition. All choices precede SPI computation.
CONTROLS = np.array([2.68, 2.6805, 2.681, 2.6814, 2.6817, 2.6818,
                     2.6819, 2.682, 2.6821, 2.6822, 2.6823, 2.6824,
                     2.6825, 2.6826, 2.6828, 2.683, 2.6835, 2.684,
                     2.685, 2.6865, 2.688])


@njit(cache=True)
def rhs(state, b):
    x, y, z = state
    return np.array([y - x*x*x + b*x*x - z + 2.4,
                     1. - 5.*x*x - y, .01*(4.*(x + 1.6) - z)])


@njit(cache=True)
def rk4_step(state, b, dt):
    k1 = rhs(state, b)
    k2 = rhs(state + .5*dt*k1, b)
    k3 = rhs(state + .5*dt*k2, b)
    k4 = rhs(state + dt*k3, b)
    return state + dt*(k1 + 2*k2 + 2*k3 + k4)/6.


@njit(cache=True)
def integrate(state, b, dt, steps, stride=0):
    state = state.copy()
    out = np.empty((3, steps//stride if stride else 0))
    for k in range(steps):
        state = rk4_step(state, b, dt)
        if not np.isfinite(state).all() or np.max(np.abs(state)) > 1e6:
            raise ValueError("HR trajectory became nonfinite or unbounded")
        if stride and (k+1) % stride == 0:
            out[:, (k+1)//stride-1] = state
    return state, out


def crossing_times(x, sample_dt, level, upward):
    left, right = x[:-1], x[1:]
    selected = ((left < level) & (right >= level) if upward
                else (left >= level) & (right < level))
    i = np.flatnonzero(selected)
    # Samples are numbered from zero here; only time differences enter Q.
    return (i + (level-left[i])/(right[i]-left[i]))*sample_dt


def burst_statistics(x, sample_dt, *, spike_threshold=0., separator=-1.2):
    """Upward separator crossings delimit complete bursts, excluding edges.

    Quiescent troughs are below the separator; within-burst troughs stay above it
    in the selected source regime. Sensitivities test this assumption explicitly.
    Duty cycle is the active envelope from up-crossing to next down-crossing,
    divided by the up-to-up period (not cumulative time above spike threshold).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 3 or sample_dt <= 0 or not np.isfinite(x).all():
        raise ValueError("finite one-dimensional trace and positive sample_dt required")
    peaks = np.flatnonzero((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:])) + 1
    peak_times = peaks*sample_dt
    peak_heights = x[peaks]
    up = crossing_times(x, sample_dt, separator, True)
    down = crossing_times(x, sample_dt, separator, False)
    counts, all_counts, periods, duties, minimum_spikes = [], [], [], [], []
    for start, end in zip(up[:-1], up[1:]):
        segment = (peak_times >= start) & (peak_times < end)
        heights = peak_heights[segment]
        accepted = heights[heights > spike_threshold]
        falls = down[(down > start) & (down < end)]
        # More than one downcrossing means the assumed active-envelope separator
        # is not valid; retain the interval but do not invent a duty cycle.
        counts.append(len(accepted))
        all_counts.append(len(heights))
        periods.append(end-start)
        duties.append((falls[0]-start)/(end-start) if len(falls) == 1 else np.nan)
        minimum_spikes.append(float(accepted.min()) if len(accepted) else np.nan)
    def mean_or_none(values):
        return float(np.mean(values)) if len(values) and np.isfinite(values).all() else None
    histogram = {str(int(k)): int(v) for k, v in zip(*np.unique(counts, return_counts=True))}
    valid = len(counts) >= 2 and np.isfinite(duties).all() and min(counts, default=0) >= 1
    return dict(Q=mean_or_none(counts), complete_bursts=len(counts),
                spike_count_histogram=histogram, spike_counts=counts,
                all_local_maxima_counts=all_counts, burst_periods=periods,
                duty_cycles=[float(d) if np.isfinite(d) else None for d in duties],
                mean_period=mean_or_none(periods),
                mean_duty_cycle=mean_or_none(duties) if np.isfinite(duties).all() else None,
                minimum_counted_peak=(float(np.min(minimum_spikes))
                    if len(minimum_spikes) and np.isfinite(minimum_spikes).all() else None),
                spike_threshold=spike_threshold, separator=separator,
                valid_burst_segmentation=bool(valid)), peak_times, peak_heights


def diagnostics(x, sample_dt):
    summary, times, heights = burst_statistics(x, sample_dt)
    halves = [burst_statistics(part, sample_dt)[0] for part in np.array_split(x, 2)]
    summary["halves"] = halves
    summary["Q_half_difference"] = (abs(halves[0]["Q"]-halves[1]["Q"])
                                      if all(h["Q"] is not None for h in halves) else None)
    sensitivity = [burst_statistics(x, sample_dt, spike_threshold=threshold, separator=sep)[0]
                   for threshold, sep in [(-.5, -1.2), (.5, -1.2), (0., -1.), (0., -1.5)]]
    summary["threshold_sensitivities"] = sensitivity
    summary["threshold_agreement"] = bool(summary["Q"] is not None and all(
        s["valid_burst_segmentation"] and s["Q"] == summary["Q"] for s in sensitivity))
    return summary, times, heights


def _steps(duration, dt, label):
    n = int(round(duration/dt))
    if n < 1 or not np.isclose(n*dt, duration, atol=1e-9, rtol=0):
        raise ValueError(f"{label} must be a positive integer multiple of dt")
    return n


def initial_state(seed):
    rng = np.random.default_rng(seed)
    return np.array([-1., -5., 3.]) + rng.uniform(-1, 1, 3)*np.array([.5, 1., .5])


def simulate(b, seed, *, dt=.01, sample_dt=.25, observation_samples=4000,
             burn=5000., reference=20000., diagnostic_dt=.05, preparation="independent",
             continuation_dwell=1000.):
    if not np.isfinite([b, dt, sample_dt, burn, reference, diagnostic_dt]).all() or dt <= 0:
        raise ValueError("parameters must be finite and dt positive")
    if preparation not in ("independent", "up", "down") or observation_samples < 3:
        raise ValueError("invalid preparation or observation size")
    stride = _steps(sample_dt, dt, "sample_dt")
    diag_stride = _steps(diagnostic_dt, dt, "diagnostic_dt")
    ref_steps = _steps(reference, dt, "reference")
    burn_steps = _steps(burn, dt, "burn")
    if ref_steps % diag_stride or stride % diag_stride:
        raise ValueError("measurement windows must be divisible by diagnostic_dt")
    tic = time.perf_counter()
    initial = initial_state(seed)
    state = initial.copy()
    path = []
    if preparation != "independent":
        path = (CONTROLS[CONTROLS < b].tolist() if preparation == "up"
                else CONTROLS[CONTROLS > b][::-1].tolist())
        dwell = _steps(continuation_dwell, dt, "continuation_dwell")
        for index, control in enumerate(path):
            state, _ = integrate(state, control, dt, burn_steps if index == 0 else dwell)
    state, _ = integrate(state, b, dt, burn_steps)
    # Keep a high-rate diagnostic trace separate from the coarser p90 view.
    state, observation_dense = integrate(state, b, dt, observation_samples*stride, diag_stride)
    X = observation_dense[:, stride//diag_stride-1::stride//diag_stride].copy()
    reference_start = state.copy()
    state, dense = integrate(state, b, dt, ref_steps, diag_stride)
    summary, times, heights = diagnostics(dense[0], diagnostic_dt)
    observation_summary, _, _ = diagnostics(observation_dense[0], diagnostic_dt)
    meta = dict(b=float(b), control=float(b), seed=int(seed), M=3, N_state=3, N_neurons=1,
                channels=CHANNELS, parameters=PARAMETERS, source=SOURCE, dt=dt,
                sample_dt=sample_dt, diagnostic_dt=diagnostic_dt,
                observation_samples=observation_samples, burn=burn, reference=reference,
                preparation=preparation, continuation_path=path,
                continuation_dwell=continuation_dwell, reference_summary=summary,
                initial_condition_definition="our seed-fixed jitter of (-1,-5,3), half-widths (.5,1,.5); source Fig6 does not specify initial conditions",
                time_units="physical time since end of optional preparation; p90 sample_dt differs from numerical dt",
                observation_interval=[burn+sample_dt, burn+observation_samples*sample_dt],
                reference_interval=[burn+observation_samples*sample_dt,
                                    burn+observation_samples*sample_dt+reference],
                observation=observation_summary, Q=summary["Q"],
                channel_standard_deviations=np.std(X, axis=1).tolist(),
                interpretation="large-spike count per burst; tiny terminal local maxima remain separate; no exact bifurcation or thermodynamic-order claim",
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                input_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
                elapsed_seconds=time.perf_counter()-tic)
    return dict(X=X, initial_state=initial, reference_start_state=reference_start,
                final_state=state, reference_peak_times=times, reference_peak_heights=heights,
                illustration_time=np.arange(min(20000, dense.shape[1]))*diagnostic_dt,
                illustration=dense[:, :20000]), meta


def pilot_cases(seed_base=260915201, seeds=8):
    if seeds < 1:
        raise ValueError("seeds must be positive")
    cases = [dict(b=float(b), seed=seed_base+s, dt=.01, preparation="independent", arm="primary")
             for b in CONTROLS for s in range(seeds)]
    cases += [dict(b=b, seed=seed_base, dt=.005, preparation="independent", arm="half-dt")
              for b in (2.68, 2.6819, 2.688)]
    # Preparation audit is physics-only by default, not pooled with primary rows.
    cases += [dict(b=float(b), seed=seed_base+s, dt=.01, preparation=direction, arm="preparation")
              for direction in ("up", "down") for b in CONTROLS for s in range(min(2, seeds))]
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--case-index", type=int)
    p.add_argument("--seed", type=int, default=260915201)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--burn", type=float, default=5000.)
    p.add_argument("--reference", type=float, default=20000.)
    p.add_argument("--sample-dt", type=float, default=.25)
    p.add_argument("--observation-samples", type=int, default=4000)
    p.add_argument("--count-only", action="store_true")
    p.add_argument("--manifest-only", action="store_true")
    args = p.parse_args()
    cases = pilot_cases(args.seed, args.seeds)
    common = dict(burn=args.burn, reference=args.reference, sample_dt=args.sample_dt,
                  observation_samples=args.observation_samples)
    if args.count_only:
        print(len(cases)); return
    if args.manifest_only:
        print(json.dumps(dict(cases=cases, common=common, source=SOURCE), indent=2)); return
    if args.output_dir is None or args.case_index is None or not 0 <= args.case_index < len(cases):
        p.error("provide --output-dir and a valid --case-index")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"case-{args.case_index:04d}"
    if stem.with_suffix(".npz").exists() or stem.with_suffix(".json").exists():
        p.error("output exists; refusing to overwrite")
    case = cases[args.case_index].copy()
    arm = case.pop("arm")
    arrays, meta = simulate(**case, **common)
    meta.update(case_index=args.case_index, arm=arm)
    np.savez_compressed(stem.with_suffix(".npz"), **arrays, metadata_json=json.dumps(meta, allow_nan=False))
    meta["archive_sha256"] = hashlib.sha256(stem.with_suffix(".npz").read_bytes()).hexdigest()
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(case=args.case_index, Q=meta["Q"],
                         elapsed_seconds=meta["elapsed_seconds"])))


if __name__ == "__main__":
    main()
