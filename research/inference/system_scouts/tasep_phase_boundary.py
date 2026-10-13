"""Continuous-time open TASEP and exact finite-size stationary density.

Particles hop right at rate one; injection/removal rates are alpha/beta.
DEHP (1993), J. Phys. A 26, 1493, equations (8)--(10): DE=D+E,
<W|E=alpha^-1<W| and D|V>=beta^-1|V>. Normal ordering of
(E+yD)^N and its y derivative gives the mean particle count exactly.
No thermodynamic discontinuity is assumed for finite N.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from numba import njit
from scipy.special import logsumexp
import yaml


def exact_stationary(N, alpha, beta):
    """Positive DEHP normal-order recurrence, rescaled each step.

    Coefficient a[p,q] multiplies E^p D^q; b is its fugacity derivative.
    Appending E uses D^q E = E + D + ... + D^q for q>=1.
    A common positive scale preserves the derivative/partition ratio.
    """
    if int(N) != N or N < 1 or alpha <= 0 or beta <= 0:
        raise ValueError("positive integer N and positive rates required")
    N = int(N)
    a = np.zeros((N+1, N+1)); a[0, 0] = 1.
    b = np.zeros_like(a)
    scale = 0.
    previous_logZ = 0.
    for k in range(1, N+1):
        out = np.zeros_like(a); derivative = np.zeros_like(a)
        # Append D, including the derivative of its extra fugacity y.
        out[:, 1:] += a[:, :-1]
        derivative[:, 1:] += b[:, :-1] + a[:, :-1]
        # Append E. Each normal-ordered monomial contributes one E term.
        out[1:, 0] += a[:-1].sum(axis=1)
        derivative[1:, 0] += b[:-1].sum(axis=1)
        out[:, 1:] += np.cumsum(a[:, :0:-1], axis=1)[:, ::-1]
        derivative[:, 1:] += np.cumsum(b[:, :0:-1], axis=1)[:, ::-1]
        factor = out.max()
        a, b = out/factor, derivative/factor
        scale += np.log(factor)
        p, q = np.nonzero(a)
        logweights = np.log(a[p, q])-p*np.log(alpha)-q*np.log(beta)
        norm = logsumexp(logweights)
        logZ = scale+norm
        if k == N:
            mean_count = np.sum(np.exp(logweights-norm)*b[p, q]/a[p, q])
            return dict(density=float(mean_count/N), current=float(np.exp(previous_logZ-logZ)),
                        log_partition=float(logZ))
        previous_logZ = logZ


def small_generator_stationary(N, alpha, beta):
    """Independent dense CTMC oracle, restricted to N<=8 for tests."""
    if not 1 <= N <= 8 or alpha <= 0 or beta <= 0:
        raise ValueError("oracle requires 1<=N<=8 and positive rates")
    states = ((np.arange(2**N)[:, None] >> np.arange(N)) & 1).astype(np.uint8)
    generator = np.zeros((2**N, 2**N))
    for i, x in enumerate(states):
        if not x[0]: generator[i, i | 1] += alpha
        if x[-1]: generator[i, i & ~(1 << (N-1))] += beta
        for j in range(N-1):
            if x[j] and not x[j+1]:
                generator[i, i ^ (3 << j)] += 1.
        generator[i, i] = -generator[i].sum()
    system = generator.T.copy(); system[-1] = 1.
    rhs = np.zeros(2**N); rhs[-1] = 1.
    stationary = np.linalg.solve(system, rhs)
    return states, stationary, generator


@njit(cache=True)
def gillespie(initial, alpha, beta, burn, observation_steps, dt, reference_time,
              blocks, trace_dt, seed):
    """Exact jump times; observations use physical time, not event index.

    Q integrates density over event holding times, split exactly at reference
    block boundaries. The separate uniform reference trace is diagnostic only.
    """
    np.random.seed(seed)
    x = initial.copy()
    N = len(x)
    observed = np.empty((N, observation_steps), dtype=np.uint8)
    start = burn+observation_steps*dt
    stop = start+reference_time
    block_length = reference_time/blocks
    integrals = np.zeros(blocks)
    trace = np.empty(int(np.ceil(reference_time/trace_dt)))
    rates = np.empty(N+1)
    count = int(x.sum())
    t = 0.; obs_index = 0; trace_index = 0; events = 0
    while t < stop:
        rates[0] = alpha*(1-x[0])
        for j in range(N-1): rates[j+1] = x[j]*(1-x[j+1])
        rates[N] = beta*x[-1]
        total = rates.sum()
        event_time = t+np.random.exponential(1./total)
        end = min(event_time, stop)
        while obs_index < observation_steps and burn+obs_index*dt < end:
            observed[:, obs_index] = x
            obs_index += 1
        while trace_index < len(trace) and start+trace_index*trace_dt < end:
            trace[trace_index] = count/N
            trace_index += 1
        cursor = max(t, start)
        while cursor < end:
            block = min(int((cursor-start)/block_length), blocks-1)
            boundary = start+(block+1)*block_length
            # Floating point may round an exact block boundary down one index.
            if boundary <= cursor and block < blocks-1:
                block += 1
                boundary = start+(block+1)*block_length
            segment_end = min(end, boundary)
            integrals[block] += (segment_end-cursor)*count/N
            cursor = segment_end
        if event_time >= stop: break
        draw = np.random.random()*total
        event = 0
        while draw >= rates[event]:
            draw -= rates[event]
            event += 1
        if event == 0:
            x[0] = 1; count += 1
        elif event == N:
            x[-1] = 0; count -= 1
        else:
            x[event-1] = 0; x[event] = 1
        t = event_time
        events += 1
    return observed, integrals/block_length, trace, x, events


def mixing_summary(trace, sample_dt):
    """Initial-positive ACF estimate, descriptive rather than a mixing proof."""
    centered = np.asarray(trace, dtype=float)-np.mean(trace)
    if len(centered) < 4 or np.var(centered) == 0:
        return dict(estimated_tau_int=None, estimated_effective_samples=None)
    size = 1 << (2*len(centered)-1).bit_length()
    spectrum = np.fft.rfft(centered, n=size)
    acf = np.fft.irfft(spectrum*spectrum.conj(), n=size)[:len(centered)]
    acf /= np.arange(len(centered), 0, -1)
    acf /= acf[0]
    negative = np.flatnonzero(acf[1:] <= 0)
    cutoff = int(negative[0]+1) if len(negative) else len(acf)//2
    tau_samples = max(1., 1+2*acf[1:cutoff].sum())
    return dict(estimated_tau_int=float(tau_samples*sample_dt),
                estimated_effective_samples=float(len(trace)/tau_samples))


def simulate(case, simulation=None):
    params = dict(beta=.2, burn=100000., observation_steps=2000, dt=1.,
                  reference_time=1000000., reference_blocks=32, reference_trace_dt=10.,
                  initial="random")
    params.update(simulation or {}); params.update(case)
    N, alpha, beta = int(params['N']), float(params['alpha']), float(params['beta'])
    burn, dt, duration, trace_dt = (float(params[k]) for k in
        ('burn', 'dt', 'reference_time', 'reference_trace_dt'))
    obs, blocks = int(params['observation_steps']), int(params['reference_blocks'])
    if N < 1 or min(alpha, beta, dt, duration, trace_dt) <= 0 or burn < 0 or obs < 2 or blocks < 2 or blocks % 2:
        raise ValueError('invalid TASEP rates, sizes or recording contract')
    seeds = np.random.SeedSequence(int(params['seed'])).generate_state(2)
    rng = np.random.default_rng(int(seeds[0]))
    initial = params['initial']
    if initial == 'random': x = rng.integers(0, 2, N, dtype=np.uint8)
    elif initial == 'empty': x = np.zeros(N, dtype=np.uint8)
    elif initial == 'full': x = np.ones(N, dtype=np.uint8)
    else: raise ValueError('initial must be random, empty or full')
    started = time.perf_counter()
    observed, qblocks, trace, last, events = gillespie(x, alpha, beta, burn, obs, dt,
        duration, blocks, trace_dt, int(seeds[1]))
    exact = exact_stationary(N, alpha, beta)
    variance = observed.var(axis=1)
    metadata = dict(**params, M=N, Q_reference=float(qblocks.mean()), Q_exact=exact['density'],
        exact_current=exact['current'], Q_window=float(observed.mean()), Q_blocks=qblocks.tolist(),
        Q_first_half=float(qblocks[:blocks//2].mean()), Q_second_half=float(qblocks[blocks//2:].mean()),
        block_mean_se=float(qblocks.std(ddof=1)/np.sqrt(blocks)),
        reference_absolute_error=float(abs(qblocks.mean()-exact['density'])),
        minimum_channel_variance=float(variance.min()), constant_channels=int(np.count_nonzero(variance == 0)),
        reference_start=burn+obs*dt, events=int(events), elapsed_seconds=time.perf_counter()-started,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        Q_definition='time integral of full-system occupation density over disjoint future reference',
        observation_definition='binary site occupancy at equally spaced physical times; no added noise',
        **mixing_summary(trace, trace_dt))
    return dict(observed=observed, reference_density=trace, final_state=last), metadata


def cases_from_config(config):
    grid = config.get('grid', dict(N=[32, 64], alpha=np.round(np.linspace(.15,.25,21), 8).tolist(),
                                  seed=list(range(260915201, 260915209))))
    keys = list(grid)
    cases = [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))] if grid else []
    return cases+config.get('cases', [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--case-index', type=int)
    parser.add_argument('--count-only', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text()) if args.config else {}
    cases = cases_from_config(config)
    if args.count_only:
        print(len(cases)); return
    if args.output_dir is None: parser.error('--output-dir required')
    selected = list(enumerate(cases))
    if args.case_index is not None:
        if not 0 <= args.case_index < len(cases): parser.error('case index out of range')
        selected = [selected[args.case_index]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, case in selected:
        path = args.output_dir/f'case-{index:04d}.npz'
        if path.exists(): raise FileExistsError(path)
        arrays, metadata = simulate(case, config.get('simulation'))
        metadata['case_index'] = index
        metadata['config_sha256'] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        with path.open('xb') as handle:
            np.savez_compressed(handle, **arrays, metadata_json=json.dumps(metadata, sort_keys=True))
        print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == '__main__': main()
