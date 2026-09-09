"""Streaming physics scout for Matthews--Strogatz (1990); no SPI fitting.

Keep the complex collective trace, activity, and a nested random sensor view.
The laboratory carrier is explicit; R is invariant under common rotation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def rhs(w, omega, coupling):
    return (1 + 1j * omega - np.abs(w) ** 2) * w + coupling * (w.mean() - w)


def step(w, omega, coupling, dt):
    a = rhs(w, omega, coupling)
    b = rhs(w + dt * a / 2, omega, coupling)
    c = rhs(w + dt * b / 2, omega, coupling)
    d = rhs(w + dt * c, omega, coupling)
    return w + dt * (a + 2 * b + 2 * c + d) / 6


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--N', type=int, required=True)
    p.add_argument('--gamma', type=float, required=True)
    p.add_argument('--seed', type=int, default=910001)
    p.add_argument('--coupling', type=float, default=.8)
    p.add_argument('--carrier', type=float, default=2.)
    p.add_argument('--dt', type=float, default=.02)
    p.add_argument('--sample-dt', type=float, default=.1)
    p.add_argument('--burn', type=float, default=200.)
    p.add_argument('--samples', type=int, default=8000)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.N < 2 or args.samples < 8 or args.dt <= 0 or args.burn < 0:
        p.error('invalid population, sample count or integration interval')
    stride = int(round(args.sample_dt / args.dt))
    burn_steps = int(round(args.burn / args.dt))
    if stride < 1 or not np.isclose(stride * args.dt, args.sample_dt):
        p.error('sample-dt must be a positive integer multiple of dt')
    if not np.isclose(burn_steps * args.dt, args.burn):
        p.error('burn must be an integer multiple of dt')
    if args.output.exists():
        p.error(f'refusing to overwrite {args.output}')
    rng = np.random.default_rng(args.seed)
    omega = args.carrier + np.linspace(-args.gamma, args.gamma, args.N)[rng.permutation(args.N)]
    w = rng.uniform(-1, 1, args.N) + 1j * rng.uniform(-1, 1, args.N)
    sensors = rng.permutation(args.N)[:min(32, args.N)]
    start = time.monotonic()
    for _ in range(burn_steps):
        w = step(w, omega, args.coupling, args.dt)
    Z = np.empty(args.samples, complex)
    activity = np.empty(args.samples)
    observed = np.empty((args.samples, len(sensors)), complex)
    for t in range(args.samples):
        for _ in range(stride):
            w = step(w, omega, args.coupling, args.dt)
        Z[t] = w.mean()
        activity[t] = np.mean(np.abs(w) ** 2)
        observed[t] = w[sensors]
    if not np.isfinite(Z).all() or not np.isfinite(observed).all():
        raise RuntimeError('nonfinite trajectory')
    R = np.abs(Z)
    blocks = np.array_split(R, 8)
    metadata = {**vars(args), 'output': str(args.output),
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'elapsed_seconds': time.monotonic() - start,
                'R_mean': float(R.mean()), 'R_std': float(R.std()),
                'R_block_means': [float(b.mean()) for b in blocks],
                'R_block_stds': [float(b.std()) for b in blocks],
                'purpose': 'exploratory physics and observation audit, not confirmation'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        np.savez_compressed(handle, Z=Z, activity=activity, observed=observed,
                            observation_indices=sensors, sensor_frequencies=omega[sensors],
                            metadata_json=json.dumps(metadata, sort_keys=True))
    print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
