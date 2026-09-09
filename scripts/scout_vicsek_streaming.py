"""Physics/observation scout; angular-noise Vicsek, Chate et al. PRE 77, 046113.

Equations (1),(2): synchronous alignment including self within radius 1,
noise uniform on [-pi*eta, pi*eta], then stream using NEW velocity. Periodic
square; dt=1. Linked cells avoid a dense N-by-N distance matrix. No SPI fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from numba import njit
import yaml


@njit(cache=True)
def advance(position, heading, noise, side, speed):
    """Return one synchronous step. Inputs are unchanged; noise is in radians."""
    n = len(heading)
    cells = int(side)  # cell width >= interaction radius (1)
    width = side / cells
    head = np.full(cells * cells, -1, dtype=np.int64)
    link = np.empty(n, dtype=np.int64)
    cosine, sine = np.cos(heading), np.sin(heading)
    for i in range(n):
        cell = int(position[i, 0] / width) * cells + int(position[i, 1] / width)
        link[i], head[cell] = head[cell], i
    new_heading = np.empty_like(heading)
    new_position = np.empty_like(position)
    for i in range(n):
        cx, cy = int(position[i, 0] / width), int(position[i, 1] / width)
        sx, sy = 0.0, 0.0
        for ox in range(-1, 2):
            for oy in range(-1, 2):
                j = head[((cx + ox) % cells) * cells + (cy + oy) % cells]
                while j != -1:
                    dx = position[j, 0] - position[i, 0]
                    dy = position[j, 1] - position[i, 1]
                    dx -= side * np.rint(dx / side)
                    dy -= side * np.rint(dy / side)
                    if dx * dx + dy * dy <= 1.0:
                        sx += cosine[j]
                        sy += sine[j]
                    j = link[j]
        angle = np.arctan2(sy, sx) + noise[i]
        new_heading[i] = (angle + np.pi) % (2 * np.pi) - np.pi
        new_position[i, 0] = (position[i, 0] + speed * np.cos(angle)) % side
        new_position[i, 1] = (position[i, 1] + speed * np.sin(angle)) % side
    return new_position, new_heading


@njit(cache=True)
def advance_many(position, heading, steps, eta, side, speed, seed):
    # One call per trajectory; observation RNG never affects dynamic noise.
    np.random.seed(seed)
    for _ in range(steps):
        noise = (np.random.random(len(heading)) - .5) * (2 * np.pi * eta)
        position, heading = advance(position, heading, noise, side, speed)
    return position, heading


@njit(cache=True)
def record(position, heading, samples, stride, eta, side, speed, seed,
           particle_ids, bin_ids, bin_width):
    np.random.seed(seed)
    phi = np.empty(samples)
    global_velocity = np.empty((samples, 2))
    # views: dispersed tracked particles, initially-local tracked particles.
    velocity = np.empty((samples, 2, 32, 2), dtype=np.float32)
    counts = np.empty((samples, 2, 32), dtype=np.int32)
    current = np.empty((samples, 2, 32, 2), dtype=np.float32)
    bins = int(round(side / bin_width))
    for t in range(samples):
        for _ in range(stride):
            noise = (np.random.random(len(heading)) - .5) * (2 * np.pi * eta)
            position, heading = advance(position, heading, noise, side, speed)
        cosines, sines = np.cos(heading), np.sin(heading)
        mx, my = np.mean(cosines), np.mean(sines)
        global_velocity[t, 0], global_velocity[t, 1] = mx, my
        phi[t] = np.sqrt(mx * mx + my * my)
        all_counts = np.zeros(bins * bins, dtype=np.int32)
        all_current = np.zeros((bins * bins, 2))
        for i in range(len(heading)):
            b = int(position[i, 0] / bin_width) * bins + int(position[i, 1] / bin_width)
            all_counts[b] += 1
            all_current[b, 0] += cosines[i]
            all_current[b, 1] += sines[i]
        for view in range(2):
            for j in range(32):
                i = particle_ids[view, j]
                velocity[t, view, j, 0], velocity[t, view, j, 1] = cosines[i], sines[i]
                b = bin_ids[view, j]
                counts[t, view, j] = all_counts[b]
                current[t, view, j, 0] = all_current[b, 0] / bin_width**2
                current[t, view, j, 1] = all_current[b, 1] / bin_width**2
    return phi, global_velocity, velocity, counts, current, position, heading


def observation_ids(position, side, bin_width, seed):
    rng = np.random.default_rng(seed)
    n = len(position)
    dispersed = rng.permutation(n)[:32]
    delta = position - rng.uniform(0, side, 2)
    delta -= side * np.rint(delta / side)
    local = np.argsort(np.sum(delta**2, axis=1))[:32]
    bins = int(round(side / bin_width))
    # Nested rectangles: first 8=2x4, first 16=4x4, all 32=4x8.
    coords = [(x, y) for x in range(2) for y in range(4)]
    coords += [(x, y) for x in range(2, 4) for y in range(4)]
    coords += [(x, y) for x in range(4) for y in range(4, 8)]
    origin = rng.integers(bins, size=2)
    patch = np.array([((x + origin[0]) % bins) * bins + (y + origin[1]) % bins
                      for x, y in coords])
    scattered = rng.permutation(bins * bins)[:32]
    return np.stack([dispersed, local]), np.stack([scattered, patch])


def simulate(case, config):
    side = float(case['L'])
    n_float = config['density'] * side**2
    n = int(round(n_float))
    if side < 3 or n < 32 or not np.isclose(n, n_float):
        raise ValueError('L>=3 and integral N>=32 required')
    bw = float(config['bin_width'])
    if bw <= 0 or side / bw < 8 or not np.isclose(side / bw, round(side / bw)):
        raise ValueError('L/bin_width must be integral and >=8 for nested spatial views')
    if not 0 <= case['eta'] <= 1 or config['speed'] <= 0:
        raise ValueError('require eta in [0,1], speed>0')
    if config['burn'] < 0 or config['samples'] < 8 or config['stride'] < 1:
        raise ValueError('invalid burn, samples or stride')
    if case['start'] not in ('ordered', 'random'):
        raise ValueError('start must be ordered or random')
    seeds = np.random.SeedSequence(int(case['seed'])).generate_state(4)
    rng = np.random.default_rng(int(seeds[0]))
    position = rng.uniform(0, side, (n, 2))
    heading = rng.uniform(-np.pi, np.pi, n)
    if case['start'] == 'ordered':
        heading[:] = 0
    start_time = time.monotonic()
    position, heading = advance_many(position, heading, config['burn'], case['eta'],
                                     side, config['speed'], int(seeds[1]))
    particles, bins = observation_ids(position, side, bw, int(seeds[2]))
    result = record(position, heading, config['samples'], config['stride'], case['eta'],
                    side, config['speed'], int(seeds[3]), particles, bins, bw)
    phi, global_velocity, velocity, counts, current, final_position, final_heading = result
    metadata = {**case, **config, 'N': n, 'elapsed_seconds': time.monotonic() - start_time,
                'phi_mean': float(phi.mean()), 'phi_std': float(phi.std()),
                'phi_block_means': [float(x.mean()) for x in np.array_split(phi, 8)],
                'binder': float(1 - np.mean(phi**4) / (3 * np.mean(phi**2)**2)),
                'particle_view_order': ['dispersed_tracked', 'initially_local_tracked'],
                'field_view_order': ['dispersed_fixed_bins', 'contiguous_fixed_bins'],
                'current_definition': 'sum(unit headings)/bin area, not conditional mean',
                'purpose': 'exploratory; no stationarity, discontinuity or SPI claim',
                'source': 'https://doi.org/10.1103/PhysRevE.77.046113',
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return dict(phi=phi, global_velocity=global_velocity, particle_velocity=velocity,
                bin_counts=counts, bin_current=current, particle_ids=particles, bin_ids=bins,
                final_position=final_position, final_heading=final_heading,
                metadata_json=json.dumps(metadata, sort_keys=True)), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--case-index', type=int, help='Run one zero-based case (scheduler use)')
    args = parser.parse_args()
    specification = yaml.safe_load(args.config.read_text())
    selected = list(enumerate(specification['cases']))
    if args.case_index is not None:
        if not 0 <= args.case_index < len(selected):
            parser.error('case-index outside configured cases')
        selected = [selected[args.case_index]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Never overwrite or silently reuse possibly incompatible experiments.
    for index, case in selected:
        path = args.output_dir / f'case-{index:03d}.npz'
        if path.exists():
            raise FileExistsError(path)
    for index, case in selected:
        arrays, metadata = simulate(case, specification['simulation'])
        if not all(np.isfinite(a).all() for a in arrays.values() if isinstance(a, np.ndarray)):
            raise RuntimeError('nonfinite simulation output')
        metadata['config_sha256'] = hashlib.sha256(args.config.read_bytes()).hexdigest()
        arrays['metadata_json'] = json.dumps(metadata, sort_keys=True)
        with (args.output_dir / f'case-{index:03d}.npz').open('xb') as handle:
            np.savez_compressed(handle, **arrays)
        print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
