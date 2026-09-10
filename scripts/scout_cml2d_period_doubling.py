"""Streaming 2D synchronous logistic CML, Marcq--Chate--Manneville (2006).

Retains global order traces and nested fixed sensor views, not an N x T field.
No pyspi, fitted features, or subsampling in time. Source equation uses r, not
the affine-conjugate quadratic-map alpha used by the existing 1D generator.
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
import yaml


@njit(cache=True)
def step(state, mapped, output, r, g):
    side = state.shape[0]
    for i in range(side):
        for j in range(side):
            mapped[i, j] = r * state[i, j] * (1.0 - state[i, j])
    for i in range(side):
        up, down = (i - 1) % side, (i + 1) % side
        for j in range(side):
            left, right = (j - 1) % side, (j + 1) % side
            output[i, j] = ((1 - 4*g) * mapped[i, j] + g *
                (mapped[up, j] + mapped[down, j] + mapped[i, left] + mapped[i, right]))


@njit(cache=True)
def evolve(state, r, g, burn, record_steps, observation_steps, indices):
    mapped = np.empty_like(state)
    output = np.empty_like(state)
    means = np.empty(record_steps)
    observed = np.empty((observation_steps, 2, indices.shape[1]))
    side = state.shape[0]
    for t in range(burn + record_steps):
        step(state, mapped, output, r, g)
        state, output = output, state
        k = t - burn
        if k >= 0:
            means[k] = np.mean(state)
            if k < observation_steps:
                for view in range(2):
                    for sensor in range(indices.shape[1]):
                        index = indices[view, sensor]
                        observed[k, view, sensor] = state[index // side, index % side]
    return means, observed, state


def sensor_indices(side, seed):
    if side < 8:
        raise ValueError('L must be at least 8 for nested 2x4,4x4,4x8,8x8 patches')
    rng = np.random.default_rng(seed)
    dispersed = rng.permutation(side**2)[:64]
    coords = [(i,j) for i in range(2) for j in range(4)]
    coords += [(i,j) for i in range(2,4) for j in range(4)]
    coords += [(i,j) for i in range(4) for j in range(4,8)]
    coords += [(i,j) for i in range(4,8) for j in range(8)]
    origin = rng.integers(side, size=2)
    patch = np.array([((i+origin[0]) % side)*side + (j+origin[1]) % side for i,j in coords])
    return np.stack([dispersed, patch])


def order_summary(means, observation_steps):
    if len(means) % 2 or observation_steps % 2 or len(means) <= observation_steps:
        raise ValueError('even recording/observation lengths and a disjoint future required')
    amplitude = np.abs(means[1::2] - means[::2])
    future = amplitude[observation_steps//2:]
    return dict(Q=float(future.mean()), Q_record=float(amplitude.mean()),
        Q_blocks=[float(b.mean()) for b in np.array_split(future, 8)],
        Q_first_half=float(future[:len(future)//2].mean()),
        Q_second_half=float(future[len(future)//2:].mean()),
        binder=float(1 - np.mean(future**4)/(3*np.mean(future**2)**2)) if np.mean(future**2)>0 else None)


def cases_from_config(config):
    grid = config.get('grid', {})
    keys = list(grid)
    cases = [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))] if keys else []
    return cases + config.get('cases', [])


def simulate(case, simulation):
    params = {**simulation, **case}
    side, r, g = int(params['L']), float(params['r']), float(params['g'])
    burn, steps, obs = (int(params[k]) for k in ('burn','record_steps','observation_steps'))
    if side < 8 or not 0 < r <= 4 or not 0 <= g <= .25 or burn < 0 or not 0 < obs < steps or steps % 2 or obs % 2:
        raise ValueError('invalid lattice, map, coupling or time contract')
    seeds = np.random.SeedSequence(int(params['seed'])).generate_state(2)
    state = np.random.default_rng(int(seeds[0])).random((side,side))
    indices = sensor_indices(side, int(seeds[1]))
    start = time.perf_counter()
    pre_steps = int(params.get('pre_steps', 0))
    if pre_steps:
        _, _, state = evolve(state, float(params['pre_r']), g, pre_steps, 0, 0, indices)
    means, observed, final_state = evolve(state, r, g, burn, steps, obs, indices)
    for array in (means, observed, final_state):
        if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1):
            raise ValueError('state left invariant interval [0,1]')
    meta = {**params, 'N': side**2, 'stride': 1, 'views':['dispersed','contiguous'],
        'elapsed_seconds':time.perf_counter()-start,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'order_definition':'mean(abs(global_mean[2t+1]-global_mean[2t]))',
        'Q_window':'after observation_steps through record_steps, disjoint from all sensor prefixes',
        **order_summary(means, obs)}
    return dict(global_mean=means, observed=observed, sensor_indices=indices, final_state=final_state), meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--case-index',type=int)
    p.add_argument('--count-only',action='store_true')
    a=p.parse_args()
    config=yaml.safe_load(a.config.read_text()); cases=cases_from_config(config)
    if a.count_only:
        print(len(cases)); return
    if a.output_dir is None: p.error('--output-dir required')
    selected=list(enumerate(cases))
    if a.case_index is not None:
        if not 0 <= a.case_index < len(cases): p.error('case-index outside config')
        selected=[selected[a.case_index]]
    for index,_ in selected:
        if (a.output_dir/f'case-{index:03d}.npz').exists(): raise FileExistsError(index)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for index,case in selected:
        arrays,meta=simulate(case,config['simulation'])
        meta.update(case_index=index,config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest())
        with (a.output_dir/f'case-{index:03d}.npz').open('xb') as f:
            np.savez_compressed(f,**arrays,metadata_json=json.dumps(meta,sort_keys=True))
        print(json.dumps(meta,sort_keys=True),flush=True)


if __name__ == '__main__': main()
