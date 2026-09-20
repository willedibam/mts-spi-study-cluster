"""Fixed illustrative seeds; save matched observed and full-system snapshots.

These are illustrations, not additional SPI evidence or selected successful runs.
Reuses the exact benchmark generators and the previous figure-export seeds.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
from src.generators.order_parameter import (
    generate_kuramoto_order_parameter, generate_miller_huse, generate_stuart_landau)
from src.generators.dynamical import generate_cml_logistic

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'data/order_parameter/lean_snapshots_260916'


def one(task):
    system, control = task
    path = OUTPUT/f'{system}-{control:g}.npz'
    if path.exists():
        return str(path)
    common = dict(M=32, T=1000, zscore=False, return_internals=True)
    if system == 'miller-huse':
        seed = 725911
        x, a = generate_miller_huse(**common, coupling=control, mu=3,
            lattice_side=128, transients=400000, sample_every=1, future_truth_T=0,
            truth_start_T=1000, observation_mode='distributed', initial_state='random',
            rng=np.random.default_rng(seed))
        arrays = dict(X=x.T, field=a.final_field, sensors=a.patch_indices,
                      trace=a.spin_magnetization)
    elif system == 'quadratic-cml':
        seed = 355974
        x, full, indices = generate_cml_logistic(M=32, T=1000, alpha=control,
            eps=.3, transients=2000000, sample_every=1, lattice_size=512,
            observation_mode='distributed', rng=np.random.default_rng(seed),
            zscore=False, return_full_lattice=True, return_observation_indices=True)
        arrays = dict(X=x.T, field=full[:100].T, sensors=indices)
        np.testing.assert_array_equal(x, full[:,indices])
    elif system == 'stuart-landau':
        seed = 991377
        x, a = generate_stuart_landau(**common, coupling=.8,
            frequency_half_width=control, N_full=32, omega_mean=2, dt=.02,
            sample_dt=.1, burn_time=200, future_truth_T=0, output='real',
            rng=np.random.default_rng(seed))
        arrays = dict(X=x.T, trace=np.abs(a.order_parameter))
    elif system == 'kuramoto':
        slug = f'{control:g}'.replace('.', 'p')
        source = ROOT/f'data/order_parameter/kuramoto_figure_examples/M20_T1000_I0_kappa{slug}'
        meta = json.loads((source/'meta.json').read_text())
        params = dict(meta['generator']['resolved_params'])
        params.update(future_truth_T=0, store_full_phases=True)
        seed = meta['generator']['seed']
        x, a = generate_kuramoto_order_parameter(M=20,T=1000,**params,
            rng=np.random.default_rng(seed), return_internals=True)
        stored = np.load(source/'timeseries.npy')
        # Stored legacy inputs are float32; compare at their storage precision.
        np.testing.assert_allclose(x, stored, atol=4*np.finfo(stored.dtype).eps, rtol=0)
        arrays = dict(X=stored.T, field=np.cos(a.full_phases[:100]).T,
                      sensors=a.observation_indices, trace=a.r_full)
    else:
        raise ValueError(system)
    assert arrays['X'].shape[1] == 1000
    np.savez_compressed(path, **arrays, metadata_json=json.dumps(dict(
        system=system, control=control, seed=seed,
        purpose='fixed-seed illustration, not an additional q/Q evaluation',
        display_steps=100, lattice_time='end of observation' if system == 'miller-huse' else 'first 100 observations')))
    return str(path)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    tasks=[(s,c) for s,cs in [('kuramoto',[.625,1.0075,1.65]),
        ('stuart-landau',[.55,.725,1.25]), ('miller-huse',[.185,.20517,.225]),
        ('quadratic-cml',[1.60,1.75,2.00])] for c in cs]
    with ProcessPoolExecutor(args.workers) as pool:
        for output in pool.map(one,tasks):
            print(output,flush=True)
