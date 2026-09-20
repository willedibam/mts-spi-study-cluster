"""Prespecified full-population Kuramoto pilot; reuse the validated generator."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scripts.finite_regime_pipeline import export_arrays, write_json
from src.generators.order_parameter import generate_kuramoto_order_parameter

CONTROLS = (.625, .725, .825, .8875, .9425, .9775, .9925, 1.0075,
            1.0225, 1.0575, 1.0875, 1.175, 1.25, 1.375, 1.525, 1.65)


def simulate(case):
    kappa, seed = case
    x, truth = generate_kuramoto_order_parameter(
        M=32, T=1000, N_full=32, K=kappa*np.sqrt(8/np.pi), dt=.02,
        sample_dt=.1, burn_time=200, future_truth_T=10000,
        omega_mean=1, omega_std=1, frequency_distribution='gaussian',
        frequency_sampling='random', output='cos', rng=np.random.default_rng(seed),
        return_internals=True, store_full_phases=False)
    x = np.ascontiguousarray(x.T)  # generators return time x process
    assert x.shape == (32, 1000) and np.isfinite(x).all()
    assert np.all(x.std(axis=1) > 1e-8)
    np.testing.assert_allclose(truth.r_observed, truth.r_full, atol=1e-14)
    blocks = np.array([b.mean() for b in np.array_split(truth.r_full_future, 10)])
    return x, dict(control=kappa, seed=seed, M=32, N_state=32, T=1000,
        system='kuramoto-full', view='all-oscillators-cos',
        role='development' if seed < 260916009 else 'evaluation',
        Q_reference=float(truth.r_full_future.mean()), Q_window=float(truth.r_full.mean()),
        reference_block_means=blocks.tolist(),
        reference_half_difference=float(abs(blocks[:5].mean()-blocks[5:].mean())))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    cases = [(c, s) for s in range(260916001, 260916017) for c in CONTROLS]
    arrays, rows = {}, []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (x, row) in enumerate(pool.map(simulate, cases), 1):
            name = f"kuramoto-k{row['control']:.4f}-s{row['seed']}-m32-t1000"
            arrays[name] = x
            rows.append(dict(row, row_id=name, corpus_index=i))
            print(f"{i}/{len(cases)} Q={row['Q_reference']:.4f}", flush=True)
    low = np.mean([r['Q_reference'] for r in rows if r['control'] == CONTROLS[0]])
    high = np.mean([r['Q_reference'] for r in rows if r['control'] == CONTROLS[-1]])
    # Physics/observability gate only; no q, feature, or per-seed outcome selection.
    assert high-low > .2, (low, high)
    export_arrays(args.output, arrays, rows, dict(
        system='kuramoto-full', control_label='reduced coupling kappa',
        quantity_label='future mean global phase coherence',
        source='https://doi.org/10.1103/PhysRevE.102.042310',
        M=32, N=32, T=1000, controls=list(CONTROLS), seeds=16,
        development_seeds=8, held_seeds=8, dt=.02, sample_dt=.1, burn_time=200,
        future_truth_T=10000, frequency_distribution='Gaussian(mean=1, SD=1)',
        pairing='same frequencies and initial phases across controls within seed',
        physics_gate='all raw finite/nonconstant, R_M=R_N, endpoint mean rise >0.2',
        inference='finite-population pilot; kappa=1 is continuum reference, not exact N32 threshold'))
    write_json(args.output/'physics-gate.json', dict(passes=True, rows=len(rows),
        low_mean=low, high_mean=high,
        max_reference_half_difference=max(r['reference_half_difference'] for r in rows)))


if __name__ == '__main__':
    main()
