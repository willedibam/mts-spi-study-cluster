"""Replace cancelled TASEP baselines with stable sparse/dense Gaussian VAR(1).

Reuse the frozen CML/MEG inputs without regenerating them. This probes pair
sampling, not phase transitions or a new physical inference benchmark.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
import yaml

from scripts.large_m_pair_baselines import sha, standardize


def transition_matrix(m, topology, seed):
    if m < 3 or topology not in ('sparse', 'dense'):
        raise ValueError('M>=3 and sparse/dense topology required')
    rng = np.random.default_rng(seed)
    if topology == 'sparse':
        order = rng.permutation(m)
        weights = np.zeros((m, m))
        weights[order, np.roll(order, 1)] = 1
    else:
        weights = rng.exponential(size=(m, m))
        np.fill_diagonal(weights, 0)
        weights /= weights.sum(axis=1, keepdims=True)
    # Every row sums to .85, all entries nonnegative: Perron root exactly .85.
    return .55*np.eye(m) + .30*weights


def generate(m, topology, seed, steps=1000):
    topology_seed, noise_seed = np.random.SeedSequence(seed).spawn(2)
    matrix = transition_matrix(m, topology, topology_seed)
    covariance = solve_discrete_lyapunov(matrix, np.eye(m))
    covariance = (covariance+covariance.T)/2
    residual = np.linalg.norm(covariance-matrix@covariance@matrix.T-np.eye(m))/np.linalg.norm(covariance)
    radius = float(np.max(abs(np.linalg.eigvals(matrix))))
    assert residual < 1e-10 and abs(radius-.85) < 1e-10
    rng = np.random.default_rng(noise_seed)
    state = np.linalg.cholesky(covariance) @ rng.standard_normal(m)
    noise = rng.standard_normal((steps, m))
    raw = np.empty((steps, m))
    for t in range(steps):
        state = matrix@state + noise[t]
        raw[t] = state
    np.testing.assert_allclose(raw[1:]-raw[:-1]@matrix.T, noise[1:], rtol=1e-12, atol=1e-12)
    x = standardize(raw)
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    oracle = dict(transition=matrix, stationary_covariance=covariance,
                  sample_mean=raw.mean(axis=0), sample_sd=raw.std(axis=0))
    metadata = dict(system='var', topology=topology, seed=seed, M=m, N=m, T=steps,
        spectral_radius=radius, lyapunov_relative_residual=float(residual),
        nonzero_directed_couplings=int(np.count_nonzero(off_diagonal)),
        minimum_channel_variance=float(raw.var(axis=0).min()),
        equation='x[t] = A x[t-1] + epsilon[t]; epsilon iid N(0,I); A=.55I+.30W',
        initial_distribution='Exact stationary Gaussian N(0,Sigma), with Sigma=A Sigma A.T+I',
        normalization='Per-record channel z-score. Saved oracle matrices describe the raw pre-standardization process.',
        scope='Topology and edge-strength distribution differ; not a pure density effect or an order-parameter claim.')
    return x, oracle, metadata


def prepare(original, output):
    assert not (output/'preparation.json').exists(), 'Keep previous preparation immutable'
    output.mkdir(parents=True, exist_ok=True)
    prior = json.loads((original/'preparation.json').read_text())
    old_rows = {r['name']:r for r in prior['rows']}
    all_rows, all_metadata, oracles = [], [], {}
    for group, m in [('m64',64), ('m100',100), ('large',256)]:
        old_config = yaml.safe_load((original/f'{group}.yaml').read_text())
        assert sha(original/f'{group}.npz') == old_config['source']['sha256']
        arrays, rows = {}, []
        with np.load(original/f'{group}.npz') as bank:
            for name in bank['__dataset_names__'].tolist():
                row = old_rows[name]
                if row['system'] != 'tasep':
                    assert row['system'] in ('cml','meg')
                    arrays[name] = bank[name]
                    rows.append(dict(row, reused_source=str(original/f'{group}.npz')))
        assert len(arrays) == 10
        for topology in ['sparse', 'dense']:
            for seed in range(260917601,260917604):
                name = f'var-{topology}-s{seed}-M{m}'
                arrays[name], oracle, metadata = generate(m, topology, seed)
                rows.append(dict(name=name, system='var', M=m, T=1000, topology=topology, seed=seed))
                all_metadata.append(metadata)
                oracles.update({f'{name}__{key}':value for key,value in oracle.items()})
        names = [r['name'] for r in rows]
        assert len(names) == len(set(names)) == 16
        path = output/f'{group}.npz'
        assert not path.exists()
        np.savez_compressed(path, **arrays, __dataset_names__=np.array(names),
            __labels_json__=np.array([json.dumps([r['system']]) for r in rows]),
            __shapes__=np.array([arrays[n].shape for n in names]),
            __axis_order__=np.array(['observation','process']))
        config = dict(old_config, name=f'large-m-var-panel-{group}-260917', base_output_dir=str(output/'pyspi'))
        config['source'] = dict(old_config['source'], archive=str(path), sha256=sha(path))
        (output/f'{group}.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        smoke = [next(i for i,r in enumerate(rows,1) if r['system']==system) for system in ['cml','meg','var']]
        for tag, indices in [('smoke',smoke), ('rest',[i for i in range(1,17) if i not in smoke])]:
            (output/f'{group}-{tag}.txt').write_text(''.join(f'{i}\n' for i in indices))
        all_rows.extend(rows)
    np.savez_compressed(output/'var-oracles.npz', **oracles)
    report = dict(rows=all_rows, var_metadata=all_metadata, replaced_system='tasep',
        original_root=str(original), original_preparation_sha256=sha(original/'preparation.json'),
        script_sha256=sha(__file__), oracle_sha256=sha(output/'var-oracles.npz'),
        scope='48 exact-reference recordings/views:18 CML,12 dependent MEG,18 VAR. No TASEP extraction authorized.',
        motivation='Known stable linear stochastic reference with sparse/dense directed edge patterns, alongside nonlinear CML and real MEG.',
        reference='https://www.statsmodels.org/stable/vector_ar.html')
    (output/'preparation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(records=len(all_rows), var_records=len(all_metadata),
        maximum_lyapunov_residual=max(r['lyapunov_relative_residual'] for r in all_metadata)), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.original, args.output)
