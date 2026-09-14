"""Isolated finite-system corpus export and target-blind SPI–SPI analysis.

No simulation or SPI extraction occurs here. Physical eligibility is established
before export; failures remain readable results rather than relaxed thresholds.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import yaml

from scripts.analyze_cml2d_spi import assemble, corr
from scripts.rossler_phase_sync import pilot_cases
from src.spi_spi_analysis import fit_feature_transform
from src.order_parameter_analysis import clustered_bootstrap_spearman
from src.spi_spi_contract import UNIFIED_CONTRACT_VERSION


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def rossler_physics(physics, output):
    """Prospective pilot gates, not a proof of an asymptotic phase threshold."""
    records = []
    expected = pilot_cases()
    paths = sorted(physics.glob('case-*.json'))
    if len(paths) != len(expected):
        raise ValueError(f'incomplete physics: {len(paths)}/{len(expected)}')
    source_hashes = set()
    for i, path in enumerate(paths):
        meta = json.loads(path.read_text())
        case = expected[i]
        assert meta['case_index'] == i
        for key, value in case.items():
            if isinstance(value, float):
                assert np.isclose(meta[key], value, atol=1e-14, rtol=0)
            else:
                assert meta[key] == value
        assert meta['reference'] == 100000 and meta['burn'] == 2000
        source_hashes.add(meta['source_sha256'])
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as a:
            x = a['X']
            assert x.shape == (6, 2000)
            assert hashlib.sha256(x.tobytes()).hexdigest() == meta['input_sha256']
            raw_ok = np.isfinite(x).all() and np.all(x[:, :1000].std(axis=1) > 1e-8)
        ref = meta['reference_summary']
        records.append(dict(case_index=i, arm=meta['arm'], seed=meta['seed'],
            control=meta['coupling'], dt=meta['dt'], Q=ref['Q'], PLV=ref['PLV'],
            half_difference=ref['Q_half_difference'], phase_span=ref['phase_difference_span'],
            slip_rate=ref['slip_rate'], raw_ok=bool(raw_ok),
            minimum_radius=min(ref['minimum_radius']),
            max_increment=max(ref['maximum_phase_increment']),
            poincare_discrepancy=max(ref['poincare_frequency_discrepancy'])))
    assert len(source_hashes) == 1
    frame = pd.DataFrame(records)
    primary = frame.query("arm == 'primary'")
    means = primary.groupby('control').Q.mean()
    contrast = float(means.iloc[0] - means.iloc[-1])
    dt_errors = []
    for row in frame.query("arm == 'half-dt'").itertuples():
        peer = primary[(primary.seed == row.seed) & np.isclose(primary.control, row.control)]
        assert len(peer) == 1
        dt_errors.append(abs(float(peer.Q.iloc[0]) - row.Q))
    checks = dict(all_raw_channels_vary=bool(frame.raw_ok.all()),
        phase_geometry=bool((frame.minimum_radius > 1e-3).all() and (frame.max_increment < .1).all()),
        poincare_agreement=bool((frame.poincare_discrepancy < 2*np.pi/100000 + 1e-9).all()),
        sizeable_contrast=bool(contrast > .01),
        entrained_endpoint=bool(means.iloc[-1] < .0002),
        decreasing_control_curve=bool(corr(means.index, means.values) < -.8),
        reference_stability=bool(primary.half_difference.quantile(.95) < .15*max(contrast, 0)),
        timestep_check=bool(max(dt_errors) < .1*max(contrast, 0)))
    result = dict(system='rossler-phase', passes=all(checks.values()), checks=checks,
        records=len(frame), primary_records=len(primary), Q_contrast=contrast,
        half_difference_p95=float(primary.half_difference.quantile(.95)),
        maximum_anchor_dt_difference=float(max(dt_errors)),
        control_mean_Q=means.to_dict(), source_sha256=next(iter(source_hashes)),
        interpretation='finite-time frequency entrainment; no exact asymptotic locking threshold claimed',
        gate_source_sha256=digest(__file__))
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output/'physics.csv', index=False)
    write_json(output/'physics-gate.json', result)
    print(json.dumps(result, indent=2))
    return result


def export_rossler(physics, gate_dir, output, T=1000):
    gate = json.loads((gate_dir/'physics-gate.json').read_text())
    if not gate['passes']:
        raise ValueError('physics gate failed; no corpus exported')
    if T not in (500, 1000, 2000):
        raise ValueError('unsupported observation length')
    rows, arrays = [], {}
    for i, case in enumerate(pilot_cases()[:168]):
        path = physics/f'case-{i:04d}.npz'
        meta_path = path.with_suffix('.json')
        meta = json.loads(meta_path.read_text())
        with np.load(path, allow_pickle=False) as a:
            x = np.ascontiguousarray(a['X'][:, :T])
        assert np.isfinite(x).all() and np.all(x.std(axis=1) > 1e-8)
        phase = np.unwrap(np.arctan2(x[[1,4]], x[[0,3]]), axis=1)
        q_window = float(abs(np.diff(phase[:, [0,-1]], axis=1).ravel().dot([1,-1])) / ((T-1)*meta['sample_dt']))
        name = f"rossler-c{case['coupling']:.6f}-s{case['seed']}-m6-t{T}"
        arrays[name] = x
        rows.append(dict(row_id=name, corpus_index=len(rows)+1,
            system='rossler-phase', control=case['coupling'], seed=case['seed'], M=6,
            N_state=6, N_oscillators=2, T=T, view='full-state',
            role='development' if case['seed'] < 260915105 else 'evaluation',
            Q_reference=meta['reference_summary']['Q'], Q_window=q_window,
            PLV_reference=meta['reference_summary']['PLV'],
            master=str(path), master_sha256=digest(path), metadata_sha256=digest(meta_path)))
    export_arrays(output, arrays, rows, dict(system='rossler-phase', control_label='coupling C',
        quantity_label='mean angular-frequency mismatch', source='https://doi.org/10.1103/PhysRevLett.76.1804',
        physics_gate_sha256=digest(gate_dir/'physics-gate.json')))


def export_arrays(output, arrays, rows, description):
    """Shared output schema; physics-specific exporters must pass their own gate."""
    if output.exists():
        raise FileExistsError(output)
    if len(arrays) != len(rows) or not rows:
        raise ValueError('empty/duplicate corpus')
    arrays.update(__dataset_names__=np.asarray(list(arrays)),
        __labels_json__=np.asarray([json.dumps([row['system'], row['view']]) for row in rows]),
        __shapes__=np.asarray([[row['M'],row['T']] for row in rows]),
        __axis_order__=np.asarray(['process','observation']))
    output.mkdir(parents=True)
    np.savez_compressed(output/'observations.npz', **arrays)
    archive_hash = digest(output/'observations.npz')
    write_json(output/'manifest.json', dict(rows=rows, archive_sha256=archive_hash,
        description=description, exporter_sha256=digest(__file__)))
    config = dict(name='finite-regime', source=dict(format='named-npz-v1',
        archive=str(output/'observations.npz'), sha256=archive_hash, axis_order=['process','observation']),
        base_output_dir=str(output/'mpi'), pyspi_config='configs/pyspi/benchmarked_p90.yaml',
        normalise=False, random_seed=260915)
    (output/'corpus.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    # Fixed low/high-control development smoke rows; no Q-based selection.
    development = [r for r in rows if r['role'] == 'development']
    smoke = [development[0]['corpus_index'], development[-1]['corpus_index']]
    (output/'smoke-indices.txt').write_text('\n'.join(map(str, smoke))+'\n')
    candidates = [r['corpus_index'] for r in development if r['corpus_index'] not in smoke]
    node = [candidates[i] for i in np.linspace(0,len(candidates)-1,min(48,len(candidates)),dtype=int)]
    (output/'node-indices.txt').write_text('\n'.join(map(str,node))+'\n')
    print(json.dumps(dict(rows=len(rows), config=str(output/'corpus.yaml'))))


def tasep_physics(physics, output):
    from scripts.tasep_phase_boundary import cases_from_config, exact_stationary
    expected = cases_from_config({})
    paths = sorted(physics.glob('case-*.npz'))
    if len(paths) != len(expected):
        raise ValueError(f'incomplete physics: {len(paths)}/{len(expected)}')
    records = []
    source_hashes = set()
    for i, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as a:
            meta = json.loads(str(a['metadata_json']))
            x = a['observed']
            assert x.shape == (meta['N'],2000) and np.isin(x,[0,1]).all()
            raw_ok = bool((x[:,:1000].var(axis=1) > 0).all())
        assert meta['case_index'] == i
        for key, value in expected[i].items():
            assert np.isclose(meta[key],value,rtol=0,atol=1e-12)
        assert meta['reference_time'] == 1000000 and meta['burn'] == 100000
        exact = exact_stationary(meta['N'],meta['alpha'],meta['beta'])['density']
        assert abs(exact-meta['Q_exact']) < 1e-12
        assert abs(np.mean(meta['Q_blocks'])-meta['Q_reference']) < 1e-12
        source_hashes.add(meta['source_sha256'])
        records.append(dict(case_index=i,N=meta['N'],control=meta['alpha'],seed=meta['seed'],
            Q=meta['Q_reference'],Q_exact=exact,Q_window=meta['Q_window'],raw_ok=raw_ok,
            reference_error=abs(exact-meta['Q_reference']),
            half_difference=abs(meta['Q_first_half']-meta['Q_second_half']),
            tau_int=meta['estimated_tau_int'],effective_samples=meta['estimated_effective_samples']))
    assert len(source_hashes) == 1
    frame = pd.DataFrame(records)
    arms = {}
    for N, part in frame.groupby('N'):
        means = part.groupby('control').Q.mean()
        contrast = float(means.iloc[-1]-means.iloc[0])
        checks = dict(raw_channels_vary=bool(part.raw_ok.all()),
            physical_contrast=bool(contrast > .3),
            increasing_curve=bool(corr(means.index,means.values) > .9),
            exact_agreement=bool(part.reference_error.quantile(.95) < .05),
            reference_stability=bool(part.half_difference.quantile(.95) < .08))
        arms[str(N)] = dict(passes=all(checks.values()),checks=checks,contrast=contrast,
            reference_error_p95=float(part.reference_error.quantile(.95)),
            half_difference_p95=float(part.half_difference.quantile(.95)),
            maximum_tau_int=float(part.tau_int.max()))
    output.mkdir(parents=True,exist_ok=False)
    frame.to_csv(output/'physics.csv',index=False)
    result = dict(system='tasep',arms=arms,records=len(frame),source_sha256=next(iter(source_hashes)),
        interpretation='finite-size density crossover across exact thermodynamic coexistence line; no finite-N discontinuity',
        gate_source_sha256=digest(__file__))
    write_json(output/'physics-gate.json',result)
    print(json.dumps(result,indent=2))


def export_tasep(physics, gate_dir, output, N, T=1000):
    gate = json.loads((gate_dir/'physics-gate.json').read_text())
    if not gate['arms'][str(N)]['passes']:
        raise ValueError('physical arm failed; no corpus exported')
    rows, arrays = [], {}
    for path in sorted(physics.glob('case-*.npz')):
        with np.load(path,allow_pickle=False) as a:
            meta = json.loads(str(a['metadata_json']))
            if meta['N'] != N:
                continue
            x = np.ascontiguousarray(a['observed'][:,:T],dtype=np.float64)
        assert x.shape == (N,T) and (x.var(axis=1)>0).all()
        name=f"tasep-n{N}-a{meta['alpha']:.6f}-s{meta['seed']}-t{T}"
        arrays[name] = x
        rows.append(dict(row_id=name,corpus_index=len(rows)+1,system='tasep',
            control=meta['alpha'],seed=meta['seed'],M=N,N_state=N,N_sites=N,T=T,view='full-state',
            role='development' if meta['seed'] < 260915205 else 'evaluation',
            Q_reference=meta['Q_reference'],Q_window=float(x.mean()),Q_exact=meta['Q_exact'],
            master=str(path),master_sha256=digest(path)))
    assert len(rows) == 168
    export_arrays(output,arrays,rows,dict(system='tasep',control_label='entry rate alpha',
        quantity_label='particle density',fixed_beta=.2,thermodynamic_boundary=.2,
        source='https://www.lps.ens.fr/~derrida/PAPIERS/1993/DEHP-93.pdf',
        physics_gate_sha256=digest(gate_dir/'physics-gate.json')))


def fit_coordinate(z, dev, seeds):
    """Only features and independent-seed labels enter this function."""
    transform = fit_feature_transform(z[dev], ['unified']*z.shape[1],
        minimum_valid_fraction=.99, variance_threshold=.05, block_balanced=False)
    training = transform.transform(z[dev])
    if min(training.shape) < 3:
        raise RuntimeError('too few valid features/rows for geometry assessment')
    pca = PCA(n_components=min(5, *training.shape), svd_solver='full').fit(training)
    component = pca.components_[0].copy()
    if component[np.argmax(abs(component))] < 0:
        component *= -1
    scale = float(np.std(training@component))
    cosines = [float(abs(PCA(n_components=1, svd_solver='full').fit(
        training[np.asarray(seeds)[dev] != seed]).components_[0]@component))
        for seed in np.unique(np.asarray(seeds)[dev])]
    evr = pca.explained_variance_ratio_.tolist()
    geometry = dict(evr=evr, leave_seed_loading_cosines=cosines,
        passes=bool(evr[0] >= .2 and evr[0]/max(evr[1],1e-15) >= 1.5 and min(cosines) >= .8))
    model = dict(keep=transform.keep_indices, impute=transform.impute_values,
        center=transform.center, component=component, score_scale=np.asarray(scale))
    return model, geometry


def apply_coordinate(z, model):
    selected = z[:,model['keep']]
    missing = np.mean(~np.isfinite(selected), axis=1)
    q = ((np.where(np.isfinite(selected), selected, model['impute'])-model['center'])
         @model['component'])/float(model['score_scale'])
    return q, missing


def analyze(corpus, mpi_root, output, frozen=None):
    if output.exists():
        raise FileExistsError(output)
    rows, z, order, sources = assemble(corpus, mpi_root)
    frame = pd.DataFrame([{k:r[k] for k in
        ('row_id','control','seed','M','T','view','role','system')} for r in rows])
    dev = (frame.role == 'development').to_numpy()
    output.mkdir(parents=True)
    np.savez_compressed(output/'features.npz', z=z,
        row_id=np.asarray(frame.row_id,dtype=str), spi_order=np.asarray(order))
    try:
        if frozen:
            with np.load(frozen/'model.npz', allow_pickle=False) as a:
                assert a['spi_order'].tolist() == order
                model = {key:a[key] for key in ('keep','impute','center','component','score_scale')}
            geometry = json.loads((frozen/'geometry.json').read_text())
        else:
            model, geometry = fit_coordinate(z, dev, frame.seed.to_numpy())
        q, missing = apply_coordinate(z, model)
    except RuntimeError as error:
        write_json(output/'summary.json', dict(status='feature geometry unavailable',
            passes=False, reason=str(error), rows=len(rows), sources=sources))
        return
    eligible = missing <= .05
    frame['q'] = q
    frame['selected_missingness'] = missing
    frame['eligible'] = eligible
    counts = frame.groupby(['role','control','M','T','view']).eligible.agg(['sum','size'])
    coverage = bool((counts['sum'] >= np.ceil(.75*counts['size'])).all()
                    and all((~part.eligible).mean() <= .1 for _,part in frame.groupby('role')))
    np.savez_compressed(output/'model.npz', **model, spi_order=np.asarray(order))
    write_json(output/'geometry.json', geometry)
    # Persist target-blind fitted model and coverage BEFORE attaching Q.
    write_json(output/'eligibility.json', dict(contract=UNIFIED_CONTRACT_VERSION,
        fit_uses_targets_or_controls=False, sources=sources, geometry=geometry,
        coverage_passes=coverage, model_sha256=digest(output/'model.npz'),
        manifest_sha256=digest(corpus/'manifest.json'), code_sha256=digest(__file__),
        frozen_source=str(frozen) if frozen else None))
    for key in ('Q_reference','Q_window'):
        frame[key] = [r[key] for r in rows]
    with np.load(corpus/'observations.npz', allow_pickle=False) as a:
        frame['mean_abs_correlation'] = [float(np.abs(np.corrcoef(a[r['row_id']])[
            np.triu_indices(r['M'],1)]).mean()) for r in rows]
    if frozen:
        sign = json.loads((frozen/'summary.json').read_text())['display_sign']
    else:
        sign = -1 if corr(frame.loc[dev & eligible,'q'],frame.loc[dev & eligible,'Q_reference']) < 0 else 1
    part = frame[~dev & eligible]
    results = []
    for keys, group in part.groupby(['M','T','view']):
        boot, within = clustered_bootstrap_spearman(sign*group.q, group.Q_reference,
            group.control, group.seed, n_resamples=2000, seed=260915)
        means = group.groupby('control')[['q','Q_reference']].mean()
        intervals = {}
        for quantity in ('q','Q_reference'):
            if len(means) > 1:
                index = np.argmax(abs(np.diff(means[quantity])/np.diff(means.index)))
                intervals[quantity] = means.index[index:index+2].tolist()
        results.append(dict(M=int(keys[0]), T=int(keys[1]), view=keys[2], rows=len(group),
            rho=corr(sign*group.q,group.Q_reference),
            rho_ci=np.nanquantile(boot,[.025,.975]).tolist(),
            within_control_ci=np.nanquantile(within,[.025,.975]).tolist(),
            control_mean_rho=corr(sign*means.q,means.Q_reference),
            steepest_intervals=intervals,
            window_Q_rho=corr(group.Q_window,group.Q_reference),
            mean_abs_correlation_rho=corr(group.mean_abs_correlation,group.Q_reference)))
    frame.to_csv(output/'scores.csv', index=False)
    summary = dict(status='frozen confirmation' if frozen else 'exploratory held-seed diagnostic',
        passes=coverage and geometry['passes'], coverage_passes=coverage, geometry=geometry,
        selected_features=len(model['keep']), spis=len(order), rows=len(rows),
        excluded_rows=int((~eligible).sum()), display_sign=sign, results=results,
        interpretation='failed gates make q descriptive only; not numerical Q calibration')
    write_json(output/'summary.json', summary)
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['rossler-physics','rossler-export','tasep-physics','tasep-export','analyze'])
    p.add_argument('--physics', type=Path)
    p.add_argument('--gate-dir', type=Path)
    p.add_argument('--corpus', type=Path)
    p.add_argument('--mpi-root', type=Path)
    p.add_argument('--frozen', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--T', type=int, default=1000)
    p.add_argument('--N', type=int, choices=[32,64],default=32)
    a = p.parse_args()
    if a.stage == 'rossler-physics':
        rossler_physics(a.physics,a.output)
    elif a.stage == 'rossler-export':
        export_rossler(a.physics,a.gate_dir,a.output,a.T)
    elif a.stage == 'tasep-physics':
        tasep_physics(a.physics,a.output)
    elif a.stage == 'tasep-export':
        export_tasep(a.physics,a.gate_dir,a.output,a.N,a.T)
    else:
        analyze(a.corpus,a.mpi_root,a.output,a.frozen)


if __name__ == '__main__':
    main()
