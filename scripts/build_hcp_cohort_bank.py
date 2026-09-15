"""Assemble verified HCP inputs and fixed features; no fitted preprocessing/models."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.signal import welch

from src.run_external_corpus import ExternalCorpusConfig, load_inventory, completion_error
from src.spi_spi_contract import build_unified_features, schema_sha256
from src.representation_attribution import rich_marginals


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_features(x, matrices, order):
    unified = build_unified_features(matrices, order, metric='pearson')
    f, power = welch(x, fs=250, nperseg=1000, axis=0)
    total = power[(f >= 1) & (f <= 100)].sum(axis=0)
    ratios = np.array([power[(f >= lo) & (f < hi)].sum(axis=0)/total
                       for lo, hi in [(1,4),(4,8),(8,13),(13,30),(30,45),(65,100)]])
    return unified, rich_marginals(matrices, order), np.quantile(ratios, [.1,.5,.9], axis=1).ravel()


def read_record(config, entry, waveform):
    error = completion_error(config, entry)
    assert error is None, (entry.name, error)
    directory = entry.output_dir(config)
    meta = json.loads((directory/'meta.json').read_text())
    order = [r['name'] for r in meta['pyspi']['spis']]
    assert len(order) == 289 and len(set(order)) == 289
    assert waveform.shape == (entry.T, entry.M) and np.isfinite(waveform).all()
    with np.load(directory/'spi_mpis.npz', allow_pickle=False) as matrices:
        unified, marginal, spectra = fixed_features(waveform, matrices, order)
    evidence = dict(mpi_sha256=sha(directory/'spi_mpis.npz'), meta_sha256=sha(directory/'meta.json'),
                    compute_seconds=meta['job']['compute_seconds'], spi_errors=meta['pyspi']['errors'])
    return order, unified, marginal, spectra, evidence


def check_scout(root):
    """Check two actual inputs against the already independently verified scout bank."""
    config = ExternalCorpusConfig.from_file(root/'external.yaml')
    entries = load_inventory(config)
    with np.load(config.archive, allow_pickle=False) as raw, np.load(root/'analysis/feature-bank.npz', allow_pickle=False) as saved:
        for i in (0, 1):
            order, unified, m, spectra, _ = read_record(config, entries[i], raw[entries[i].name])
            assert entries[i].name == saved['names'][i]
            assert order == saved['spi_order'].tolist()
            for actual, expected in [(unified.z,saved['z'][i]),(m,saved['m'][i]),(spectra,saved['spectra'][i])]:
                np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12, equal_nan=True)
    print('Two scout records reproduce all three frozen feature families', flush=True)


def build(root):
    plan_path = root/'cohort-T4000-extraction.json'
    plan = json.loads(plan_path.read_text())
    assert plan['status'] == 'frozen_extraction_inputs' and plan['M'] == 32 and plan['T'] == 4000
    assert plan['layout'] == 'coverage_a'
    output = root/'cohort-primary-bank'
    identity = dict(plan_sha256=sha(plan_path), script_sha256=sha(Path(__file__)),
                    feature_code_sha256={function.__name__:sha(Path(function.__code__.co_filename))
                                         for function in (build_unified_features, rich_marginals)})
    if output.exists():
        report = json.loads((output/'manifest.json').read_text())
        assert report['identity'] == identity and sha(output/'bank.npz') == report['bank_sha256']
        for row in report['rows']:
            directory = Path(row['output'])
            assert sha(directory/'spi_mpis.npz') == row['mpi_sha256']
            assert sha(directory/'meta.json') == row['meta_sha256']
        print('Verified existing cohort bank; no rebuild', flush=True)
        return
    arrays = {k:[] for k in ['x','z','m','spectra']}
    rows, spi_order, schema_hash = [], None, None
    # One source archive open per run; preserve the frozen extraction-plan order.
    for source in plan['sources']:
        config = ExternalCorpusConfig.from_file(source['config'])
        assert sha(config.archive) == config.archive_sha256 == source['archive_sha256']
        manifest_path = Path(source['config']).parent/'manifest.json'
        assert sha(manifest_path) == source['manifest_sha256']
        manifest = json.loads(manifest_path.read_text())
        assert sha(Path(source['config'])) == manifest['external_config_sha256']
        by_name = {r['name']:r for r in manifest['rows']}
        entries = load_inventory(config)
        with np.load(config.archive, allow_pickle=False) as bank:
            for task in [t for t in plan['tasks'] if t['config'] == source['config']]:
                entry = entries[task['job_index']-1]
                row = by_name[entry.name]
                assert entry.name == task['name'] and row['layout'] == 'coverage_a'
                assert row['memory'] in (1,2) and row['image'] in (1,2)
                x = bank[entry.name]
                order, unified, m, spectra, evidence = read_record(config, entry, x)
                if spi_order is None:
                    spi_order, schema_hash = order, schema_sha256(unified.schema)
                assert order == spi_order and unified.z.shape == (41616,)
                arrays['x'].append(x.astype(np.float32))
                arrays['z'].append(unified.z)
                arrays['m'].append(m)
                arrays['spectra'].append(spectra)
                rows.append(dict(task, **evidence, memory_code=row['memory'], image_code=row['image'],
                                 finite_z=int(np.isfinite(unified.z).sum())))
        print(f'Packed {len(rows)}/{len(plan["tasks"])} observations', flush=True)
    assert [r['name'] for r in rows] == [r['name'] for r in plan['tasks']]
    assert len(rows) == len({r['name'] for r in rows}) == 590
    arrays = {k:np.asarray(v) for k,v in arrays.items()}
    assert arrays['x'].shape == (590,4000,32) and arrays['m'].shape == (590,289*23)
    assert arrays['spectra'].shape == (590,18)
    arrays.update(record_id=np.array([r['name'] for r in rows]), participant=np.array([r['subject'] for r in rows]),
                  run=np.array([r['run'] for r in rows]), block=np.array([r['block'] for r in rows]),
                  y=np.array([int(r['memory_code']==2) for r in rows]),
                  image=np.array([r['image_code'] for r in rows]), spi_order=np.array(spi_order))
    with tempfile.TemporaryDirectory(prefix='.cohort-bank-', dir=root) as temporary:
        stage = Path(temporary)
        np.savez_compressed(stage/'bank.npz', **arrays)
        report = dict(status='verified_fixed_features', identity=identity, rows=rows,
                      bank_sha256=sha(stage/'bank.npz'), schema_sha256=schema_hash,
                      shapes={k:list(v.shape) for k,v in arrays.items()},
                      label_mapping={'memory_code_1': '0-back; y=0', 'memory_code_2':'2-back; y=1',
                                     'image_code_1':'faces', 'image_code_2':'tools'},
                      scope='No imputation, scaling, PCA, family assignment or model fit. NaNs preserved. Waveforms float32 for matched neural/library inputs; fixed spectral statistics computed from original float64 windows.')
        (stage/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
        stage.rename(output)
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    check_scout(args.root/'m32-short')
    if not args.check_only:
        build(args.root)
