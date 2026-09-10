"""Evaluate all frozen models once; no fitting or target-driven selection."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml

from src.neurotycho_evaluation import audited_cuda_tolerance, summarize_run, verify_pearson_edges
from src.neurotycho_learning import predict_binary
from src.neurotycho_statistical import METHODS, predict_fitted
from src.representation_state_neural import make_encoder
from src.spi_edge_pool import pack_inputs


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_file(folders, filename):
    matches = [folder / filename for folder in folders if (folder / filename).exists()]
    if len(matches) != 1:
        raise ValueError(f'Expected one {filename}, found {len(matches)}')
    return matches[0]


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    config = yaml.safe_load(args.config.read_text())
    torch.set_num_threads(2)
    provenance = json.loads(args.target_bank.with_suffix('.json').read_text())
    assert 'PF' in provenance['phase'] and sha(args.target_bank) == provenance['sha256']
    manifest_path = args.target_bank.parent / 'manifest.json'
    assert sha(manifest_path) == provenance['manifest_sha256']
    manifest = json.loads(manifest_path.read_text())
    raw_provenance = {Path(r['metadata']).name: r for r in manifest['sources']}
    with np.load(args.target_bank) as values:
        bank = {key: values[key] for key in values.files}
    target_gram_error = verify_pearson_edges(bank)
    assert set(bank['animal']) == set(config['target_animals'])
    identities = list(zip(bank['record_id'], bank['M'], bank['T'], strict=True))
    assert len(set(identities)) == len(identities)
    spectral_report = json.loads(args.spectral.with_suffix('.json').read_text())
    assert spectral_report['phase'] == 'evaluation' and sha(args.spectral) == spectral_report['sha256']
    with np.load(args.spectral) as values:
        lookup = {(r, m, t): i for i, (r, m, t) in enumerate(zip(values['record_id'], values['M'], values['T'], strict=True))}
        positions = [lookup[key] for key in identities]
        for key in ['y', 'animal', 'archive']:
            np.testing.assert_array_equal(bank[key], values[key][positions])
        bank['spectral'] = values['spectral'][positions]
    raw, excluded_dates = {}, []
    for path in sorted(args.target_windows.glob('201*.json')):
        meta = json.loads(path.read_text())
        assert 'pf' in meta['archive']
        if not meta['usable']:
            excluded_dates.append(meta['archive'])
            continue
        expected = raw_provenance[path.name]
        assert sha(path) == expected['metadata_sha256']
        assert sha(path.with_suffix('.npz')) == expected['bank_sha256']
        with np.load(path.with_suffix('.npz')) as values:
            for r in meta['records']:
                if r['quality']['accepted']:
                    key = f'{r["archive"]}/{r["session"]}/{r["state"]}/{r["start"]}'
                    assert key not in raw
                    raw[key] = values['x'][r['array_row']]
    assert set(raw) == set(bank['record_id'])
    # Require every prespecified model before producing any target score.
    stat_paths = {(method, a): args.statistical / f'{method}-{a}.json'
                  for method in METHODS for a in config['target_animals']}
    neural_paths = {(kind, a, seed): unique_file(args.neural, f'{kind}-{a}-s{seed}.json')
                    for kind in ['matched', 'enriched', 'pool'] for a in config['target_animals'] for seed in config['neural_seeds']}
    for path in [*stat_paths.values(), *neural_paths.values()]:
        assert path.exists()
        report = json.loads(path.read_text())
        assert report['identity']['config_sha256'] == sha(args.config) and report['target_data_used'] is False
        assert report['identity']['animal'] not in report['source_animals'] and len(report['source_animals']) == 3
    outputs, audits = [], []
    def store_prediction(method, seed, a, m, t, indices, probability):
        outputs.append(dict(method=method, seed=seed, animal=a, M=m, T=t, indices=indices, probability=probability))
    for (method, animal), path in stat_paths.items():
        report = json.loads(path.read_text())
        assert sha(path.with_suffix('.joblib')) == report['model_sha256']
        fitted = joblib.load(path.with_suffix('.joblib'))
        for m, t in [(16, 2000), (8, 1000)]:
            ix = np.flatnonzero((bank['animal'] == animal) & (bank['M'] == m) & (bank['T'] == t))
            store_prediction(method, 0, animal, m, t, ix, predict_fitted(fitted, bank, ix))
        audits.append(dict(model=str(path), report_sha256=sha(path), model_sha256=report['model_sha256']))
    source_cache = {}
    cuda_audit_path = args.source_data / 'cuda-replay-audit.json'
    cuda_audit = json.loads(cuda_audit_path.read_text())
    for (kind, animal, seed), path in neural_paths.items():
        report = json.loads(path.read_text())
        assert sha(path.with_suffix('.pt')) == report['checkpoint_sha256']
        assert sha(path.with_suffix('.npz')) == report['predictions_sha256']
        for filename, digest in report['identity']['inputs'].items():
            assert sha(args.source_data / filename) == digest
        filename = 'edges.npz' if kind == 'pool' else 'dense.npz' if kind == 'enriched' else 'matched.npz'
        if filename not in source_cache:
            with np.load(args.source_data / filename) as values:
                source_cache[filename] = {k: values[k] for k in ['x', 'record_id', 'y', 'animal']}
        source = source_cache[filename]
        checkpoint = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=True)
        model = make_encoder(checkpoint['spec']); model.load_state_dict(checkpoint['state_dict'])
        with np.load(path.with_suffix('.npz')) as saved:
            lookup = {r: i for i, r in enumerate(source['record_id'])}
            ix = np.array([lookup[r] for r in saved['training_ids']])
            assert np.all(source['animal'][ix] != animal)
            np.testing.assert_array_equal(saved['y'], source['y'][ix])
            replay = predict_binary(model, torch.from_numpy(source['x'][ix]))
            difference = float(np.max(np.abs(replay - saved['probability'])))
            tolerance = (audited_cuda_tolerance(cuda_audit, path.name, report['checkpoint_sha256'])
                         if report['identity']['device'] == 'cuda' else 2e-5)
            np.testing.assert_allclose(replay, saved['probability'], atol=tolerance, rtol=0)
        for m, t in [(16, 2000), (8, 1000)]:
            ix = np.flatnonzero((bank['animal'] == animal) & (bank['M'] == m) & (bank['T'] == t))
            if kind == 'pool':
                x = pack_inputs(bank['edges'][ix, :m*(m-1)], bank['validity'][ix])
            else:
                x = np.stack([raw[str(bank['record_id'][i])][:m, -t:].T for i in ix]).astype(np.float64)
                x = ((x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)).astype(np.float32)
            probability = predict_binary(model, torch.from_numpy(x))
            store_prediction(kind, seed, animal, m, t, ix, probability)
        audits.append(dict(model=str(path), report_sha256=sha(path), model_sha256=report['checkpoint_sha256'],
                           source_cpu_replay_max_difference=difference, source_cpu_replay_tolerance=tolerance))
    scores = []
    for method, seeds in [(m, [0]) for m in METHODS] + [(k, config['neural_seeds']) for k in ['matched', 'enriched', 'pool']]:
        for seed in seeds:
            for m, t in [(16, 2000), (8, 1000)]:
                selected = [r for r in outputs if (r['method'], r['seed'], r['M'], r['T']) == (method, seed, m, t)]
                assert len(selected) == 2
                ix = np.concatenate([r['indices'] for r in selected]); p = np.concatenate([r['probability'] for r in selected])
                scores.append(dict(method=method, seed=seed, **summarize_run(bank['y'][ix], p, bank['animal'][ix], bank['archive'][ix], m, t)))
    args.output.mkdir(parents=True)
    flat = {k: [] for k in ['method', 'seed', 'record_id', 'animal', 'archive', 'y', 'M', 'T', 'probability']}
    for row in outputs:
        ix = row['indices']
        for key in ['method', 'seed', 'M', 'T']:
            flat[key].extend([row[key]] * len(ix))
        for key in ['record_id', 'animal', 'archive', 'y']:
            flat[key].extend(bank[key][ix].tolist())
        flat['probability'].extend(row['probability'].tolist())
    np.savez_compressed(args.output / 'predictions.npz', **{k: np.asarray(v) for k, v in flat.items()})
    result = dict(protocol_sha256=sha(args.config), target_bank_sha256=sha(args.target_bank),
        source_cuda_precision_audit_sha256=sha(cuda_audit_path),
        target_pearson_edge_gram_max_difference=target_gram_error,
        target_spectral_sha256=sha(args.spectral), models=audits, scores=scores,
        prediction_sha256=sha(args.output / 'predictions.npz'), evaluation_script_sha256=sha(Path(__file__)),
        unit=f"{len(set(bank['animal']))} target animals/{len(set(bank['archive']))} dates; windows are not independent subjects",
        excluded_target_dates=excluded_dates, threshold=.5,
        target_calibration=False, source_model_selection_only=True)
    (args.output / 'report.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# NeuroTycho prospective transfer', '', 'Mean of dates within animal, then animals; neural rows average three initialization results.',
             '', '| Method | M,T | Balanced accuracy | AUROC | Balanced Brier |', '|---|---|---:|---:|---:|']
    for method in list(METHODS) + ['matched', 'enriched', 'pool']:
        for m, t in [(16, 2000), (8, 1000)]:
            values = [s['mean'] for s in scores if s['method'] == method and s['M'] == m]
            numbers = [np.mean([v[k] for v in values]) for k in ['balanced_accuracy', 'auroc', 'balanced_brier']]
            lines.append(f'| {method} | {m},{t} | '+ ' | '.join(f'{n:.4f}' for n in numbers) + ' |')
    lines += ['', 'These two evaluation animals also appeared in exploratory source-data development, but each fitted model excludes its target animal.',
              'Anaesthetic and calendar period are confounded. Reduced analysis windows inherit 28 seconds of filtering context.',
              'Per-date, per-animal and per-seed results are retained in report.json; no population-level uncertainty or clinical validity is established.']
    (args.output / 'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/analysis/neurotycho-transfer-260910.yaml'))
    parser.add_argument('--target-bank', type=Path, required=True)
    parser.add_argument('--target-windows', type=Path, required=True)
    parser.add_argument('--spectral', type=Path, required=True)
    parser.add_argument('--statistical', type=Path, required=True)
    parser.add_argument('--neural', type=Path, nargs='+', required=True)
    parser.add_argument('--source-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
