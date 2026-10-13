"""Full-duration sampling audit and lean SPI–SPI analysis for measured oscillators."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert, resample_poly, welch

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = (2000, 5000, 10000)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resample_record(x, target):
    if x.shape[1] % target:
        raise ValueError("This audit requires an integer resampling ratio")
    return resample_poly(x, 1, x.shape[1] // target, axis=1, padtype="line")


def coordination(x, fs, peak_hz):
    centred = x - x.mean(axis=1, keepdims=True)
    analytic = hilbert(centred, axis=1)
    amplitude = np.abs(analytic)
    unit = analytic / np.maximum(amplitude, 1e-12)
    plv = np.abs(unit @ unit.conj().T / x.shape[1])
    r = np.abs(unit.mean(axis=0))
    # Five-cycle smoothing defines an operational intermittency diagnostic.
    r_smooth = uniform_filter1d(r, max(1, round(5 * fs / peak_hz)), mode="nearest")
    active = r_smooth > .8
    edge = ~np.eye(len(x), dtype=bool)
    return dict(r=r, r_smooth=r_smooth, plv=plv[edge],
                envelope=np.corrcoef(amplitude)[edge], pearson=np.corrcoef(centred)[edge],
                mean_r=float(r.mean()), sd_r=float(r_smooth.std()),
                high_fraction=float(active.mean()), switches=int(np.count_nonzero(np.diff(active))),
                mean_abs_pearson=float(np.abs(np.corrcoef(centred)[edge]).mean()))


def audit_record(x, fs, targets=CANDIDATES):
    """Compare each resampling with the SAME complete, high-resolution record."""
    f, psd = welch(x, fs=fs, nperseg=min(x.shape[1], round(fs)), axis=1)
    peaks = f[1:][np.argmax(psd[:, 1:], axis=1)]
    peak = float(np.median(peaks))
    if peak <= 0 or np.any(x.std(axis=1) <= 1e-12):
        raise ValueError("Degenerate channels or spectrum")
    reference = coordination(x, fs, peak)
    summaries, views, traces = [], {}, {}
    for target in targets:
        factor = x.shape[1] // target
        y = resample_record(x, target)
        out_fs = fs / factor
        replay = resample_poly(y, factor, 1, axis=1, padtype="line")[:, :x.shape[1]]
        trim = max(1, round(.05 * fs))
        nrmse = np.sqrt(np.mean((replay[:, trim:-trim] - x[:, trim:-trim]) ** 2, axis=1)) / x[:, trim:-trim].std(axis=1)
        loss = psd[:, f >= .4 * out_fs].sum(axis=1) / psd.sum(axis=1)
        measured = coordination(y, out_fs, peak)
        trim_out = max(2, round(.05 * out_fs))
        sl = slice(trim_out, -trim_out)
        r_error = float(np.sqrt(np.mean((measured['r_smooth'][sl] - reference['r_smooth'][::factor][sl]) ** 2)))
        row = dict(T=target, fs=out_fs, seconds=x.shape[1] / fs, peak_hz=peak,
                   power_above_passband_p95=float(np.quantile(loss, .95)),
                   reconstruction_nrmse_p95=float(np.quantile(nrmse, .95)),
                   plv_rmse=float(np.sqrt(np.mean((measured['plv'] - reference['plv']) ** 2))),
                   envelope_rmse=float(np.sqrt(np.mean((measured['envelope'] - reference['envelope']) ** 2))),
                   pearson_rmse=float(np.sqrt(np.mean((measured['pearson'] - reference['pearson']) ** 2))),
                   synchrony_trace_rmse=r_error,
                   high_fraction_error=abs(measured['high_fraction'] - reference['high_fraction']),
                   switches_raw=reference['switches'], switches_resampled=measured['switches'],
                   mean_r=reference['mean_r'], sd_r=reference['sd_r'], high_fraction=reference['high_fraction'])
        row['passes'] = bool(row['power_above_passband_p95'] <= .01 and row['reconstruction_nrmse_p95'] <= .05
                             and row['plv_rmse'] <= .03 and row['envelope_rmse'] <= .03
                             and row['pearson_rmse'] <= .02 and r_error <= .03 and row['high_fraction_error'] <= .03)
        summaries.append(row)
        views[target] = y
        traces[f'r_T{target}'] = measured['r_smooth']
    # Modest display products; full-resolution inputs remain in the raw bank.
    display = dict(f=f, psd=psd, peaks=peaks,
                   r_reference=reference['r_smooth'][::max(1, x.shape[1] // 10000)],
                   waveform_reference=x[:3, :min(x.shape[1], round(.1 * fs))],
                   fs_reference=fs, **traces)
    return summaries, views, display


def scope(raw_root, output):
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(raw_root.glob('step-???.npz'))
    if not paths:
        raise FileNotFoundError(f"No verified raw records in {raw_root}")
    rows, sources = [], []
    for path in paths:
        with np.load(path) as bank:
            x, fs, step, record = bank['x'], float(bank['fs']), int(bank['step']), int(bank['record'])
        result, _, display = audit_record(x, fs)
        rows.extend(dict(r, step=step, record=record, voltage=step * .005) for r in result)
        np.savez_compressed(output / f'display-{record}-{step:03d}.npz', **display)
        sources.append(dict(path=str(path), sha256=sha(path), step=step, M=x.shape[0], T=x.shape[1], fs=fs))
        print(f"Audited step {step}: {[ (r['T'], r['passes']) for r in result ]}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'sampling-audit.csv', index=False)
    acceptable = [t for t in CANDIDATES if frame.loc[frame['T'] == t, 'passes'].all()]
    decision = dict(selected_T=min(acceptable) if acceptable else None, records=len(paths),
                    duration_preserved=True, filter='scipy.signal.resample_poly default Kaiser FIR; padtype=line',
                    source_files=sources, script_sha256=sha(__file__),
                    interpretation='Operational screening tolerances; inspect spectra and dwell traces before extraction.')
    (output / 'sampling-decision.json').write_text(json.dumps(decision, indent=2) + '\n')
    return decision


def build(raw_root, output, target, record):
    import yaml
    output.mkdir(parents=True, exist_ok=True)
    arrays, rows = {}, []
    paths = sorted(raw_root.glob('step-???.npz'))
    if len(paths) < 2:
        raise ValueError('Need at least two source settings')
    for i, path in enumerate(paths):
        with np.load(path) as bank:
            x, fs, step = bank['x'], float(bank['fs']), int(bank['step'])
            assert int(bank['record']) == record
        y = resample_record(x, target)
        y = (y - y.mean(axis=1, keepdims=True)) / y.std(axis=1, keepdims=True)
        name = f'r{record}-s{step:03d}-t{target}'
        arrays[name] = y
        rows.append(dict(name=name, record=record, step=step, voltage=step * .005, M=len(y), T=target,
                         fs=fs * target / x.shape[1], seconds=x.shape[1] / fs,
                         development=i % 2 == 0, source_sha256=sha(path)))
    arrays.update(__dataset_names__=np.array([r['name'] for r in rows]),
                  __labels_json__=np.array([json.dumps([str(record), f"step-{r['step']}"]) for r in rows]),
                  __shapes__=np.array([(r['M'], r['T']) for r in rows]),
                  __axis_order__=np.array(['process', 'observation']))
    archive = output / 'observations.npz'
    np.savez_compressed(archive, **arrays)
    name = f'hardware-{record}-t{target}'
    config = dict(name=name, source=dict(format='named-npz-v1', archive=str(archive.resolve()),
                                        sha256=sha(archive), axis_order=['process', 'observation']),
                  base_output_dir=str((output / 'mpis').resolve()), pyspi_config='configs/pyspi/benchmarked_p90.yaml',
                  normalise=False, random_seed=261012)
    (output / 'external.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    (output / 'manifest.json').write_text(json.dumps(dict(rows=rows, archive_sha256=sha(archive),
                                                       catalogue_sha256=sha(ROOT / config['pyspi_config']),
                                                       standardization='per-channel full-record z score'), indent=2) + '\n')
    return config


def correlation_endpoint(name):
    return name.startswith(('corr_', 'xcorr')) or name in {
        'spearmanr', 'spearmanr-sq', 'kendalltau', 'kendalltau-sq',
        'cov_EmpiricalCovariance', 'cov-sq_EmpiricalCovariance'}


def analyse(root, mirror_revision=None):
    from mtsspi.corpus import ArchiveCorpus, load_inventory
    from mtsspi.provenance import array_sha256
    from mtsspi.run_corpus import completion_error
    from mtsspi.spi_spi_contract import build_unified_features
    config = ArchiveCorpus.from_file(root / 'external.yaml')
    identity = None
    if mirror_revision:
        # Validate historical outputs against their frozen source, not today's editable install.
        from dataclasses import replace
        import subprocess
        revision = subprocess.check_output(['git', 'rev-parse', mirror_revision], cwd=ROOT, text=True).strip()
        def frozen_hash(path):
            return hashlib.sha256(subprocess.check_output(['git', 'show', f'{revision}:{path}'], cwd=ROOT)).hexdigest()
        config = replace(config, archive=root / 'observations.npz', base_output_dir=root / 'mpis')
        identity = dict(corpus_config_sha256=sha(config.path),
                        pyspi_config_sha256=frozen_hash('configs/pyspi/benchmarked_p90.yaml'),
                        runner_sha256=frozen_hash('src/run_corpus.py'),
                        compute_sha256=frozen_hash('src/compute.py'),
                        pyspi_version={'dist': '3.0.0', 'computation': '3.0.0.r7'})
        assert sha(config.pyspi_config) == identity['pyspi_config_sha256']
    manifest = json.loads((root / 'manifest.json').read_text())
    assert sha(config.archive) == manifest['archive_sha256']
    entries = load_inventory(config)
    rows = manifest['rows']
    values, means, sources, order = [], [], [], None
    example_indices = {min(range(len(rows)), key=lambda j: abs(rows[j]['step'] - step)) for step in (0, 23, 24, 350, 700)}
    examples, example_steps = [], []
    for entry, row in zip(entries, rows, strict=True):
        error = completion_error(config, entry, identity=identity)
        if error:
            raise ValueError(f'{entry.name}: {error}')
        p = entry.output_dir(config)
        metadata = json.loads((p / 'meta.json').read_text())
        if mirror_revision:
            assert metadata['experiment']['git_commit'] == revision and not metadata['experiment']['git_dirty']
            assert metadata['spectral_memory_repair']['runner_sha256'] == frozen_hash('scripts/run_hardware_bounded_corpus.py')
        assert (entry.M, entry.T) == (28, 5000) and metadata.get('spectral_memory_repair')
        with np.load(config.archive) as bank:
            assert array_sha256(bank[entry.name]) == metadata['source']['member_sha256']
        names = [s['name'] for s in metadata['pyspi']['spis']]
        if order is None:
            order = names
        assert names == order and len(order) == 289 and row['name'] == entry.name
        with np.load(p / 'spi_mpis.npz') as bank:
            mpis = {s: bank[s] for s in names}
        features = build_unified_features(mpis, names)
        values.append(features.z)
        if entry.index - 1 in example_indices:
            examples.append(np.array([mpis[s] for s in names]))
            example_steps.append(row['step'])
        edge = ~np.eye(entry.M, dtype=bool)
        finite_fraction = float(np.mean([np.isfinite(mpis[s][edge]).mean() for s in names]))
        assert finite_fraction >= .9 and len(metadata['pyspi']['errors']) <= 15
        means.append([float(np.mean(mpis[s][edge])) if np.isfinite(mpis[s][edge]).all() else np.nan for s in names])
        sources.append(dict(name=entry.name, mpi_sha256=sha(p / 'spi_mpis.npz'),
                            seconds=metadata['job']['compute_seconds'], valid_features=int(np.isfinite(features.z).sum()),
                            failed_spis=list(metadata['pyspi']['errors']), finite_edge_fraction=finite_fraction))
    z = np.array(values)
    dev = np.array([r['development'] for r in rows], bool)
    # Strict development completeness prevents missingness from manufacturing a PC.
    keep = np.isfinite(z[dev]).all(axis=0)
    eligible = np.flatnonzero(keep)
    keep[eligible] = z[dev][:, eligible].std(axis=0) > 1e-8
    if keep.sum() < 2 or dev.sum() < 3:
        raise ValueError('Insufficient eligible features/development observations')
    center = z[dev][:, keep].mean(axis=0)
    selected = z[:, keep].astype(np.float64)
    missing = (~np.isfinite(selected)).mean(axis=1)
    if missing.max() > .05:
        raise ValueError(f'More than 5% frozen features missing in a recording: {missing.max():.3f}')
    transformed = np.where(np.isfinite(selected), selected, center) - center
    _, s, vt = np.linalg.svd(transformed[dev], full_matrices=False)
    for j in range(min(2, len(vt))):
        if vt[j, np.argmax(np.abs(vt[j]))] < 0:
            vt[j] *= -1
    scores = transformed @ vt[:2].T
    indices = np.flatnonzero(keep)
    specs = [features.schema[j] for j in indices]
    ranks = np.argsort(-np.abs(vt[0]), kind='stable')
    table = pd.DataFrame([dict(feature_index=int(indices[j]), spi_a=specs[j].spi_a, spi_b=specs[j].spi_b,
                               weight=float(vt[0, j]), both_correlation=correlation_endpoint(specs[j].spi_a) and correlation_endpoint(specs[j].spi_b)) for j in ranks])
    table.insert(0, 'rank', np.arange(1, len(table) + 1))
    out = root / 'analysis'; out.mkdir(exist_ok=True)
    table.to_csv(out / 'feature-rankings.csv', index=False)
    loading_cosines = []
    for leave in np.flatnonzero(dev):
        mask = dev.copy(); mask[leave] = False
        q = transformed[mask] - transformed[mask].mean(axis=0)
        _, _, v = np.linalg.svd(q, full_matrices=False)
        loading_cosines.append(float(abs(v[0] @ vt[0])))
    # Sensitivity refits the representation; a filtered display alone cannot test redundancy.
    noncorr = np.array([not (correlation_endpoint(s.spi_a) and correlation_endpoint(s.spi_b)) for s in specs])
    _, sf, vf = np.linalg.svd(transformed[dev][:, noncorr], full_matrices=False)
    filtered_scores = transformed[:, noncorr] @ vf[0]
    if np.dot(filtered_scores[dev], scores[dev, 0]) < 0:
        filtered_scores *= -1
    sensitivity = dict(excluded_features=int((~noncorr).sum()),
                       score_correlation=float(np.corrcoef(filtered_scores, scores[:, 0])[0, 1]),
                       explained_variance=float(sf[0]**2 / np.sum(sf**2)))
    contrast = None
    steps = [r['step'] for r in rows]
    if 23 in steps and 24 in steps:
        left, right = steps.index(23), steps.index(24)
        table['delta_23_24'] = z[right, table.feature_index] - z[left, table.feature_index]
        table['pc1_contribution_23_24'] = table.weight * table.delta_23_24
        table.sort_values('pc1_contribution_23_24', ascending=False).to_csv(out / 'contrast-contributions.csv', index=False)
        chosen = pd.concat([table.head(1), table.loc[~table.both_correlation].nlargest(2, 'pc1_contribution_23_24')]).drop_duplicates('feature_index')
        diagnostics = []
        for feature in chosen.itertuples():
            for step in (23, 24):
                matrix = examples[example_steps.index(step)]
                a, b = matrix[order.index(feature.spi_a)], matrix[order.index(feature.spi_b)]
                edge = ~np.eye(len(a), dtype=bool)
                deletion = []
                for channel in range(len(a)):
                    mask = edge.copy(); mask[channel, :] = False; mask[:, channel] = False
                    deletion.append(float(np.corrcoef(a[mask], b[mask])[0, 1]))
                from scipy.stats import spearmanr
                diagnostics.append(dict(rank=feature.rank, spi_a=feature.spi_a, spi_b=feature.spi_b,
                    weight=feature.weight, step=step, z=float(np.corrcoef(a[edge], b[edge])[0, 1]),
                    edge_spearman=float(spearmanr(a[edge], b[edge]).statistic),
                    channel_deletion_min=min(deletion), channel_deletion_max=max(deletion),
                    mean_a=float(a[edge].mean()), mean_b=float(b[edge].mean()),
                    sd_a=float(a[edge].std()), sd_b=float(b[edge].std())))
        pd.DataFrame(diagnostics).to_csv(out / 'contrast-edges.csv', index=False)
        delta = np.diff(scores[:, 0])
        contrast = dict(steps=[23, 24], pc1_delta=float(scores[right, 0]-scores[left, 0]),
                        largest_adjacent_absolute_jump=bool(np.argmax(abs(delta)) == left),
                        filtered_pc1_delta=float(filtered_scores[right]-filtered_scores[left]),
                        summed_feature_contributions=float(table.pc1_contribution_23_24.sum()),
                        diagnostic_note='Channel deletion removes existing MPI edges; it does not refit population-dependent SPIs. Ranges are sensitivity checks, not confidence intervals.')
    np.savez_compressed(out / 'features.npz', z=z, means=np.array(means), spi_order=np.array(order),
                        scores=scores, selected=keep, center=center, loadings=vt[:2], eigenvalues=s*s,
                        missing_fraction=missing, development=dev, filtered_scores=filtered_scores)
    np.savez_compressed(out / 'edge-examples.npz', matrices=np.array(examples), steps=np.array(example_steps), spi_order=np.array(order))
    report = dict(rows=rows, sources=sources, retained_features=int(keep.sum()),
                  explained_variance=(s*s / (s*s).sum()).tolist(), leave_setting_loading_cosines=loading_cosines,
                  max_imputed_fraction=float(missing.max()),
                  top20_double_correlation=int(table.head(20).both_correlation.sum()),
                  correlation_exclusion_sensitivity=sensitivity, contrast_23_24=contrast,
                  scope='One measured sweep; held settings test interpolation, not independent physical replication.',
                  pc_sign='Largest absolute loading positive; no coupling/physics orientation',
                  validation=dict(records=len(entries), source_revision=mirror_revision, execution_identity=identity,
                                  checks='Archive/member hashes, source revision, repair runner, catalogue, shape, infinities, seed, normalization and coverage'),
                  script_sha256=sha(__file__))
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('scope'); a.add_argument('--raw', type=Path, required=True); a.add_argument('--output', type=Path, required=True)
    a = sub.add_parser('build'); a.add_argument('--raw', type=Path, required=True); a.add_argument('--output', type=Path, required=True)
    a.add_argument('--T', type=int, choices=CANDIDATES, required=True); a.add_argument('--record', type=int, default=17451096)
    a = sub.add_parser('analyse'); a.add_argument('--root', type=Path, required=True)
    a.add_argument('--mirror-revision', help='Frozen main-repo revision for a local mirror of the original r7 pilot')
    a = p.parse_args()
    if a.command == 'scope':
        print(json.dumps(scope(a.raw, a.output), indent=2))
    elif a.command == 'build':
        print(json.dumps(build(a.raw, a.output, a.T, a.record), indent=2))
    else:
        print(json.dumps(analyse(a.root, a.mirror_revision), indent=2))
