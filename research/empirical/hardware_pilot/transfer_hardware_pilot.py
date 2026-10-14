"""Frozen two-statistic transfer to the small-world hardware sweep; no pyspi."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.signal import butter, hilbert, sosfiltfilt

from research.empirical.hardware_pilot.calibrate_hardware_pilot import (
    BANDS, I, J, edge_corr, lag_covariance, lag_matrix, phase_properties,
)
from research.empirical.hardware_pilot.hardware_pilot import audit_record

RECORD = 17451530
INITIAL_STEPS = (0, 7, 14, 21, 28, 42, 63, 70, 87, 175, 350, 700)
TARGETS = (5000, 10000)
LAGS_SECONDS = (.002, .020)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def band_properties(x, fs, limits):
    """Same filter/phase rule as ER, preserving physical units at either resolution."""
    if fs == 500:
        return phase_properties(x, limits)
    trim = round(.5 * fs)
    signal = sosfiltfilt(butter(4, limits, fs=fs, btype="bandpass", output="sos"), x, axis=1)
    analytic = hilbert(signal, axis=1)[:, trim:-trim]
    unit = analytic / np.maximum(abs(analytic), 1e-12)
    cross = unit @ unit.conj().T / unit.shape[1]
    time = np.arange(unit.shape[1]) / fs
    time -= time.mean()
    frequency = np.unwrap(np.angle(unit), axis=1) @ time / (time @ time) / (2 * np.pi)
    return dict(signal=signal[:, trim:-trim], unit=unit, plv=abs(cross),
                frequency=frequency, offset=np.angle(cross[:, 0]),
                R1=float(abs(unit.mean(axis=0)).mean()),
                R2=float(abs((unit ** 2).mean(axis=0)).mean()))


def lag_profiles(x, fs):
    taus = [round(fs * seconds) for seconds in LAGS_SECONDS]
    if not np.allclose(np.array(taus) / fs, LAGS_SECONDS):
        raise ValueError("Sampling frequency cannot express both frozen physical lags")
    return np.array([lag_matrix(x, tau)[I, J] for tau in taus])


def profile_summary(x, fs):
    profiles = lag_profiles(x, fs)
    deletes = [edge_corr(*profiles[:, (I != k) & (J != k)]) for k in range(28)]
    half = x.shape[1] // 2
    halves = [edge_corr(*lag_profiles(v, fs)) for v in (x[:, :half], x[:, half:])]
    return dict(z=edge_corr(*profiles), mean_pearson=float(lag_matrix(x, 0)[I, J].mean()),
                sd_short=float(profiles[0].std()), sd_long=float(profiles[1].std()),
                half_1=halves[0], half_2=halves[1],
                delete_min=float(np.min(deletes)), delete_max=float(np.max(deletes))), profiles


def explain_record(x, fs):
    """Unfitted ER covariance/cosine decomposition, using the same fixed bands."""
    trim = round(.5 * fs)
    central = x[:, trim:-trim]
    properties = {f: band_properties(x, fs, limits) for f, limits in BANDS.items()}
    normalizer = np.outer(central.std(axis=1), central.std(axis=1))
    components = {f: lag_covariance(p['signal'], 0) / normalizer for f, p in properties.items()}
    predicted = np.array([
        sum(np.cos(2 * np.pi * np.median(properties[f]['frequency']) * lag) * matrix
            for f, matrix in components.items())[I, J]
        for lag in LAGS_SECONDS
    ])
    summary, actual = profile_summary(central, fs)
    summary.update(predicted_z=edge_corr(*predicted),
                   z_error=abs(edge_corr(*predicted) - summary['z']),
                   profile_r_short=edge_corr(actual[0], predicted[0]),
                   profile_r_long=edge_corr(actual[1], predicted[1]),
                   profile_rmse_short=float(np.sqrt(np.mean((actual[0] - predicted[0]) ** 2))),
                   profile_rmse_long=float(np.sqrt(np.mean((actual[1] - predicted[1]) ** 2))),
                   z_carrier_only=edge_corr(*lag_profiles(properties[116]['signal'], fs)),
                   z_full_record=edge_corr(*lag_profiles(x, fs)),
                   frequency_range=float(np.ptp(properties[116]['frequency'])),
                   mean_R116=properties[116]['R1'], mean_PLV116=float(properties[116]['plv'][I, J].mean()),
                   R1_58=properties[58]['R1'], R2_58=properties[58]['R2'],
                   group_positive_58=int((np.cos(properties[58]['offset']) > 0).sum()))
    summary['reconstruction_target_met'] = bool(summary['z_error'] <= .05
                                               and min(summary['profile_r_short'], summary['profile_r_long']) >= .95)
    harmonic_rows = []
    for f, p in properties.items():
        fraction = p['signal'].var(axis=1) / central.var(axis=1)
        harmonic_rows.append(dict(band=f, median_frequency=float(np.median(p['frequency'])),
                                  power_fraction_median=float(np.median(fraction)),
                                  profile_sd=float(components[f][I, J].std()),
                                  R1=p['R1'], R2=p['R2'], mean_PLV=float(p['plv'][I, J].mean())))
    return summary, actual, predicted, properties, harmonic_rows


def _prepare_one(args):
    raw, out, step = args
    path = Path(raw) / f'step-{step:03d}.npz'
    source = json.loads(path.with_suffix('.json').read_text())
    source_hash = sha(path)
    if source_hash != source['numeric_sha256'] or source['record'] != RECORD:
        raise ValueError(f"Unverified source {path}")
    destination = Path(out) / 'prepared' / f'step-{step:03d}.npz'
    receipt_path = destination.with_suffix('.json')
    identity = dict(source_sha256=source_hash, audit_sha256=sha(Path(audit_record.__code__.co_filename)),
                    transfer_sha256=sha(__file__))
    if destination.exists() and receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if all(old.get(k) == v for k, v in identity.items()) and sha(destination) == old['prepared_sha256']:
            return old
    with np.load(path) as bank:
        x, fs = bank['x'], float(bank['fs'])
        if x.shape != (28, 500000) or fs != 50000 or int(bank['step']) != step or int(bank['record']) != RECORD:
            raise ValueError(f"Unexpected recording identity/shape: {path}")
    audits, views, display = audit_record(x, fs, targets=TARGETS)
    arrays = {f'x_{target}': (y - y.mean(axis=1, keepdims=True)) / y.std(axis=1, keepdims=True)
              for target, y in views.items()}
    arrays.update(f=display['f'], psd=display['psd'], peaks=display['peaks'],
                  r_reference=display['r_reference'], r_T5000=display['r_T5000'], r_T10000=display['r_T10000'])
    np.savez_compressed(destination, **arrays)
    receipt = dict(step=step, record=RECORD, raw_T=500000, M=28, seconds=10,
                   audits=audits, prepared_sha256=sha(destination), **identity)
    write_json(receipt_path, receipt)
    print(f'Prepared {step}: ' + ', '.join(f"T={a["T"]} passes={a["passes"]}" for a in audits), flush=True)
    return receipt


def prepare(raw, out, steps, workers):
    """Acquire no features: only sampling checks and physical coordination diagnostics."""
    out = Path(out)
    (out / 'prepared').mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        receipts = list(pool.map(_prepare_one, [(raw, out, step) for step in sorted(steps)]))
    audits = pd.DataFrame([dict(step=r['step'], **a) for r in receipts for a in r['audits']])
    audits.to_csv(out / 'sampling-audit.csv', index=False)
    acceptable = [target for target in TARGETS if audits.loc[audits['T'] == target, 'passes'].all()]
    chosen = min(acceptable) if acceptable else None
    write_json(out / 'sampling-decision.json', dict(selected_T=chosen, steps=sorted(steps),
                                                   physical_lags_seconds=LAGS_SECONDS, receipts=receipts))
    if chosen is None:
        raise ValueError('Neither T=5000 nor T=10000 passes the frozen audit for the whole panel')
    physical = []
    for step in sorted(steps):
        with np.load(out / 'prepared' / f'step-{step:03d}.npz') as bank:
            x = bank[f'x_{chosen}']
        p = band_properties(x, chosen / 10, BANDS[116])
        physical.append(dict(step=step, voltage=step * .005,
                             frequency_range=float(np.ptp(p['frequency'])),
                             frequency_median=float(np.median(p['frequency'])),
                             R1=p['R1'], mean_PLV=float(p['plv'][I, J].mean())))
    pd.DataFrame(physical).to_csv(out / 'physical-screen.csv', index=False)
    print(json.dumps(dict(selected_T=chosen, settings=len(steps))), flush=True)


def select_refinements(out):
    """Choose at most four interior settings using physical dispersion alone."""
    out = Path(out)
    frame = pd.read_csv(out / 'physical-screen.csv').sort_values('step')
    if tuple(frame.step.astype(int)) != INITIAL_STEPS:
        raise ValueError('Refinement selection requires exactly the frozen initial panel')
    drops = np.log10(np.maximum(frame.frequency_range.to_numpy(), 1e-4))
    delta = drops[:-1] - drops[1:]
    index = int(np.argmax(delta))
    left, right = [int(v) for v in frame.step.iloc[index:index + 2]]
    extra = []
    if delta[index] >= .5:  # At least a threefold dispersion decrease.
        candidates = np.rint(np.linspace(left, right, 6)[1:-1]).astype(int)
        extra = sorted(set(candidates.tolist()) - set(INITIAL_STEPS))
        extra = [step for step in extra if left < step < right]
    selection = dict(record=RECORD, initial_steps=INITIAL_STEPS, additional_steps=extra,
                     all_steps=sorted(set(INITIAL_STEPS) | set(extra)),
                     rule='Up to four equally spaced interior settings in the largest log-frequency-range drop; minimum 0.5 log10, floor 1e-4 Hz.',
                     bracket=[left, right], log10_drop=float(delta[index]),
                     physical_screen_sha256=sha(out / 'physical-screen.csv'), features_not_computed=True)
    path = out / 'selection.json'
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(selection)):
        raise ValueError('Refusing to replace a different frozen selection')
    write_json(path, selection)
    # Preserve the physical-only selection evidence when final preparation adds records.
    (out / 'initial-physical-screen.csv').write_bytes((out / 'physical-screen.csv').read_bytes())
    print(json.dumps(selection), flush=True)


def matched_pairs(frame):
    rows = []
    for a, b in itertools.combinations(frame.to_dict('records'), 2):
        if abs(a['mean_pearson'] - b['mean_pearson']) > .01:
            continue
        delta = b['z'] - a['z']
        signs_match = all(np.isfinite(v) and np.sign(v) == np.sign(delta)
                          for v in (b['half_1'] - a['half_1'], b['half_2'] - a['half_2']))
        ranges_separate = b['delete_min'] > a['delete_max'] if delta > 0 else a['delete_min'] > b['delete_max']
        a_range = max(a['z'], a['half_1'], a['half_2']) - min(a['z'], a['half_1'], a['half_2'])
        b_range = max(b['z'], b['half_1'], b['half_2']) - min(b['z'], b['half_1'], b['half_2'])
        rows.append(dict(step_a=int(a['step']), step_b=int(b['step']),
                         mean_pearson_difference=b['mean_pearson'] - a['mean_pearson'],
                         delta_z=delta, same_direction_halves=signs_match,
                         deletion_ranges_separate=bool(ranges_separate),
                         larger_than_half_variation=bool(abs(delta) > max(a_range, b_range)),
                         working_contrast_target=bool(abs(delta) >= .1 and signs_match and ranges_separate
                                                      and abs(delta) > max(a_range, b_range)),
                         both_reconstructed=bool(a['reconstruction_target_met'] and b['reconstruction_target_met'])))
    return pd.DataFrame(rows, columns=['step_a', 'step_b', 'mean_pearson_difference', 'delta_z',
                                      'same_direction_halves', 'deletion_ranges_separate',
                                      'larger_than_half_variation', 'working_contrast_target', 'both_reconstructed'])


def analyse(out, revision):
    out = Path(out)
    selection = json.loads((out / 'selection.json').read_text())
    decision = json.loads((out / 'sampling-decision.json').read_text())
    if decision['steps'] != selection['all_steps'] or decision['selected_T'] not in TARGETS:
        raise ValueError('Analysis requires the full frozen panel and an accepted common resolution')
    target, rows, harmonics, resolutions, profiles = decision['selected_T'], [], [], [], {}
    for step in selection['all_steps']:
        path = out / 'prepared' / f'step-{step:03d}.npz'
        receipt = json.loads(path.with_suffix('.json').read_text())
        if sha(path) != receipt['prepared_sha256']:
            raise ValueError(f'Prepared-record hash mismatch: {step}')
        with np.load(path) as bank:
            x = bank[f'x_{target}']
            summary, actual, prediction, props, band_rows = explain_record(x, target / 10)
            by_resolution = {}
            for t in TARGETS:
                y = bank[f'x_{t}']; trim = round(t / 20)
                by_resolution[t] = lag_profiles(y[:, trim:-trim], t / 10)
            resolutions.append(dict(step=step,
                z_5000=edge_corr(*by_resolution[5000]), z_10000=edge_corr(*by_resolution[10000]),
                z_absolute_difference=abs(edge_corr(*by_resolution[5000]) - edge_corr(*by_resolution[10000])),
                profile_r_short=edge_corr(by_resolution[5000][0], by_resolution[10000][0]),
                profile_r_long=edge_corr(by_resolution[5000][1], by_resolution[10000][1])))
        rows.append(dict(step=step, voltage=step * .005, T=target, **summary))
        harmonics.extend(dict(step=step, **band) for band in band_rows)
        profiles[f'actual_{step}'] = actual
        profiles[f'predicted_{step}'] = prediction
        for frequency in (58, 116):
            unit = props[frequency]['unit']
            profiles[f'alignment_{frequency}_{step}'] = (unit @ unit.conj().T / unit.shape[1]).real
        profiles[f'group58_{step}'] = np.cos(props[58]['offset']) > 0
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'features.csv', index=False)
    pd.DataFrame(harmonics).to_csv(out / 'harmonics.csv', index=False)
    pd.DataFrame(resolutions).to_csv(out / 'resolution-sensitivity.csv', index=False)
    pairs = matched_pairs(frame)
    pairs.to_csv(out / 'matched-pairs.csv', index=False)
    np.savez_compressed(out / 'edge-profiles.npz', **profiles)
    result = dict(record=RECORD, steps=selection['all_steps'], M=28, seconds=10, T=target,
                  feature=['corr_pearson_2ms', 'corr_pearson_20ms'], bands_hz=BANDS,
                  reconstruction_targets_met=int(frame.reconstruction_target_met.sum()), records=len(frame),
                  maximum_z_error=float(frame.z_error.max()), median_z_error=float(frame.z_error.median()),
                  z_range=float(frame.z.max() - frame.z.min()),
                  eligible_matched_pairs=len(pairs), positive_matched_pairs=int(pairs.working_contrast_target.sum()),
                  reconstruction_and_contrast_pairs=int((pairs.working_contrast_target & pairs.both_reconstructed).sum()),
                  max_resolution_z_difference=float(pd.DataFrame(resolutions).z_absolute_difference.max()),
                  source_revision=revision, source_hashes={p.name: sha(p) for p in
                    (Path(__file__), Path(lag_matrix.__code__.co_filename), Path(audit_record.__code__.co_filename))},
                  selection_sha256=sha(out / 'selection.json'), sampling_decision_sha256=sha(out / 'sampling-decision.json'),
                  versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                  interpretation='Fixed-feature exploratory transfer between wirings on one apparatus; all selected records reported.')
    write_json(out / 'summary.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def plot_transfer(out):
    from research.empirical.hardware_pilot.calibrate_hardware_pilot import _plot_tools, _save
    plt, box = _plot_tools()
    out = Path(out)
    frame = pd.read_csv(out / 'features.csv')
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.3))
    axes[0].plot(frame.voltage, frame.z, 's-', color='#26828e', label='Measured')
    axes[0].plot(frame.voltage, frame.predicted_z, '.--', color='#d95f02', label='Frozen spectral explanation')
    axes[0].set(xscale='symlog', xlabel='Control voltage (V)', ylabel='Lag-profile agreement',
                title='A  Transfer without refitting', ylim=(-1.05, 1.05))
    axes[0].set_xscale('symlog', linthresh=.05)
    axes[0].legend(fontsize=6)
    points = axes[1].scatter(frame.z, frame.predicted_z, c=frame.voltage, cmap='viridis', s=15)
    axes[1].plot([-1, 1], [-1, 1], ':', color='.55', lw=.8)
    axes[1].set(xlabel='Measured agreement', ylabel='Predicted agreement',
                title='B  Every selected recording', xlim=(-1.05, 1.05), ylim=(-1.05, 1.05))
    fig.colorbar(points, ax=axes[1], shrink=.6, label='Control voltage (V)')
    axes[2].plot(frame.voltage, frame.frequency_range, 'o-', color='#222222')
    axes[2].set(xlabel='Control voltage (V)', ylabel='116 Hz frequency range (Hz)',
                title='C  Independent locking diagnostic', yscale='log')
    axes[2].set_xscale('symlog', linthresh=.05)
    for ax in axes:
        box(ax)
    _save(fig, out, 'transfer-summary')
    pairs = pd.read_csv(out / 'matched-pairs.csv')
    eligible = pairs.loc[pairs.working_contrast_target & pairs.both_reconstructed]
    if eligible.empty:
        return None
    selected = eligible.loc[eligible.delta_z.abs().idxmax()]
    steps = [int(selected.step_a), int(selected.step_b)]
    with np.load(out / 'edge-profiles.npz') as bank:
        actual = [bank[f'actual_{step}'] for step in steps]
        groups = [bank[f'group58_{step}'] for step in steps]
        alignment = [bank[f'alignment_58_{step}'] for step in steps]
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 6.1))
    limits = []
    for j in (0, 1):
        v = np.concatenate([p[j] for p in actual]); pad = .06 * np.ptp(v)
        limits.append((v.min() - pad, v.max() + pad))
    for row, step in enumerate(steps):
        values = frame.set_index('step').loc[step]
        im = axes[row, 0].imshow(alignment[row], origin='lower', vmin=-1, vmax=1,
                                 extent=(.5, 28.5, .5, 28.5), cmap='RdBu_r', interpolation='nearest', rasterized=True)
        axes[row, 0].set(title=f'Setting {step}: 58 Hz organisation', xlabel='Channel', ylabel='Channel',
                         xticks=[1, 14, 28], yticks=[1, 14, 28])
        same = groups[row][I] == groups[row][J]
        for mask, color, label in ((same, '#31688e', 'Same 58 Hz group'), (~same, '#d95f02', 'Opposite groups')):
            axes[row, 1].scatter(actual[row][0, mask], actual[row][1, mask], s=7, alpha=.7,
                                  color=color, linewidths=0, label=label, rasterized=True)
        axes[row, 1].set(title=f'mean r={values.mean_pearson:.3f}; z={values.z:.3f}',
                         xlabel='Pearson, 2 ms', ylabel='Pearson, 20 ms', xlim=limits[0], ylim=limits[1])
        axes[row, 1].legend(fontsize=6)
    for ax in axes.flat:
        box(ax)
    fig.colorbar(im, ax=axes[:, 0], shrink=.55, label='Phase alignment (cosine)')
    _save(fig, out, 'transfer-matched-contrast')
    return steps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'select', 'analyse', 'plot'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--raw', type=Path)
    parser.add_argument('--steps', type=int, nargs='+', default=INITIAL_STEPS)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--revision', default='unrecorded')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.raw, args.output, args.steps, args.workers)
    elif args.command == 'select':
        select_refinements(args.output)
    elif args.command == 'analyse':
        analyse(args.output, args.revision)
    else:
        print(plot_transfer(args.output))
