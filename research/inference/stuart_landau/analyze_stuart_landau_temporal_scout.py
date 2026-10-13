"""Retain temporal collective order; spectral summaries are not chaos tests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import periodogram


def summarize(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata_json']))
        z, observed = data['Z'], data['observed']
    r = np.abs(z)
    times = meta['burn'] + (np.arange(len(z)) + 1) * meta['sample_dt']
    rotation = np.exp(-1j * meta['carrier'] * times)
    z_rot = z * rotation
    sd = float(r.std())
    peak_frequency = entropy = peak_fraction = np.nan
    # Do not normalize a negligible residual into an apparent rich spectrum.
    if sd > 1e-8:
        frequency, power = periodogram(r, fs=1/meta['sample_dt'], window='hann', detrend='constant')
        frequency, power = frequency[1:], power[1:]
        p = power / power.sum()
        peak_frequency = float(frequency[np.argmax(p)])
        peak_fraction = float(p.max())
        positive = p[p > 0]
        entropy = float(-np.sum(positive * np.log(positive)) / np.log(len(p)))
    return {
        **{k: meta[k] for k in ('N', 'gamma', 'seed', 'burn', 'dt', 'sample_dt', 'samples', 'elapsed_seconds')},
        'path': str(path), 'R_mean': float(r.mean()), 'R_std': sd,
        'R_block_mean_range': float(np.ptp([x.mean() for x in np.array_split(r, 8)])),
        'R_half_mean_difference': float(abs(r[:len(r)//2].mean() - r[len(r)//2:].mean())),
        'R_peak_frequency': peak_frequency, 'R_peak_power_fraction': peak_fraction,
        'R_spectral_entropy': entropy,
        'Z_rot_fluctuation_rms': float(np.sqrt(np.mean(abs(z_rot - z_rot.mean())**2))),
        'lab_real_channel_sd_median': float(np.median(observed.real.std(axis=0))),
        'rotating_real_channel_sd_median': float(np.median((observed * rotation[:, None]).real.std(axis=0))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sensitivity-dir', type=Path)
    args = parser.parse_args()
    frame = pd.DataFrame([summarize(p) for p in sorted(args.input_dir.glob('*.npz'))])
    if frame.empty:
        raise ValueError('no archives')
    frame = frame.sort_values(['N', 'gamma', 'seed', 'dt', 'burn'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / 'temporal_order.csv', index=False)
    if args.sensitivity_dir:
        checks = []
        for path in sorted(args.sensitivity_dir.glob('*.npz')):
            row = summarize(path)
            reference = frame[(frame.N == row['N']) & np.isclose(frame.gamma, row['gamma'])
                              & (frame.seed == row['seed'])]
            if len(reference) != 1:
                raise ValueError(f'Need one matched baseline for {path}')
            base = reference.iloc[0]
            checks.append({**row, 'reference_path': base.path,
                           'delta_R_mean': row['R_mean'] - base.R_mean,
                           'delta_R_std': row['R_std'] - base.R_std})
        pd.DataFrame(checks).to_csv(args.output_dir / 'integration_sensitivity.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True)
    for n, group in frame.groupby('N'):
        means = group.groupby('gamma').mean(numeric_only=True)
        for ax, key, label in zip(axes, ['R_mean', 'R_std', 'R_spectral_entropy'],
                                  ['Mean |Z|', 'Temporal SD of |Z|', 'Spectral entropy of |Z|']):
            ax.plot(means.index, means[key], 'o-', ms=4, label=f'N={n}')
            ax.set(xlabel='Frequency half-width gamma', ylabel=label)
            ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend()
    axes[2].set_title('Descriptive; not a chaos diagnosis', fontsize=9)
    fig.savefig(args.output_dir / 'temporal_order.png', dpi=160)
    plt.close(fig)
    ns = sorted(frame.N.unique())
    controls = [.70, .80, 1., 1.2]
    fig, axes = plt.subplots(len(ns), len(controls), figsize=(12, 5), squeeze=False, constrained_layout=True)
    for i, n in enumerate(ns):
        for j, gamma in enumerate(controls):
            rows = frame[(frame.N == n) & np.isclose(frame.gamma, gamma)]
            for _, row in rows.iterrows():
                with np.load(row.path, allow_pickle=False) as data:
                    r = np.abs(data['Z'])[:2000]
                axes[i, j].plot(np.arange(len(r)) * row.sample_dt, r, lw=.7, alpha=.8,
                                label=f'seed {int(row.seed)}')
            axes[i, j].set(title=f'N={n}, gamma={gamma:g}', xlabel='Time after burn-in', ylim=(0, .75))
            axes[i, j].spines[['top', 'right']].set_visible(False)
        axes[i, 0].set_ylabel('|Z(t)|')
    axes[0, 0].legend(loc='lower left', fontsize=7)
    fig.savefig(args.output_dir / 'order_traces.png', dpi=160)
    plt.close(fig)
    print(frame[['N', 'gamma', 'seed', 'R_mean', 'R_std', 'R_block_mean_range',
                 'lab_real_channel_sd_median', 'rotating_real_channel_sd_median']].to_string(index=False))


if __name__ == '__main__':
    main()
