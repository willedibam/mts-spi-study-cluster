"""Validate a frozen physics cohort and show all seeds/starts, not just a mean."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    config_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
    expected_files = {f'case-{i:03d}.npz' for i in range(len(config['cases']))}
    actual_files = {p.name for p in args.input_dir.glob('case-*.npz')}
    if actual_files != expected_files:
        raise ValueError('Archive set does not match the complete frozen cohort')
    rows, traces = [], {}
    for index, case in enumerate(config['cases']):
        path = args.input_dir / f'case-{index:03d}.npz'
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata_json']))
            assert meta['script_sha256'] == args.source_sha256
            assert meta['config_sha256'] == config_hash
            for key, value in {**case, **config['simulation']}.items():
                assert meta[key] == value, (path, key)
            for name in data.files:
                if name != 'metadata_json':
                    assert np.isfinite(data[name]).all(), (path, name)
            phi = data['phi']
            t = config['simulation']['samples']
            assert phi.shape == (t,)
            assert data['particle_velocity'].shape == (t, 2, 32, 2)
            assert data['bin_counts'].shape == (t, 2, 32)
            assert data['bin_current'].shape == (t, 2, 32, 2)
            assert np.all((phi >= 0) & (phi <= 1))
            np.testing.assert_allclose(phi.mean(), meta['phi_mean'], atol=1e-12, rtol=0)
        block_means = np.array([x.mean() for x in np.array_split(phi, 8)])
        row = {**case, 'phi_mean': phi.mean(), 'phi_std': phi.std(),
               'block_range': np.ptp(block_means),
               'half_mean_difference': abs(phi[:t//2].mean() - phi[t//2:].mean()),
               'binder': meta['binder']}
        rows.append(row)
        traces[case['eta'], case['seed'], case['start']] = phi
    frame = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / 'verified_replicates.csv', index=False)
    summary = frame.groupby('eta').agg(mean=('phi_mean', 'mean'),
        minimum=('phi_mean', 'min'), maximum=('phi_mean', 'max'),
        maximum_block_range=('block_range', 'max'),
        maximum_half_shift=('half_mean_difference', 'max'))
    summary.to_csv(args.output_dir / 'control_summary.csv')
    paired = frame.pivot(index=['eta', 'seed'], columns='start', values='phi_mean')
    paired['absolute_start_gap'] = abs(paired.ordered - paired.random)
    paired.to_csv(args.output_dir / 'start_sensitivity.csv')
    slopes = []
    for (seed, start), group in frame.groupby(['seed', 'start']):
        group = group.sort_values('eta')
        q = group.phi_mean.to_numpy()
        eta = group.eta.to_numpy()
        j = np.argmax(abs(np.diff(q) / np.diff(eta)))
        slopes.append(dict(seed=int(seed), start=start, monotone_decrease=bool(np.all(np.diff(q) < 0)),
                           steepest_interval=eta[j:j+2].tolist()))
    (args.output_dir / 'validation.json').write_text(json.dumps({
        'verified_archives': len(rows), 'source_sha256': args.source_sha256,
        'config_sha256': config_hash, 'replicate_slopes': slopes,
        'SPI_computed': False, 'thermodynamic_discontinuity_established': False,
        'range_is_confidence_interval': False}, indent=2) + '\n')

    seeds = sorted(frame.seed.unique())
    colors = dict(zip(seeds, ['#31688e', '#b35d2f']))
    fig, ax = plt.subplots(figsize=(6.5, 3.6), constrained_layout=True)
    for (seed, start), group in frame.groupby(['seed', 'start']):
        group = group.sort_values('eta')
        ax.plot(group.eta, group.phi_mean, 'o-' if start == 'ordered' else 's--',
                color=colors[seed], ms=4, label=f'{seed}, {start}')
    ax.set(xlabel='Angular noise eta', ylabel='Mean global polarization',
           title='Independent-seed physics refinement, N=32768')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8)
    fig.savefig(args.output_dir / 'replicate_curves.png', dpi=170)
    plt.close(fig)
    controls = sorted(frame.eta.unique())
    fig, axes = plt.subplots(2, len(controls), figsize=(12, 5), constrained_layout=True)
    for j, eta in enumerate(controls):
        for seed in seeds:
            for start in ('ordered', 'random'):
                phi = traces[eta, seed, start]
                style = '-' if start == 'ordered' else '--'
                axes[0, j].plot(np.arange(1, 9), [x.mean() for x in np.array_split(phi, 8)],
                                style, color=colors[seed], label=f'{seed}, {start}', lw=1)
                axes[1, j].hist(phi, bins=np.linspace(0, .45, 46), density=True,
                                histtype='step', linestyle=style, color=colors[seed], lw=1)
        axes[0, j].set(title=f'eta={eta:g}', xlabel='Consecutive 12,500-step block', ylim=(0, .4))
        axes[1, j].set(xlabel='Global polarization', xlim=(0, .45))
        for ax in axes[:, j]:
            ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].set_ylabel('Block-mean polarization')
    axes[1, 0].set_ylabel('Time-occupancy density')
    axes[0, -1].legend(fontsize=6, loc='upper right')
    fig.savefig(args.output_dir / 'blocks_and_distributions.png', dpi=170)
    plt.close(fig)
    print(summary.to_string())
    print(paired.to_string())
    print(json.dumps(slopes))


if __name__ == '__main__':
    main()
