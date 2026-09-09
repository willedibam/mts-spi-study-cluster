"""Descriptive physics/observation checks only; no feature fitting or Q selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def channel_diagnostics(x):
    sd = np.std(x, axis=0)
    valid = sd > 1e-8
    if valid.sum() >= 2:
        c = np.corrcoef(x[:, valid], rowvar=False)
        edges = c[np.triu_indices_from(c, k=1)]
        mean_abs, spread = float(np.mean(np.abs(edges))), float(np.std(edges))
    else:
        mean_abs, spread = np.nan, np.nan
    return dict(constant_fraction=float(np.mean(~valid)),
                channel_sd_median=float(np.median(sd)),
                mean_abs_correlation=mean_abs, dyad_correlation_sd=spread)


def inspect(path):
    observations = []
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata_json']))
        phi = data['phi']
        velocities = data['particle_velocity']
        current = data['bin_current']
        density = data['bin_counts'] / meta['bin_width']**2
        blocks = np.array_split(phi, 8)
        row = {key: meta[key] for key in ('N', 'L', 'eta', 'start', 'seed', 'burn',
                                         'samples', 'stride', 'elapsed_seconds',
                                         'phi_mean', 'phi_std', 'binder')}
        row.update(path=str(path), block_range=float(np.ptp([b.mean() for b in blocks])),
                   half_mean_difference=float(abs(phi[:len(phi)//2].mean() - phi[len(phi)//2:].mean())))
        for t in (100, 500, 1000, 2000):
            if t > len(phi):
                continue
            for m in (8, 16, 32):
                for view in range(2):
                    sample_phi = np.linalg.norm(velocities[:t, view, :m].mean(axis=1), axis=1)
                    for axis in range(2):
                        label = meta['particle_view_order'][view] + ('_x' if axis == 0 else '_y')
                        observations.append({**row, 'M': m, 'T': t, 'view': label,
                            'global_phi_window_mean': float(phi[:t].mean()),
                            'sample_phi_bias': float(np.mean(sample_phi - phi[:t])),
                            'sample_phi_rmse': float(np.sqrt(np.mean((sample_phi - phi[:t])**2))),
                            **channel_diagnostics(velocities[:t, view, :m, axis])})
                    for label, array in [('density', density[:, view]),
                                         ('current_x', current[:, view, :, 0]),
                                         ('current_y', current[:, view, :, 1])]:
                        observations.append({**row, 'M': m, 'T': t,
                            'view': meta['field_view_order'][view] + '_' + label,
                            'global_phi_window_mean': float(phi[:t].mean()),
                            'sample_phi_bias': np.nan, 'sample_phi_rmse': np.nan,
                            **channel_diagnostics(array[:t, :m])})
    return row, observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    paths = sorted({p for folder in args.inputs for p in folder.glob('case-*.npz')})
    if not paths:
        raise ValueError('no case archives found')
    rows, observations = [], []
    for path in paths:
        row, obs = inspect(path)
        rows.append(row)
        observations.extend(obs)
    frame = pd.DataFrame(rows).sort_values(['N', 'eta', 'start', 'seed'])
    obs = pd.DataFrame(observations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / 'physics.csv', index=False)
    obs.to_csv(args.output_dir / 'observation_checks.csv', index=False)
    summary = {
        'status': 'exploratory physics and raw-observation audit only',
        'cases': len(frame), 'N': sorted(frame.N.unique().tolist()),
        'seeds': sorted(frame.seed.unique().tolist()),
        'total_simulation_seconds': float(frame.elapsed_seconds.sum()),
        'max_block_range': float(frame.block_range.max()),
        'maximum_constant_channel_fraction': float(obs.constant_fraction.max()),
        'independent_confirmation': False, 'SPI_computed': False,
        'limitations': ['Short horizons cannot establish stationary first-order coexistence.',
                        'Matched starts/controls and nested views are not independent replicates.',
                        'Raw polarization and correlation diagnostics do not validate SPI-SPI.',
                        'Eulerian bins observe aggregates, not single particles.'],
    }
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(.15, .85, len(frame.N.unique())))
    for color, (n, group) in zip(colors, frame.groupby('N')):
        for start, curve in group.groupby('start'):
            means = curve.groupby('eta').phi_mean.mean()
            axes[0].plot(means.index, means, 'o-' if start == 'ordered' else 's--',
                         color=color, ms=4, label=f'N={n}, {start}')
    axes[0].set(xlabel='Angular noise eta', ylabel='Mean global polarization',
                title='Exploratory physics scout')
    axes[0].legend(fontsize=7)
    eta_colors = dict(zip(sorted(frame.eta.unique()), plt.cm.plasma(np.linspace(.1, .8, frame.eta.nunique()))))
    for _, row in frame[frame.N == frame.N.max()].iterrows():
        with np.load(row.path, allow_pickle=False) as data:
            block = np.array_split(data['phi'], 8)
            axes[1].plot(np.arange(1, 9), [b.mean() for b in block],
                         '-' if row.start == 'ordered' else '--', color=eta_colors[row.eta],
                         label=f'eta={row.eta:g}, {row.start}, seed={row.seed}')
    axes[1].set(xlabel='Consecutive recording block', ylabel='Mean global polarization',
                title=f'Time stability, N={frame.N.max()}')
    axes[1].legend(fontsize=6, loc='upper left', bbox_to_anchor=(1.02, 1))
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(args.output_dir / 'physics.png', dpi=160)
    plt.close(fig)
    # Do not aggregate different observation families or use Q to select a winner.
    show = obs[(obs['M'] == 32) & (obs['T'] == 1000)]
    table = show.groupby('view').agg(
        constant_fraction_max=('constant_fraction', 'max'),
        median_channel_sd=('channel_sd_median', 'median'),
        median_abs_correlation=('mean_abs_correlation', 'median'),
        median_polarization_bias=('sample_phi_bias', 'median'))
    table.to_csv(args.output_dir / 'observation_summary.csv')
    print(json.dumps(summary, indent=2))
    print(frame[['N', 'eta', 'start', 'seed', 'phi_mean', 'block_range', 'elapsed_seconds']].to_string(index=False))


if __name__ == '__main__':
    main()
