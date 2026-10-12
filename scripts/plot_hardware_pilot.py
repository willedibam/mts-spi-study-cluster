"""Compact figures for the hardware pilot; plots cached scientific results only."""
import json
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def style():
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['CMU Serif', 'DejaVu Serif'],
                         'mathtext.fontset': 'cm', 'font.size': 8, 'axes.labelsize': 8,
                         'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'legend.fontsize': 7, 'legend.frameon': False,
                         'lines.linewidth': 1.1, 'lines.markersize': 3,
                         'figure.dpi': 180, 'figure.constrained_layout.use': True})


def box(ax):
    ax.set_box_aspect(1)
    ax.tick_params(direction='out', width=.6, length=2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(.6)


def save(fig, output, name):
    output.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(output / f'{name}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def short(name):
    return name.replace('cov_EmpiricalCovariance', 'Covariance').replace('corr_', '').replace('coint_johansen_', 'Johansen ').replace('te_kraskov_', 'TE KSG ').replace('DCE-AUTO_', '').replace('constraint-', '').replace('_', ' ')


def sampling_figures(scope, trace_steps=None, focus_max_step=None):
    style()
    scope = Path(scope)
    frame = pd.read_csv(scope / 'sampling-audit.csv')
    out = scope / 'figures'
    colors = {2000: '#440154', 5000: '#26828e', 10000: '#8a9c19'}
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.8))
    checks = [('reconstruction_nrmse_p95', 'Waveform NRMSE (channel p95)', .05),
              ('plv_rmse', 'Phase-locking edge RMSE', .03),
              ('synchrony_trace_rmse', 'Synchrony-trace RMSE', .03)]
    for ax, (column, label, tolerance) in zip(axes, checks):
        for target, group in frame.groupby('T'):
            ax.plot(group.voltage, group[column], 'o-', color=colors[target], label=f'T={target}')
        ax.axhline(tolerance, color='.4', lw=.7, ls=':')
        ax.set(xlabel='Control voltage (V)', ylabel=label)
        box(ax)
    axes[0].legend()
    save(fig, out, 'sampling-fidelity')
    paths = sorted(scope.glob('display-*.npz'))
    if trace_steps is None:
        chosen = [paths[i] for i in sorted({0, len(paths)//2, len(paths)-1})]
    else:
        by_step = {int(p.stem.rsplit('-', 1)[-1]): p for p in paths}
        chosen = [by_step[step] for step in trace_steps]
    fig, axes = plt.subplots(1, len(chosen), figsize=(2.5 * len(chosen), 2.7), squeeze=False)
    for ax, path in zip(axes[0], chosen):
        with np.load(path) as d:
            normalized = d['psd'] / d['psd'].sum(axis=1, keepdims=True)
            ax.semilogy(d['f'][1:], np.median(normalized[:, 1:], axis=0), color='.15')
            for target, color in colors.items():
                ax.axvline(target / 20, color=color, ls=':', lw=.8)
            ax.set(xlim=(0, 1200), ylim=(1e-8, 1), xlabel='Frequency (Hz)', ylabel='Median normalized PSD',
                   title=f"Setting {int(path.stem.rsplit('-', 1)[-1])}")
            box(ax)
    save(fig, out, 'spectra')
    fig, axes = plt.subplots(len(chosen), 1, figsize=(7.3, 1.45 * len(chosen)), squeeze=False)
    for ax, path in zip(axes[:, 0], chosen):
        with np.load(path) as d:
            r = d['r_reference']
            ax.plot(np.arange(len(r)) * 10 / len(r), r, color='.15', label='Original')
            for target, color in colors.items():
                r = d[f'r_T{target}']
                ax.plot(np.arange(len(r)) * 10 / len(r), r, color=color, alpha=.7, lw=.7, label=f'T={target}')
            ax.axhline(.8, color='.6', ls=':', lw=.6)
            ax.set(ylim=(0, 1), ylabel=r'$R_s(t)$', title=f"Setting {int(path.stem.rsplit('-', 1)[-1])}")
    axes[-1, 0].set_xlabel('Time (s)'); axes[0, 0].legend(ncol=4, loc='upper right')
    save(fig, out, 'synchrony-traces')
    if focus_max_step is not None:
        physical = frame.drop_duplicates(['record', 'step']).sort_values('step')
        physical = physical.loc[physical.step <= focus_max_step]
        fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.8))
        for ax, column, label in zip(axes, ['mean_r', 'sd_r', 'high_fraction'],
                                    [r'Mean $R$', r'SD of smoothed $R$', r'Fraction of time: smoothed $R>0.8$']):
            ax.plot(physical.voltage, physical[column], 'o-', color='.15')
            ax.set(xlabel='Control voltage (V)', ylabel=label)
            ax.set_ylim(bottom=0)
            box(ax)
        save(fig, out, 'coordination-profile')
    return out


def result_figures(root, scope=None):
    style()
    root = Path(root); out = root / 'figures'
    report = json.loads((root / 'summary.json').read_text())
    rows = pd.DataFrame(report['rows'])
    table = pd.read_csv(root / 'feature-rankings.csv')
    with np.load(root / 'features.npz') as bank:
        z, scores, dev = bank['z'], bank['scores'], bank['development']
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8))
    axes[0].plot(scores[:, 0], scores[:, 1], color='.7', lw=.7)
    im = axes[0].scatter(scores[:, 0], scores[:, 1], c=rows.voltage, cmap='viridis', s=18)
    fig.colorbar(im, ax=axes[0], label='Control voltage (V)', shrink=.7)
    axes[0].set(xlabel='PC1', ylabel='PC2')
    axes[1].plot(rows.voltage, scores[:, 0], '.-', color='#26828e', label='SPI–SPI PC1')
    axes[1].set(xlabel='Control voltage (V)', ylabel='PC1 score')
    if scope is not None and (Path(scope) / 'sampling-audit.csv').exists():
        physical = pd.read_csv(Path(scope) / 'sampling-audit.csv').drop_duplicates(['record', 'step']).set_index(['record', 'step'])
        keys = [(int(r.record), int(r.step)) for r in rows.itertuples()]
        available = [i for i, key in enumerate(keys) if key in physical.index]
        if available:
            twin = axes[1].twinx()
            twin.plot(rows.iloc[available].voltage, [physical.loc[keys[i], 'mean_r'] for i in available], 'o-', color='.2', lw=.8)
            twin.set(ylabel=r'Mean $R$', ylim=(0, 1))
    evr = np.array(report['explained_variance'])[:8]
    axes[2].bar(np.arange(1, len(evr)+1), evr, color='.5')
    axes[2].set(xlabel='Principal component', ylabel='Explained variance fraction')
    for ax in axes: box(ax)
    save(fig, out, 'geometry')
    for filtered in (False, True):
        chosen = table.loc[~table.both_correlation] if filtered else table
        chosen = chosen.head(9)
        fig, axes = plt.subplots(3, 3, figsize=(6.5, 7.5))
        for ax, (_, row) in zip(axes.flat, chosen.iterrows()):
            values = z[:, int(row.feature_index)]
            ax.plot(rows.voltage, values, color='.6', lw=.7)
            ax.scatter(rows.voltage[dev], values[dev], s=12, facecolors='none', edgecolors='#26828e', label='Development')
            ax.scatter(rows.voltage[~dev], values[~dev], s=12, marker='s', color='#26828e', label='Held settings')
            label = f'{short(row.spi_a)} × {short(row.spi_b)}'
            ax.set(title=f'#{row["rank"]}  w={row.weight:+.3f}\n' + '\n'.join(textwrap.wrap(label, 35)),
                   xlabel='Control voltage (V)', ylabel=r'$z_{ab}$', ylim=(-1.05, 1.05))
            ax.title.set_fontsize(6.4); box(ax)
        for ax in list(axes.flat)[len(chosen):]: ax.set_visible(False)
        save(fig, out, 'features-filtered' if filtered else 'features-top')
    with np.load(root / 'edge-examples.npz') as bank:
        matrices, steps, order = bank['matrices'], bank['steps'], bank['spi_order'].tolist()
    chosen = table.loc[~table.both_correlation].head(3)
    fig, axes = plt.subplots(len(chosen), len(steps), figsize=(2.5 * len(steps), 2.6 * len(chosen)), squeeze=False)
    edge = ~np.eye(matrices.shape[-1], dtype=bool)
    for row_index, (_, feature) in enumerate(chosen.iterrows()):
        a, b = order.index(feature.spi_a), order.index(feature.spi_b)
        xs, ys = matrices[:, a][:, edge], matrices[:, b][:, edge]
        for k, (ax, x, y) in enumerate(zip(axes[row_index], xs, ys)):
            ax.scatter(x, y, s=3, alpha=.25, color='#26828e', linewidths=0, rasterized=True)
            valid = np.isfinite(x) & np.isfinite(y)
            corr = np.corrcoef(x, y)[0, 1] if valid.all() else np.nan
            ax.set(xlabel='\n'.join(textwrap.wrap(short(feature.spi_a), 32)),
                   ylabel='\n'.join(textwrap.wrap(short(feature.spi_b), 32)),
                   title=f'Setting {steps[k]} · z={corr:.2f}')
            for setter, all_values in ((ax.set_xlim, xs), (ax.set_ylim, ys)):
                lo, hi = np.nanmin(all_values), np.nanmax(all_values)
                pad = max((hi-lo)*.04, 1e-8)
                setter(lo-pad, hi+pad)
            box(ax)
    save(fig, out, 'edge-profiles')
    return out
