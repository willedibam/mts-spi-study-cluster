"""Construction and mechanism diagnostics for the dated focused case viewers."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from tslearn.metrics import dtw_path

from mtsspi.figures import save, style
from mtsspi.generators import generate_filter_roll_mts, generate_lagged_warping_mts
from research.cases.refresh_case_figures import R, RHO, MI, ED, XD, DTW, COLORS, corr, rmi_edges


def replay(record):
    """Recover generating maps only after checking against the archived observation."""
    path = Path(record['path'])
    meta = json.loads((path / 'meta.json').read_text())
    gen = meta['generator']
    fn = {'filter_roll_mts': generate_filter_roll_mts,
          'lagged_warping_mts': generate_lagged_warping_mts}[gen['name']]
    generated, internals = fn(M=meta['M'], T=meta['T'], rng=np.random.default_rng(gen['seed']),
                             return_internals=True, **gen['resolved_params'])
    observed = np.load(path / 'timeseries.npy')
    np.testing.assert_allclose(generated, observed, rtol=0, atol=2e-14,
                               err_msg=f'Generator replay differs: {path}')
    return observed, meta, internals, float(np.max(abs(generated - observed)))


def rmi_example(frame, beta_tag='beta-5'):
    return frame[(frame.instance == 0) & (frame.beta_tag == beta_tag) &
                 (frame['class'] == 'iter3_LMNM')].iloc[0]


def dtw_examples(frame):
    return [frame[(frame.instance == 0) & (frame.max_lag == lag) &
                  np.isclose(frame.p_step, p)].iloc[0] for lag, p in [(0, 0), (5, 0), (5, .9)]]


def square(ax):
    ax.set_box_aspect(1)
    ax.spines[['top', 'right']].set_visible(True)


def plot_rmi_construction(frame, output, *, beta_tag='beta-5'):
    x, meta, internal, error = replay(rmi_example(frame, beta_tag))
    start, stop = 100, 250
    style(); fig, axes = plt.subplots(2, 1, figsize=(7, 4.3), sharex=True,
                                     height_ratios=[1, 2], layout='constrained')
    axes[0].plot(np.arange(start, stop), internal.mother[start:stop], color='.25', lw=1)
    axes[0].set(ylabel='Mother (SD units)', title='One mother → three channel families → independent channel noise')
    im = axes[1].imshow(x[start:stop].T, origin='lower', aspect='auto', cmap='RdBu_r',
                        vmin=-3, vmax=3, extent=(start-.5, stop-.5, .5, meta['M']+.5))
    p = meta['generator']['params']; nl, nm = p['n_linear'], p['n_monotonic']
    for boundary in [nl+.5, nl+nm+.5]: axes[1].axhline(boundary, color='black', lw=.8)
    axes[1].set(xlabel='Sample index', ylabel='Channel family', yticks=[(nl+1)/2, nl+(nm+1)/2, (nl+nm+1+meta['M'])/2],
                yticklabels=[f'L: 1–{nl}', f'M: {nl+1}–{nl+nm}', f'Q: {nl+nm+1}–{meta["M"]}'])
    fig.colorbar(im, ax=axes[1], label='Observed channel (SD units)', extend='both', shrink=.8)
    return save(fig, Path(output), 'mts-construction')


def plot_rmi_matrices(frame, output, *, beta_tag='beta-5'):
    record = rmi_example(frame, beta_tag); path = Path(record['path'])
    meta = json.loads((path / 'meta.json').read_text()); p = meta['generator']['params']
    style(); fig, axes = plt.subplots(1, 3, figsize=(9, 3.3), layout='constrained')
    with np.load(path / 'spi_mpis.npz') as bank:
        for ax, key, label in zip(axes, [R, RHO, MI], ['Pearson r', r'Spearman $\rho$', 'MI (nats)']):
            matrix = bank[key].copy(); np.fill_diagonal(matrix, np.nan)
            signed = key != MI
            im = ax.imshow(matrix, origin='lower', vmin=-1 if signed else 0,
                           vmax=1 if signed else np.nanmax(matrix), cmap='RdBu_r' if signed else 'viridis',
                           extent=(.5, meta['M']+.5, .5, meta['M']+.5))
            for b in [p['n_linear']+.5, p['n_linear']+p['n_monotonic']+.5]:
                ax.axhline(b, color='.3', lw=.6); ax.axvline(b, color='.3', lw=.6)
            for tag, (i, j) in zip('ABC', rmi_edges(meta)):
                ax.plot(j+1, i+1, 'o', ms=5, mfc='none', mec='black', mew=.8)
                ax.annotate(tag, (j+1, i+1), xytext=(3, 5), textcoords='offset points', fontsize=7,
                            bbox={'facecolor':'white','edgecolor':'none','alpha':.8,'pad':.5})
            ax.set(title=label, xlabel='Channel j', ylabel='Channel i'); square(ax)
            fig.colorbar(im, ax=ax, shrink=.65)
    return save(fig, Path(output), 'spi-matrices')


def correlation_decomposition(x, y, groups):
    """Exact empirical within/between-group decomposition, using population moments."""
    x, y, groups = np.asarray(x), np.asarray(y), np.asarray(groups)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y, groups = x[valid], y[valid], groups[valid]
    denominator = x.std() * y.std()
    if len(x) < 3 or denominator < 1e-12:
        return np.nan, np.nan, np.nan
    within = between = 0.
    for group in np.unique(groups):
        mask = groups == group; xx, yy = x[mask], y[mask]; weight = mask.mean()
        within += weight * np.mean((xx-xx.mean()) * (yy-yy.mean()))
        between += weight * (xx.mean()-x.mean()) * (yy.mean()-y.mean())
    total = corr(x, y)
    np.testing.assert_allclose((within+between)/denominator, total, atol=1e-12)
    return within/denominator, between/denominator, total


def analyze_rmi_composition(frame, output, *, beta_tag='beta-5'):
    rows = []
    for _, record in frame[(frame.beta_tag == beta_tag) | (frame['class'] == 'iter1_L')].iterrows():
        path = Path(record['path']); meta = json.loads((path/'meta.json').read_text())
        types = meta['generator']['types']; iu = np.triu_indices(meta['M'], 1)
        groups = ['|'.join(sorted((types[i], types[j]))) for i, j in zip(*iu)]
        with np.load(path/'spi_mpis.npz') as bank:
            within, between, total = correlation_decomposition(bank[R][iu], bank[RHO][iu], groups)
        rows.append({'class':record['class'], 'instance':record.instance,
                     'within':within, 'between':between, 'total':total})
    result = pd.DataFrame(rows); result.to_csv(Path(output)/'pair-type-decomposition.csv', index=False)
    return result


def plot_rmi_composition(result, output):
    style(); fig, ax = plt.subplots(figsize=(4, 3.8), layout='constrained')
    for k, (key, label, color) in enumerate(zip(['within', 'between', 'total'],
            ['Within pair types', 'Between pair types', 'Total z'], [COLORS[0], COLORS[1], '.25'])):
        for j, cls in enumerate(['iter1_L', 'iter2_LM', 'iter3_LMNM']):
            values = result[result['class'] == cls][key]
            ci = student_t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))
            ax.errorbar(j+(k-1)*.16, values.mean(), yerr=ci, color=color, fmt='o', ms=4,
                        capsize=2, label=label if j == 0 else None)
    ax.axhline(0, color='.8', lw=.7)
    ax.set(xticks=range(3), xticklabels=['Linear', '+ sigmoid', '+ quadratic'], ylabel=r'Contribution to $z_{r,\rho}$')
    square(ax); fig.legend(*ax.get_legend_handles_labels(), loc='outside upper center', ncols=3, fontsize=7)
    return save(fig, Path(output), 'pair-type-decomposition')


def plot_dtw_construction(frame, output, *, full_maps=False):
    style(); fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.5), layout='constrained')
    start, stop = 100, 160
    for col, (record, title) in enumerate(zip(dtw_examples(frame), ['No lag / no warp', 'Lag only', 'Lag + heavy warp'])):
        x, meta, internal, error = replay(record)
        ax = axes[0, col]
        im = ax.imshow(x[start:stop].T, origin='lower', aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3,
                       extent=(start-.5, stop-.5, .5, meta['M']+.5))
        ax.set(title=title, xlabel='Observed sample (detail)', ylabel='Channel', yticks=[1, 5, 10, 15, 20]); square(ax)
        ax = axes[1, col]; t = np.arange(meta['T']) if full_maps else np.arange(start, stop)
        limits = (0, meta['T']-1) if full_maps else (start-7, stop+2)
        ax.plot(limits, limits, color='.6', ls='--', lw=.8, label='Identity')
        for i, color in zip([0, meta['M']-1], COLORS[:2]):
            ax.step(t, internal.mappings[i][t]-internal.lags[i], where='mid',
                    color=color, lw=1.1, label=f'Channel {i+1}')
        ax.set(xlabel=r'$t_{\rm warp}$ (observed index)', ylabel=r'$t_{\rm orig}$ (mother index)',
               xlim=limits, ylim=(-int(internal.lags.max()), meta['T']-1) if full_maps else limits); square(ax)
        if full_maps:
            inset = ax.inset_axes([.12, .53, .40, .40])
            zoom_t = np.arange(start, stop)
            inset.plot([start-7, stop+2], [start-7, stop+2], '--', color='.6', lw=.6)
            for i, color in zip([0, meta['M']-1], COLORS[:2]):
                inset.step(zoom_t, internal.mappings[i][zoom_t]-internal.lags[i], where='mid', color=color, lw=.8)
            inset.set(xlim=(start-7, stop+2), ylim=(start-7, stop+2), xticks=[100, 160], yticks=[100, 160])
            inset.tick_params(labelsize=6, length=2); square(inset)
            ax.indicate_inset_zoom(inset, edgecolor='.6', alpha=.7)
    fig.colorbar(im, ax=axes[0].tolist(), label='Channel SD units', shrink=.65, extend='both')
    fig.legend(*axes[1, 0].get_legend_handles_labels(), loc='outside lower center', ncols=3, fontsize=8)
    return save(fig, Path(output), 'mts-and-generating-maps')


def shifted_path(length, shift):
    """Endpoint-matched path for the archived shifted-distance definition."""
    if abs(shift) >= length: raise ValueError('Shift must be smaller than record length')
    if shift < 0: return shifted_path(length, -shift)[:, ::-1]
    middle = [(i+shift, i) for i in range(length-shift)]
    return np.array([(i, 0) for i in range(shift)] + middle +
                    [(length-1, j) for j in range(length-shift, length)], dtype=int)


def path_distance(x, y, path):
    return float(np.sqrt(np.sum((x[path[:, 0]]-y[path[:, 1]])**2)/len(x)))


def inspect_dtw_pair(record):
    x, meta, internal, replay_error = replay(record)
    i, j = 0, meta['M']-1; a, b = x[:, i], x[:, j]
    shifts = np.arange(-10, 11)
    costs = np.array([path_distance(a, b, shifted_path(len(a), s)) for s in shifts])
    shift = int(shifts[np.argmin(costs)])
    optimal_path, cost = dtw_path(a, b); optimal_path = np.asarray(optimal_path)
    path = Path(record['path'])
    with np.load(path/'spi_mpis.npz') as bank:
        computed = [np.sqrt(np.mean((a-b)**2)), costs.min(), cost/np.sqrt(len(a))]
        np.testing.assert_allclose(computed, [bank[k][i, j] for k in [ED, XD, DTW]], atol=1e-10, rtol=1e-10)
    return x, optimal_path, shift, computed, replay_error


def plot_dtw_pair_inspections(frame, output):
    style(); fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.5), layout='constrained')
    rows = []; start, stop = 100, 160
    for col, (record, title) in enumerate(zip(dtw_examples(frame), ['No lag / no warp', 'Lag only', 'Lag + heavy warp'])):
        x, path, shift, distances, error = inspect_dtw_pair(record)
        rows.append({'class':record['class'], 'instance':0, 'channel_i':1, 'channel_j':x.shape[1],
                     'euclidean':distances[0], 'shifted':distances[1], 'dtw':distances[2],
                     'best_shift':shift, 'replay_max_abs_error':error})
        ax = axes[0, col]
        for i, color in zip([0, x.shape[1]-1], COLORS[:2]):
            ax.plot(np.arange(start, stop), x[start:stop, i], color=color, lw=.9, label=f'Channel {i+1}')
        ax.set(xlabel='Observed sample', ylabel='Channel SD units',
               title=f'{title}\nE={distances[0]:.3f}, shifted={distances[1]:.3f}, DTW={distances[2]:.3f}'); square(ax)
        ax = axes[1, col]
        ax.plot([start-10, stop+10], [start-10, stop+10], '--', color='.65', lw=.8, label='Identity')
        sp = shifted_path(len(x), shift)
        ax.plot(sp[:, 0], sp[:, 1], color=COLORS[0], lw=1.2, label=f'Best global shift')
        ax.plot(path[:, 0], path[:, 1], color='.15', lw=.8, label='Fitted DTW')
        ax.set(xlim=(start, stop), ylim=(start, stop), xlabel='Channel 1 sample',
               ylabel=f'Channel {x.shape[1]} sample', title=f'Optimal global shift s={shift}'); square(ax)
    pd.DataFrame(rows).to_csv(Path(output)/'inspected-pairs.csv', index=False)
    fig.legend(*axes[1, 0].get_legend_handles_labels(), loc='outside lower center', ncols=3, fontsize=8)
    return save(fig, Path(output), 'edge-pair-inspections')


def analyze_realized_warp(frame, output):
    rows = []
    for _, record in frame.iterrows():
        x, meta, internal, error = replay(record)
        delta = np.asarray(internal.mappings)-np.arange(meta['T'])
        rows.append({'class':record['class'], 'instance':record.instance, 'max_lag':record.max_lag,
                     'p_step':record.p_step, 'mean_abs_offset':np.mean(abs(delta)),
                     'mean_signed_offset':np.mean(delta),
                     'mean_centered_rms_offset':np.mean(np.std(delta, axis=1)),
                     'fraction_displaced':np.mean(delta != 0), 'dtw_xd':record.dtw_xd,
                     'replay_max_abs_error':error})
    result = pd.DataFrame(rows); result.to_csv(Path(output)/'realized-warp.csv', index=False)
    return result


def plot_realized_warp(result, output):
    style(); fig, axes = plt.subplots(1, 2, figsize=(7, 3.6), layout='constrained')
    lagged = result[result.max_lag == 5]
    grouped = lagged.groupby('p_step').mean(numeric_only=True)
    errors = lagged.groupby('p_step').mean_centered_rms_offset.sem()*student_t.ppf(.975, 49)
    axes[0].errorbar(grouped.index, grouped.mean_centered_rms_offset, yerr=errors, color='.25', marker='o', ms=3, capsize=2)
    axes[0].set(xlabel=r'Path decision probability $p_{\rm step}$', ylabel='Time-varying warp (RMS samples)')
    sc = axes[1].scatter(lagged.mean_centered_rms_offset, lagged.dtw_xd, c=lagged.p_step, cmap='viridis', vmin=0, vmax=.9,
                        s=8, alpha=.55, linewidths=0)
    axes[1].set(xlabel='Time-varying warp (RMS samples)', ylabel=r'$z_{\rm DTW,shifted}$')
    for ax in axes: square(ax)
    fig.colorbar(sc, ax=axes[1], label=r'$p_{\rm step}$', shrink=.65)
    return save(fig, Path(output), 'realized-warp')
