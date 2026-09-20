"""Plot-only helpers for the separate baseline-comparison notebook."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.order_parameter_simple_baselines import OUT, FEATURES, LABELS

CORRELATIONS = [('mean_correlation', '#d95f02', '--', 'mean Pearson $r$'),
                ('mean_abs_correlation', '#7570b3', ':', 'mean $|r|$')]


def load_scores(path, *args, **kwargs):
    registry = json.loads((OUT / 'sources.json').read_text())
    bank = registry.get(str(Path(path).resolve()))
    return pd.read_csv(OUT / f'{bank}.csv' if bank else path, *args, **kwargs)


def install(namespace):
    bootstrap = namespace['bootstrap_curve']
    paper_axis = namespace['paper_axis']
    colors = namespace['COLORS']
    alpha = {100: .38, 500: .68, 1000: 1.0}

    def line(ax, frame, control, column, *, sign=1, color, label, ls='-', opacity=1, marker=None):
        curve = bootstrap(frame.assign(_value=sign * frame[column]), control, '_value')
        # Preserve missing control levels in failed patch analyses rather than bridge them.
        bank = frame['_baseline_bank'].iloc[0]
        full = pd.read_csv(OUT / f'{bank}.csv') if bank == 'cml_patch' else frame
        expected = np.sort(full[control].unique())
        curve = curve.set_index(control).reindex(expected).reset_index()
        handle, = ax.plot(curve[control], curve['mean'], color=color, label=label,
                          ls=ls, alpha=opacity, lw=1.5, marker=marker, ms=2.2)
        ax.fill_between(curve[control], curve.lower, curve.upper, color=color,
                        alpha=.09 * opacity, linewidth=0)
        return handle

    def dual_tracking(ax, frame, *, control, Q, q, q_sign=1, q_color='#2b6cb0',
                      boundary=None, title='', Q_label='physical $Q$'):
        right = ax.twinx()
        handles = [line(ax, frame, control, Q, color='#222222', label=Q_label, marker='o'),
                   line(right, frame, control, 'display_' + q, sign=q_sign,
                        color=q_color, label='frozen $q$', marker='s')]
        for col, color, ls, label in CORRELATIONS:
            handles.append(line(right, frame, control, 'display_' + col, color=color, label=label, ls=ls))
        if boundary is not None:
            ax.axvline(boundary, color='.5', lw=.9, ls=':')
        ax.set(xlabel=namespace['CONTROL_LABELS'].get(control, control), ylabel='physical $Q$', title=title)
        right.set_ylabel('standardized scores (display only)')
        paper_axis(ax)
        paper_axis(right, right=True)
        ax.legend(handles, [h.get_label() for h in handles], frameon=False, fontsize=6.5, ncol=2)
        return right

    def baseline_mt(frame, *, control, Q, q='q', q_sign=1, boundary=None, title='', reference=None):
        group_cols = ['view', 'M'] if 'view' in frame else ['M']
        groups = list(frame.groupby(group_cols))
        fig, axes = plt.subplots(1, len(groups), figsize=(4.8 * len(groups), 3.9), squeeze=False, constrained_layout=True)
        for ax, (key, group) in zip(axes.flat, groups):
            M = int(group.M.iloc[0])
            right = ax.twinx()
            ref = reference if reference is not None else group.query('T == T.max()')
            handles = [line(ax, ref, control, Q, color='#222222', label='physical $Q$', marker='o')]
            for T, part in group.groupby('T'):
                handles.append(line(right, part, control, 'display_' + q, sign=q_sign,
                    color=colors.get(M, '#35b779'), label=f'$q$, T={T}', marker='s', opacity=alpha[T]))
                for col, color, ls, label in CORRELATIONS:
                    handles.append(line(right, part, control, 'display_' + col, color=color,
                                        label=f'{label}, T={T}', ls=ls, opacity=alpha[T]))
            if boundary is not None:
                ax.axvline(boundary, color='.5', ls=':', lw=.9)
            ax.set(xlabel=namespace['CONTROL_LABELS'].get(control, control), ylabel='physical $Q$',
                   title=f'{group["view"].iloc[0] + ", " if "view" in group else ""}M={M}')
            right.set_ylabel('standardized scores (display only)')
            ax.legend(handles, [h.get_label() for h in handles], fontsize=6, ncol=1,
                      loc='upper right', frameon=False)
            paper_axis(ax)
            paper_axis(right, right=True)
        fig.suptitle(title)
        return fig

    namespace.update(dual_tracking=dual_tracking, baseline_mt=baseline_mt)


def summary_figure():
    metrics = pd.read_csv(OUT / 'metrics.csv')
    order = list(dict.fromkeys(metrics.system))
    columns = ['q', *FEATURES]
    table = metrics.pivot(index='system', columns='feature', values='abs_rho').loc[order, columns]
    fig, ax = plt.subplots(figsize=(10, 5.3), constrained_layout=True)
    im = ax.imshow(table, vmin=0, vmax=1, cmap='viridis', aspect='auto')
    for i in range(len(table)):
        for j in range(len(columns)):
            ax.text(j, i, f'{table.iloc[i,j]:.3f}', ha='center', va='center',
                    color='white' if table.iloc[i,j] < .65 else 'black', fontsize=8)
    ax.set_xticks(range(len(columns)), ['SPI--SPI q'] + [LABELS[c].replace(' ', '\n', 1) for c in FEATURES])
    ax.set_yticks(range(len(table)), table.index)
    ax.set_title('Same-record comparison: absolute Spearman association with physical Q')
    fig.colorbar(im, ax=ax, label=r'$|\rho|$')
    return fig
