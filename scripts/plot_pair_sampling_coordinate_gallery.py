"""Semantic coordinate examples from cached MPIs; no new pyspi extraction."""
import hashlib
import argparse
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.plot_large_m_pair_sampling import COLORS, OUT, ROOT, X_LABEL, draw_paths, figure_style
from matplotlib.lines import Line2D
from scripts.scout_spi_pair_sampling import correlation_features, pair_vectors

PEARSON = 'cov_EmpiricalCovariance'  # Inputs have channel variance one (ddof=0).
PLV = 'plv_multitaper_mean_fs-1_fmin-0_fmax-0-5'
COH = 'cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-5'
GC = 'gc_gaussian_k-1_kt-1_l-1_lt-1'
TE = 'te_kraskov_NN-4_k-1_kt-1_l-1_lt-1'
# Chosen before inspecting these coordinates' errors, not ranked by recovery.
EXAMPLES = [
    ('Spearman × Kendall', 'spearmanr', 'kendalltau'),
    ('Pearson × Spearman', PEARSON, 'spearmanr'),
    ('Pearson × Kendall', PEARSON, 'kendalltau'),
    ('Pearson × lag-1 Pearson', PEARSON, 'corr_pearson_tau-1'),
    ('Pearson × distance correlation', PEARSON, 'dcorr'),
    ('Pearson × Gaussian MI', PEARSON, 'mi_gaussian'),
    ('Gaussian MI × KSG MI', 'mi_gaussian', 'mi_kraskov_NN-4'),
    ('Pearson × PLV', PEARSON, PLV),
    ('Coherence magnitude × PLV', COH, PLV),
    ('Pearson × DTW', PEARSON, 'dtw'),
    ('Granger causality × transfer entropy', GC, TE),
    ('Lag-1 Pearson × Granger causality', 'corr_pearson_tau-1', GC),
]


def sample_examples(matrices, budgets, seed, repeats=128):
    """Same seeded, nested dyads as the full-catalogue experiment."""
    pairs = pair_vectors(matrices)
    d = pairs.shape[1]
    permutations = [np.random.default_rng(np.random.SeedSequence([260917801, seed, r])).permutation(d)
                    for r in range(repeats)]
    gold = correlation_features(pairs, np.arange(d))
    values = np.stack([np.broadcast_to(gold, (repeats, len(gold))) if b == d else
                       np.stack([correlation_features(pairs, p[:b]) for p in permutations])
                       for b in budgets])
    return gold, values


def main(source=OUT):
    output = source / 'coordinate-gallery'
    output.mkdir(exist_ok=True)
    prior = json.loads((source / 'report.json').read_text())
    sizes = sorted({c['M'] for c in prior['cases'] if c['name']==f'meg-105923-run6-block01-M{c["M"]}'})
    selected = [next(c for c in prior['cases'] if c['name'] == f'meg-105923-run6-block01-M{m}')
                for m in sizes]
    names = list(dict.fromkeys(n for _, a, b in EXAMPLES for n in [a, b]))
    lookup = {frozenset((names[i], names[j])): k for k, (i, j) in
              enumerate(zip(*np.triu_indices(len(names), 1)))}
    indices = [lookup[frozenset((a, b))] for _, a, b in EXAMPLES]
    figure_style()
    rows = []
    for c in selected:
        m = c['M']
        with np.load(ROOT / c['path']) as bank:
            matrices = np.stack([bank[n] for n in names]).astype(float)
        with np.load(source / c['coordinate_file']) as old:
            budgets = old['budgets']
            full_lookup = {frozenset((str(old['spi_order'][i]), str(old['spi_order'][j]))): k
                           for k, (i, j) in enumerate(zip(*np.triu_indices(289, 1)))}
            reference = old['gold'][[full_lookup[frozenset((a, b))] for _, a, b in EXAMPLES]]
            old_spearman_kendall = old['example_estimates'][:, :, 0]
        gold, values = sample_examples(matrices, budgets, c['seed'])
        gold, values = gold[indices], values[:, :, indices]
        np.testing.assert_allclose(gold, reference, atol=2e-7, rtol=0, equal_nan=True)
        np.testing.assert_allclose(values[:, :, 0], old_spearman_kendall, atol=2e-7, rtol=0, equal_nan=True)
        np.savez_compressed(output / f'meg-M{m}.npz', budgets=budgets, gold=gold,
                            estimates=values, labels=np.array([x[0] for x in EXAMPLES]))
        fig, axes = plt.subplots(3, 4, figsize=(12.6, 7.8), sharex=True)
        for k, (ax, (label, a, b)) in enumerate(zip(axes.flat, EXAMPLES)):
            v = values[:, :, k]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                low, med, high = np.nanquantile(v, [.025, .5, .975], axis=1)
            draw_paths(ax, budgets, v)
            ax.fill_between(budgets, low, high, color=COLORS[m], alpha=.12, label=r'2.5--97.5\% of draws')
            ax.plot(budgets, med, 'o-', color=COLORS[m], ms=2.7, lw=1.7, label='Median of 128 draws')
            ax.axhline(gold[k], color='.15', ls='--', lw=1, label='Exact all-pairs value')
            valid = np.isfinite(v).mean(axis=1)
            extra = '' if valid.min() == 1 else rf'; min. valid {100*valid.min():.0f}\%'
            ax.set_title(label.replace(' × ', r' $\times$ ')+'\n'+rf'exact $z = {gold[k]:.4f}${extra}',fontsize=9)
            ax.set_xscale('log')
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            ax.grid(axis='y', alpha=.12)
            for n, budget in enumerate(budgets):
                rows.append(dict(M=m, label=label, spi_a=a, spi_b=b, dyads=int(budget),
                    exact=float(gold[k]), valid_draw_fraction=float(valid[n]),
                    median=float(med[n]), q025=float(low[n]), q975=float(high[n]),
                    rmse=float(np.sqrt(np.nanmean((v[n]-gold[k])**2)))))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        handles.append(Line2D([], [], color='.6', lw=.6)); labels.append('128 nested paths (faint)')
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, -.015), ncol=4, fontsize=8)
        fig.supxlabel(X_LABEL)
        fig.supylabel(r'Estimated SPI-SPI feature value $\hat{z}$')
        fig.suptitle(rf'select examples $\cdot$ data=MEG ($M={m}$, $T=1000$, 1 instance)'
                     r' $\cdot$ 128 draws $\cdot$ vertical scales differ', fontsize=11)
        for ext in ['png', 'svg']:
            fig.savefig(output / f'coordinate-gallery-M{m}.{ext}', dpi=180)
        plt.close(fig)
    pd.DataFrame(rows).to_csv(output / 'summary.csv', index=False)
    report = dict(examples=EXAMPLES, records=selected, repeats=128,
        figure_style=dict(latex=True, individual_paths=128, path_alpha=.05),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference='Same exact fixed-record MPI bank and dyad permutations as original full-vector experiment.',
        checks=f'All {len(selected)*len(EXAMPLES)} census coordinates and original Spearman/Kendall trajectories agree within 2e-7.',
        pearson='Empirical covariance equals zero-lag Pearson because prepared channels have variance one with ddof=0; pyspi normalise=False. Not a general covariance/Pearson equivalence.',
        caveats='Selected by statistic meaning, not recovery. One MEG block at nested sizes, not broad redundancy validation. Nonfinite draws excluded from bands and counted in summary.')
    (output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(pd.DataFrame(rows).query('dyads == 50')[['M','label','exact','rmse','valid_draw_fraction']].to_string(index=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=OUT)
    main(parser.parse_args().source)
