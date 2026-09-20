"""Coordinate-level cached-MPI diagnostic; no pyspi edits or new extraction.

Two fixed recordings, selected by control value before inspecting coordinate
errors. Example coordinates include a named reference pair and explicitly
post-hoc median/90th-percentile difficulty examples at 50 sampled dyads.
"""
import argparse
import hashlib
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.scout_spi_pair_sampling import ROOT, correlation_features, pair_vectors

OUT = ROOT / 'results/spi_pair_sampling_260917'


def coordinate_statistics(estimates, gold):
    """Summaries are conditional on finite estimates; coverage is never hidden."""
    errors = np.asarray(estimates, dtype=float) - gold
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return dict(bias=np.nanmean(errors, axis=0),
                    rmse=np.sqrt(np.nanmean(errors**2, axis=0)),
                    sd=np.nanstd(errors, axis=0),
                    valid_fraction=np.isfinite(errors).mean(axis=0))


def run(repeats=128):
    OUT.mkdir(parents=True, exist_ok=True)
    earlier = json.loads((ROOT / 'results/spi_pair_sampling_260916/report.json').read_text())
    # CML near the published boundary and central TASEP control, first cached seed.
    selected = [earlier['cases'][14], earlier['cases'][22]]
    summaries, examples, provenance = [], [], []
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    error_fig, error_axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for row, source in enumerate(selected):
        path = ROOT / source['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256'], 'Cached source changed'
        meta = json.loads(path.with_name('meta.json').read_text())
        names = [s['name'] for s in meta['pyspi']['spis']]
        with np.load(path, allow_pickle=False) as archive:
            pairs = pair_vectors(np.stack([archive[n] for n in names]).astype(float))
        D = pairs.shape[1]
        gold = correlation_features(pairs, np.arange(D))
        valid = np.isfinite(gold)
        budgets = np.array(sorted({min(D, b) for b in [8, 16, 32, 50, 100, 200, 400, 800, 1600, D]}))
        permutations = [np.random.default_rng(260917000 + source['case'] * 10000 + d).permutation(D)
                        for d in range(repeats)]
        all_stats, estimates_by_budget = [], []
        for b in budgets:
            estimates = np.stack([correlation_features(pairs, p[:b]) for p in permutations])
            estimates_by_budget.append(estimates)
            stats = coordinate_statistics(estimates, gold)
            all_stats.append(stats)
            complete = valid & (stats['valid_fraction'] == 1)
            quantiles = np.quantile(stats['rmse'][complete], [.1, .5, .9, .95])
            summaries.append(dict(system=source['system'], M=source['M'], T=source['T'],
                dyads=int(b), ordered_entries=int(2*b), full_valid=int(valid.sum()),
                valid_in_every_draw=int(complete.sum()),
                rmse_p10=quantiles[0], rmse_median=quantiles[1], rmse_p90=quantiles[2], rmse_p95=quantiles[3],
                fraction_full_valid_with_complete_rmse_le_005=float(np.sum(complete & (stats['rmse'] <= .05))/valid.sum()),
                mean_sample_valid_fraction=float(stats['valid_fraction'][valid].mean())))
            print(source['system'], 'dyads', b, 'coordinate RMSE p10/50/90/95', quantiles, flush=True)
        estimates_by_budget = np.stack(estimates_by_budget)
        np.testing.assert_allclose(estimates_by_budget[-1], np.broadcast_to(gold, estimates_by_budget[-1].shape), atol=2e-7, equal_nan=True)
        stacked = {key: np.stack([s[key] for s in all_stats]) for key in all_stats[0]}
        ia, ib = np.triu_indices(len(names), 1)
        signed = np.flatnonzero((np.array(names)[ia] == 'spearmanr') & (np.array(names)[ib] == 'kendalltau'))
        assert len(signed) == 1
        index = int(np.flatnonzero(budgets == 50)[0])
        complete = valid & (stacked['valid_fraction'][index] == 1)
        candidates = np.flatnonzero(complete)
        ranked = candidates[np.argsort(stacked['rmse'][index, candidates], kind='stable')]
        coords = [int(signed[0]), int(ranked[round(.5*(len(ranked)-1))]), int(ranked[round(.9*(len(ranked)-1))])]
        labels = ['Prespecified rank-correlation pair', 'Post-hoc median difficulty', 'Post-hoc 90th-percentile difficulty']
        for col, (coordinate, label) in enumerate(zip(coords, labels)):
            values = estimates_by_budget[:, :, coordinate]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                lo, mid, hi = np.nanquantile(values, [.025, .5, .975], axis=1)
            ax = axes[row, col]
            ax.fill_between(2*budgets, lo, hi, color='#2879a8', alpha=.18)
            ax.plot(2*budgets, mid, 'o-', color='#2879a8', ms=3, label='Median sampled estimate')
            ax.axhline(gold[coordinate], color='black', ls='--', lw=1, label='Exact all-pairs value')
            ax.set_title(f'{label}\n{names[ia[coordinate]]}\n× {names[ib[coordinate]]}', fontsize=9)
            ax.set_ylabel(f"{source['system']}, M={source['M']}\nSPI–SPI correlation")
            ax.set_xlabel('Sampled ordered entries (2 per dyad)')
            ax.set_xscale('log')
            ax.spines[['top', 'right']].set_visible(False)
            examples.append(dict(system=source['system'], selection=label, feature=int(coordinate),
                spi_a=names[ia[coordinate]], spi_b=names[ib[coordinate]], gold=float(gold[coordinate]),
                rmse_at_100_entries=float(stacked['rmse'][index, coordinate])))
        axes[row, 0].legend(fontsize=7, frameon=False)
        sub = pd.DataFrame(summaries[-len(budgets):])
        ax = error_axes[row]
        ax.fill_between(2*budgets, sub.rmse_p10, sub.rmse_p90, alpha=.18, label='10–90% of coordinates')
        ax.plot(2*budgets, sub.rmse_median, 'o-', label='Median coordinate')
        ax.plot(2*budgets, sub.rmse_p95, ':', label='95th-percentile coordinate')
        ax.set(xscale='log', xlabel='Sampled ordered entries (2 per dyad)', ylabel='Coordinate RMSE over pair draws',
               title=f"{source['system']}, M={source['M']}, T={source['T']}")
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
        np.savez_compressed(OUT / f'case-{source["case"]}-coordinate-statistics.npz',
            budgets=budgets, gold=gold, full_valid=valid, spi_order=np.asarray(names),
            example_indices=coords, example_estimates=estimates_by_budget[:, :, coords], **stacked)
        provenance.append(dict(source, example_indices=coords))
    fig.suptitle(f'Fixed recordings, {repeats} common-dyad draws: medians and 2.5–97.5% draw bands\n'
                 'Different vertical scales; difficulty examples selected at 100 entries, not held-out validation', fontsize=11)
    error_fig.suptitle('Error differs across individual SPI–SPI coordinates\n'
                       'Distributions use coordinates finite in every draw at each budget; coverage saved separately', fontsize=10)
    for f, name in [(fig, 'individual-features'), (error_fig, 'coordinate-errors')]:
        f.savefig(OUT / f'{name}.png', dpi=180)
        f.savefig(OUT / f'{name}.svg')
        plt.close(f)
    pd.DataFrame(summaries).to_csv(OUT/'coordinate-summary.csv', index=False)
    report = dict(repeats=repeats, cases=provenance, examples=examples,
        sampling='Common uniform dyads without replacement, both orientations, nested prefixes per draw',
        reference='Exact finite-record all-pairs descriptor; not population truth',
        rng_rule='260917000 + original_case_index * 10000 + draw_index',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        caveats=['Two illustrative fixed records; not controlled size scaling or corpus-wide calibration.',
                 'Statistics conditional on finite estimates; per-coordinate coverage saved explicitly.',
                 'Post-hoc example selection is descriptive, not independent confirmation.',
                 'No sparse extraction, timing claim, pyspi edit, catalogue reduction or cluster job.'])
    (OUT/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=128)
    run(parser.parse_args().repeats)
