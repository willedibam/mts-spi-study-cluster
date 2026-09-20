"""Descriptive signed-Pearson redundancy screen, not held-source selection.

Uses the existing trusted local feature bank. Missingness is reported separately
from conditional redundancy; real/synthetic splits are broad strata, not source
groups or independent classes. No reduced catalogue is fitted or exported.
"""
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/spi_pair_sampling_260917/zenodo-screen'


def screen(values):
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return dict(valid_fraction=valid.mean(axis=0), mean=np.nanmean(values, axis=0),
                    p05=np.nanquantile(values, .05, axis=0),
                    fraction_valid_ge_095=np.divide((values >= .95).sum(axis=0), count,
                        out=np.full(values.shape[1], np.nan), where=count > 0))


def run():
    path = ROOT / 'data/zenodo_7118947/features/pearson-unified-v3-seed1729.npz'
    OUT.mkdir(parents=True, exist_ok=True)
    # Object metadata originates from this project's feature builder, not a new download.
    with np.load(path, allow_pickle=True) as bank:
        x = bank['X'].astype(float)
        a, b = bank['feature_spi_a'].astype(str), bank['feature_spi_b'].astype(str)
        labels = bank['labels'].tolist()
        names = bank['y'].astype(str)
    real = np.array(['real' in tags for tags in labels])
    synthetic = np.array(['synthetic' in tags for tags in labels])
    groups = dict(all=np.ones(len(x), dtype=bool), real=real, synthetic=synthetic)
    columns = dict(spi_a=a, spi_b=b)
    report = dict(status='descriptive candidate screen only; no catalogue selected',
        source=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        thresholds=dict(signed_similarity=.95, valid_record_fraction=.95, fraction_valid_above_similarity=.95),
        strata={})
    gates = {}
    for label, mask in groups.items():
        stats = screen(x[mask])
        columns.update({f'{label}_{key}': value for key, value in stats.items()})
        available = stats['valid_fraction'] >= .95
        mean_gate = available & (stats['mean'] >= .95)
        tail_gate = available & (stats['fraction_valid_ge_095'] >= .95)
        gates[label] = tail_gate
        report['strata'][label] = dict(records=int(mask.sum()), available_pairs=int(available.sum()),
            mean_gate_pairs=int(mean_gate.sum()), tail_gate_pairs=int(tail_gate.sum()),
            mean_pass_tail_fail=int((mean_gate & ~tail_gate).sum()),
            touched_spis=len(set(a[tail_gate]) | set(b[tail_gate])))
    frame = pd.DataFrame(columns)
    # Broad origin sensitivity is not an adequate replacement for source-blocked validation.
    report['pooled_tail_gate_failing_either_origin'] = int((gates['all'] & ~(gates['real'] & gates['synthetic'])).sum())
    report['all'] = dict(records=len(x), spis=289, coordinates=x.shape[1],
        distinct_tag_sets=len({tuple(sorted(tags)) for tags in labels}),
        unclassified_records=int((~(real | synthetic)).sum()))
    candidates = frame[gates['all']].sort_values(['all_p05', 'all_valid_fraction'], ascending=False)
    candidates.to_csv(OUT/'candidate-pairs.csv', index=False)
    np.savez_compressed(OUT/'all-pair-statistics.npz', **{k:np.asarray(v) for k,v in columns.items()})
    pd.DataFrame(dict(name=names, broad_origin=np.where(real, 'real', np.where(synthetic, 'synthetic', 'unclassified')),
                      tags=[json.dumps(tags) for tags in labels])).to_csv(OUT/'record-label-audit.csv', index=False)
    report['caveats'] = [
        'Record weighted and in-corpus; no grouped-development or held-source validation.',
        'Duplicate/nested recordings remain in the source bank; grouping must precede selection.',
        'Positive signed-Pearson screen differs from historical absolute-Spearman modules.',
        'Pairwise near-redundancy is not transitive and does not prove catalogue substitutability.',
        'No cost, module coverage, row-profile replacement, geometry or downstream gate applied yet.',
        'Thresholds are descriptive engineering choices, not universal scientific cutoffs.']
    (OUT/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run()
