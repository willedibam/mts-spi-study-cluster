"""NeuroTycho reference values for the decoupling generator: class means (over units) of the generator's alignment
readouts, from the bank's pyspi edge profiles, for real data and the spectrum-final IAAFT surrogate. Run on Gadi."""
import argparse
import json
from pathlib import Path

import numpy as np

from mtsspi.character_axes import edge_profiles, profile_correlation
from research.representation.character_axes.direct_marginal import units

PAIRS = dict(align_robust_spearman=('prec-sq_MinCovDet', 'spearmanr-sq'), align_classical_spearman=('prec-sq_EmpiricalCovariance', 'spearmanr-sq'),
             align_robust_ids=('prec-sq_MinCovDet', 'ids'), align_classical_ids=('prec-sq_EmpiricalCovariance', 'ids'),
             robust_vs_classical=('prec-sq_MinCovDet', 'prec-sq_EmpiricalCovariance'))


def class_means(bank, g):
    spi = [str(s) for s in bank['spi_order']]
    P = edge_profiles(bank['aligned'], int(bank['M'][0]), len(spi))
    out = {}
    for name, (a, b) in PAIRS.items():
        v = profile_correlation(P[:, spi.index(a)], P[:, spi.index(b)])
        out[name] = {int(c): float(np.mean([np.nanmean(v[(g == u) & (bank['y'] == c)]) for u in np.unique(g[bank['y'] == c])])) for c in (0, 1)}
    return out


def main(root, output):
    bank = np.load(root / 'character-transfer-261031/neurotycho-primary/bank.npz')
    g = units(bank, 'neurotycho')
    report = dict(classes={0: 'awake', 1: 'anaesthesia'}, real=class_means(bank, g),
                  surrogate=class_means(np.load(root / 'character-axes-261102/surr-neurotycho-spectrum/bank.npz'), g))
    print(json.dumps(report, indent=1))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    main(a.root, a.output)
