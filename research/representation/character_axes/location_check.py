"""Does the coupling location frequency shift beyond the channels' own spectra?

Stage 3 found that channel spectra move location_frequency 0.36-0.53 times as much as coupling frequency, so
it is reported beside the channel shift. Per window: location_frequency (core), the mean over channels of the
spectral centroid and of the peak bin in 0.04-0.2 cycles per sample (4-20 Hz at 100 Hz), from the bank's
32-bin log spectra. Per contrast: each one's state effect, and location_frequency's effect after adjusting for
both channel measures (as in sufficiency.adjust). Run on Gadi (jobs/location_check.pbs).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from mtsspi.character_axes import core_scores, edge_profiles
from research.representation.character_axes.evaluate_cohorts import CONTRASTS, effect
from research.representation.character_axes.sufficiency import adjust

BINS = 32
CENTRES = (np.arange(BINS) + .5) * .5 / BINS          # equal-width bins over (0, 0.5] cycles per sample
WINDOW = (CENTRES >= .04) & (CENTRES <= .2)


def channel_frequency(bank):
    M = int(bank['M'][0])
    log_power = np.asarray(bank['spectral'], dtype=np.float64).reshape(len(bank['y']), BINS, M)
    power = np.expm1(log_power)[:, WINDOW]
    centroid = (power * CENTRES[WINDOW, None]).sum(1) / power.sum(1)
    peak = CENTRES[WINDOW][np.argmax(power, axis=1)]
    return centroid.mean(1), peak.mean(1)


def main(cohort, root, output):
    bank = np.load(root / 'bank.npz')
    spi = [str(s) for s in bank['spi_order']]
    core, names = core_scores(edge_profiles(bank['aligned'], int(bank['M'][0]), len(spi)), spi)
    location = core[:, names.index('location_frequency')]
    centroid, peak = channel_frequency(bank)
    y_all, groups_all = bank['y'], bank['group'].astype(str)
    if cohort == 'neurotycho':
        groups_all = np.array([f'{g}-{r}' for g, r in zip(groups_all, bank['role'].astype(str))])
    ok = np.isfinite(location)
    report = dict(cohort=cohort, location_missing=float(np.mean(~ok)),
                  window_correlation=dict(centroid=float(np.corrcoef(location[ok], centroid[ok])[0, 1]),
                                          peak=float(np.corrcoef(location[ok], peak[ok])[0, 1])), contrasts={})
    for label, positive, negative in CONTRASTS[cohort]:
        ix = np.isin(y_all, [positive, negative])
        y, groups = (y_all[ix] == positive).astype(int), groups_all[ix]
        values = np.column_stack([location[ix], centroid[ix], peak[ix]])
        raw = effect(values, y, groups)
        adjusted = effect(adjust(location[ix, None], np.column_stack([centroid[ix], peak[ix]]), y), y, groups)[0]
        report['contrasts'][label] = dict(location=raw[0], channel_centroid=raw[1], channel_peak=raw[2],
                                          location_adjusted_for_channels=adjusted)
        f = lambda e: f"{e['standardized']:+.2f} (p {e['p']:.3f})" if e['standardized'] is not None else 'n/a'
        print(cohort, label, 'location', f(raw[0]), 'centroid', f(raw[1]), 'peak', f(raw[2]), 'adjusted', f(adjusted), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1, default=float) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort', choices=sorted(CONTRASTS))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    main(a.cohort, a.root, a.output)
