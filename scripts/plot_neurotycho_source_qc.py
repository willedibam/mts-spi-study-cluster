"""Inspect fixed midpoint windows and spectra for the source-only ECoG scout."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def main(root):
    fig, axes = plt.subplots(4, 2, figsize=(12, 12))
    reports = []
    metadata = [json.loads(p.read_text()) for p in root.glob('201*.json')]
    if len(metadata) != 4:
        raise ValueError('expected four source scout archives')
    for row, meta in enumerate(sorted(metadata, key=lambda m: m['animal'])):
        bank = np.load(root / (meta['archive'] + '.npz'))
        for target, color, label in [(0, 'C0', 'Awake, eyes closed'), (1, 'C1', 'Anaesthetized')]:
            records = [r for r in meta['records'] if r['target'] == target and r['quality']['accepted']]
            selected = min(records, key=lambda r: abs(r['window'] - 8))
            x = bank['x'][selected['array_row']].astype(float)
            f, p = signal.welch(x, fs=250, nperseg=500, noverlap=250)
            for channel in range(3):
                wave = (x[channel] - x[channel].mean()) / x[channel].std()
                axes[row, 0].plot(np.arange(1000)/250, wave[:1000] + 7*channel + 3*target,
                                  color=color, linewidth=.5, alpha=.9,
                                  label=label if channel == 0 else None)
            axes[row, 1].semilogy(f, np.median(p, axis=0), color=color, label=label)
            reports.append(dict(animal=meta['animal'], state=label, accepted=len(records),
                rms_median=float(np.median([r['quality']['rms_median'] for r in records])),
                raw_line_fraction_median=float(np.median([r['quality']['raw_line_fraction_median'] for r in records])),
                largest_flat_run=max(r['quality']['longest_constant_raw_run'] for r in records),
                largest_crest_factor=max(r['quality']['crest_factor_max'] for r in records),
                plotted_window=selected['window']))
        axes[row, 0].set_title(meta['animal'] + ': first three bipolar channels')
        axes[row, 0].set(xlabel='Time within window (s)', ylabel='Within-window SD + display offset')
        axes[row, 1].set(xlim=(.5, 100), xlabel='Frequency (Hz)', ylabel='Median PSD (stored amplitude²/Hz)')
    axes[0, 0].legend(loc='upper right', fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle('Source-only QC: fixed midpoint windows, identical preprocessing\nAmplitude units remain those of the public files; no physical-voltage calibration assumed')
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(root / 'source-qc.png', dpi=150)
    (root / 'quality-summary.json').write_text(json.dumps(reports, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/neurotycho_source_pilot_260910'))
    main(parser.parse_args().root)
