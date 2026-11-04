"""CHBMP figures. `scope`: cohort quality, spectra, vigilance and an example recording (blind; no WAIS)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mtsspi.figures import save, style
from research.representation.character_axes.prepare import CANON

MONTAGE = {62: '#4C72B0', 120: '#DD8452'}


def scope(out, raw_root):
    style()
    d = pd.read_csv(out / 'records.csv')
    figs = out / 'figures'
    fig, ax = plt.subplots(2, 3, figsize=(10.5, 6.2), constrained_layout=True)

    a = ax[0, 0]
    for m, c in MONTAGE.items():
        a.hist(d.loc[d.montage == m, 'clean_s_150'], bins=np.arange(0, 1000, 25), color=c, alpha=.75, label=f'{m} ch (n={sum(d.montage == m)})')
    a.set(xlabel='clean eyes-closed seconds (1 s blocks, p2p ≤ 150 µV)', ylabel='recordings', title='Clean data in the first eyes-closed block')
    a.legend()

    a = ax[0, 1]
    for L, c in zip((10, 20, 40, 60), plt.cm.viridis(np.linspace(.1, .85, 4))):
        k = np.arange(1, 21)
        a.plot(k, [(d[f'runs{L}s_150'] >= n).mean() for n in k], color=c, label=f'{L} s windows (T={L * 100})')
    a.axhline(1, color='.8', lw=.8)
    a.set(xlabel='disjoint clean windows required per recording', ylabel='fraction of recordings retained', ylim=(0, 1.02),
          title='Window budget: length × count')
    a.legend(loc='lower left')

    a = ax[0, 2]
    bad = pd.Series([c for v in d.bad_canonical.dropna() for c in str(v).split()]).value_counts().reindex(CANON, fill_value=0)
    a.bar(range(19), bad.values / len(d), color='.35')
    a.set_xticks(range(19), CANON, rotation=90)
    a.set(ylabel='fraction of recordings', title=f'Bad canonical channels ({(d.bad_canonical.notna()).mean():.0%} with any)')

    a = ax[1, 0]
    a.hist(d.iaf.dropna(), bins=np.arange(7, 13.5, .25), color='.35')
    a.set(xlabel='individual alpha frequency (Hz, O1/O2/Pz)', ylabel='recordings', title=f'IAF, median {d.iaf.median():.2f} Hz')

    a = ax[1, 1]
    with np.load(out / 'vigilance.npz') as v:
        r = v['alpha_theta']; t = np.arange(r.shape[1]) * int(v['step_s']) / 60
    ok = np.isfinite(r).sum(0) >= 10
    a.plot(t[ok], np.nanmedian(r, 0)[ok], color='k', label='median')
    a.fill_between(t[ok], *np.nanpercentile(r, [25, 75], 0)[:, ok], color='.8', label='IQR')
    a.set(xlabel='minutes into eyes-closed block', ylabel=r'$\log_{10}$ alpha / theta power (occipital)', title='Vigilance proxy over the block')
    a.legend()

    a = ax[1, 2]
    a.scatter(d.age, d.alpha_theta_last - d.alpha_theta_first, s=8, c=[MONTAGE.get(m, 'k') for m in d.montage])
    a.axhline(0, color='.7', lw=.8)
    a.set(xlabel='age (years)', ylabel='Δ log alpha/theta, last − first minute', title='Drowsiness drift by age')
    save(fig, figs, 'scope-overview'); plt.close(fig)

    with np.load(out / 'spectra.npz') as s:
        f, S, subj = s['freq'], s['spectrum'], s['subject']
    order = np.argsort(d.set_index('subject').loc[subj, 'iaf'].values)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.4, 1]))
    keep = (f >= .5) & (f <= 60)
    im = ax[0].imshow(S[order][:, keep], aspect='auto', origin='lower', cmap='magma',
                      extent=[f[keep][0], f[keep][-1], 0, len(order)], vmin=np.percentile(S[:, keep], 2), vmax=np.percentile(S[:, keep], 99.5))
    ax[0].set(xlabel='frequency (Hz)', ylabel='recording (sorted by IAF)', title='Median canonical-channel spectrum')
    fig.colorbar(im, ax=ax[0], label=r'$\log_{10}$ PSD (µV²/Hz)')
    a = ax[1]
    a.plot(f[keep], np.median(S[:, keep], 0), color='k')
    a.fill_between(f[keep], *np.percentile(S[:, keep], [10, 90], 0), color='.85')
    for x, lab in ((25, 'fs/4 at 100 Hz'), (45, 'band-pass edge'), (50, 'hardware low-pass')):
        a.axvline(x, color='.5', ls=':', lw=1); a.text(x + .5, a.get_ylim()[1] - .3, lab, rotation=90, va='top', fontsize=7, color='.4')
    a.set(xlabel='frequency (Hz)', ylabel=r'$\log_{10}$ PSD', title='Cohort median, 10–90%')
    save(fig, figs, 'scope-spectra'); plt.close(fig)

    if raw_root is not None:
        example(d, raw_root, figs)


def example(d, raw_root, figs):
    """Typical and worst-artefact recordings: 20 s of the 19 canonical channels with rejected seconds marked."""
    from research.representation.chbmp_iq.prepare import canonical, first_eyes_closed, read_edf, segment, tsv
    d = d.dropna(subset=['clean_s_150'])
    typical = d.iloc[(d.clean_s_150 - d.clean_s_150.median()).abs().argmin()].subject
    picks = [typical] + [s for s in d.sort_values('clean_s_150').subject if s != typical][:1]
    fig, ax = plt.subplots(len(picks), 1, figsize=(10, 3.2 * len(picks)), constrained_layout=True, squeeze=False)
    for a, sid in zip(ax[:, 0], picks):
        edf = next((raw_root / 'bids' / sid / 'eeg').glob('*_eeg.edf'))
        raw = read_edf(edf, preload=True); fs = raw.info['sfreq']
        labels = [c.split('-', 1)[0] for c in raw.ch_names]
        idx = {canonical(l): k for k, l in enumerate(labels) if canonical(l)}
        x = raw.get_data(picks=[idx[c] for c in CANON]) * 1e6
        segs = first_eyes_closed(tsv(edf.with_name(edf.name.replace('_eeg.edf', '_events.tsv'))), fs, raw.n_times)
        y, _ = segment(x, fs, *segs[0])
        nb = y.shape[1] // int(fs)
        ptp = np.ptp(y[:, :nb * int(fs)].reshape(19, nb, int(fs)), axis=2).max(0)
        span = 20
        start = max(0, int(np.argmax(ptp)) - span // 2) if sid != picks[0] else 120
        start = min(start, nb - span)
        w = y[:, start * int(fs):(start + span) * int(fs)]
        im = a.imshow(w, aspect='auto', cmap='RdBu_r', vmin=-80, vmax=80, extent=[start, start + span, 18.5, -.5], interpolation='none')
        for k in np.flatnonzero(ptp[start:start + span] > 150):
            a.axvspan(start + k, start + k + 1, ymin=0, ymax=.03, color='k')
        a.set_yticks(range(19), CANON, fontsize=6)
        a.set(xlabel='seconds into eyes-closed block', title=f'{sid}: {int(d.set_index("subject").loc[sid, "clean_s_150"])} clean s; black ticks = rejected seconds')
        fig.colorbar(im, ax=a, label='µV (0.5–45 Hz)', shrink=.8)
    save(fig, figs, 'scope-example'); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('scope'); p.add_argument('--scope', type=Path, required=True); p.add_argument('--raw', type=Path)
    args = parser.parse_args()
    if args.command == 'scope':
        scope(args.scope, args.raw)
