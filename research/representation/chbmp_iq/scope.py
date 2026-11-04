"""Blind cohort scoping of the CHBMP eyes-closed EEG: states, channels, artefacts, spectra, vigilance.

Reads the downloaded BIDS tree and the demographic table (age, sex, education); never WAIS.
Per recording, over the first eyes-closed block (first eyes_closed event to the next
eyes_open, split at discontinuity markers), at the native 200 Hz after a 0.5-45 Hz
zero-phase band-pass:

- channel quality for the 19 canonical channels and the 62-channel common montage:
  SD, maximum |x|, flat fraction, 25-45 Hz power share (muscle), highest absolute
  correlation with another channel (isolation and bridging);
- 1 s artefact blocks: the peak-to-peak range of any good canonical channel above
  100, 150 or 200 uV, and the longest clean contiguous runs;
- the median canonical-channel spectrum, individual alpha frequency and a 10 s
  occipital alpha/theta course as a vigilance proxy.

Writes <out>/records.csv, spectra.npz, vigilance.npz, channels.npz, summary.json and figures.
"""
from __future__ import annotations

import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import signal

from research.representation.character_axes.prepare import CANON
from research.representation.chbmp_iq.prepare import (bad_channels, canonical, demographics, family, first_eyes_closed,
                                                      read_edf, recordings, reference, sidecar, tsv)

THRESHOLDS = (100, 150, 200)
RUNS = (10, 20, 30, 40, 60, 120)
FMAX = 80


def eeg_names(edf):
    """EEG channel labels from channels.tsv (the EDF itself types every channel as EEG)."""
    rows = tsv(sidecar(edf, '_channels.tsv'))
    return [r['name'] for r in rows if r['type'].upper() == 'EEG'], {r['name']: r.get('status', '') for r in rows}


def longest_runs(good, lengths):
    """Number of disjoint clean runs of each length (in blocks) that fit in the good-block mask."""
    runs, k = [], 0
    for g in list(good) + [False]:
        if g:
            k += 1
        elif k:
            runs.append(k); k = 0
    return {L: int(sum(r // L for r in runs)) for L in lengths}, (max(runs) if runs else 0)


def scope_one(args):
    sid, edf = args
    raw = read_edf(edf, preload=True)
    fs = raw.info['sfreq']
    names, status = eeg_names(edf)
    labels = [c.split('-', 1)[0] for c in raw.ch_names]
    pick = [labels.index(n) for n in names if n in labels]
    eeg = [labels[k] for k in pick]
    events = tsv(sidecar(edf, '_events.tsv'))
    segs = first_eyes_closed(events, fs, raw.n_times)
    rec = dict(subject=sid, sfreq=fs, montage=family(len(names)), eeg_channels=len(names),
               reference=' '.join(sorted(reference(raw.ch_names))), seconds=raw.n_times / fs,
               ec_seconds=sum(b - a for a, b in segs) / fs, ec_segments=len(segs),
               status_bad=sum(1 for v in status.values() if v.lower() == 'bad'))
    if not segs:
        return rec, None
    sos = signal.butter(4, [.5, 45], btype='bandpass', fs=fs, output='sos')
    x = raw.get_data(picks=pick) * 1e6
    y = np.concatenate([signal.sosfiltfilt(sos, x[:, a:b], axis=1) for a, b in segs], axis=1)
    can = {canonical(n): k for k, n in enumerate(eeg) if canonical(n)}
    ci = [can[c] for c in CANON if c in can]
    rec['missing_canonical'] = ' '.join(c for c in CANON if c not in can)

    sd = y.std(1)
    f, p = signal.welch(y, fs=fs, nperseg=int(4 * fs), axis=1)
    hf = p[:, (f >= 25) & (f < 45)].sum(1) / p[:, (f >= 1) & (f < 45)].sum(1)
    r = np.corrcoef(y[:, ::2]); np.fill_diagonal(r, 0)
    bad = bad_channels(y, fs)
    bridged = np.abs(r).max(1) > .99
    rec.update(bad_channels=' '.join(eeg[k] for k in np.flatnonzero(bad)),
               bad_canonical=' '.join(eeg[k] for k in ci if bad[k]), n_bad=int(bad.sum()),
               n_bridged=int(bridged.sum()), hf_share_median=float(np.median(hf[ci])))

    block = int(fs)
    nb = y.shape[1] // block
    good_ci = [k for k in ci if not bad[k]]
    b = y[good_ci, :nb * block].reshape(len(good_ci), nb, block)
    ptp = (b.max(2) - b.min(2)).max(0)
    for t in THRESHOLDS:
        ok = ptp <= t
        counts, longest = longest_runs(ok, RUNS)
        rec[f'clean_s_{t}'] = int(ok.sum()); rec[f'longest_s_{t}'] = int(longest)
        rec.update({f'runs{L}s_{t}': v for L, v in counts.items()})

    spec = np.log10(np.median(p[ci], axis=0))
    occ = [can[c] for c in ('O1', 'O2', 'Pz') if c in can]
    po = p[occ].mean(0)
    k = (f >= 7) & (f <= 13)
    rec['iaf'] = float(f[k][po[k].argmax()])
    kk = (f >= 2) & (f <= 40) & ~((f >= 7) & (f <= 14))
    rec['aperiodic_slope'] = float(np.polyfit(np.log10(f[kk]), np.log10(po[kk]), 1)[0])
    step = int(10 * fs)
    ratio = []
    for s in range(0, y.shape[1] - step + 1, step):
        fw, pw = signal.welch(y[occ, s:s + step], fs=fs, nperseg=int(2 * fs), axis=1)
        pw = pw.mean(0)
        ratio.append(np.log10(pw[(fw >= 8) & (fw < 13)].mean() / pw[(fw >= 4) & (fw < 8)].mean()))
    rec['alpha_theta_first'] = float(np.mean(ratio[:6])); rec['alpha_theta_last'] = float(np.mean(ratio[-6:]))
    keep = f <= FMAX
    detail = dict(freq=f[keep], spectrum=spec[keep], vigilance=np.array(ratio), ptp=ptp.astype(np.float32),
                  channel_sd=dict(zip(eeg, sd)), channel_bad=dict(zip(eeg, bad)))
    return rec, detail


def main(raw_root, out, workers):
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(sid, edf) for sid, edf, _ in recordings(raw_root / 'bids')]
    with Pool(workers) as pool:
        results = pool.map(scope_one, tasks, chunksize=1)
    demo = demographics(raw_root)
    recs = [dict(r, **demo.get(r['subject'], {})) for r, _ in results]
    keys = list(dict.fromkeys(k for r in recs for k in r))
    with open(out / 'records.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(recs)
    ok = [(r['subject'], d) for r, d in results if d is not None]
    np.savez_compressed(out / 'spectra.npz', subject=np.array([s for s, _ in ok]), freq=ok[0][1]['freq'],
                        spectrum=np.stack([d['spectrum'] for _, d in ok]))
    n = max(len(d['vigilance']) for _, d in ok)
    vig = np.full((len(ok), n), np.nan)
    for i, (_, d) in enumerate(ok):
        vig[i, :len(d['vigilance'])] = d['vigilance']
    np.savez_compressed(out / 'vigilance.npz', subject=np.array([s for s, _ in ok]), alpha_theta=vig, step_s=10)
    names = sorted({c for _, d in ok for c in d['channel_sd']})
    np.savez_compressed(out / 'channels.npz', subject=np.array([s for s, _ in ok]), channel=np.array(names),
                        sd=np.array([[d['channel_sd'].get(c, np.nan) for c in names] for _, d in ok]),
                        bad=np.array([[bool(d['channel_bad'].get(c, False)) for c in names] for _, d in ok]))
    np.savez_compressed(out / 'ptp.npz', **{s: d['ptp'] for s, d in ok})
    summary = dict(recordings=len(recs), with_ec=len(ok),
                   montage={str(m): sum(r['montage'] == m for r in recs) for m in sorted({r['montage'] for r in recs})},
                   sfreq=sorted({r['sfreq'] for r in recs}))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    main(args.raw, args.out, args.workers)
