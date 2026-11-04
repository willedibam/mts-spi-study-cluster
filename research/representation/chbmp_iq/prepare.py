"""Prepare CHBMP eyes-closed EEG windows for pyspi.

Contract (reasons in protocol.md §2):
- **State:** the first eyes-closed block, from the first eyes_closed event to the next eyes_open (about 10 min), split at discontinuity markers.
- **Channels:** the 19 canonical 10-20 channels or the 62-channel common montage, in fixed order, on the recording's own common reference. Channel types come from channels.tsv; the EDF itself types ECG/EOG as EEG.
- **Filtering:** 0.5-45 Hz zero-phase band-pass, then resampling to 100 Hz, each contiguous segment separately so that no filter runs across a discontinuity.
- **Artefacts:** a 1 s block is clean when every selected channel's peak-to-peak range, at native rate after band-pass, is at most 150 uV. Windows lie inside runs of clean blocks.
- **Exclusions:** a recording is excluded when a selected channel is bad by the scope rule (median 1 s block SD over 3x or under 1/5 of the recording's median channel, a flat fraction over 0.2, or a highest |r| with any other channel under 0.3). Excluded recordings are listed in issues; there is no interpolation.
- **Windows:** windows of length T are spread evenly over the clean runs, then standardised per channel. With --anchor, starts are placed for the anchor length and each window is that anchor window's first T samples, so a T-sweep compares nested windows from the same data.

IQ is not joined. Every row has target 0 and carries age, sex, education and montage from the demographic table; WAIS stays unread until the analysis is frozen.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
from pathlib import Path

import numpy as np
from scipy import signal

from research.representation.character_axes.prepare import ALIAS, CANON, FS, cut, finalize, windows

THRESHOLD_UV, MARGIN = 150, 200
LOWER = {c.lower(): c for c in CANON} | {k.lower(): v for k, v in ALIAS.items()}
# The 62-channel montage, in its channels.tsv order (sub-CBM00002); the 120-channel montage contains it.
COMMON62 = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz',
            'Cz', 'Pz', 'F1', 'F2', 'P1', 'P2', 'AF3', 'AF4', 'P5', 'P6', 'FC5', 'FC6', 'C5', 'C6', 'FT7', 'TP7',
            'TP8', 'PO5', 'PO6', 'AF7', 'AF8', 'FpZ', 'FCZ', 'CPZ', 'POZ', 'OZ', 'PO3', 'PO4', 'CP1', 'CP2', 'CP3',
            'CP4', 'C1', 'C2', 'F5', 'F6', 'FT8', 'FC3', 'FC4', 'FC1', 'FC2', 'CP5', 'CP6', 'PO7', 'PO8']


def canonical(label):
    """'EEG FP1-REF' -> 'Fp1'; None when the label is not a canonical 10-20 position."""
    name = label.strip()
    for prefix in ('EEG ', 'EEG-'):
        if name.upper().startswith(prefix):
            name = name[len(prefix):]
    return LOWER.get(name.split('-')[0].strip().lower())


def tsv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def recordings(bids):
    for eeg in sorted(bids.glob('sub-*/eeg')):
        edf = sorted(eeg.glob('*_eeg.edf'))
        if edf:
            yield eeg.parent.name, edf[0], eeg


def sidecar(edf, suffix):
    path = edf.with_name(edf.name.replace('_eeg.edf', suffix))
    return path if path.exists() else None


def read_edf(edf, preload):
    import mne
    return mne.io.read_raw_edf(edf, preload=preload, verbose='error')


def first_eyes_closed(events, fs, n):
    """Native-sample segments of the first eyes-closed block, split at discontinuities."""
    ev = sorted(((float(e['onset']), e['trial_type']) for e in events), key=lambda x: x[0])
    start = next((t for t, l in ev if l == 'eyes_closed'), None)
    if start is None:
        return []
    end = next((t for t, l in ev if l == 'eyes_open' and t > start), n / fs)
    edges = [start] + sorted(t for t, l in ev if l == 'discontinuity' and start < t < end) + [end]
    return [(int(round(a * fs)), min(n, int(round(b * fs)))) for a, b in zip(edges[:-1], edges[1:]) if b - a > 1]


def demographics(raw_root):
    rows = list(csv.reader(open(raw_root / 'cognitive' / 'Demographic_data.csv', encoding='utf-8-sig')))[1:]
    head = [h.strip() for h in rows[0]]
    out = {}
    for r in rows[1:]:
        d = dict(zip(head, r))
        if d.get('Code', '').strip():
            out['sub-' + d['Code'].strip()] = dict(age=float(d['Age']), sex=d['Gender'].strip(),
                                                   education=d['Education Level'].strip())
    return out


def bad_channels(y, fs):
    """Scope rule on (channel, sample) band-passed data: gross, flat or isolated channels.

    SD and correlation are medians over 1 s blocks, so transient artefacts are left to block rejection."""
    block = int(fs); nb = y.shape[1] // block
    b = y[:, :nb * block].reshape(len(y), nb, block)
    sd = np.median(b.std(2), axis=1); med = np.median(sd)
    flat = (np.abs(np.diff(y, axis=1)) < 1e-3).mean(1)
    c = b - b.mean(2, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        c = c / np.linalg.norm(c, axis=2, keepdims=True)
    r = np.nanmedian(np.einsum('itk,jtk->tij', c, c), axis=0); np.fill_diagonal(r, 0)
    return (sd > 3 * med) | (sd < med / 5) | (flat > .2) | (np.abs(r).max(1) < .3)


def clean_runs(y, fs):
    """Sample intervals (at FS) of runs of 1 s blocks whose peak-to-peak range is at most THRESHOLD_UV on all channels."""
    block = int(fs); nb = y.shape[1] // block
    b = y[:, :nb * block].reshape(len(y), nb, block)
    ok = (b.max(2) - b.min(2)).max(0) <= THRESHOLD_UV
    runs, start = [], None
    for k, g in enumerate(list(ok) + [False]):
        if g and start is None:
            start = k
        elif not g and start is not None:
            runs.append((start * FS, k * FS)); start = None
    return runs


def segment(x, fs, a, b):
    """Band-pass one native segment, return (native band-passed, resampled to FS as (sample, channel))."""
    sos = signal.butter(4, [.5, 45], btype='bandpass', fs=fs, output='sos')
    y = signal.sosfiltfilt(sos, x[:, a:b], axis=1)
    ratio = Fraction(FS / fs).limit_denominator(1000)
    z = signal.resample_poly(y, ratio.numerator, ratio.denominator, axis=1) if ratio != 1 else y
    return y, z.T


def prepare(raw_root, root, channels, length, count, anchor=None, subjects=None):
    bids, demo = raw_root / 'bids', demographics(raw_root)
    rows, arrays, power, issues, sources = [], {}, {}, [], []
    for sid, edf, eeg in recordings(bids):
        if subjects and sid not in subjects:
            continue
        types = {r['name']: r['type'].upper() for r in tsv(sidecar(edf, '_channels.tsv'))}
        raw = read_edf(edf, preload=True)
        labels = [c.replace('-REF', '') for c in raw.ch_names]
        if channels == 'canonical':
            have = {canonical(l): k for k, l in enumerate(labels) if types.get(l) == 'EEG' and canonical(l)}
            names = CANON
        else:
            have = {l: k for k, l in enumerate(labels) if types.get(l) == 'EEG'}
            names = COMMON62
        missing = [c for c in names if c not in have]
        if missing:
            issues.append(dict(subject=sid, missing=missing)); continue
        fs = raw.info['sfreq']
        x = raw.get_data(picks=[have[c] for c in names]) * 1e6
        segs = first_eyes_closed(tsv(sidecar(edf, '_events.tsv')), fs, raw.n_times)
        pieces = [segment(x, fs, a, b) for a, b in segs]
        if not pieces:
            issues.append(dict(subject=sid, reason='no eyes-closed block')); continue
        bad = bad_channels(np.concatenate([p[0] for p in pieces], axis=1), fs)
        if bad.any():
            issues.append(dict(subject=sid, bad=[names[k] for k in np.flatnonzero(bad)])); continue
        # Concatenate segments at FS with a NaN gap so that no window can straddle two segments.
        ys, runs, offset = [], [], 0
        for native, z in pieces:
            runs += [(offset + a, offset + min(b, len(z))) for a, b in clean_runs(native, fs)]
            ys += [z, np.full((MARGIN, len(names)), np.nan)]; offset += len(z) + MARGIN
        y = np.concatenate(ys)
        starts = windows(runs, anchor or length, count)
        d = demo.get(sid, {})
        r, a, w = cut(y, starts, length, sid, sid, 0, 'cohort',
                      dict(age=d.get('age'), sex=d.get('sex'), education=d.get('education'), montage=sum(v == 'EEG' for v in types.values())))
        rows += r; arrays.update(a); power.update(w)
        sources.append(dict(path=str(edf.relative_to(raw_root)), sha256=hashlib.sha256(edf.read_bytes()).hexdigest()))
        if len(r) < count:
            issues.append(dict(subject=sid, windows=len(r), clean_seconds=sum(b - a for a, b in runs) / FS))
    finalize(root, f'chbmp-ec-{channels}-T{length}', rows, arrays, power,
             dict(dataset='CHBMP Synapse syn28129311 (ds_bids_cbm_loris_24_11_21), first eyes-closed block',
                  receipt_sha256=hashlib.sha256((raw_root / 'receipt.json').read_bytes()).hexdigest(),
                  channel_set=channels, channel_names=names, length=length, anchor_length=anchor, count=count,
                  artefact_rule=f'1 s blocks, peak-to-peak <= {THRESHOLD_UV} uV on every selected channel',
                  sources=sources, target='0 for every row: IQ is joined only after the protocol is frozen'),
             issues)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True, help='download root holding bids/, cognitive/ and receipt.json')
    parser.add_argument('--root', type=Path, required=True, help='output corpus root')
    parser.add_argument('--channels', choices=['canonical', 'common62'], default='canonical')
    parser.add_argument('--length', type=int, required=True, help='window length T in samples at 100 Hz')
    parser.add_argument('--count', type=int, required=True, help='windows per recording')
    parser.add_argument('--anchor', type=int, help='place starts for this length and keep each window\'s first T samples')
    parser.add_argument('--subjects', nargs='*', help='restrict to these participant ids')
    args = parser.parse_args()
    if (args.root / 'manifest.json').exists():
        raise FileExistsError('prepared corpus already exists')
    prepare(args.raw, args.root, args.channels, args.length, args.count, args.anchor, set(args.subjects or []))
