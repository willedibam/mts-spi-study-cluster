"""Prepare the CHBMP eyes-closed EEG corpus under the character-axes window contract.

Same contract as the AD/FTD, LEMON and propofol cohorts (character_axes/prepare.py):
19 canonical 10-20 channels in fixed order, the recording's own reference,
0.5-45 Hz zero-phase band-pass, 100 Hz, per-channel standardization, six 20 s
windows inside clean contiguous data, no interpolation.

IQ is deliberately not joined here. Every row has target 0 and carries only
age, sex and montage; WAIS stays in the downloaded table until the analysis
protocol is frozen (protocol.md).

--inspect summarises sampling rates, references, channel names, event labels
and their durations for every recording, without windowing or reading WAIS.
The event labels that mark eyes-closed state A and artefacts are fixed from
that output and passed explicitly; the default mode refuses to run without them.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from mtsspi.provenance import atomic_json
from research.representation.character_axes.prepare import ALIAS, CANON, FS, condition, cut, finalize, windows

LENGTH, COUNT = 2000, 6
LOWER = {c.lower(): c for c in CANON} | {k.lower(): v for k, v in ALIAS.items()}


def canonical(label):
    """'EEG FP1-REF' -> 'Fp1'; None when the label is not a canonical 10-20 position."""
    name = label.strip()
    for prefix in ('EEG ', 'EEG-'):
        if name.upper().startswith(prefix):
            name = name[len(prefix):]
    name = name.split('-')[0].strip()
    return LOWER.get(name.lower())


def channel_index(labels):
    found = {}
    for k, label in enumerate(labels):
        c = canonical(label)
        if c is not None and c not in found:
            found[c] = k
    missing = [c for c in CANON if c not in found]
    return ([] if missing else [found[c] for c in CANON]), missing


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


def intervals(events, column, labels, fs):
    """Sample intervals at fs for events whose `column` value is in labels."""
    out = []
    for e in events:
        if e.get(column) in labels:
            onset, duration = float(e['onset']), float(e.get('duration') or 0)
            out.append((int(round(onset * fs)), int(round((onset + duration) * fs))))
    return sorted(out)


def subtract(segments, cuts):
    out = []
    for a, b in segments:
        pieces = [(a, b)]
        for c, d in cuts:
            pieces = [p for x, y in pieces for p in ((x, min(y, c)), (max(x, d), y)) if p[1] > p[0]]
        out += pieces
    return out


def inspect(raw_root, output):
    bids = raw_root / 'bids'
    rows, labels, channels = [], Counter(), Counter()
    for sid, edf, eeg in recordings(bids):
        info = json.loads(sidecar(edf, '_eeg.json').read_text()) if sidecar(edf, '_eeg.json') else {}
        raw = read_edf(edf, preload=False)
        events = tsv(sidecar(edf, '_events.tsv')) if sidecar(edf, '_events.tsv') else []
        column = next((c for c in ('trial_type', 'value', 'description') if events and c in events[0]), None)
        per = Counter()
        for e in events:
            per[e.get(column)] += float(e.get('duration') or 0)
        labels.update({k: 1 for k in per})
        names = list(raw.ch_names)
        channels.update(names)
        _, missing = channel_index(names)
        rows.append(dict(subject=sid, sfreq=raw.info['sfreq'], seconds=raw.n_times / raw.info['sfreq'],
                         n_channels=len(names), missing_canonical=missing, reference=info.get('EEGReference'),
                         event_column=column, event_seconds=dict(per),
                         annotations=dict(Counter(raw.annotations.description))))
    summary = dict(recordings=len(rows), sfreq=dict(Counter(r['sfreq'] for r in rows)),
                   n_channels=dict(Counter(r['n_channels'] for r in rows)),
                   missing_any_canonical=sum(bool(r['missing_canonical']) for r in rows),
                   event_labels_by_recordings=dict(labels.most_common()), channel_names=dict(channels.most_common()))
    atomic_json(output, dict(summary=summary, recordings=rows))
    print(json.dumps(summary, indent=1)[:4000])


def prepare(raw_root, root, eyes_closed, artifacts, column):
    bids = raw_root / 'bids'
    people = {r['participant_id']: r for r in tsv(bids / 'participants.tsv')}
    rows, arrays, power, issues, sources = [], {}, {}, [], []
    for sid, edf, eeg in recordings(bids):
        raw = read_edf(edf, preload=True)
        index, missing = channel_index(raw.ch_names)
        sources.append(dict(path=str(edf.relative_to(raw_root)), sha256=hashlib.sha256(edf.read_bytes()).hexdigest()))
        if missing:
            issues.append(dict(subject=sid, missing=missing)); continue
        events = tsv(sidecar(edf, '_events.tsv'))
        state = intervals(events, column, eyes_closed, FS)
        clean = subtract(state, intervals(events, column, artifacts, FS))
        y = condition(raw.get_data()[index], raw.info['sfreq'])
        clean = [(a, min(b, len(y))) for a, b in clean if a < len(y)]
        starts = windows(clean, LENGTH, COUNT)
        p = people.get(sid, {})
        r, a, w = cut(y, starts, LENGTH, sid, sid, 0, 'cohort',
                      dict(age=p.get('age'), sex=p.get('sex'), montage=len(raw.ch_names)))
        rows += r; arrays.update(a); power.update(w)
        if len(r) < COUNT:
            issues.append(dict(subject=sid, windows=len(r), clean_seconds=sum(b - a for a, b in clean) / FS))
    finalize(root, 'chbmp-ec-r10', rows, arrays, power,
             dict(dataset='CHBMP Synapse syn28129311 (ds_bids_cbm_loris_24_11_21), state A eyes closed',
                  receipt_sha256=hashlib.sha256((raw_root / 'receipt.json').read_bytes()).hexdigest(),
                  event_column=column, eyes_closed_labels=sorted(eyes_closed), artifact_labels=sorted(artifacts),
                  sources=sources, target='0 for every row: IQ is joined only after the protocol is frozen'),
             issues)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True, help='download root holding bids/ and receipt.json')
    parser.add_argument('--root', type=Path, help='output corpus root (default mode)')
    parser.add_argument('--inspect', type=Path, help='write the inspection summary here and stop')
    parser.add_argument('--eyes-closed', nargs='+', help='event labels marking state A')
    parser.add_argument('--artifact', nargs='*', default=[], help='event labels marking artefacts')
    parser.add_argument('--event-column', default='trial_type')
    args = parser.parse_args()
    if args.inspect:
        inspect(args.raw, args.inspect)
    else:
        if not (args.root and args.eyes_closed):
            parser.error('--root and --eyes-closed are required; fix the labels from --inspect output first')
        args.root.mkdir(parents=True, exist_ok=True)
        if (args.root / 'manifest.json').exists():
            raise FileExistsError('prepared corpus already exists')
        prepare(args.raw, args.root, set(args.eyes_closed), set(args.artifact), args.event_column)
