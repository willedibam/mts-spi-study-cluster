"""Run-specific HCP label and ICA checks, including QC-removed trials."""
import re

import numpy as np
import pandas as pd


def eprime_events(frame):
    """Recover global task/rest block IDs despite the within-run counter reset."""
    onset = pd.to_numeric(frame['Stim.OnsetTime'], errors='coerce')
    events = frame[onset.notna() & (onset > 0)].copy()
    events['onset_ms'] = onset.loc[events.index]
    events = events.sort_values('onset_ms')
    numbers = pd.to_numeric(events['BlockNumber'], errors='raise').astype(int)
    group = numbers.ne(numbers.shift()).cumsum()
    starts = [(float(rows['onset_ms'].iloc[0]), int(key)) for key, rows in events.groupby(group)]
    rests = pd.to_numeric(frame['Fix15sec.OnsetTime'], errors='coerce').dropna().unique()
    # Initial/final resting periods are outside the numbered task interval.
    starts += [(float(value), None) for value in rests if events['onset_ms'].min() < value < events['onset_ms'].max()]
    starts.sort()
    mapping = {key: number for number, (_, key) in enumerate(starts, 1) if key is not None}
    events['block_id'] = group.map(mapping).astype(int)
    events['trial_id'] = events.groupby('block_id').cumcount()+1
    return events


def align_eprime(task, frame, sfreq):
    """Match retained TIM rows by block and within-block trial, not row position.

    TIM columns: block=1, image=3, memory=4, first sample=6, trial=8.
    This permits QC omissions without shifting subsequent event associations.
    """
    events = eprime_events(frame)
    lookup = {(int(row.block_id), int(row.trial_id)): row for row in events.itertuples()}
    if len(lookup) != len(events):
        raise ValueError('Duplicate E-Prime event identifiers')
    matched_onsets = []
    for row in task:
        key = (int(row[1]), int(row[8]))
        if key not in lookup:
            raise ValueError(f'Retained TIM event missing from E-Prime: {key}')
        event = lookup[key]
        if event.BlockType != {1: '0-Back', 2: '2-Back'}[int(row[4])]:
            raise ValueError(f'Memory label mismatch for {key}')
        if event.StimType != {1: 'Face', 2: 'Tools'}[int(row[3])]:
            raise ValueError(f'Stimulus label mismatch for {key}')
        matched_onsets.append(float(event.onset_ms)/1000)
    matched_onsets = np.asarray(matched_onsets)
    samples = (np.asarray(task)[:, 6]-1)/sfreq
    if len(task) < 3 or len(set(zip(task[:, 1], task[:, 8]))) != len(task):
        raise ValueError('Need at least three unique retained events')
    slope, offset = np.polyfit(matched_onsets, samples, 1)
    return dict(retained_events=len(task), eprime_events=len(events),
                clock_slope=float(slope), offset_seconds=float(offset),
                max_affine_timing_residual_seconds=float(np.max(np.abs(samples-(slope*matched_onsets+offset)))))


def ica_projection(comp, annotation):
    """Remove only this run's explicitly labelled cardiac/ocular components."""
    labels = np.atleast_1d(comp['topolabel']).tolist()
    mixing, unmixing = np.asarray(comp['topo']), np.asarray(comp['unmixing'])
    if mixing.shape != unmixing.T.shape or mixing.shape[0] != len(labels) or len(set(labels)) != len(labels):
        raise ValueError('ICA sensor labels/matrix shapes do not match')
    error = float(np.max(np.abs(unmixing@mixing-np.eye(unmixing.shape[0]))))
    if not np.isfinite(error) or error >= 1e-5:
        raise ValueError('ICA mixing/unmixing inverse check failed')
    match = re.search(r'vs\.ecg_eog_ic\s*=\s*\[([^]]*)\]', annotation)
    if match is None:
        raise ValueError('Missing explicit ECG/EOG component annotation')
    tokens = match[1].split()
    excluded = np.asarray([int(token)-1 for token in tokens], dtype=int)
    if (len(set(excluded)) != len(excluded) or (excluded < 0).any()
            or (excluded >= unmixing.shape[0]).any()):
        raise ValueError('Invalid or duplicate ICA component index')
    projection = np.eye(len(labels))-mixing[:, excluded]@unmixing[excluded]
    if excluded.size and np.max(np.abs(unmixing[excluded]@projection)) >= 1e-5*max(1, np.max(np.abs(unmixing))):
        raise ValueError('Excluded ICA components were not removed')
    return labels, projection, excluded, error
