"""Windows from the MNI Open iEEG Atlas (NEMAR nm000362) for the human-sleep MI-pair test (rules: ../preregs/sleep-prereg.md).

Each recording is a concatenation of artifact-free segments separated by 2-s zero buffers (README, "Known caveats").
Every segment is processed on its own, never across a buffer: high-pass 0.5 Hz, low-pass 40 Hz (zero-phase Butterworth,
the atlas long-arm filters of ../../prepare.py, with reflection padding of up to 2 s), polyphase resampling 200 -> 100 Hz.
Non-overlapping 10-s windows (T = 1000) are tiled from each segment's start; the remainder is dropped. Each window is
linearly detrended. Bipolar channels are kept as released (no re-referencing).

    from extract import recording, windows, channel_set
"""
import os, zlib, numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt, resample_poly, detrend
FS_SRC, FS, T = 200, 100, 1000
burst = lambda c: float(np.median(1.4826 * np.median(np.abs(c - np.median(c, 0)), 0) / c.std(0)))  # as ../human/extract.py


def path(root, sub, task, acq, suffix):
    return f'{root}/{sub}/ieeg/{sub}_task-{task}_acq-{acq}_{suffix}'


def segments(root, sub, task, acq):
    ev = pd.read_csv(path(root, sub, task, acq, 'events.tsv'), sep='\t')
    s = ev[ev.trial_type == 'artifact_free_segment']
    return [(int(a), int(n)) for a, n in zip(s['sample'], s.n_samples)]


def names(root, sub, task, acq):
    return list(pd.read_csv(path(root, sub, task, acq, 'channels.tsv'), sep='\t').name)


def recording(root, sub, task, acq):
    """(n, M) µV float64 at 200 Hz, channel names. BrainVision IEEE_FLOAT_32, multiplexed, µV (README, Conversion)."""
    import mne
    raw = mne.io.read_raw_brainvision(path(root, sub, task, acq, 'ieeg.vhdr'), preload=True, verbose='ERROR')
    assert raw.info['sfreq'] == FS_SRC
    return raw.get_data().T * 1e6, raw.ch_names


def prep(x):
    """One contiguous segment (n, M) at 200 Hz -> 100 Hz, filtered as the atlas long arm, never across a buffer."""
    pad = min(len(x) - 1, 2 * FS_SRC)
    hp = butter(2, 0.5, btype='highpass', fs=FS_SRC, output='sos'); lp = butter(8, 40.0, fs=FS_SRC, output='sos')
    y = sosfiltfilt(hp, x, axis=0, padtype='even', padlen=pad); y = sosfiltfilt(lp, y, axis=0, padtype='even', padlen=pad)
    return resample_poly(y, 1, 2, axis=0)


def windows(root, sub, task, acq, chans):
    """List of (window (T, len(chans)) float32, info) for every whole 10-s window inside one segment, in time order."""
    x, nm = recording(root, sub, task, acq); idx = [nm.index(c) for c in chans]; out = []
    for k, (a, n) in enumerate(segments(root, sub, task, acq)):
        seg = x[a:a + n][:, idx]
        if n < T * FS_SRC // FS: continue
        y = prep(seg)
        for j in range(len(y) // T):
            w = detrend(y[j * T:(j + 1) * T], axis=0)
            out.append((w.astype(np.float32), dict(segment=k, start_s=a / FS_SRC + j * T / FS, burst=burst(w))))
    return out


def channel_set(root, sub, states, seed0=20261127):
    """Channels present in every state's recording, in wake order; > 32 -> 32 drawn with a fixed seed (protocol, KSG subset).
    The draw uses names only (no data), so it is fixed before any window is read."""
    common = [c for c in names(root, sub, *states[0]) if all(c in names(root, sub, *s) for s in states[1:])]
    if len(common) <= 32: return common
    rng = np.random.default_rng(seed0 + int(sub[4:])); keep = sorted(rng.choice(len(common), 32, replace=False))
    return [common[i] for i in keep]


def shared_contact(chans):
    """(i, j) upper-triangle mask of pairs of bipolar derivations on one electrode with adjacent first contacts
    (e.g. Rh1 = Rh1-Rh2 and Rh2 = Rh2-Rh3 share contact 2); for the shared-contact sensitivity row."""
    import re
    p = [re.match(r'^(.*?)(\d+)$', c) for c in chans]; e = [(m.group(1), int(m.group(2))) if m else (c, -99) for m, c in zip(p, chans)]
    iu = np.triu_indices(len(chans), 1)
    return np.array([e[i][0] == e[j][0] and abs(e[i][1] - e[j][1]) == 1 for i, j in zip(*iu)])


seed = lambda sub, task, acq, i: zlib.crc32(f'{sub}:{task}:{acq}:{i}'.encode()) * 10
