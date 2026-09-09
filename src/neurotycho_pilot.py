"""Source-only ECoG windowing, quality checks and spectral descriptors."""
import numpy as np
from scipy import signal

BANDS = ((.5, 4), (4, 8), (8, 13), (13, 30), (30, 45), (55, 100))


def window_specs(condition, count=16):
    labels = [str(x) for x in np.atleast_1d(condition['ConditionLabel'])]
    indices = np.atleast_1d(condition['ConditionIndex']).astype(int) - 1
    times = np.atleast_1d(condition['ConditionTime'])
    if len(set(labels)) != len(labels) or not np.allclose(indices / 1000, times, rtol=0, atol=1e-8):
        raise ValueError('ambiguous labels or timestamp convention')
    lookup = dict(zip(labels, indices))
    result = []
    for state, target in [('AwakeEyesClosed', 0), ('Anesthetized', 1)]:
        if state + '-Start' not in lookup:
            continue
        lower, upper = lookup[state + '-Start'], lookup[state + '-End']
        first, last = lower + 30000, upper - 30000 - 8000
        starts = np.linspace(first, last, count).astype(int)
        if lower < 0 or first > last or np.any(np.diff(starts) < 8000):
            raise ValueError('insufficient stable-state interval')
        result.extend(dict(state=state, target=target, window=i, start=int(start),
                           context_start=int(start - 10000), context_stop=int(start + 18000))
                      for i, start in enumerate(starts))
    return result


def longest_constant_run(row):
    changes = np.flatnonzero(np.diff(row) != 0) + 1
    return int(np.diff(np.r_[0, changes, len(row)]).max())


def preprocess(raw, pair_indices):
    """raw is electrode x 28s at 1kHz; output is bipolar x central 8s at250Hz."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 28000:
        raise ValueError('expected 28-second context at1kHz')
    core = raw[:, 10000:18000]
    finite = bool(np.isfinite(raw).all())
    flat = max(longest_constant_run(row) for row in core) if finite else 8000
    pairs = np.asarray(pair_indices)
    bipolar = raw[pairs[:, 0]] - raw[pairs[:, 1]]
    constant_pair = bool(np.any(np.ptp(bipolar[:, 10000:18000], axis=1) == 0))
    quality = dict(finite=finite, longest_constant_raw_run=flat, constant_pair=constant_pair,
                   repeated_raw_fraction=float(np.mean(np.diff(core, axis=1) == 0)),
                   accepted=finite and flat < 1000 and not constant_pair)
    if not quality['accepted']:
        return None, quality
    filtered = signal.detrend(bipolar, axis=-1)
    filtered = signal.sosfiltfilt(signal.butter(4, [.5, 100], btype='bandpass', fs=1000,
                                              output='sos'), filtered, axis=-1)
    b, a = signal.iirnotch(50, 30, fs=1000)
    filtered = signal.filtfilt(b, a, filtered, axis=-1)
    filtered = signal.resample_poly(filtered, 1, 4, axis=-1)[:, 2500:4500]
    rms = np.sqrt(np.mean(filtered**2, axis=1))
    quality.update(rms_median=float(np.median(rms)),
                   amplitude_mad_median=float(np.median(np.median(np.abs(filtered -
                        np.median(filtered, axis=1, keepdims=True)), axis=1))),
                   crest_factor_max=float(np.max(np.max(np.abs(filtered), axis=1) / rms)))
    f, power = signal.welch(bipolar[:, 10000:18000], fs=1000, nperseg=2000, noverlap=1000)
    quality['raw_line_fraction_median'] = float(np.median(
        power[:, (f >= 49) & (f <= 51)].sum(1) / power[:, (f >= .5) & (f <= 100)].sum(1)))
    # KSG/Kozachenko require tie-free coordinates: float32 rounding can create
    # exact ties absent from the filtered float64 signal. Preserve this precision
    # through SPI extraction; neural tensors may be cast at model input later.
    return filtered, quality


def spectral_features(x):
    f, power = signal.welch(x.astype(float), fs=250, nperseg=500, noverlap=250)
    positive = (f >= .5) & (f <= 100)
    spectrum = power[:, positive]
    total = spectrum.sum(1)
    if np.any(total <= 0) or not np.isfinite(total).all():
        raise ValueError('degenerate spectrum')
    bandpower = np.stack([power[:, (f >= lo) & (f < hi)].sum(1) for lo, hi in BANDS], axis=1)
    normalized = spectrum / total[:, None]
    entropy = -(normalized * np.log(np.maximum(normalized, 1e-300))).sum(1) / np.log(spectrum.shape[1])
    edge = f[positive][np.argmax(np.cumsum(normalized, axis=1) >= .95, axis=1)]
    per_channel = np.column_stack([np.log10(np.maximum(bandpower * (f[1] - f[0]), 1e-300)),
                                   bandpower / total[:, None], entropy, edge])
    return np.concatenate([per_channel.mean(0), per_channel.std(0),
                            np.quantile(per_channel, [.25, .5, .75], axis=0).ravel()])
