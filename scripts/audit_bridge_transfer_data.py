"""Dataset eligibility audit, not a prediction benchmark or a pyspi run.

Uses the public version-1 Figshare inventory, small file tails for all four
configuration indices, and the checksum-verified B3 raw file. External pickle
globals are restricted to the NumPy constructors needed by this dataset.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import pickle
import pickletools

import numpy as np
from scipy.signal import periodogram


class NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        constructors = {
            ('numpy.core.multiarray', '_reconstruct'): np._core.multiarray._reconstruct,
            ('numpy._core.multiarray', '_reconstruct'): np._core.multiarray._reconstruct,
            ('numpy', 'ndarray'): np.ndarray,
            ('numpy', 'dtype'): np.dtype,
        }
        if (module, name) not in constructors:
            raise pickle.UnpicklingError(f'Unsupported global: {module}.{name}')
        return constructors[module, name]


def unicode_field(tail, key):
    """Read a known Unicode ndarray payload without executing pickle opcodes."""
    marker = b'\x8c' + bytes([len(key)]) + key.encode()
    start = tail.index(marker)
    dtype = None
    for op, arg, _ in pickletools.genops(tail[start:]):
        if isinstance(arg, str) and arg.startswith('U') and arg[1:].isdigit():
            dtype = '<' + arg
        if op.name in ['SHORT_BINBYTES', 'BINBYTES', 'BINBYTES8'] and dtype is not None:
            return np.frombuffer(arg, dtype=dtype).copy()
    raise ValueError(f'No payload for {key}')


def peaks(record, centers, fs):
    frequency, power = periodogram(record, fs=fs, window='hann', axis=0)
    power = power.mean(axis=1)
    result = []
    for center in centers:
        candidates = np.flatnonzero(abs(frequency - center) <= 2)
        k = candidates[np.argmax(power[candidates])]
        # Log-parabolic interpolation avoids interpreting FFT grid spacing as
        # a hard bound on frequency-estimation accuracy.
        a, b, c = np.log(np.maximum(power[k-1:k+2], np.finfo(float).tiny))
        offset = .5 * (a-c) / (a-2*b+c) if a-2*b+c else 0.
        result.append(frequency[k] + np.clip(offset, -.5, .5) * (frequency[1]-frequency[0]))
    return np.asarray(result)


def audit(root, output):
    metadata = json.loads((root / 'bridge-metadata.json').read_text())
    assert metadata['version'] == 1
    inventory = {}
    indices = {}
    for name in ['B1', 'B2', 'B3', 'B4']:
        tail = (root / (name + '.pkl.tail1m')).read_bytes()
        files = unicode_field(tail, 'lms_file_name')
        timestamps = unicode_field(tail, 'time_stamp')
        assert len(files) == len(timestamps)
        labels = np.array([f.split('_')[2] for f in files])
        temperatures = np.array([int(f.split('_')[1][1:]) for f in files])
        order = sorted(range(len(files)), key=lambda i: datetime.strptime(timestamps[i], '%d-%b-%y %H:%M:%S'))
        blocks = []
        for i in order:
            if not blocks or labels[i] != blocks[-1]['label']:
                blocks.append(dict(label=labels[i], start=timestamps[i], records=0))
            blocks[-1]['records'] += 1
            blocks[-1]['end'] = timestamps[i]
        selected = np.isin(labels, ['N', 'M1', 'M2', 'M3', 'M4'])
        positive = labels[selected] != 'N'
        tag = temperatures[selected]
        # Descriptive, post-inspection confound audit; no test performance claim.
        threshold_prediction = tag > 28
        inventory[name] = dict(
            records=len(files), label_counts=dict(Counter(labels.tolist())),
            dates=dict(Counter(t.split()[0] for t in timestamps)),
            temperature_tags_by_label={label: dict(Counter(temperatures[labels == label].tolist())) for label in sorted(set(labels))},
            chronological_label_runs=blocks,
            normal_vs_M1_M4=dict(records=int(selected.sum()),
                normal_tag_range=[int(tag[~positive].min()), int(tag[~positive].max())],
                mass_tag_range=[int(tag[positive].min()), int(tag[positive].max())],
                tag_above_28_descriptive_accuracy=float(np.mean(threshold_prediction == positive))),
            tail_sha256=hashlib.sha256(tail).hexdigest())
        indices[name] = files, timestamps
    path = root / 'B3.pkl'
    spec = next(f for f in metadata['files'] if f['name'] == path.name)
    assert path.stat().st_size == spec['size']
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    assert digest == spec['computed_md5']
    with path.open('rb') as stream:
        raw = NumpyOnlyUnpickler(stream).load()
    meta = raw['meta']
    np.testing.assert_array_equal(indices['B3'][0], meta['lms_file_name'])
    np.testing.assert_array_equal(indices['B3'][1], meta['time_stamp'])
    np.testing.assert_array_equal([f.split('_')[2] for f in indices['B3'][0]], meta['damage'])
    tags = [int(f.split('_')[1][1:]) for f in indices['B3'][0]]
    sensor_names = [f'A{i}' for i in range(1, 21)]
    x = np.stack([raw[name] for name in sensor_names], axis=-1)
    assert x.shape == (70, 24600, 20) and np.isfinite(x).all()
    for scale in meta['scale']:
        assert len(scale) >= x.shape[1]
        np.testing.assert_allclose(np.diff(scale), 1/256, atol=1e-12, rtol=0)
    centers = [14.78, 41.33, 47.53, 81.77]  # Published B3 reference frequencies.
    full = np.stack([peaks(record, centers, 256) for record in x])
    window_summary = {}
    for length in [1024, 4096]:
        start = (x.shape[1] - length) // 2
        estimated = np.stack([peaks(record[start:start+length], centers, 256) for record in x])
        error = abs(estimated - full)
        window_summary[str(length)] = dict(duration_seconds=length/256,
            median_absolute_difference_Hz=np.median(error, axis=0).tolist(),
            p90_absolute_difference_Hz=np.quantile(error, .9, axis=0).tolist())
    raw_audit = dict(md5=digest, observed_array_shapes={k: list(v.shape) for k,v in raw.items() if isinstance(v,np.ndarray)},
        analysis_channels=sensor_names, sampling_rate_from_time_vectors=256,
        stored_samples=x.shape[1], duration_seconds=x.shape[1]/256,
        time_vector_lengths=sorted({len(t) for t in meta['scale']}),
        filename_temperature_matches_measured_metadata=bool(np.array_equal(tags, meta['temp_data'])),
        median_peak_Hz_by_label={label: np.median(full[meta['damage']==label],axis=0).tolist() for label in np.unique(meta['damage'])},
        window_sensitivity=window_summary,
        caveat='Reference-anchored spectral peaks, not identified modal parameters. '
               'Same-record window comparison, not out-of-sample prediction. '
               'Time-vector tails exceed stored signal lengths; no padding or inferred extra samples used.')
    output.mkdir(parents=True, exist_ok=True)
    result = dict(status='eligibility_audit_no_pyspi_no_predictive_claim', dataset_doi=metadata['doi'],
        metadata_sha256=hashlib.sha256((root/'bridge-metadata.json').read_bytes()).hexdigest(),
        file_indices=inventory, raw_B3=raw_audit,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output / 'bridge-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(output/'B3-spectral-scout.npz', peaks=full, labels=meta['damage'],
                        temperature=meta['temp_data'], filename=meta['lms_file_name'])
    plot_temperature_support(inventory, output)
    print(json.dumps({'records': {k:v['records'] for k,v in inventory.items()},
                     'B4_tag_audit': inventory['B4']['normal_vs_M1_M4'], 'raw_B3':raw_audit}, indent=2))


def plot_temperature_support(inventory, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = ['N', 'M1', 'M2', 'M3', 'M4']
    fig, axes = plt.subplots(1, 4, figsize=(11, 4), sharex=True, sharey=True)
    for ax, (name, item) in zip(axes, inventory.items(), strict=True):
        for row, label in enumerate(labels):
            counts = item['temperature_tags_by_label'][label]
            ax.scatter(list(map(int, counts)), [row]*len(counts),
                       s=12 + 4*np.sqrt(list(counts.values())),
                       color='#287EA5' if label == 'N' else '#BD663C')
        ax.set_title(name)
        ax.set_yticks(range(5), ['Normal', 'Mass M1', 'Mass M2', 'Mass M3', 'Mass M4'])
        ax.set_xlim(-13, 33)
        ax.set_xticks([-10, 0, 10, 20, 30])
        ax.set_xlabel('Temperature tag (°C)')
        ax.grid(axis='x', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle('Temperature and condition support in the released bridge indices', x=.02, ha='left')
    fig.text(.02, .02, 'Filename tags; B3 checked against measured metadata. Marker area increases with record count. No prediction results.', fontsize=8)
    fig.tight_layout(rect=[0, .07, 1, .93])
    for suffix in ['png', 'svg']:
        fig.savefig(output / ('temperature-support.' + suffix), dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/representation_application_scout_260908'))
    parser.add_argument('--output', type=Path, default=Path('results/representation_application_scout_260908'))
    args = parser.parse_args()
    audit(args.data, args.output)
