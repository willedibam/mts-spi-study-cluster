"""Preliminary linear dimension and held-block-out sensor reconstruction, not neural ID."""
import argparse
import hashlib
import json
from pathlib import Path

import mne
import numpy as np
from scipy.linalg import qr
from scipy.signal import butter, sosfiltfilt


def spectrum(cov):
    vals = np.maximum(np.linalg.eigvalsh(cov)[::-1], 0)
    p = vals / vals.sum()
    return dict(participation_ratio=float(1 / (p @ p)),
                entropy_rank=float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0])))),
                components_for_variance={str(q): int(np.searchsorted(np.cumsum(p), q) + 1)
                                         for q in [.90, .95, .99]}, eigenvalue_fractions=p.tolist())


def reconstruction(train, test, count):
    # Scale is fitted to training blocks; report both raw-energy and equal-channel views outside.
    _, vectors = np.linalg.eigh(train)
    basis = vectors[:, -count:]
    _, _, order = qr(basis.T, pivoting=True, mode='economic')
    selected = order[:count]
    gram = train[np.ix_(selected, selected)]
    ridge = 1e-6 * np.trace(gram) / count
    weights = np.linalg.solve(gram + ridge * np.eye(count), train[selected, :]).T
    residual = test - weights @ test[selected, :] - test[:, selected] @ weights.T
    residual += weights @ test[np.ix_(selected, selected)] @ weights.T
    omitted = np.setdiff1d(np.arange(len(train)), selected)
    return dict(selected_indices=selected.tolist(),
                pca_retained_variance=float(np.trace(basis.T @ test @ basis) / np.trace(test)),
                sensor_reconstruction_retained_variance=float(1 - np.trace(residual) / np.trace(test)),
                omitted_sensor_retained_variance=float(1 - np.trace(residual[np.ix_(omitted, omitted)])
                                                       / np.trace(test[np.ix_(omitted, omitted)])))


def main(root):
    audit = json.loads((root / 'continuous-audit.json').read_text())
    verified = json.loads((root / 'download-verification.json').read_text())
    files = {Path(row['key']).name: Path(row['path']) for row in verified['files']}
    raw = mne.io.read_raw_bti(files['c,rfDC'], config_fname=files['config'], head_shape_fname=None,
                            convert=False, rename_channels=False, sort_by_ch_name=False,
                            preload=False, verbose='ERROR')
    bad = set(audit['marked_bad_meg_channels'])
    picks = [i for i in mne.pick_types(raw.info, meg=True, ref_meg=False, exclude=[])
             if raw.ch_names[i] not in bad]
    fs = float(raw.info['sfreq'])
    sos = butter(4, [1, 100], fs=fs, btype='bandpass', output='sos')
    covariance = {'raw': [], 'bandpass_1_100_hz': []}
    for block in audit['blocks']:
        start, stop = block['start_sample'], block['stop_sample']
        lo, hi = max(0, start - int(3 * fs)), min(raw.n_times, stop + int(3 * fs))
        x = raw.get_data(picks=picks, start=lo, stop=hi, verbose='ERROR')
        assert np.isfinite(x).all()
        filtered = sosfiltfilt(sos, x, axis=1)
        for key, data in [('raw', x), ('bandpass_1_100_hz', filtered)]:
            segment = data[:, start-lo:stop-lo].copy()
            segment -= segment.mean(axis=1, keepdims=True)
            covariance[key].append(segment @ segment.T / (segment.shape[1] - 1))
        print('completed block', block['block'], flush=True)
    # Two temporal halves within each condition: no random splitting of correlated time points.
    early, late = [], []
    for image in [1, 2]:
        for memory in [1, 2]:
            ix = [i for i, b in enumerate(audit['blocks'])
                  if b['image_type'] == image and b['memory_type'] == memory]
            assert len(ix) == 4
            early.extend(ix[:2]); late.extend(ix[2:])
    results = {}
    for key, items in covariance.items():
        covs = np.asarray(items)
        pooled = covs.mean(axis=0)
        diag = np.sqrt(np.diag(pooled))
        result = dict(pooled_covariance=spectrum(pooled),
                      pooled_correlation=spectrum(pooled / np.outer(diag, diag)),
                      per_block=[spectrum(c) for c in covs], folds=[])
        for train_ix, test_ix in [(early, late), (late, early)]:
            train, test = covs[train_ix].mean(axis=0), covs[test_ix].mean(axis=0)
            for scaling in ['raw_energy', 'equal_channel']:
                scale = np.sqrt(np.diag(train)) if scaling == 'equal_channel' else np.ones(len(train))
                a, b = train / np.outer(scale, scale), test / np.outer(scale, scale)
                for count in [16, 32, 64]:
                    row = reconstruction(a, b, count)
                    row.update(channels=count, scaling=scaling,
                               train_blocks=[audit['blocks'][i]['block'] for i in train_ix],
                               test_blocks=[audit['blocks'][i]['block'] for i in test_ix])
                    result['folds'].append(row)
        results[key] = result
    np.savez_compressed(root / 'dimension-covariances.npz', **{k: np.asarray(v) for k, v in covariance.items()})
    report = dict(status='complete', channels=[raw.ch_names[i] for i in picks], blocks=16,
                  participant='105923', run='6-Wrkmem', sampling_hz=fs, results=results,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  audit_sha256=hashlib.sha256((root / 'continuous-audit.json').read_bytes()).hexdigest(),
                  limitations=['One participant/run; not intrinsic neural dimension or task information.',
                               'Marked bad channels excluded; no ICA/reference correction or full artifact rejection.',
                               '1-100 Hz is a diagnostic filter, not a frozen scientific passband; 3 seconds context.',
                               'Block-centered covariance averaged equally; no temporal concatenation for SPI.',
                               'Selected sensors and reconstruction fit only training blocks; same-person validation.',
                               'No pyspi extraction or evidence of z preservation.'])
    (root / 'dimension-probe.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
