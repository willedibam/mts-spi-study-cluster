"""Prepare a bounded, single-run M32 spatial-coverage feasibility corpus."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import yaml


def farthest_sensors(positions, first, count=32):
    """Deterministic Euclidean coverage on the helmet; no signal or label fitting."""
    distances = np.sum((positions[:, None] - positions[None, :]) ** 2, axis=2)
    selected = [int(first)]
    while len(selected) < count:
        nearest = distances[:, selected].min(axis=1)
        nearest[selected] = -np.inf
        selected.append(int(np.argmax(nearest)))
    return np.asarray(selected), float(np.sqrt(distances[:, selected].min(axis=1).max()))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root):
    output = root / 'm32-spatial'
    output.mkdir(exist_ok=True)
    assert not (output / 'views.npz').exists(), 'Keep existing prepared corpus immutable'
    audit = json.loads((root / 'continuous-audit.json').read_text())
    downloads = json.loads((root / 'download-verification.json').read_text())
    ica_download = json.loads((root / 'ica/download-verification.json').read_text())
    assert downloads['status'] == ica_download['status'] == 'complete'
    files = {Path(r['key']).name: Path(r['path']) for r in downloads['files']}
    ica_file = Path(ica_download['files'][0]['path'])
    assert sha(ica_file) == ica_download['files'][0]['sha256']
    comp = loadmat(ica_file, simplify_cells=True)['comp_class']
    labels = np.asarray(comp['topolabel']).tolist()
    mixing, unmixing = np.asarray(comp['topo']), np.asarray(comp['unmixing'])
    assert mixing.shape == unmixing.T.shape and mixing.shape[0] == len(labels)
    identity_error = float(np.max(np.abs(unmixing @ mixing - np.eye(unmixing.shape[0]))))
    assert identity_error < 1e-5
    annotation = (root / 'metadata/105923_MEG_6-Wrkmem_icaclass_vs.txt').read_text()
    # Remove only explicitly identified cardiac/ocular components for this feasibility stage.
    excluded = np.asarray([int(v)-1 for v in re.search(r'vs.ecg_eog_ic = \[([^]]*)\]', annotation)[1].split()])
    projection = np.eye(len(labels)) - mixing[:, excluded] @ unmixing[excluded]
    assert np.max(np.abs(unmixing[excluded] @ projection)) < 1e-5 * max(1, np.max(np.abs(unmixing)))
    raw = mne.io.read_raw_bti(files['c,rfDC'], config_fname=files['config'], head_shape_fname=None,
                            convert=False, rename_channels=False, sort_by_ch_name=False,
                            preload=False, verbose='ERROR')
    fs = float(raw.info['sfreq'])
    picks = [raw.ch_names.index(name) for name in labels]
    ref = mne.pick_types(raw.info, meg=False, ref_meg=True, exclude=[]).tolist()
    assert ref and not set(picks).intersection(ref)
    good = np.asarray([i for i, name in enumerate(labels) if name not in audit['marked_bad_meg_channels']])
    good_names = [labels[i] for i in good]
    positions = np.asarray([raw.info['chs'][picks[i]]['loc'][:3] for i in good])
    assert np.isfinite(positions).all() and len(np.unique(positions, axis=0)) == len(positions)
    assert 0.01 < np.max(np.linalg.norm(positions - positions.mean(axis=0), axis=1)) < 1
    first = min(range(len(good)), key=lambda i: int(good_names[i][1:]))
    second = int(np.argmax(np.linalg.norm(positions - positions[first], axis=1)))
    layouts = {}
    for tag, start in [('coverage_a', first), ('coverage_b', second)]:
        selected, radius = farthest_sensors(positions, start)
        layouts[tag] = dict(indices=selected.tolist(), channels=[good_names[i] for i in selected],
                            max_distance_to_selected_m=radius)
    blocks = sorted(audit['blocks'], key=lambda b: b['start_sample'])
    assert all(b['bad_segment_overlap_seconds'] == 0 for b in blocks)
    frame = pd.read_csv(files['105923_MEG_Wrkmem_run1.tab'], sep='\t')
    onset = pd.to_numeric(frame['Stim.OnsetTime'], errors='coerce')
    events = frame[onset.notna() & (onset > 0)].assign(onset=onset).sort_values('onset')
    assert events['BlockType'].tolist() == [({1:'0-Back', 2:'2-Back'}[b['memory_type']]) for b in blocks for _ in range(10)]
    assert events['StimType'].tolist() == [({1:'Face', 2:'Tools'}[b['image_type']]) for b in blocks for _ in range(10)]
    # HCP/MNE-HCP reference regression, fit to sparsely sampled task data across this run.
    # This is per-record unsupervised preprocessing, not fitting a predictive model.
    sampled = []
    for b in blocks:
        sampled.append(raw.get_data(picks=picks+ref, start=b['start_sample'], stop=b['stop_sample'], verbose='ERROR')[:, ::100])
    fit = np.concatenate(sampled, axis=1)
    estimator = make_pipeline(StandardScaler(), LinearRegression()).fit(fit[len(picks):].T, fit[:len(picks)].T)
    del sampled, fit
    sos = butter(4, [1, 100], fs=fs, btype='bandpass', output='sos')
    arrays, entries, metrics = {}, [], []
    for b in blocks:
        start, stop = b['start_sample'], b['stop_sample']
        lo, hi = max(0, start-int(3*fs)), min(raw.n_times, stop+int(3*fs))
        data = raw.get_data(picks=picks+ref, start=lo, stop=hi, verbose='ERROR')
        corrected = data[:len(picks)] - estimator.predict(data[len(picks):].T).T
        cleaned = projection @ corrected
        clean_block = cleaned[:, start-lo:stop-lo]
        assert np.isfinite(clean_block).all() and np.all(clean_block[good].std(axis=1) > 0)
        metrics.append(dict(block=b['block'], reference_residual_energy_ratio=float(np.sum(corrected**2)/np.sum(data[:len(picks)]**2)),
                            ica_residual_energy_ratio=float(np.sum(cleaned**2)/np.sum(corrected**2))))
        filtered = sosfiltfilt(sos, cleaned[good], axis=1)
        # FFT resampling includes anti-aliasing; retain the full continuous task span.
        resampled = mne.filter.resample(filtered, down=fs/250, npad='auto', verbose='ERROR')
        begin = round((start-lo)*250/fs)
        length = round((stop-start)*250/fs)
        for tag, layout in layouts.items():
            x = resampled[np.asarray(layout['indices']), begin:begin+length].copy()
            assert x.shape == (32, length) and np.isfinite(x).all()
            # Explicit per-channel standardisation, consistent across all comparator inputs.
            x -= x.mean(axis=1, keepdims=True)
            x /= x.std(axis=1, keepdims=True)
            name = f"105923_run6_block{b['block']:02d}_{tag}"
            arrays[name] = x.T
            entries.append(dict(name=name, block=b['block'], layout=tag, memory=b['memory_type'], image=b['image_type'], M=32, T=length))
        print('prepared block', b['block'], 'T', length, flush=True)
    names = list(arrays)
    arrays.update(__dataset_names__=np.asarray(names), __labels_json__=np.asarray([json.dumps([f"memory{r['memory']}", f"image{r['image']}", r['layout']]) for r in entries]),
                  __shapes__=np.asarray([arrays[name].shape for name in names]), __axis_order__=np.asarray(['observation','process']))
    np.savez_compressed(output/'views.npz', **arrays)
    config = dict(name='hcp-m32-spatial-260914', source=dict(format='named-npz-v1', archive=str(output/'views.npz'),
                  sha256=sha(output/'views.npz'), axis_order=['observation','process']), base_output_dir=str(output/'pyspi'),
                  pyspi_config='configs/pyspi/benchmarked_p90.yaml', normalise=False, random_seed=260914311)
    (output/'external.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    report = dict(status='prepared_requires_QC_review', entries=entries, layouts=layouts, good_channel_names=good_names,
                  good_channel_positions_m=positions.tolist(), ica_labels=labels, excluded_ica_zero_based=excluded.tolist(),
                  ica_left_inverse_max_error=identity_error, diagnostics=metrics, input_audit_sha256=sha(root/'continuous-audit.json'),
                  script_sha256=sha(Path(__file__)), archive_sha256=config['source']['sha256'],
                  limitations=['One-run feasibility only; no task learning or cohort claim.',
                               'Coordinate coverage is not anatomical source parcellation.',
                               'Only cardiac/ocular labelled ICA components removed; other artifact classes need assessment.',
                               'Per-run reference regression uses all task blocks without labels.',
                               'Diagnostic 1-100 Hz, 250 Hz sampling; no claim of lossless transformation.',
                               'Selection fixed across blocks; alternate layout is a sensitivity comparison, not extra independent recordings.'])
    (output/'preparation.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
