"""Bounded full-p90 reference panel for future pair sampling, not sparse pyspi.

CML: nested dispersed views of a fixed large lattice. TASEP: full observation
of each finite chain. MEG: nested spatial coverage from four fixed blocks of
one already-audited run. These are not 48 independent physical realizations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import yaml


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def standardize(x):
    x = np.array(x, dtype=float, copy=True)
    assert x.ndim == 2 and np.isfinite(x).all()
    scale = x.std(axis=0)
    assert np.all(scale > 0), 'Constant channels: preserve and inspect, do not add noise'
    return (x-x.mean(axis=0))/scale


def tasks():
    cml = [dict(system='cml', r=r, seed=260917301+s) for r in [3.84, 3.89] for s in range(3)]
    tasep = [dict(system='tasep', M=m, alpha=a, seed=260917401+s)
             for m in [64, 100, 256] for a in [.15, .25] for s in range(3)]
    return cml + tasep + [dict(system='meg')]


def write_part(root, index, arrays, rows, metadata):
    directory = root/'prepared'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/f'part-{index:02d}.npz'
    assert not path.exists(), 'Preserve existing prepared part'
    for name, x in arrays.items():
        assert x.shape[0] == 1000 and np.isfinite(x).all() and np.all(x.std(axis=0) > 0)
    np.savez_compressed(path, **arrays)
    report = dict(task=tasks()[index], rows=rows, metadata=metadata, archive_sha256=sha(path),
                  script_sha256=sha(__file__))
    path.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')


def prepare_synthetic(root, index):
    task = tasks()[index]
    arrays, rows = {}, []
    if task['system'] == 'cml':
        from scripts.scout_cml2d_period_doubling import evolve, order_summary
        rng = np.random.default_rng(task['seed'])
        state = rng.random((256, 256))
        indices = rng.permutation(256**2)[:256].reshape(1, -1)
        means, observed, _ = evolve(state, task['r'], .2, 100000, 6000, 1000, indices)
        assert np.isfinite(means).all() and 0 <= observed.min() <= observed.max() <= 1
        for m in [64, 100, 256]:
            name = f'cml-r{task["r"]}-s{task["seed"]}-M{m}'
            arrays[name] = standardize(observed[:, 0, :m])
            rows.append(dict(name=name, M=m, T=1000, system='cml', seed=task['seed'], r=task['r']))
        metadata = dict(N=256**2, L=256, g=.2, burn=100000, record_steps=6000,
                        indices=indices.tolist(), **order_summary(means, 1000),
                        generator_sha256=sha('scripts/scout_cml2d_period_doubling.py'))
    else:
        from scripts.tasep_phase_boundary import simulate
        m = task['M']
        values, metadata = simulate(dict(N=m, alpha=task['alpha'], seed=task['seed']),
            dict(beta=.2, burn=100000., observation_steps=1000, dt=1.,
                 reference_time=100000., reference_blocks=32, reference_trace_dt=10.))
        assert metadata['constant_channels'] == 0
        # Broad stationary-density sanity gate, not a new precision physics claim.
        assert metadata['reference_absolute_error'] < .05, metadata
        name = f'tasep-a{task["alpha"]}-s{task["seed"]}-M{m}'
        arrays[name] = standardize(values['observed'].T)
        rows.append(dict(name=name, M=m, T=1000, system='tasep', seed=task['seed'], alpha=task['alpha']))
    write_part(root, index, arrays, rows, metadata)


def prepare_meg(root, source, index):
    """Repeat audited preprocessing on all good channels; don't modify old views."""
    import mne
    from scipy.io import loadmat
    from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch, welch
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from scripts.prepare_hcp_m32 import farthest_sensors
    from src.hcp_run_audit import ica_projection
    audit = json.loads((source/'continuous-audit.json').read_text())
    downloads = json.loads((source/'download-verification.json').read_text())
    ica = json.loads((source/'ica/download-verification.json').read_text())
    assert downloads['status'] == ica['status'] == 'complete'
    for row in downloads['files'] + ica['files']:
        assert sha(row['path']) == row['sha256']
    files = {Path(r['key']).name: Path(r['path']) for r in downloads['files']}
    comp = loadmat(ica['files'][0]['path'], simplify_cells=True)['comp_class']
    annotation = (source/'metadata/105923_MEG_6-Wrkmem_icaclass_vs.txt').read_text()
    labels, projection, excluded, error = ica_projection(comp, annotation)
    raw = mne.io.read_raw_bti(files['c,rfDC'], config_fname=files['config'], head_shape_fname=None,
        convert=False, rename_channels=False, sort_by_ch_name=False, preload=False, verbose='ERROR')
    fs = float(raw.info['sfreq'])
    picks = [raw.ch_names.index(n) for n in labels]
    refs = mne.pick_types(raw.info, meg=False, ref_meg=True, exclude=[]).tolist()
    good = np.array([i for i, n in enumerate(labels) if n not in audit['marked_bad_meg_channels']])
    assert len(good) == 243 and refs and not set(picks).intersection(refs)
    positions = np.array([raw.info['chs'][picks[i]]['loc'][:3] for i in good])
    first = min(range(len(good)), key=lambda i: int(labels[good[i]][1:]))
    order, _ = farthest_sensors(positions, first, count=len(good))
    blocks = sorted(audit['blocks'], key=lambda b:b['start_sample'])
    assert all(b['bad_segment_overlap_seconds'] == 0 for b in blocks)
    fit = np.concatenate([raw.get_data(picks=picks+refs, start=b['start_sample'],
        stop=b['stop_sample'], verbose='ERROR')[:, ::100] for b in blocks], axis=1)
    estimator = make_pipeline(StandardScaler(), LinearRegression()).fit(fit[len(picks):].T, fit[:len(picks)].T)
    del fit
    sos = butter(4, [1, 100], fs=fs, btype='bandpass', output='sos')
    notch_b, notch_a = iirnotch(60, 30, fs=250)
    arrays, rows, qc = {}, [], []
    for block in blocks[:4]:  # chronological, no selection using signals or z
        start, stop = block['start_sample'], block['stop_sample']
        lo, hi = max(0, start-int(3*fs)), min(raw.n_times, stop+int(3*fs))
        data = raw.get_data(picks=picks+refs, start=lo, stop=hi, verbose='ERROR')
        corrected = projection @ (data[:len(picks)]-estimator.predict(data[len(picks):].T).T)
        filtered = sosfiltfilt(sos, corrected[good], axis=1)
        resampled = mne.filter.resample(filtered, down=fs/250, npad='auto', verbose='ERROR')
        begin, length = round((start-lo)*250/fs), round((stop-start)*250/fs)
        parent = resampled[:, begin:begin+length].T
        notched = filtfilt(notch_b, notch_a, parent, axis=0)
        crop = (length-1000)//2
        x = standardize(notched[crop:crop+1000, order])
        frequencies, power = welch(notched, fs=250, nperseg=1000, axis=0)
        line = (frequencies >= 59) & (frequencies <= 61)
        band = (frequencies >= 1) & (frequencies <= 100)
        line_fraction = np.median(power[line].sum(axis=0)/power[band].sum(axis=0))
        assert np.isfinite(line_fraction) and line_fraction < .1
        qc.append(dict(block=block['block'], median_line_power_fraction=float(line_fraction),
                       max_abs_standardized=float(abs(x).max())))
        for m in [64, 100, 243]:
            name = f'meg-105923-run6-block{block["block"]:02d}-M{m}'
            arrays[name] = x[:, :m]
            rows.append(dict(name=name, system='meg', M=m, T=1000, block=block['block'],
                memory=block['memory_type'], image=block['image_type'], crop_start=crop,
                subject='105923', run=6))
    metadata = dict(source_root=str(source), input_audit_sha256=sha(source/'continuous-audit.json'),
        raw_sha256={r['key']:r['sha256'] for r in downloads['files']+ica['files']},
        channel_order=[labels[good[i]] for i in order], excluded_ica=excluded.tolist(),
        ica_identity_error=error, qc=qc,
        preprocessing='Audited reference regression and ECG/EOG ICA projection; 1–100 Hz; resample250Hz; notch60HzQ30; central1000; channel z-score.',
        scope='Four dependent blocks from one run; no cohort or task-inference claim. Existing HCP analysis unchanged.')
    write_part(root, index, arrays, rows, metadata)


def assemble(root):
    arrays, rows, reports = {}, [], []
    for index in range(len(tasks())):
        path = root/'prepared'/f'part-{index:02d}.npz'
        report = json.loads(path.with_suffix('.json').read_text())
        assert sha(path) == report['archive_sha256']
        with np.load(path) as bank:
            for name in bank.files:
                assert name not in arrays
                arrays[name] = bank[name]
        rows.extend(report['rows'])
        reports.append(report)
    assert len(rows) == len(arrays) == 48
    # Per size: first instance of every family is the smoke; 13 others follow.
    for group, sizes in [('m64', [64]), ('m100', [100]), ('large', [243, 256])]:
        subset = [r for r in rows if r['M'] in sizes]
        assert len(subset) == 16
        names = [r['name'] for r in subset]
        path = root/f'{group}.npz'
        assert not path.exists()
        np.savez_compressed(path, **{n:arrays[n] for n in names}, __dataset_names__=np.array(names),
            __labels_json__=np.array([json.dumps([r['system']]) for r in subset]),
            __shapes__=np.array([arrays[n].shape for n in names]),
            __axis_order__=np.array(['observation', 'process']))
        config = dict(name=f'large-m-{group}-260917', source=dict(format='named-npz-v1', archive=str(path),
            sha256=sha(path), axis_order=['observation', 'process']), base_output_dir=str(root/'pyspi'),
            pyspi_config='configs/pyspi/benchmarked_p90.yaml', normalise=False, random_seed=260917501)
        (root/f'{group}.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        smoke = [next(i for i,r in enumerate(subset, 1) if r['system'] == s) for s in ['cml', 'tasep', 'meg']]
        for tag, indices in [('smoke', smoke), ('rest', [i for i in range(1,17) if i not in smoke])]:
            (root/f'{group}-{tag}.txt').write_text(''.join(f'{i}\n' for i in indices))
    (root/'preparation.json').write_text(json.dumps(dict(rows=rows, reports=reports,
        interpretation='Exact within-M references for numerical approximation; not invariant z across M or 48 independent records.'), indent=2)+'\n')


def extract(root, group, index):
    from src import run_external_corpus as corpus
    from src.hcp_spectral_memory import bounded_psi_memory
    config = corpus.ExternalCorpusConfig.from_file(root/f'{group}.yaml')
    entries = corpus.load_inventory(config)
    entry = entries[index-1]
    assert 1 <= index <= len(entries)
    assert corpus.completion_error(config, entry) is not None, 'Do not overwrite completed output'
    started = time.perf_counter()
    with bounded_psi_memory() as repair:
        directory = corpus.run_dataset(config, entry, n_jobs=1)
    assert corpus.completion_error(config, entry) is None
    meta = json.loads((directory/'meta.json').read_text())
    meta['spectral_memory_repair'] = repair
    corpus._atomic_json(directory/'meta.json', meta)
    order = [s['name'] for s in meta['pyspi']['spis']]
    valid_spis = 0
    with np.load(directory/'spi_mpis.npz') as bank:
        for name in order:
            matrix = bank[name]
            vector = matrix[~np.eye(len(matrix), dtype=bool)]
            valid_spis += bool(np.isfinite(vector).all() and np.linalg.norm(vector-vector.mean()) >= 1e-12)
    status = dict(index=index, group=group, output_dir=str(directory), seconds=time.perf_counter()-started,
                  full_valid_spis=valid_spis, total_spis=len(order),
                  peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                  script_sha256=sha(__file__))
    (root/f'{group}-resource-{index:02d}.json').write_text(json.dumps(status, indent=2)+'\n')
    print(json.dumps(status), flush=True)


def gate(root, group):
    from src import run_external_corpus as corpus
    config = corpus.ExternalCorpusConfig.from_file(root/f'{group}.yaml')
    entries = corpus.load_inventory(config)
    smoke = [int(i) for i in (root/f'{group}-smoke.txt').read_text().split()]
    for index in smoke:
        assert corpus.completion_error(config, entries[index-1]) is None
        status = json.loads((root/f'{group}-resource-{index:02d}.json').read_text())
        assert status['peak_rss_gib'] < 20, 'Stop for memory resizing before parallel continuation'
        assert status['total_spis'] == 289 and status['full_valid_spis'] >= 200, 'Stop for validity inspection'
    print('SMOKE_GATE_PASS', group, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'assemble', 'extract', 'gate'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--hcp-root', type=Path)
    parser.add_argument('--index', type=int)
    parser.add_argument('--group', choices=['m64', 'm100', 'large'])
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.action == 'prepare':
        if tasks()[args.index]['system'] == 'meg':
            prepare_meg(args.root, args.hcp_root, args.index)
        else:
            prepare_synthetic(args.root, args.index)
    elif args.action == 'assemble':
        assemble(args.root)
    elif args.action == 'extract':
        extract(args.root, args.group, args.index)
    else:
        gate(args.root, args.group)
