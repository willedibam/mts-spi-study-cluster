"""Readability, label, timing and QC audit of the single HCP feasibility recording."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat
from src.hcp_run_audit import align_eprime


def main(root, subject='105923', run='6-Wrkmem'):
    verified = json.loads((root / 'download-verification.json').read_text())
    assert verified['status'] == 'complete'
    files = {Path(row['key']).name: Path(row['path']) for row in verified['files']}
    for row in verified['files']:
        assert Path(row['path']).stat().st_size == row['size']
    raw = mne.io.read_raw_bti(files['c,rfDC'], config_fname=files['config'], head_shape_fname=None,
                            convert=False, rename_channels=False, sort_by_ch_name=False,
                            preload=False, verbose='ERROR')
    metadata = root / 'metadata'
    trl_path = metadata / f'{subject}_MEG_{run}_tmegpreproc_trialinfo.mat'
    trl = loadmat(trl_path, simplify_cells=True)['trlInfo']
    a = trl['lockTrl'][list(trl['lockNames']).index('TIM')]
    task = a[np.isin(a[:, 3], [1, 2]) & np.isin(a[:, 4], [1, 2]) & np.isin(a[:, 8], np.arange(1, 11))]
    task = task[np.argsort(task[:, 6])]
    assert len(task) > 0
    sfreq = float(raw.info['sfreq'])
    bad_text = (metadata / f'{subject}_MEG_{run}_baddata_badsegments.txt').read_text()
    block = re.search(r'badsegment\.all\s*=\s*\[([^\]]*)\]', bad_text).group(1)
    segments = np.asarray([float(x) for x in re.findall(r'\d+', block)]).reshape(-1, 2).astype(int)
    # HCP sample indices are one-based and inclusive.
    segments[:, 0] -= 1
    bad_channels = set(re.findall(r'A\d+', (metadata / f'{subject}_MEG_{run}_baddata_badchannels.txt').read_text()))
    meg = mne.pick_types(raw.info, meg=True, ref_meg=False, exclude=[])
    good = [int(ix) for ix in meg if raw.ch_names[ix] not in bad_channels]
    picks = np.asarray(good)[np.linspace(0, len(good)-1, min(32, len(good)), dtype=int)]
    blocks, excluded = [], []
    for number in np.unique(task[:, 1]):
        rows = task[task[:, 1] == number]
        complete = len(rows) == 10 and set(rows[:, 8]) == set(range(1, 11))
        consistent = len(np.unique(rows[:, [3, 4]], axis=0)) == 1
        start, stop = int(rows[:, 6].min())-1, int(rows[:, 7].max())
        assert 0 <= start < stop <= raw.n_times
        overlap = int(sum(max(0, min(stop, end)-max(start, begin)) for begin, end in segments))
        if not complete or not consistent or overlap:
            excluded.append(dict(block=int(number), trials=len(rows), complete=complete,
                                 consistent_condition=consistent, bad_overlap_samples=overlap))
            continue
        center = (start+stop)//2
        sample = raw.get_data(picks=picks, start=center-int(sfreq), stop=center+int(sfreq), verbose='ERROR')
        finite = bool(np.isfinite(sample).all())
        varying = bool(np.all(sample.std(axis=1) > 0))
        if not finite or not varying:
            excluded.append(dict(block=int(number), sample_finite=finite, sample_nonconstant=varying))
            continue
        blocks.append(dict(block=int(number), image_type=int(rows[0, 3]), memory_type=int(rows[0, 4]),
                           trials=len(rows), start_sample=start, stop_sample=stop, duration_seconds=(stop-start)/sfreq,
                           bad_segment_overlap_seconds=overlap/sfreq, sample_finite=finite, sample_nonconstant=varying))
    eprime_files = [path for name, path in files.items() if name.endswith('.tab')]
    assert len(eprime_files) == 1
    frame = pd.read_csv(eprime_files[0], sep='\t')
    eprime = align_eprime(task, frame, sfreq)
    counts = {f'image{image}_memory{memory}':sum(b['image_type']==image and b['memory_type']==memory for b in blocks)
              for image in [1, 2] for memory in [1, 2]}
    assert all(count > 0 for count in counts.values()), 'A condition is absent after run-level QC'
    result = dict(status='readability_and_condition_coverage_gate_passed', mne_version=mne.__version__,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_verification_sha256=hashlib.sha256((root/'download-verification.json').read_bytes()).hexdigest(),
                  sfreq=sfreq, samples=int(raw.n_times), recording_seconds=raw.n_times/sfreq,
                  channels=len(raw.ch_names), meg_channels=len(meg), marked_bad_meg_channels=sorted(bad_channels),
                  good_meg_channels=len(good), sampled_channels=[raw.ch_names[ix] for ix in picks],
                  subject=subject, run=run, blocks=blocks, excluded_blocks=excluded,
                  condition_block_counts=counts, eprime=eprime,
                  preprocessing_complete=False, learning_or_pyspi_started=False,
                  remaining='Verify per-run ICA/reference correction and spatial preparation; finish cohort eligibility and family grouping before learning.')
    (root / 'continuous-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['blocks','sampled_channels']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--subject', default='105923')
    parser.add_argument('--run', choices=['6-Wrkmem', '7-Wrkmem'], default='6-Wrkmem')
    args = parser.parse_args()
    main(args.root, args.subject, args.run)
