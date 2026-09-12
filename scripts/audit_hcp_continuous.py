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


def main(root):
    verified = json.loads((root / 'download-verification.json').read_text())
    assert verified['status'] == 'complete'
    files = {Path(row['key']).name: Path(row['path']) for row in verified['files']}
    for row in verified['files']:
        assert Path(row['path']).stat().st_size == row['size']
    raw = mne.io.read_raw_bti(files['c,rfDC'], config_fname=files['config'], head_shape_fname=None,
                            convert=False, rename_channels=False, sort_by_ch_name=False,
                            preload=False, verbose='ERROR')
    metadata = root / 'metadata'
    trl_path = metadata / '105923_MEG_6-Wrkmem_tmegpreproc_trialinfo.mat'
    trl = loadmat(trl_path, simplify_cells=True)['trlInfo']
    a = trl['lockTrl'][list(trl['lockNames']).index('TIM')]
    task = a[np.isin(a[:, 3], [1, 2]) & np.isin(a[:, 4], [1, 2]) & np.isin(a[:, 8], np.arange(1, 11))]
    task = task[np.argsort(task[:, 6])]
    assert len(task) == 160
    sfreq = float(raw.info['sfreq'])
    bad_text = (metadata / '105923_MEG_6-Wrkmem_baddata_badsegments.txt').read_text()
    block = re.search(r'badsegment\.all\s*=\s*\[([^\]]*)\]', bad_text).group(1)
    segments = np.asarray([float(x) for x in re.findall(r'\d+', block)]).reshape(-1, 2).astype(int)
    # HCP sample indices are one-based and inclusive.
    segments[:, 0] -= 1
    bad_channels = set(re.findall(r'A\d+', (metadata / '105923_MEG_6-Wrkmem_baddata_badchannels.txt').read_text()))
    meg = mne.pick_types(raw.info, meg=True, ref_meg=False, exclude=[])
    good = [int(ix) for ix in meg if raw.ch_names[ix] not in bad_channels]
    picks = np.asarray(good)[np.linspace(0, len(good)-1, min(32, len(good)), dtype=int)]
    blocks = []
    for number in np.unique(task[:, 1]):
        rows = task[task[:, 1] == number]
        assert len(rows) == 10 and len(np.unique(rows[:, [3, 4]], axis=0)) == 1
        start, stop = int(rows[:, 6].min())-1, int(rows[:, 7].max())
        assert 0 <= start < stop <= raw.n_times
        center = (start+stop)//2
        sample = raw.get_data(picks=picks, start=center-int(sfreq), stop=center+int(sfreq), verbose='ERROR')
        finite = bool(np.isfinite(sample).all())
        varying = bool(np.all(sample.std(axis=1) > 0))
        assert finite and varying
        overlap = int(sum(max(0, min(stop, end)-max(start, begin)) for begin, end in segments))
        blocks.append(dict(block=int(number), image_type=int(rows[0, 3]), memory_type=int(rows[0, 4]),
                           trials=len(rows), start_sample=start, stop_sample=stop, duration_seconds=(stop-start)/sfreq,
                           bad_segment_overlap_seconds=overlap/sfreq, sample_finite=finite, sample_nonconstant=varying))
    frame = pd.read_csv(files['105923_MEG_Wrkmem_run1.tab'], sep='\t')
    onset = pd.to_numeric(frame['Stim.OnsetTime'], errors='coerce')
    ep = frame[onset.notna() & (onset > 0)].copy()
    ep['onset_ms'] = onset.loc[ep.index]
    ep = ep.sort_values('onset_ms')
    eprime = dict(stimulus_rows=len(ep), block_types=sorted(ep['BlockType'].dropna().astype(str).unique()),
                  stimulus_types=sorted(ep['StimType'].dropna().astype(str).unique()))
    if len(ep) == len(task):
        x = ep['onset_ms'].to_numpy(float)
        y = (task[:, 6]-1)/sfreq
        slope, intercept = np.polyfit(x/1000, y, 1)
        eprime.update(clock_slope=float(slope), offset_seconds=float(intercept),
                      max_affine_timing_residual_seconds=float(np.max(np.abs(y-(slope*x/1000+intercept)))))
    counts = {f'image{image}_memory{memory}':sum(b['image_type']==image and b['memory_type']==memory for b in blocks)
              for image in [1, 2] for memory in [1, 2]}
    assert set(counts.values()) == {4}
    result = dict(status='readability_and_balanced_block_gate_passed', mne_version=mne.__version__,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_verification_sha256=hashlib.sha256((root/'download-verification.json').read_bytes()).hexdigest(),
                  sfreq=sfreq, samples=int(raw.n_times), recording_seconds=raw.n_times/sfreq,
                  channels=len(raw.ch_names), meg_channels=len(meg), marked_bad_meg_channels=sorted(bad_channels),
                  good_meg_channels=len(good), sampled_channels=[raw.ch_names[ix] for ix in picks],
                  blocks=blocks, condition_block_counts=counts, eprime=eprime,
                  preprocessing_complete=False, learning_or_pyspi_started=False,
                  remaining='Verify condition labels against E-Prime, timing residuals, ICA/reference correction and spatial mixing before learning; audit cohort and family grouping.')
    (root / 'continuous-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['blocks','sampled_channels']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
