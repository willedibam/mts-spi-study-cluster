"""Freeze HCP M32 central 16/8-second views and compact preprocessing QC."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import filtfilt, iirnotch, welch
import yaml


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def center_window(data, length):
    begin = (len(data) - length) // 2
    assert begin >= 0
    x = np.array(data[begin:begin+length], dtype=np.float64, copy=True)
    x -= x.mean(axis=0)
    scale = x.std(axis=0)
    assert np.all(scale > 0)
    x /= scale
    assert x.shape == (length, 32) and np.isfinite(x).all()
    return x, begin


def main(root):
    source = root/'m32-spatial'
    output = root/'m32-short'
    output.mkdir(exist_ok=True)
    assert not (output/'views.npz').exists(), 'Do not overwrite a frozen corpus'
    prep = json.loads((source/'preparation.json').read_text())
    assert digest(source/'views.npz') == prep['archive_sha256']
    bank = np.load(source/'views.npz', allow_pickle=False)
    b, a = iirnotch(60, 30, fs=250)
    corrected, qc = {}, []
    for entry in prep['entries']:
        x = bank[entry['name']]
        y = filtfilt(b, a, x, axis=0)
        f, before = welch(x, fs=250, nperseg=1000, axis=0)
        _, after = welch(y, fs=250, nperseg=1000, axis=0)
        band, line = (f >= 1) & (f <= 100), (f >= 59) & (f <= 61)
        corrected[entry['name']] = y
        qc.append(dict(name=entry['name'],
                       line_power_fraction_before=float(np.median(before[line].sum(axis=0)/before[band].sum(axis=0))),
                       line_power_fraction_after=float(np.median(after[line].sum(axis=0)/after[band].sum(axis=0))),
                       power_80_100_fraction=float(np.median(after[(f >= 80)&(f <= 100)].sum(axis=0)/after[band].sum(axis=0)))))
    arrays, rows = {}, []
    # Primary 16-second view first; chronological ordering permits a simple two-input smoke.
    for length in [4000, 2000]:
        for layout in ['coverage_a', 'coverage_b']:
            for entry in prep['entries']:
                if entry['layout'] != layout:
                    continue
                name = entry['name'] + f'_T{length}'
                x, start = center_window(corrected[entry['name']], length)
                arrays[name] = x
                rows.append(dict(entry, name=name, T=length, source_name=entry['name'], source_start=start,
                                 start_after_block_onset_seconds=start/250,
                                 duration_seconds=length/250, array_sha256=hashlib.sha256(x.tobytes()).hexdigest()))
    names = list(arrays)
    labels = [json.dumps([f"memory{r['memory']}",f"image{r['image']}",r['layout'],f"T{r['T']}"]) for r in rows]
    arrays.update(__dataset_names__=np.array(names),__labels_json__=np.array(labels),
                  __shapes__=np.array([arrays[n].shape for n in names]),__axis_order__=np.array(['observation','process']))
    np.savez_compressed(output/'views.npz', **arrays)
    config = yaml.safe_load((source/'external.yaml').read_text())
    config['name']='hcp-m32-short-260914'
    config['source'].update(archive=str(output/'views.npz'),sha256=digest(output/'views.npz'))
    config['base_output_dir']=str(output/'pyspi')
    (output/'external.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    manifest=dict(status='prepared',rows=rows,archive_sha256=config['source']['sha256'],
                  source_preparation_sha256=digest(source/'preparation.json'),script_sha256=digest(Path(__file__)),
                  preprocessing='Reuse reference/ICA corrected, 1-100 Hz filtered, 250 Hz parent. Fixed 60 Hz Q30 zero-phase notch before central cropping; restandardize each channel in each window.',
                  scope='16 blocks, one person/run; 2 layouts x 2 durations = 64 dependent views. No new sampling rate or M64 arm.',
                  qc=qc)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'smoke-indices.txt').write_text('1\n2\n')
    (output/'production-indices.txt').write_text(''.join(f'{i}\n' for i in range(3,65)))
    print(json.dumps(dict(inputs=len(rows),shapes=sorted(set((r['M'],r['T']) for r in rows)),
                          median_line_before=float(np.median([r['line_power_fraction_before'] for r in qc])),
                          median_line_after=float(np.median([r['line_power_fraction_after'] for r in qc]))),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
