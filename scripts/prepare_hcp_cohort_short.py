"""Build one verified cohort run's contiguous inputs at an explicitly chosen length.

This command does not select a duration, assign a split, or run pyspi/learning.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.signal import filtfilt, iirnotch
import yaml

from scripts.prepare_hcp_short_views import center_window, digest


def prepare(root, audit_path, length):
    if length not in (2000, 4000):
        raise ValueError('Choose the declared T2000 or T4000 window explicitly')
    root = root.resolve()
    audit = json.loads(audit_path.read_text())
    if audit['status'] != 'verified' or audit['errors']:
        raise ValueError('Cohort preparation audit has not passed')
    identity = json.loads((root/'preparation-identity.json').read_text())
    matched = [r for r in audit['runs']
               if r['subject'] == identity['subject'] and r['run'] == identity['run']]
    if len(matched) != 1:
        raise ValueError('Expected exactly one audited participant/run')
    source = root/'m32-spatial'
    prepared = json.loads((source/'preparation.json').read_text())
    if (digest(source/'preparation.json') != matched[0]['preparation_sha256']
            or digest(source/'views.npz') != matched[0]['archive_sha256']
            or prepared['archive_sha256'] != matched[0]['archive_sha256']):
        raise ValueError('Prepared source differs from audited source')
    contract = dict(subject=identity['subject'], run=identity['run'], M=32, T=length,
                    sampling_hz=250, layouts=['coverage_a', 'coverage_b'],
                    source_preparation_sha256=matched[0]['preparation_sha256'],
                    source_archive_sha256=matched[0]['archive_sha256'],
                    cohort_audit_sha256=digest(audit_path),
                    script_sha256=digest(Path(__file__)),
                    crop_helper_sha256=digest(Path(__file__).with_name('prepare_hcp_short_views.py')))
    output = root/f'm32-T{length}'
    if output.exists():
        manifest = json.loads((output/'manifest.json').read_text())
        if manifest['contract'] != contract or digest(output/'views.npz') != manifest['archive_sha256']:
            raise ValueError('Preserve existing inputs with a different identity or digest')
        if digest(output/'external.yaml') != manifest['external_config_sha256']:
            raise ValueError('Existing external configuration changed')
        return dict(status='verified_existing', output=str(output), views=len(manifest['rows']))
    b, a = iirnotch(60, 30, fs=250)
    arrays, rows = {}, []
    with np.load(source/'views.npz', allow_pickle=False) as bank:
        if bank['__dataset_names__'].tolist() != [r['name'] for r in prepared['entries']]:
            raise ValueError('Source dataset order differs from preparation manifest')
        for layout in contract['layouts']:
            for entry in prepared['entries']:
                if entry['layout'] != layout:
                    continue
                parent = bank[entry['name']]
                if parent.shape != (entry['T'], 32) or not np.isfinite(parent).all():
                    raise ValueError('Invalid parent shape or values')
                # Identical operation order to the frozen scout: notch the complete
                # parent first, then crop centrally and standardize each channel.
                x, begin = center_window(filtfilt(b, a, parent, axis=0), length)
                name = entry['name'] + f'_T{length}'
                arrays[name] = x
                rows.append(dict(entry, name=name, T=length, subject=identity['subject'],
                                 run=identity['run'], source_name=entry['name'],
                                 source_start=begin, source_stop=begin+length,
                                 start_after_block_onset_seconds=begin/250,
                                 duration_seconds=length/250,
                                 array_sha256=hashlib.sha256(x.tobytes()).hexdigest()))
    if len(rows) != matched[0]['views'] or len(arrays) != len(rows):
        raise ValueError('Missing, duplicate or unexpected cohort views')
    names = list(arrays)
    arrays.update(__dataset_names__=np.array(names),
                  __labels_json__=np.array([json.dumps([f"memory{r['memory']}", f"image{r['image']}",
                                                         r['layout'], f"T{length}"]) for r in rows]),
                  __shapes__=np.array([arrays[n].shape for n in names]),
                  __axis_order__=np.array(['observation', 'process']))
    # Publish the archive/config/manifest together; never leave a half-ready corpus.
    with tempfile.TemporaryDirectory(prefix=f'.T{length}-', dir=root) as temporary:
        stage = Path(temporary)
        np.savez_compressed(stage/'views.npz', **arrays)
        config = yaml.safe_load((source/'external.yaml').read_text())
        if config['pyspi_config'] != 'configs/pyspi/benchmarked_p90.yaml' or config['normalise']:
            raise ValueError('Unexpected catalogue or normalization contract')
        config['name'] = f"hcp-{identity['subject']}-{identity['run']}-T{length}"
        config['source'].update(archive=str(output/'views.npz'), sha256=digest(stage/'views.npz'))
        config['base_output_dir'] = str(output/'pyspi')
        (stage/'external.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        manifest = dict(contract=contract, rows=rows, archive_sha256=config['source']['sha256'],
                        external_config_sha256=digest(stage/'external.yaml'),
                        preprocessing='Fixed 60 Hz Q30 zero-phase notch on verified corrected 250 Hz parent; central contiguous crop; per-channel standardization.',
                        scope='Two sensor layouts of the same blocks are dependent views; no concatenation, split assignment or prediction.')
        (stage/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        stage.rename(output)
    return dict(status='prepared', output=str(output), views=len(rows), M=32, T=length)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='One cohort-prepared/subject_run directory')
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--length', type=int, choices=[2000, 4000], required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.audit, args.length), indent=2))
