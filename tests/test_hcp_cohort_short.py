import hashlib
import json

import numpy as np
import pytest
from scipy.signal import filtfilt, iirnotch
import yaml

from scripts.prepare_hcp_cohort_short import prepare
from scripts.prepare_hcp_short_views import digest


def source_run(tmp_path):
    root = tmp_path/'person_run'
    source = root/'m32-spatial'
    source.mkdir(parents=True)
    (root/'preparation-identity.json').write_text(json.dumps(dict(subject='test', run='6-Wrkmem')))
    rng = np.random.default_rng(731)
    entries, arrays = [], {}
    for block in [1, 2]:
        for layout in ['coverage_a', 'coverage_b']:
            name = f'test_block{block}_{layout}'
            # Different lengths test central offsets without concatenating blocks.
            length = 4100 + block
            arrays[name] = rng.normal(size=(length, 32)) + block
            entries.append(dict(name=name, block=block, layout=layout,
                                memory=0 if block == 1 else 2, image='Face', M=32, T=length))
    np.savez_compressed(source/'views.npz', **arrays, __dataset_names__=np.array(list(arrays)))
    archive_sha = digest(source/'views.npz')
    (source/'preparation.json').write_text(json.dumps(dict(entries=entries, archive_sha256=archive_sha)))
    (source/'external.yaml').write_text(yaml.safe_dump(dict(
        name='parent', source=dict(format='named-npz-v1', archive=str(source/'views.npz'), sha256=archive_sha),
        base_output_dir='unused', pyspi_config='configs/pyspi/benchmarked_p90.yaml',
        normalise=False, random_seed=260914311)))
    audit = tmp_path/'audit.json'
    audit.write_text(json.dumps(dict(status='verified', errors=[], runs=[dict(
        subject='test', run='6-Wrkmem', views=4, archive_sha256=archive_sha,
        preparation_sha256=digest(source/'preparation.json'))])))
    return root, audit, arrays


@pytest.mark.parametrize('length', [2000, 4000])
def test_contiguous_scout_equivalence_and_resume(tmp_path, length):
    root, audit, parents = source_run(tmp_path)
    result = prepare(root, audit, length)
    assert result['status'] == 'prepared' and result['views'] == 4
    output = root/f'm32-T{length}'
    manifest = json.loads((output/'manifest.json').read_text())
    b, a = iirnotch(60, 30, fs=250)
    with np.load(output/'views.npz', allow_pickle=False) as bank:
        for row in manifest['rows']:
            parent = parents[row['source_name']]
            begin = (len(parent)-length)//2
            expected = filtfilt(b, a, parent, axis=0)[begin:begin+length].copy()
            expected -= expected.mean(0)
            expected /= expected.std(0)
            np.testing.assert_array_equal(bank[row['name']], expected)
            assert row['source_start'] == begin and row['source_stop'] == begin+length
            assert row['array_sha256'] == hashlib.sha256(expected.tobytes()).hexdigest()
    original = digest(output/'views.npz')
    assert prepare(root, audit, length)['status'] == 'verified_existing'
    assert digest(output/'views.npz') == original
    # Never silently reuse a changed runner configuration.
    with (output/'external.yaml').open('a') as stream:
        stream.write('\n# changed\n')
    with pytest.raises(ValueError, match='configuration changed'):
        prepare(root, audit, length)


def test_reject_unaudited_source_and_out_of_scope_duration(tmp_path):
    root, audit, _ = source_run(tmp_path)
    with pytest.raises(ValueError, match='T2000 or T4000'):
        prepare(root, audit, 6126)
    with (root/'m32-spatial/preparation.json').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='differs from audited'):
        prepare(root, audit, 2000)
    assert not (root/'m32-T2000').exists()
