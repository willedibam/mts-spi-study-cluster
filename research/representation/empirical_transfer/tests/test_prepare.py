import json

import numpy as np

from research.representation.empirical_transfer.prepare import (
    aligned_networks, assign_fmri_roles, prepare_fmri_array, write_corpus,
)
from mtsspi.corpus import ArchiveCorpus


def test_lookup_is_aligned_by_native_identity_not_csv_row_order():
    native = 'name, x, y, z\n' + '\n'.join(f'p{i},0,0,0' for i in range(114))
    lookup = 'Index,Full_Region_Name,Yeo_17_Networks_Bilateral\n' + '\n'.join(
        f'{i+1},p{i},g{i%17:02}' for i in reversed(range(114)))
    names, groups = aligned_networks(native, lookup)
    assert len(names) == 17
    np.testing.assert_array_equal(groups[0], np.arange(0, 114, 17))


def test_partial_gaps_are_rejected_not_stitched():
    x = np.random.default_rng(4).normal(size=(1000, 114))
    x[500] = np.nan
    assert prepare_fmri_array(x, [np.arange(114)])[1] == 'partial_missing'


def test_crop_and_normalization_have_no_label_dependency():
    x = np.random.default_rng(4).normal(size=(1000, 114))
    regions = [np.arange(57), np.arange(57, 114)]
    result, reason = prepare_fmri_array(x, regions)
    expected = np.column_stack([x[26:973, ix].mean(1) for ix in regions])
    expected = (expected - expected.mean(0)) / expected.std(0)
    assert reason is None
    np.testing.assert_allclose(result, expected)


def test_person_split_and_missing_pair_eligibility():
    rows = [dict(row_id=f'{p}-{s}-{y}', group=f'p{p}', target=y, run_set=s)
            for p in range(10) for s in ['early', 'late'] for y in [0, 1]]
    rows = [r for r in rows if r['row_id'] != '9-late-1']
    kept, split = assign_fmri_roles(rows)
    assert split['excluded_incomplete_people'] == ['p9']
    source = {r['group'] for r in kept if r['role'] == 'source'}
    target = {r['group'] for r in kept if r['role'].startswith('confirmation')}
    assert source.isdisjoint(target)
    assert len(source | target) == 9
    assert (kept, split) == assign_fmri_roles(rows)


def test_prepared_corpus_roundtrip_preserves_waveform_identity(tmp_path):
    original = np.random.default_rng(9).normal(size=(120, 3))
    rows = [dict(row_id='one', group='animal', target=0, role='source')]
    write_corpus(tmp_path, 'test', rows, {'one': original}, {})
    corpus = ArchiveCorpus.from_file(tmp_path / 'corpus.yaml')
    assert corpus.validate()['datasets'] == 1
    result, metadata = corpus.load(corpus.entries()[0])
    np.testing.assert_array_equal(result, original)
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert metadata['member_sha256'] == manifest['rows'][0]['array_sha256']
