"""Prevent a larger bank from silently reusing the old 64-row development limit."""
import json
import numpy as np
from scripts import analyze_band_swap as analysis


def test_development_rows_follow_manifest_not_legacy_count(tmp_path, monkeypatch):
    rows=[dict(row_id=str(i),block=i//2,role='development' if i<256 else 'evaluation',
               development_part='train' if i<192 else 'validation' if i<256 else 'held') for i in range(384)]
    (tmp_path/'manifest.json').write_text(json.dumps(dict(rows=rows)))
    monkeypatch.setattr(analysis,'DATA',tmp_path)
    development=analysis.get_rows('development')
    assert len(development)==256
    assert development.development_part.eq('train').sum()==192
    assert not development.development_part.eq('held').any()
    final=analysis.get_rows('final')
    assert len(final)==384
    assert final.role.eq('development').sum()==256
    assert final.role.ne('development').sum()==128
    assert set(development.block).isdisjoint(final.loc[final.role.eq('evaluation'),'block'])


def test_legacy_split_is_unchanged(tmp_path, monkeypatch):
    rows=[dict(row_id=str(i),block=i//2,role='development' if i<64 else 'evaluation',
               development_part='train' if i<48 else 'validation' if i<64 else 'held') for i in range(128)]
    (tmp_path/'manifest.json').write_text(json.dumps(dict(rows=rows)))
    monkeypatch.setattr(analysis,'DATA',tmp_path)
    for stage in ['development','final']:
        selected=analysis.get_rows(stage)
        assert len(selected)==(64 if stage=='development' else 128)
        actual=(selected.development_part.eq('train') if stage=='development' else selected.role.eq('development'))
        np.testing.assert_array_equal(actual,selected.block.lt(24 if stage=='development' else 32))
