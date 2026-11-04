import json

import numpy as np
import pytest

from research.representation.character_axes.prepare import CANON
from research.representation.chbmp_iq.prepare import canonical, first_eyes_closed, prepare


def test_canonical_labels_and_aliases():
    assert canonical('EEG FP1-REF') == 'Fp1'
    assert canonical('T3') == 'T7' and canonical('t6') == 'P8'
    assert canonical('OZ') is None


def test_first_eyes_closed_splits_at_discontinuity():
    events = [dict(onset='0', trial_type='eyes_closed'), dict(onset='40', trial_type='discontinuity'),
              dict(onset='100', trial_type='eyes_open'), dict(onset='110', trial_type='eyes_closed')]
    assert first_eyes_closed(events, 10, 5000) == [(0, 400), (400, 1000)]


def write_edf(path, names, x_uv, fs):
    """Minimal EDF (1 s records, int16) so the test exercises MNE's real reader."""
    ns, n = len(names), x_uv.shape[1] // fs
    field = lambda v, w: str(v).ljust(w)[:w]
    head = (field(0, 8) + field('X', 80) + field('X', 80) + '01.01.20' + '00.00.00' + field(256 * (ns + 1), 8)
            + field('', 44) + field(n, 8) + field(1, 8) + field(ns, 4))
    for w, v in [(16, None), (80, ''), (8, 'uV'), (8, -3276.8), (8, 3276.7), (8, -32768), (8, 32767), (80, ''), (8, fs), (32, '')]:
        head += ''.join(field(name if v is None else v, w) for name in names)
    digital = np.clip(np.round(x_uv * 10), -32768, 32767).astype('<i2')
    body = digital.reshape(ns, n, fs).transpose(1, 0, 2).tobytes()
    path.write_bytes(head.encode('ascii') + body)


def fake_download(root, fs=200, seconds=200):
    pytest.importorskip('mne')
    rng = np.random.default_rng(0)
    names = [f'{c}-REF' for c in CANON] + ['ECG-REF']
    common = rng.normal(scale=10, size=fs * seconds)
    for sid in ('sub-CBM00001', 'sub-CBM00002'):
        eeg = root / 'bids' / sid / 'eeg'
        eeg.mkdir(parents=True)
        x = rng.normal(scale=10, size=(len(names), fs * seconds)) + common
        x[0, 50 * fs:51 * fs] += 400  # one artefact second on Fp1
        x[-1] *= 50                   # ECG is large and must not count as EEG
        write_edf(eeg / f'{sid}_task-protmap_eeg.edf', names, x, fs)
        (eeg / f'{sid}_task-protmap_events.tsv').write_text(
            'onset\tduration\ttrial_type\n0.005\t0\teyes_closed\n150\t0\teyes_open\n160\t0\teyes_closed\n')
        (eeg / f'{sid}_task-protmap_channels.tsv').write_text(
            'name\ttype\n' + ''.join(f'{c}\tEEG\n' for c in CANON) + 'ECG\tECG\n')
    (root / 'cognitive').mkdir()
    (root / 'cognitive' / 'Demographic_data.csv').write_text(
        'Demographic_data,,,,\nCode,Gender,Age,Weight lb,Education Level \nCBM00001,M,30,150,High School\nCBM00002,F,40,120,College School\n')
    (root / 'receipt.json').write_text('{}')


def test_prepare_windows_avoid_artefact_and_later_blocks(tmp_path):
    fake_download(tmp_path)
    out = tmp_path / 'corpus'
    prepare(tmp_path, out, 'canonical', 1000, 6)
    rows = json.loads((out / 'manifest.json').read_text())['rows']
    assert len(rows) == 12 and {r['M'] for r in rows} == {19} and {r['T'] for r in rows} == {1000}
    assert {r['target'] for r in rows} == {0} and {r['age'] for r in rows} == {30.0, 40.0}
    assert not any('wais' in k.lower() or 'iq' in k.lower() for r in rows for k in r)
    for r in rows:  # state A is 0-150 s; the artefact second is 50-51 s
        s = r['start_sample']
        assert s + 1000 <= 150 * 100 and (s + 1000 <= 50 * 100 or s >= 51 * 100)


def test_anchor_nests_shorter_windows(tmp_path):
    fake_download(tmp_path)
    prepare(tmp_path, tmp_path / 'long', 'canonical', 2000, 3)
    prepare(tmp_path, tmp_path / 'short', 'canonical', 500, 3, anchor=2000)
    long = json.loads((tmp_path / 'long' / 'manifest.json').read_text())['rows']
    short = json.loads((tmp_path / 'short' / 'manifest.json').read_text())['rows']
    assert [r['start_sample'] for r in long] == [r['start_sample'] for r in short]
    with np.load(tmp_path / 'long' / 'views.npz') as a, np.load(tmp_path / 'short' / 'views.npz') as b:
        name = long[0]['row_id']
        x, y = a[name][:500], b[name]
        assert np.allclose((x - x.mean(0)) / x.std(0), y)
