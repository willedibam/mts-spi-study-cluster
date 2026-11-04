import json

import numpy as np
import pytest

from research.representation.character_axes.prepare import CANON
from research.representation.chbmp_iq.prepare import canonical, channel_index, inspect, prepare, subtract


def test_canonical_labels_and_aliases():
    assert canonical('EEG FP1-REF') == 'Fp1'
    assert canonical('T3') == 'T7' and canonical('t6') == 'P8'
    assert canonical('OZ') is None
    index, missing = channel_index(['OZ'] + [c.upper() for c in CANON])
    assert missing == [] and index[0] == 1


def test_subtract_removes_artifacts():
    assert subtract([(0, 100)], [(10, 20), (90, 120)]) == [(0, 10), (20, 90)]


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
    names = [c.upper() for c in CANON] + ['OZ']
    for sid in ('sub-CBM00001', 'sub-CBM00002'):
        eeg = root / 'bids' / sid / 'eeg'
        eeg.mkdir(parents=True)
        write_edf(eeg / f'{sid}_task-protmap_eeg.edf', names, rng.normal(scale=20, size=(len(names), fs * seconds)), fs)
        (eeg / f'{sid}_task-protmap_events.tsv').write_text(
            'onset\tduration\ttrial_type\n0\t150\tA\n40\t5\tartifact\n150\t50\tB\n')
        (eeg / f'{sid}_task-protmap_eeg.json').write_text(json.dumps(dict(EEGReference='linked ears')))
    (root / 'bids' / 'participants.tsv').write_text('participant_id\tage\tsex\nsub-CBM00001\t30\tM\nsub-CBM00002\t40\tF\n')
    (root / 'receipt.json').write_text('{}')


def test_inspect_and_prepare_without_iq(tmp_path):
    fake_download(tmp_path)
    inspect(tmp_path, tmp_path / 'inspect.json')
    summary = json.loads((tmp_path / 'inspect.json').read_text())['summary']
    assert summary['recordings'] == 2 and summary['missing_any_canonical'] == 0
    assert summary['event_labels_by_recordings'] == {'A': 2, 'artifact': 2, 'B': 2}
    out = tmp_path / 'corpus'
    prepare(tmp_path, out, {'A'}, {'artifact'}, 'trial_type')
    manifest = json.loads((out / 'manifest.json').read_text())
    rows = manifest['rows']
    assert len(rows) == 12 and {r['M'] for r in rows} == {19} and {r['T'] for r in rows} == {2000}
    assert {r['target'] for r in rows} == {0} and not any('wais' in k.lower() or 'iq' in k.lower() for r in rows for k in r)
    # State A is 0-150 s minus the 40-45 s artefact: no window may overlap it or state B.
    for r in rows:
        s = r['start_sample']
        assert s + 2000 <= 150 * 100 and (s + 2000 <= 40 * 100 or s >= 45 * 100)
