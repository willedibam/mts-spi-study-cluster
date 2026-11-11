import io
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.io import savemat

from research.representation.atlas import extract, prepare


def mat_bytes(**variables):
    buffer = io.BytesIO()
    savemat(buffer, variables)
    return buffer.getvalue()


def synthetic_archive(path, channels=6, flat_channel=None, seed=0):
    """NeuroTycho layout: <prefix>/SessionN/{Condition.mat, ECoG_chK.mat}; 1 kHz; one-based ConditionIndex."""
    rng = np.random.default_rng(seed)
    prefix = 'X_Anesthesia+and+Sleep_Test_mat_ECoG128'
    sessions = {'Session1': (150, [('AwakeEyesClosed-Start', 1), ('AwakeEyesClosed-End', 70),
                                   ('AwakeEyesOpened-Start', 75), ('AwakeEyesOpened-End', 140)]),
                'Session2': (130, [('AnestheticInjection', 2), ('Anesthetized-Start', 5), ('Anesthetized-End', 120)])}
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr(f'{prefix}/Info-Test.txt', 'synthetic')
        for session, (seconds, events) in sessions.items():
            labels = np.array([e[0] for e in events], dtype=object)
            index = np.array([e[1] * 1000 + 1 for e in events], dtype=float)
            z.writestr(f'{prefix}/{session}/Condition.mat', mat_bytes(ConditionLabel=labels, ConditionIndex=index,
                                                                       ConditionTime=(index - 1) / 1000))
            n = seconds * 1000
            common = np.sin(2 * np.pi * 3 * np.arange(n) / 1000)
            for ch in range(1, channels + 1):
                x = 50 * rng.standard_normal(n) + 20 * common
                if ch == flat_channel:
                    x = np.zeros(n)
                z.writestr(f'{prefix}/{session}/ECoG_ch{ch}.mat', mat_bytes(**{f'ECoGData_ch{ch}': x[None, :]}))


def test_filters_keep_band_and_remove_mains():
    t = np.arange(20000) / 1000
    # 40 Hz is a soft ceiling (45 Hz about -18 dB); mains is removed and 60 Hz, which would alias to 40, is gone.
    for f, lo, hi in ((10, 0.95, 1.05), (45, 0, 0.2), (50, 0, 0.01), (60, 0, 0.01)):
        y = prepare.continuous(np.sin(2 * np.pi * f * t)[:, None], 1000, 50.0)[200:-200, 0]
        assert lo <= np.sqrt(2) * y.std() <= hi, (f, y.std())
    assert prepare.short_epoch(np.random.default_rng(0).standard_normal((2250, 4)), 1000).shape == (200, 4)


def test_reference_keeps_full_rank():
    x = np.random.default_rng(1).standard_normal((1000, 8))
    w = prepare.reference(x, 'car')
    assert w.shape == (1000, 7) and prepare.conditioning(w)[0] == 7
    assert np.linalg.matrix_rank(np.cov(x - x.mean(axis=1, keepdims=True), rowvar=False)) == 7  # why one is dropped


def test_neurotycho_archive_windows(tmp_path):
    path = tmp_path / 'synthetic.zip'
    synthetic_archive(path, flat_channel=3)
    long_rows, long_arrays, short_rows, short_arrays, meta = prepare.nt_archive(path)
    assert meta['bad_channels'] == [3] and meta['dropped_after_car'] == 6 and meta['kept_channels'] == [1, 2, 4, 5]
    counts = {s: sum(r['state'] == s for r in long_rows) for s in ('awake_ec', 'awake_eo', 'drug')}
    assert counts == dict(awake_ec=5, awake_eo=5, drug=10)
    assert all(a.shape == (1000, 4) for a in long_arrays.values())
    assert all(a.shape == (200, 4) for a in short_arrays.values())
    assert [r['id'] for r in long_rows] == [r['id'] for r in short_rows]
    assert all(r['rank'] == 4 for r in long_rows + short_rows)
    starts = sorted(r['start_s'] for r in long_rows if r['state'] == 'drug')
    assert np.all(np.diff(starts) >= prepare.LONG)  # no overlap


def test_bank_runs_through_extract(tmp_path):
    root = tmp_path / 'atlas'
    raw = root / 'neurotycho' / 'raw'
    raw.mkdir(parents=True)
    synthetic_archive(raw / 'a.zip', seed=1)
    (root / 'neurotycho' / 'acquisition.json').write_text(json.dumps(dict(files=[
        dict(path='a.zip', animal='Test', agent='KT', date='2012-01-01')])))
    prepare.neurotycho(root, tmp_path / 'banks', workers=1)
    bank = tmp_path / 'banks' / 'neurotycho-long'
    manifest = json.loads((bank / 'manifest.json').read_text())
    assert manifest['T'] == 1000 and len(manifest['rows']) == 20 and manifest['rows'][0]['agent'] == 'KT'
    small = tmp_path / 'small'
    prepare.subset(bank, [manifest['rows'][0]['id'], manifest['rows'][-1]['id']], small)
    config = Path(__file__).resolve().parents[4] / 'configs' / 'pyspi' / 'test.yaml'
    extract.run_shard(small, tmp_path / 'out', 0, 2, config)
    assert extract.audit(small, tmp_path / 'out', 2, config)
