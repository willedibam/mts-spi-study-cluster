import json
from pathlib import Path

import numpy as np

from research.representation.atlas import extract

CONFIG = Path(__file__).resolve().parents[4] / 'configs/pyspi/test.yaml'


def bank(root):
    rng = np.random.default_rng(0)
    windows = {f'u{i}-w{j}': rng.normal(size=(120, 3 + i)) for i in range(2) for j in range(3)}
    np.savez(root / 'windows.npz', **windows)
    rows = [dict(id=k, unit=k.split('-')[0], M=int(v.shape[1])) for k, v in windows.items()]
    (root / 'manifest.json').write_text(json.dumps(dict(dataset='synthetic', arm='test', rows=rows)))
    return windows


def test_shards_complete_skip_and_audit(tmp_path):
    windows = bank(tmp_path)
    out = tmp_path / 'out'
    assert extract.count(tmp_path, 4) == 2
    assert not extract.audit(tmp_path, out, 4, CONFIG)
    for k in range(2):
        extract.run_shard(tmp_path, out, k, 4, CONFIG)
    assert extract.audit(tmp_path, out, 4, CONFIG)
    with np.load(extract.shard_path(out, 1)) as archive:
        meta = json.loads(str(archive['__meta__']))
        spis = list(archive['__spis__'])
        for w in extract.shard_ids(tmp_path, 1, 4):
            M = windows[w].shape[1]
            assert archive[w].shape == (len(spis), M, M)
    assert meta['identity']['normalise'] and set(meta['windows']) == set(extract.shard_ids(tmp_path, 1, 4))
    mtime = extract.shard_path(out, 0).stat().st_mtime
    extract.run_shard(tmp_path, out, 0, 4, CONFIG)  # complete, so skipped
    assert extract.shard_path(out, 0).stat().st_mtime == mtime


def test_matches_run_pyspi(tmp_path):
    from mtsspi.compute import run_pyspi
    windows = bank(tmp_path)
    w = extract.shard_ids(tmp_path, 0, 1)[0]
    extract.run_shard(tmp_path, tmp_path / 'out', 0, 1, CONFIG)
    direct = run_pyspi(windows[w], config_path=CONFIG, normalise=True, random_seed=extract.seed(w))
    with np.load(extract.shard_path(tmp_path / 'out', 0)) as archive:
        for k, name in enumerate(archive['__spis__']):
            np.testing.assert_allclose(archive[w][k], direct.matrices[str(name)], equal_nan=True)
