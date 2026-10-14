"""Small serial/partial/parallel checkpoint gate for the frozen CCEP pyspi.

Replays fd0de71 tests/test_parallel.py's resume test without assuming its tests
were included in the cluster's computation-only snapshot. Config copied from
that exact revision; all temporary checkpoints live in the allocation jobfs.
"""
import argparse
from pathlib import Path
import numpy as np
from pyspi.calculator import Calculator


def check(root):
    root = Path(root)
    assert not root.exists(), 'Use a fresh smoke directory'
    data = np.random.default_rng(0).standard_normal((5, 300))
    config = Path(__file__).parent / 'configs/checkpoint-smoke.yaml'
    def calculator():
        return Calculator(dataset=data, config=str(config), zscore=False, verbose=False)
    first = calculator()
    first.compute(n_jobs=1, checkpoint_dir=root, progress=False)
    saved = sorted(root.glob('*.npy'))
    assert len(saved) == len(first.spis) and len(saved) >= 10
    assert np.isfinite(first.table.to_numpy()).any()
    for path in saved[:len(saved)//2]:
        path.unlink()
    second = calculator()
    second.compute(n_jobs=2, checkpoint_dir=root, resume=True, mp_context='spawn', progress=False)
    assert list(first.table.columns) == list(second.table.columns)
    np.testing.assert_allclose(first.table.to_numpy(), second.table.to_numpy(),
                               rtol=1e-10, atol=1e-12, equal_nan=True)
    print(f'Checkpoint parity passed: {len(saved)} SPIs; partial serial → parallel resume.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    check(parser.parse_args().root)
