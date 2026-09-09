"""Run the bounded 28-case temporal-order scout from overnight-260910.md.

Reuse only matching complete archives; preserve the original N32/gamma.8 pilot.
This is exploratory physics, not SPI extraction or a confirmation experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    script = Path(__file__).with_name('scout_stuart_landau_streaming.py').resolve()
    code_hash = hashlib.sha256(script.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for n, gamma, seed in itertools.product((32, 800), (.70, .74, .75, .80, .90, 1., 1.2), (910001, 910002)):
        expected = dict(N=n, gamma=gamma, seed=seed, coupling=.8, carrier=2., dt=.02,
                        sample_dt=.1, burn=200., samples=8000)
        filename = f'N{n}-gamma{gamma:.2f}-seed{seed}.npz'
        if (n, gamma, seed) == (32, .8, 910001):
            filename = 'local-pilot-N32-gamma0p8.npz'
        output = args.output_dir / filename
        if output.exists():
            with np.load(output, allow_pickle=False) as data:
                meta = json.loads(str(data['metadata_json']))
                if any(meta[k] != value for k, value in expected.items()) or meta['script_sha256'] != code_hash:
                    raise ValueError(f'Incompatible existing archive: {output}')
                if data['Z'].shape != (8000,) or not np.isfinite(data['Z']).all():
                    raise ValueError(f'Invalid existing archive: {output}')
            print(f'Reusing verified {output}', flush=True)
            continue
        command = [sys.executable, str(script)]
        for key, value in expected.items():
            command += ['--' + key.replace('_', '-'), str(value)]
        command += ['--output', str(output)]
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
