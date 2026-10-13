"""Audit finer full-duration T choices across the fixed hardware bank."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.empirical.hardware_pilot.hardware_pilot import audit_record, sha

TARGETS = (2500, 3125, 4000, 5000)


def audit(path):
    with np.load(path) as bank:
        x, fs, step, record = bank['x'], float(bank['fs']), int(bank['step']), int(bank['record'])
    rows, _, _ = audit_record(x, fs, targets=TARGETS)
    print(f'Audited {step}: {[(r["T"], r["passes"]) for r in rows]}', flush=True)
    return [dict(r, step=step, record=record, voltage=.005 * step) for r in rows], dict(path=str(path), sha256=sha(path))


def main(raw, output, workers):
    paths = sorted(raw.glob('step-???.npz'))
    if len(paths) != 29:
        raise ValueError(f'Expected the fixed 29-record bank, found {len(paths)}')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(audit, paths))
    frame = pd.DataFrame([row for rows, _ in results for row in rows])
    passed = [t for t in TARGETS if frame.loc[frame['T'] == t, 'passes'].all()]
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / 'sampling-audit.csv', index=False)
    report = dict(targets=TARGETS, records=len(paths), smallest_passing_T=min(passed) if passed else None,
                  source_files=[source for _, source in results], script_sha256=sha(__file__),
                  scope='Full 10 s, all 28 channels, same tolerances. Passing physical checks does not establish SPI equivalence.')
    (output / 'sampling-decision.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    main(args.raw, args.output, args.workers)
