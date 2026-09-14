"""Check all staged E-Prime/TIM label matches using compact downloaded metadata."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import tarfile

import numpy as np
import pandas as pd
from scipy.io import loadmat

from src.hcp_run_audit import align_eprime


def main(root):
    downloads = json.loads((root/'cohort-stage/download-verification.json').read_text())
    assert downloads['status'] == 'complete'
    expected = {Path(row['key']).name: row for row in downloads['files'] if row['key'].endswith('.tab')}
    reference_fs = json.loads((root/'continuous-audit.json').read_text())['sfreq']
    qc = json.loads((root/'cohort-qc-audit.json').read_text())
    counts = {(row['subject'], row['run']): row['retained_task_rows'] for row in qc['runs']}
    rows, errors = [], []
    with tarfile.open(root/'cohort-stage/eprime.tar.gz') as archive:
        assert set(archive.getnames()) == set(expected) and len(expected) == 48
        for name, source in sorted(expected.items()):
            match = re.fullmatch(r'(\d{6})_MEG(?:\d+)?_Wrkmem_run([12])\.tab', name)
            assert match
            subject, run = match[1], f'{int(match[2])+5}-Wrkmem'
            content = archive.extractfile(name).read()
            assert hashlib.sha256(content).hexdigest() == source['sha256']
            frame = pd.read_csv(BytesIO(content), sep='\t')
            path = root/f'cohort-qc/{subject}_MEG_{run}_tmegpreproc_trialinfo.mat'
            trl = loadmat(path, simplify_cells=True)['trlInfo']
            task = trl['lockTrl'][list(trl['lockNames']).index('TIM')]
            task = task[np.isin(task[:, 3], [1, 2]) & np.isin(task[:, 4], [1, 2]) & np.isin(task[:, 8], np.arange(1, 11))]
            assert len(task) == counts[(subject, run)]
            try:
                result = align_eprime(task, frame, reference_fs)
                rows.append(dict(subject=subject, run=run, **result,
                                 eprime_sha256=source['sha256'], tim_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            except Exception as error:
                errors.append(dict(subject=subject, run=run, error_type=type(error).__name__, detail=str(error)))
    report = dict(status='labels_verified' if not errors else 'requires_review', rows=rows, errors=errors,
                  retained_events=sum(row['retained_events'] for row in rows),
                  reference_sampling_hz=reference_fs,
                  timing_limit='Clock diagnostics use the scout sampling rate; verify each raw header before treating these as final timing values.',
                  helper_sha256=hashlib.sha256(Path('src/hcp_run_audit.py').read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'cohort-stage/label-verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], runs=len(rows), errors=errors,
                         retained_events=report['retained_events'],
                         maximum_reference_clock_residual=max((r['max_affine_timing_residual_seconds'] for r in rows), default=None)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
