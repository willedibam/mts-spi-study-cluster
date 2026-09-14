"""Audit and prepare one staged HCP run using its own verified metadata/ICA."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.audit_hcp_continuous import main as audit_run
from scripts.prepare_hcp_m32 import main as prepare_run


def main(cohort_root, metadata_root, output_root, subject, run):
    metadata_manifest = json.loads((metadata_root/'manifest.json').read_text())
    for row in metadata_manifest['rows']:
        if row['subject'] == subject and row['run'] == run:
            assert hashlib.sha256((metadata_root/row['name']).read_bytes()).hexdigest() == row['sha256']
    files = []
    source_reports = {}
    for part in range(4):
        path = cohort_root/f'part-{part}/download-verification.json'
        report = json.loads(path.read_text())
        assert report['status'] == 'complete'
        source_reports[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        files.extend(report['files'])
    raw_prefix = f'HCP_1200/{subject}/unprocessed/MEG/{run}/'
    raw = [row for row in files if row['key'].startswith(raw_prefix)]
    ica_key = f'HCP_1200/{subject}/MEG/Wrkmem/icaclass/{subject}_MEG_{run}_icaclass.mat'
    ica = [row for row in files if row['key'] == ica_key]
    assert len(raw) == 3 and len(ica) == 1
    for row in raw+ica:
        assert Path(row['path']).stat().st_size == row['size']
    root = output_root/f'{subject}_{run}'
    identity = dict(subject=subject, run=run, raw_sha256={r['key']:r['sha256'] for r in raw+ica},
                    metadata_manifest_sha256=hashlib.sha256((metadata_root/'manifest.json').read_bytes()).hexdigest(),
                    code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in map(Path,
                        ['scripts/audit_hcp_continuous.py', 'scripts/prepare_hcp_m32.py', 'src/hcp_run_audit.py'])})
    identity_path = root/'preparation-identity.json'
    if identity_path.exists():
        assert json.loads(identity_path.read_text()) == identity, 'Preserve a different prepared run'
    prepared = root/'m32-spatial/preparation.json'
    if prepared.exists():
        report = json.loads(prepared.read_text())
        archive = root/'m32-spatial/views.npz'
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == report['archive_sha256']
        print(f'VERIFIED_EXISTING {subject} {run}', flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    identity_path.write_text(json.dumps(identity, indent=2)+'\n')
    (root/'ica').mkdir(exist_ok=True)
    link = root/'metadata'
    if link.is_symlink():
        assert link.resolve() == metadata_root.resolve()
    else:
        assert not link.exists()
        link.symlink_to(metadata_root.resolve(), target_is_directory=True)
    for path, rows in [(root/'download-verification.json', raw), (root/'ica/download-verification.json', ica)]:
        payload = dict(status='complete', files=rows, original_report_sha256=source_reports,
                       scope='Subset of verified staged downloads; raw data not recopied')
        path.write_text(json.dumps(payload, indent=2)+'\n')
    audit_run(root, subject, run)
    prepare_run(root, subject, run)
    print(f'PREPARED {subject} {run}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort-root', type=Path, required=True)
    parser.add_argument('--metadata-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--subject', required=True)
    parser.add_argument('--run', choices=['6-Wrkmem', '7-Wrkmem'], required=True)
    args = parser.parse_args()
    main(args.cohort_root, args.metadata_root, args.output_root, args.subject, args.run)
