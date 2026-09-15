"""Freeze primary-layout extraction tasks from verified HCP cohort windows."""
import argparse
import hashlib
import json
from pathlib import Path

from src.run_external_corpus import ExternalCorpusConfig, load_inventory


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root, length):
    audit_path = root/'cohort-prepared-audit.json'
    audit = json.loads(audit_path.read_text())
    assert audit['status'] == 'verified' and not audit['errors']
    tasks, sources = [], []
    for run in audit['runs']:
        directory = root/'cohort-prepared'/f"{run['subject']}_{run['run']}"/f'm32-T{length}'
        manifest_path = directory/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        contract = manifest['contract']
        assert contract['cohort_audit_sha256'] == sha(audit_path)
        assert contract['subject'] == run['subject'] and contract['run'] == run['run']
        assert contract['T'] == length and contract['M'] == 32
        assert contract['source_archive_sha256'] == run['archive_sha256']
        config_path = directory/'external.yaml'
        assert sha(config_path) == manifest['external_config_sha256']
        config = ExternalCorpusConfig.from_file(config_path)
        assert sha(config.archive) == config.archive_sha256 == manifest['archive_sha256']
        entries = load_inventory(config)
        assert len(entries) == len(manifest['rows']) == run['views']
        selected = []
        for entry, row in zip(entries, manifest['rows'], strict=True):
            assert entry.name == row['name'] and entry.M == 32 and entry.T == length
            if row['layout'] != 'coverage_a':
                continue
            selected.append(entry.index)
            tasks.append(dict(subject=run['subject'], run=run['run'], block=row['block'],
                              config=str(config_path), job_index=entry.index, name=entry.name,
                              output=str(entry.output_dir(config))))
        assert len(selected) == run['clean_blocks']
        sources.append(dict(config=str(config_path), manifest_sha256=sha(manifest_path),
                            archive_sha256=config.archive_sha256, selected_indices=selected))
    assert len(tasks) == audit['clean_blocks']
    assert len({r['output'] for r in tasks}) == len(tasks)
    # Same two runs used for the independently verified preprocessing smoke;
    # choose first available chronological primary block, without viewing scores.
    smoke = [next(i for i,t in enumerate(tasks) if t['subject'] == person and t['run'] == run)
             for person,run in [('112920','6-Wrkmem'),('599671','7-Wrkmem')]]
    plan = dict(status='frozen_extraction_inputs', M=32, T=length, layout='coverage_a',
                tasks=tasks, smoke_task_offsets=smoke, sources=sources,
                cohort_audit_sha256=sha(audit_path), script_sha256=sha(Path(__file__)),
                scope='No family split or model prediction; alternate layout extraction restricted to future confirmation participants.')
    output = root/f'cohort-T{length}-extraction.json'
    if output.exists():
        assert json.loads(output.read_text()) == plan, 'Preserve existing extraction plan'
    else:
        output.write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps(dict(plan=str(output), tasks=len(tasks), smoke=[tasks[i] for i in smoke]), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--length', type=int, choices=[2000,4000], required=True)
    args = parser.parse_args()
    main(args.root, args.length)
