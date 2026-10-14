"""Audit the active pilot/cohort selections without treating unfinished work as failure."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from mtsspi.corpus import ArchiveCorpus
from .acquire import sha256
from .release import EXPECTED_SOURCE, validate_table


def audit(root, cohort=False):
    root=Path(root)
    assert os.environ.get('PYSPI_SOURCE') == EXPECTED_SOURCE
    selection=root/'active-configs.json' if cohort else Path(__file__).parent/'configs/analysis-selection.json'
    chosen=json.loads(selection.read_text())
    configs=[Path(p) for p in chosen] if cohort else [root/'corpora'/p/'external.yaml' for p in chosen['corpora']]
    rows=[]
    for config in configs:
        assert config.resolve()==(root/'corpora'/config.parent.name/'external.yaml').resolve()
        corpus=ArchiveCorpus.from_file(config)
        assert corpus.pair_tolerance is None
        for entry in corpus.entries():
            row=dict(corpus=config.parent.name,index=entry.index,name=entry.name,M=entry.M,T=entry.T)
            if not (entry.output_dir(corpus)/'meta.json').exists():
                row['status']='pending'
            else:
                try:
                    meta,coverage=validate_table(corpus,entry)
                    row.update(status='valid',finite_edge_fraction=coverage,workers=meta['job']['n_jobs'],
                               compute_seconds=meta['job']['compute_seconds'],failed_spis=len(meta['pyspi']['errors']))
                except Exception as exc:
                    row.update(status='invalid',error=f'{type(exc).__name__}: {exc}')
            rows.append(row)
    assert len(rows)==(83 if cohort else 19)
    report=dict(checked_utc=datetime.now(timezone.utc).isoformat(),root=str(root),pyspi_source=EXPECTED_SOURCE,
                selection_sha256=sha256(selection),script_sha256=sha256(__file__),counts=dict(Counter(r['status'] for r in rows)),rows=rows)
    (root/'audit-latest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(root.name,report['counts'],flush=True)
    return report['counts'].get('invalid',0)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--parent',type=Path,required=True)
    args=parser.parse_args()
    failures=audit(args.parent/'ccep-pilot-261014')+audit(args.parent/'ccep-cohort-261014',True)
    raise SystemExit(bool(failures))
