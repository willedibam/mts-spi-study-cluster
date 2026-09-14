"""Retrieve small open-access HCP QC/label metadata; no raw waveforms or family data."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import boto3
from botocore.config import Config


def main(root):
    inventory=json.loads((root/'cohort-inventory.json').read_text())
    client=boto3.Session(profile_name='hcp').client('s3',region_name='us-east-1',
        config=Config(max_pool_connections=16,retries={'mode':'standard','max_attempts':5}))
    output=root/'cohort-qc';output.mkdir(exist_ok=True)
    objects=[]
    for participant in inventory['rows']:
        subject=participant['subject']
        for run in participant['runs']:
            stem=f"{subject}_MEG_{run['run']}"
            for folder,suffix in [('baddata','baddata_badchannels.txt'),('baddata','baddata_badsegments.txt'),
                                  ('baddata','baddata_rawtrialinfo_QC.txt'),('icaclass','icaclass_vs.txt'),
                                  ('tmegpreproc','tmegpreproc_trialinfo.mat')]:
                objects.append(dict(subject=subject,run=run['run'],kind=suffix,
                    key=f'HCP_1200/{subject}/MEG/Wrkmem/{folder}/{stem}_{suffix}',path=str(output/f'{stem}_{suffix}')))
    def fetch(row):
        path=Path(row['path'])
        response=client.get_object(Bucket='hcp-openaccess',Key=row['key'])
        content=response['Body'].read()
        if len(content)!=response['ContentLength']:raise ValueError('Length mismatch')
        etag=response['ETag'].strip('"')
        if '-' not in etag and hashlib.md5(content).hexdigest()!=etag:raise ValueError('ETag mismatch')
        if path.exists():assert hashlib.sha256(path.read_bytes()).digest()==hashlib.sha256(content).digest()
        else:path.write_bytes(content)
        return dict(row,size=len(content),etag=etag,sha256=hashlib.sha256(content).hexdigest())
    rows,errors=[],[]
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures={pool.submit(fetch,row):row for row in objects}
        for i,future in enumerate(as_completed(futures),1):
            try:rows.append(future.result())
            except Exception as exc:
                row=futures[future];code=getattr(exc,'response',{}).get('Error',{}).get('Code',type(exc).__name__)
                errors.append(dict(key=row['key'],subject=row['subject'],run=row['run'],error_code=code))
            if i%100==0:print(f'QC objects checked {i}/{len(objects)}, errors {len(errors)}',flush=True)
    report=dict(status='complete' if not errors else 'partial',expected=len(objects),downloaded=len(rows),
                bytes=sum(r['size'] for r in rows),rows=sorted(rows,key=lambda r:r['key']),errors=errors,
                query_unix=time.time(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'cohort-qc-downloads.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['rows','errors']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
