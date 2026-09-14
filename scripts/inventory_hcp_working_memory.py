"""Read-only S3 cohort inventory; no waveforms, keys or signed URLs are logged."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import boto3
from botocore.config import Config


def main(output, profile):
    client = boto3.Session(profile_name=profile).client('s3', region_name='us-east-1',
            config=Config(max_pool_connections=16, retries={'mode':'standard','max_attempts':5}))
    paginator = client.get_paginator('list_objects_v2')
    subjects = []
    for page in paginator.paginate(Bucket='hcp-openaccess', Prefix='HCP_1200/', Delimiter='/'):
        subjects.extend(p['Prefix'].split('/')[1] for p in page.get('CommonPrefixes',[]) if p['Prefix'].split('/')[1].isdigit())
    subjects = sorted(set(subjects))

    def inspect(subject):
        prefix=f'HCP_1200/{subject}/unprocessed/MEG/'
        response=client.list_objects_v2(Bucket='hcp-openaccess',Prefix=prefix,Delimiter='/')
        assert not response.get('IsTruncated',False)
        runs=[p['Prefix'] for p in response.get('CommonPrefixes',[]) if 'Wrkmem' in p['Prefix']]
        selected=[]
        for run in runs:
            response=client.list_objects_v2(Bucket='hcp-openaccess',Prefix=run)
            assert not response.get('IsTruncated',False)
            files=[dict(key=r['Key'],size=r['Size'],etag=r['ETag']) for r in response.get('Contents',[])
                   if r['Key'].endswith('/c,rfDC') or r['Key'].endswith('/config') or '/EPRIME/' in r['Key'] and r['Key'].endswith('.tab')]
            selected.append(dict(run=run.split('/')[-2],files=files,
                                 has_raw=any(r['key'].endswith('/c,rfDC') for r in files),
                                 has_config=any(r['key'].endswith('/config') for r in files),
                                 has_eprime=any(r['key'].endswith('.tab') for r in files)))
        return dict(subject=subject,runs=selected)

    output.parent.mkdir(parents=True,exist_ok=True)
    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures={pool.submit(inspect,s):s for s in subjects}
        for i,future in enumerate(as_completed(futures),1):
            try:
                row=future.result()
                if row['runs']:rows.append(row)
            except Exception as exc:
                errors.append(dict(subject=futures[future],error_type=type(exc).__name__))
            if i % 100 == 0:print(f'Checked {i}/{len(subjects)} participant prefixes; working-memory found {len(rows)}; errors {len(errors)}',flush=True)
    rows.sort(key=lambda r:r['subject'])
    complete=[r for r in rows if any(x['has_raw'] and x['has_config'] and x['has_eprime'] for x in r['runs'])]
    result=dict(status='complete' if not errors else 'partial',bucket='hcp-openaccess',release='HCP_1200',
                checked_participant_prefixes=len(subjects),participants_with_working_memory=len(rows),
                participants_with_at_least_one_raw_config_eprime_run=len(complete),
                participants_with_two_raw_config_eprime_runs=sum(sum(x['has_raw'] and x['has_config'] and x['has_eprime'] for x in r['runs'])>=2 for r in rows),
                rows=rows,errors=errors,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                limitations='Availability inventory only; not QC eligibility, independent-family count, verified dimensions or labels. No waveforms downloaded.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','errors']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile',default='hcp')
    args=parser.parse_args()
    main(args.output,args.profile)
