"""Prepare and release the 45-record, all-edge hardware rerun after a measured smoke."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd
import yaml

from mtsspi.provenance import array_sha256, atomic_json, atomic_savez, file_sha256

VERSION = '3.0.0.r9'
PYSPI_REVISION = 'fd0de716ff28f6a3e8eb6ba91b6fdb991ce308e0'
WORKERS = 4
SMOKE_INDEX = 34  # Small-world setting 28: just before the independently observed locking interval.


def prepare(pilot, out, tiny_archive):
    """Reuse audited full-duration arrays without changing their preprocessing."""
    from mtsspi.corpus import ArchiveCorpus
    from mtsspi.run_corpus import execution_identity
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise FileExistsError('Prepared bank already exists; inspect instead of overwriting')
    arrays, rows = {}, []
    er = pilot / 'sweep29'
    manifest = json.loads((er / 'manifest.json').read_text())
    if file_sha256(er / 'observations.npz') != manifest['archive_sha256']:
        raise ValueError('ER archive hash mismatch')
    er_audit = pd.read_csv(pilot / 'scope-er-transition/sampling-audit.csv').query('T == 5000')
    if len(er_audit) != 29 or not er_audit.passes.all():
        raise ValueError('ER sampling audit incomplete or failed')
    with np.load(er / 'observations.npz') as bank:
        for row in manifest['rows']:
            if row['step'] not in set(er_audit.step):
                raise ValueError('ER record missing from sampling audit')
            arrays[row['name']] = bank[row['name']]
            rows.append(dict(row, topology='ER', source_archive_sha256=manifest['archive_sha256']))
    sw = pilot / 'transfer-sw-261014'
    decision = json.loads((sw / 'sampling-decision.json').read_text())
    if decision['selected_T'] != 5000 or len(decision['steps']) != 16:
        raise ValueError('Unexpected small-world sampling decision')
    for step in decision['steps']:
        path = sw / 'prepared' / f'step-{step:03d}.npz'
        receipt = json.loads(path.with_suffix('.json').read_text())
        audits = [a for a in receipt['audits'] if a['T'] == 5000]
        if (file_sha256(path) != receipt['prepared_sha256'] or len(audits) != 1
                or not audits[0]['passes'] or receipt['record'] != 17451530 or receipt['step'] != step):
            raise ValueError(f'Invalid small-world preparation: {path}')
        name = f'r17451530-s{step:03d}-t5000'
        with np.load(path) as bank:
            arrays[name] = bank['x_5000']
        rows.append(dict(name=name, record=17451530, topology='SW', step=step, voltage=step*.005,
                         M=28, T=5000, fs=500, seconds=10,
                         source_sha256=receipt['source_sha256'], prepared_sha256=receipt['prepared_sha256']))
    if len(rows) != 45 or rows[SMOKE_INDEX-1]['topology'] != 'SW' or rows[SMOKE_INDEX-1]['step'] != 28:
        raise ValueError('Unexpected recording panel or smoke identity')
    for row in rows:
        x = arrays[row['name']]
        if x.shape != (28, 5000) or not np.isfinite(x).all() or np.any(x.std(axis=1) < 1e-8):
            raise ValueError(f'Invalid data: {row["name"]}')
        row['member_sha256'] = array_sha256(x)
    arrays.update(__dataset_names__=np.array([r['name'] for r in rows]),
                  __labels_json__=np.array([json.dumps([r['topology'], str(r['record']), f"step-{r['step']}"]) for r in rows]),
                  __shapes__=np.array([(28, 5000)]*45), __axis_order__=np.array(['process', 'observation']))
    archive = out / 'observations.npz'
    atomic_savez(archive, arrays)
    config = dict(name='hardware-full-p90-r9-t5000',
                  source=dict(format='named-npz-v1', archive=str(archive.resolve()), sha256=file_sha256(archive),
                              axis_order=['process', 'observation']),
                  base_output_dir=str((out / 'mpis').resolve()),
                  pyspi_config='configs/pyspi/benchmarked_p90.yaml', normalise=False, random_seed=261012)
    (out / 'external.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    corpus = ArchiveCorpus.from_file(out / 'external.yaml')
    identity = execution_identity(corpus)
    if identity['pyspi_version']['computation'] != VERSION:
        raise ValueError('Unexpected pyspi computation version')
    atomic_json(out / 'execution-identity.json', identity)
    atomic_json(out / 'manifest.json', dict(rows=rows, archive_sha256=config['source']['sha256'],
                pyspi_revision=PYSPI_REVISION, computation=VERSION, smoke_index=SMOKE_INDEX,
                standardization='Unchanged audited per-channel full-record z scores; all 28 channels, all dyads',
                sampling_audits={str(p):file_sha256(p) for p in [pilot/'scope-er-transition/sampling-audit.csv',
                                                               sw/'sampling-audit.csv']}))
    for label in ('serial', 'parallel'):
        tiny = dict(config, name=f'hardware-r9-tiny-{label}', base_output_dir=str((out/'tiny-mpis').resolve()),
                    source=dict(format='named-npz-v1', archive=str(tiny_archive), sha256=file_sha256(tiny_archive),
                                axis_order=['process', 'observation']))
        (out / f'tiny-{label}.yaml').write_text(yaml.safe_dump(tiny, sort_keys=False))
    print(json.dumps(dict(prepared=len(rows), smoke_index=SMOKE_INDEX, identity=identity)), flush=True)


def check_tiny(out):
    from mtsspi.corpus import ArchiveCorpus
    from mtsspi.run_corpus import completion_error
    banks = []
    for label in ('serial', 'parallel'):
        corpus = ArchiveCorpus.from_file(out/f'tiny-{label}.yaml')
        entries = corpus.entries()
        for entry in entries[:1 if label == 'serial' else 2]:
            if error := completion_error(corpus, entry):
                raise ValueError(error)
        banks.append(entries[0].output_dir(corpus))
    maximum = 0.
    with np.load(banks[0]/'spi_mpis.npz') as serial, np.load(banks[1]/'spi_mpis.npz') as parallel:
        if serial.files != parallel.files:
            raise ValueError('Serial/parallel catalogues differ')
        for name in serial.files:
            a, b = serial[name], parallel[name]
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-10, equal_nan=True, err_msg=name)
            finite = np.isfinite(a)
            if finite.any(): maximum = max(maximum, float(np.max(abs(a[finite]-b[finite]))))
        count = len(serial.files)
    atomic_json(out/'parallel-validation.json', dict(spis=count, maximum_absolute_difference=maximum,
                rtol=1e-6, atol=1e-10, workers=[1,WORKERS], equal_nan_masks=True,
                scope='One tiny recording at both worker counts, plus a second parallel tiny recording'))


def validate_record(corpus, entry):
    from mtsspi.run_corpus import completion_error
    if error := completion_error(corpus, entry):
        raise ValueError(f'{entry.name}: {error}')
    directory = entry.output_dir(corpus)
    meta = json.loads((directory/'meta.json').read_text())
    if meta['pyspi']['version']['computation'] != VERSION or meta['job']['n_jobs'] != WORKERS:
        raise ValueError('Unexpected computation version or worker count')
    with np.load(directory/'spi_mpis.npz') as bank:
        edge = ~np.eye(28, dtype=bool)
        coverage = float(np.mean([np.isfinite(bank[name][edge]).mean() for name in bank.files]))
        if len(bank.files) != 289 or coverage < .9 or len(meta['pyspi']['errors']) > 15:
            raise ValueError(f'Insufficient SPI coverage: {len(bank.files)} SPIs, fraction {coverage}')
    return meta, coverage


def resource_plan(peak_mib, seconds, count):
    """Four workers, measured whole-job memory plus 25%, disjoint single-node batches."""
    if not np.isfinite([peak_mib, seconds]).all() or min(peak_mib, seconds) <= 0:
        raise ValueError('Invalid resource measurement')
    per_record = max(8, math.ceil(peak_mib / 1024 * 1.25))
    capacity = min(48 // WORKERS, 190 // per_record)
    if capacity < 1: raise ValueError('Recording exceeds normal-node memory')
    groups = min(count, capacity)
    wall = max(1800, math.ceil(2*seconds/60)*60)
    if wall > 48*3600: raise ValueError('Required runtime exceeds queue limit')
    return dict(workers=WORKERS, groups=groups, mem_gb=groups*per_record,
                walltime=f'{wall//3600:02d}:{wall//60%60:02d}:00', per_record_gb=per_record)


def submitted(command, env=None):
    result = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
    ids = re.findall(r'\b\d+\.gadi-pbs\b', result.stdout)
    if len(ids) != 1: raise RuntimeError(f'Ambiguous submission: {result.stdout} {result.stderr}')
    return ids[0]


def release(out, smoke_job):
    from mtsspi.corpus import ArchiveCorpus
    from mtsspi.run_corpus import completion_error
    config = out/'external.yaml'
    corpus = ArchiveCorpus.from_file(config)
    entries = corpus.entries()
    if len(entries) != 45 or corpus.pair_tolerance is not None:
        raise ValueError('Unexpected scope')
    # Only release from the successful, single-recording measurement job.
    job = json.loads(subprocess.check_output(['qstat','-xf','-F','json',smoke_job],text=True))['Jobs'][smoke_job]
    variables = job['Variable_List']
    if (job['job_state'] != 'F' or int(job['Exit_status']) != 0
            or Path(variables['RUN_ROOT']).resolve() != out.resolve()
            or variables['PYSPI_SOURCE'] != os.environ['PYSPI_SOURCE']):
        raise ValueError('Smoke provenance or completion mismatch')
    if not (out/'parallel-validation.json').is_file(): raise ValueError('Missing concurrency validation')
    metadata, coverage = validate_record(corpus, entries[SMOKE_INDEX-1])
    memory = re.fullmatch(r'([0-9.]+)(kb|mb|gb|b)',job['resources_used']['mem'].lower())
    if not memory: raise ValueError('Unrecognised PBS memory units')
    peak = float(memory[1])*{'b':1/2**20,'kb':1/1024,'mb':1,'gb':1024}[memory[2]]
    seconds = metadata['job']['compute_seconds']
    folder = out/'dispatch'; folder.mkdir(exist_ok=True)
    # Exclusive receipts prohibit duplicate submission, including ambiguous outcomes.
    previous = list(folder.glob('batch-*/receipt.json'))
    if previous: raise FileExistsError('Dispatch receipts exist; reconcile their jobs before retrying')
    remaining = [e.index for e in entries if completion_error(corpus,e)]
    capacity = resource_plan(peak,seconds,len(remaining))['groups']
    if not remaining: return
    launched = []
    for start in range(0,len(remaining),capacity):
        indices = remaining[start:start+capacity]
        batch = folder/f'batch-{start//capacity}';batch.mkdir()
        index_file = batch/'indices.txt';index_file.write_text(''.join(f'{i}\n' for i in indices))
        plan = resource_plan(peak,seconds,len(indices))
        record = dict(status='submitting',indices=indices,resources=plan,smoke_job=smoke_job,
                      smoke_peak_mib=peak,smoke_seconds=seconds,smoke_coverage=coverage)
        receipt = batch/'receipt.json'
        with receipt.open('x') as stream: json.dump(record,stream,indent=2)
        env = dict(os.environ,MEM_GB=str(plan['mem_gb']),WORKERS=str(plan['groups']),
                   CORES_PER_RECORDING=str(WORKERS),WALLTIME=plan['walltime'])
        env.pop('NCPUS',None)
        try:
            record.update(job=submitted(['bash','jobs/gadi/submit_farm.sh',str(config),str(index_file)],env),status='submitted')
            atomic_json(receipt,record);launched.append(record)
        except Exception as exc:
            record.update(status='requires-inspection',error=str(exc));atomic_json(receipt,record);raise
    atomic_json(out/'release.json',dict(status='released',smoke_job=smoke_job,batches=launched,
                total_recordings=45,computation=VERSION,pyspi_revision=PYSPI_REVISION))
    print(json.dumps(dict(released=len(remaining),batches=launched)),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','check-tiny','release'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--pilot',type=Path);parser.add_argument('--tiny-archive',type=Path)
    parser.add_argument('--smoke-job')
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.pilot,args.root,args.tiny_archive)
    elif args.command=='check-tiny':check_tiny(args.root)
    else:release(args.root,args.smoke_job)
